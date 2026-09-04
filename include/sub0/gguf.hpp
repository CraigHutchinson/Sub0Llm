// sub0/gguf.hpp -- minimal, engine-free GGUF file reader + import-compatibility report.
//
// SCOPE (deliberately narrow -- this is step 1 of a staged open-weight-import investigation, not an
// importer): parse a GGUF file's header, key/value metadata, and tensor table, then compare the
// tensor names/shapes against what a Llama-1-style (plain multi-head attention, no grouped-query
// attention) model would need, reporting concrete matches/mismatches. It does NOT read tensor data
// payloads, does NOT decode quantized formats, and does NOT write a Sub0Llm checkpoint -- those are
// later, separate steps once this report has actually been used to pick a real candidate file.
//
// Pure std + <cstdint>, no dependency on the generated config header, so it lives in the engine-free
// frontend layer alongside memplan.hpp/config_util.hpp and is unit-testable without a compiled model.
//
// Format reference (GGUF v2/v3, https://github.com/ggml-org/ggml/blob/master/docs/gguf.md): magic
// "GGUF" + version (u32) + tensor_count (u64) + metadata_kv_count (u64), then that many KV pairs
// (string key, typed value), then that many tensor-info entries (string name, dims, ggml dtype,
// data offset), then the (alignment-padded) tensor data section this reader never touches.

#pragma once

#include "sub0/gguf_quant_tables.hpp"   // the ggml IQ codebooks -- data only, see that file's header

#include <cstdint>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sub0::gguf {

// --- wire format enums (as defined by the GGUF spec, not this project) -----------------------

enum class ValueType : std::uint32_t {
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5,
    Float32 = 6, Bool = 7, String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
};

// GGML tensor element type IDs this reader recognizes. F32/F16/BF16 are plain storage (read as-is or
// widened); the rest are block-quantized formats this reader can DEQUANTIZE (see the decoders at the
// bottom of this file). Anything NOT listed here is still reported by its raw id, so a caller says
// "quantized, not supported yet" rather than silently misreading bytes.
//
// WHY THIS EXACT SET (docs/WP4_SCOPE.md S3a-bis): Q8_0 came first, as the simplest ggml quant format
// and what the original validated import candidate (llama-160m) shipped. The other seven are the
// formats the REAL Qwen3.8-Flash-Next UD-IQ1_S file actually uses -- read out of the downloaded file's
// own tensor table, not guessed: every raw type id present across its two tensor-bearing shards is one
// of {F32, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ1_S, IQ4_NL, BF16}. Note there is no F16 in that file at
// all; F16 stays because it is the format's own baseline and Q8_0's block scale is an f16 regardless.
//
// A caller must dispatch on a tensor's OWN type_raw, never on its name/role: unsloth's "Dynamic" (UD)
// mixed quantization gives the SAME logical tensor different formats at different layers (layers 0/3's
// ffn_gate_exps is IQ1_S, layers 1/2's is IQ2_XXS -- observed directly in the real file). The
// per-TensorInfo dispatch every function here uses already has that shape.
enum class TensorType : std::uint32_t {
    F32 = 0, F16 = 1, Q8_0 = 8,
    Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ1_S = 19, IQ4_NL = 20, BF16 = 30,
};

// One quantization format's block geometry: `elems` elements encoded in `bytes` bytes. The plain
// storage types are expressed as a degenerate 1-element "block" so ONE rule sizes every format and
// tensor_byte_size() has no special cases. {0, 0} means "this reader does not know this type".
//
// Every byte figure below is derived from ggml-common.h's own struct definitions (transcribed, not
// recalled -- AGENTS.md S5), and every one was then CHECKED against the real UD-IQ1_S file by
// differencing consecutive tensors' data offsets, which is an independent oracle the struct
// definitions cannot fake:
//   Q4_K   2*f16 + 12 scale bytes + 128 quant bytes                    = 144 / 256
//   Q5_K   2*f16 + 12 scale bytes + 32 high-bit bytes + 128 low bytes  = 176 / 256
//   Q6_K   128 low + 64 high + 16 int8 scales + f16                    = 210 / 256
//   Q8_0   f16 + 32 int8                                              =  34 /  32
//   IQ1_S  f16 + 32 grid-index bytes + 8*u16 qh                       =  50 / 256
//   IQ2_XXS f16 + 32*u16                                              =  66 / 256
//   IQ4_NL f16 + 16 packed nibbles                                    =  18 /  32
struct BlockSpec { std::uint64_t elems = 0, bytes = 0; };
inline constexpr BlockSpec block_spec(std::uint32_t type_raw) {
    switch (static_cast<TensorType>(type_raw)) {
        case TensorType::F32:     return {1, 4};
        case TensorType::F16:     return {1, 2};
        case TensorType::BF16:    return {1, 2};
        case TensorType::Q8_0:    return {32, 34};
        case TensorType::Q4_K:    return {256, 144};
        case TensorType::Q5_K:    return {256, 176};
        case TensorType::Q6_K:    return {256, 210};
        case TensorType::IQ2_XXS: return {256, 66};
        case TensorType::IQ1_S:   return {256, 50};
        case TensorType::IQ4_NL:  return {32, 18};
        default:                  return {};
    }
}

// --- parsed structures -------------------------------------------------------------------------

// A metadata value, narrowed to what this reader's callers actually need: every integer width folds
// into i64/u64 (by signedness), every float width into f64, strings and string-arrays kept verbatim
// (the tokenizer vocab is a string array), and any other array collapsed to just its element count
// (shape/compat checks only need "how many", not the values).
struct Value {
    enum class Kind { Int, UInt, Float, Bool, String, StringArray, OtherArray, Unknown } kind = Kind::Unknown;
    std::int64_t              i = 0;
    std::uint64_t              u = 0;
    double                     f = 0.0;
    bool                       b = false;
    std::string                s;
    std::vector<std::string>   sa;
    std::uint64_t               array_len = 0;   // set for StringArray/OtherArray
};

struct TensorInfo {
    std::string                name;
    std::vector<std::uint64_t> dims;      // ne[0..n_dims), GGML order: dims[0] is the fastest-varying axis
    std::uint32_t               type_raw = 0;   // raw GGML tensor-type id (see TensorType for the recognized subset)
    std::uint64_t                offset = 0;     // byte offset into the (alignment-padded) data section

    bool is(TensorType t) const { return type_raw == static_cast<std::uint32_t>(t); }
    bool is_f32() const { return is(TensorType::F32); }
    bool is_f16() const { return is(TensorType::F16); }
    bool is_bf16() const { return is(TensorType::BF16); }
    bool is_q8_0() const { return is(TensorType::Q8_0); }
    // "Plain" = stored one value per element, no block structure. BF16 joins F32/F16 here: it is a
    // straight 16-bit truncation of an f32, not a quantization, so a caller checking "is this file
    // quantized at all" must not be told yes by a bf16 tensor.
    bool is_plain_float() const { return is_f32() || is_f16() || is_bf16(); }
    std::uint64_t element_count() const {
        std::uint64_t n = 1;
        for (auto d : dims) n *= d;
        return dims.empty() ? 0 : n;
    }
};

// --- reader --------------------------------------------------------------------------------------

class Reader {
public:
    enum class Err { Ok, BadMagic, UnsupportedVersion, Truncated, BadValueType };

    // Parses `data` in place (does not copy or take ownership -- the caller keeps the buffer alive
    // for the Reader's lifetime, e.g. a memory-mapped or fully-read file). Never throws; check ok()/
    // error() before using tensors()/metadata().
    explicit Reader(std::span<const std::uint8_t> data) { parse(data); }

    // Refuse to bind to a TEMPORARY owning buffer. Reader holds spans into `data` and copies nothing,
    // so `Reader r(make_buffer());` leaves every one of them dangling the moment the full-expression
    // ends -- and because the freed pages usually still hold the old bytes, it reads correctly for
    // years and then stops. It did: 14 test sites were written this way, passed throughout C++23, and
    // only broke when a C++26 build changed stack reuse enough to clobber the buffer before it was
    // read (tensor_bytes returned the right SIZE from the wrong memory, so the size assertions still
    // passed and only the decoded values were garbage).
    //
    // Deleting the rvalue overload turns that entire class of mistake into a compile error: an rvalue
    // vector matches this exactly, which beats the user-defined conversion to span, so it is chosen
    // and then rejected. Callers must name the buffer, which is precisely the lifetime they need.
    Reader(std::vector<std::uint8_t>&&) = delete;

    Err  error() const { return err_; }
    bool ok()    const { return err_ == Err::Ok; }
    std::uint32_t version() const { return version_; }

    std::span<const TensorInfo> tensors()  const { return tensors_; }
    const std::map<std::string, Value>& metadata() const { return meta_; }
    const Value* find(std::string_view key) const {
        auto it = meta_.find(std::string(key));
        return it == meta_.end() ? nullptr : &it->second;
    }
    const TensorInfo* find_tensor(std::string_view name) const {
        for (const auto& t : tensors_) if (t.name == name) return &t;
        return nullptr;
    }

    // --- tensor data access (the part step 1's reader deliberately skipped) --------------------
    // Absolute byte offset (into the buffer this Reader was constructed over) where the tensor
    // data section begins: the tensor info table, padded up to `general.alignment` (default 32,
    // per the GGUF spec) -- every TensorInfo::offset is relative to THIS, not to the file start.
    std::uint64_t data_offset() const { return data_offset_; }

    // Byte length of one tensor's raw (still-encoded) data: ceil(n / block.elems) * block.bytes,
    // which collapses to n*4 / n*2 for the plain storage types (their "block" is one element). 0 for
    // a type this reader does not know the size rule for -- see block_spec() for the per-format table
    // and for how each figure was cross-checked against the real file.
    std::uint64_t tensor_byte_size(const TensorInfo& t) const {
        const BlockSpec b = block_spec(t.type_raw);
        if (b.elems == 0) return 0;
        const std::uint64_t n = t.element_count();
        return ((n + b.elems - 1) / b.elems) * b.bytes;
    }

    // The raw (still-encoded) bytes for one tensor, bounds-checked against the buffer -- empty if
    // the type's size is unknown (tensor_byte_size returned 0) or the file is truncated/malformed.
    std::span<const std::uint8_t> tensor_bytes(const TensorInfo& t) const {
        const std::uint64_t sz = tensor_byte_size(t);
        if (sz == 0) return {};
        const std::uint64_t off = data_offset_ + t.offset;
        if (off > buf_.size() || sz > buf_.size() - off) return {};
        return buf_.subspan(static_cast<std::size_t>(off), static_cast<std::size_t>(sz));
    }

private:
    Err                           err_ = Err::Truncated;
    std::uint32_t                  version_ = 0;
    std::uint64_t                  data_offset_ = 0;
    std::map<std::string, Value> meta_;
    std::vector<TensorInfo>      tensors_;

    // --- byte-cursor helpers: every read checks remaining length first, setting err_ = Truncated
    // and leaving the cursor where it is rather than reading past the buffer on a malformed file.
    std::span<const std::uint8_t> buf_;
    std::size_t                    pos_ = 0;

    bool need(std::size_t n) const { return pos_ + n <= buf_.size(); }

    template <class T> bool read_pod(T& out) {
        if (!need(sizeof(T))) return false;
        std::memcpy(&out, buf_.data() + pos_, sizeof(T));
        pos_ += sizeof(T);
        return true;
    }
    bool read_string(std::string& out) {
        std::uint64_t len = 0;
        if (!read_pod(len) || !need(len)) return false;
        out.assign(reinterpret_cast<const char*>(buf_.data() + pos_), static_cast<std::size_t>(len));
        pos_ += static_cast<std::size_t>(len);
        return true;
    }

    // Reads one typed value. `type` is already known (either from a KV entry's own type field, or
    // -- for an ARRAY -- the array's declared element type). Handles ARRAY by reading array_type +
    // array_len then that many elements of array_type (materializing strings, counting everything
    // else) -- GGUF arrays are one level deep, arrays-of-arrays are not part of the spec.
    bool read_value(ValueType type, Value& v) {
        switch (type) {
            case ValueType::UInt8:  { std::uint8_t  x; if (!read_pod(x)) return false; v.kind = Value::Kind::UInt; v.u = x; return true; }
            case ValueType::Int8:   { std::int8_t   x; if (!read_pod(x)) return false; v.kind = Value::Kind::Int;  v.i = x; return true; }
            case ValueType::UInt16: { std::uint16_t x; if (!read_pod(x)) return false; v.kind = Value::Kind::UInt; v.u = x; return true; }
            case ValueType::Int16:  { std::int16_t  x; if (!read_pod(x)) return false; v.kind = Value::Kind::Int;  v.i = x; return true; }
            case ValueType::UInt32: { std::uint32_t x; if (!read_pod(x)) return false; v.kind = Value::Kind::UInt; v.u = x; return true; }
            case ValueType::Int32:  { std::int32_t  x; if (!read_pod(x)) return false; v.kind = Value::Kind::Int;  v.i = x; return true; }
            case ValueType::UInt64: { std::uint64_t x; if (!read_pod(x)) return false; v.kind = Value::Kind::UInt; v.u = x; return true; }
            case ValueType::Int64:  { std::int64_t  x; if (!read_pod(x)) return false; v.kind = Value::Kind::Int;  v.i = x; return true; }
            case ValueType::Float32:{ float          x; if (!read_pod(x)) return false; v.kind = Value::Kind::Float; v.f = x; return true; }
            case ValueType::Float64:{ double         x; if (!read_pod(x)) return false; v.kind = Value::Kind::Float; v.f = x; return true; }
            case ValueType::Bool:   { std::uint8_t  x; if (!read_pod(x)) return false; v.kind = Value::Kind::Bool; v.b = (x != 0); return true; }
            case ValueType::String: { v.kind = Value::Kind::String; return read_string(v.s); }
            case ValueType::Array: {
                std::uint32_t elem_type_raw = 0; std::uint64_t len = 0;
                if (!read_pod(elem_type_raw) || !read_pod(len)) return false;
                const auto elem_type = static_cast<ValueType>(elem_type_raw);
                if (elem_type == ValueType::String) {
                    v.kind = Value::Kind::StringArray; v.array_len = len;
                    v.sa.reserve(static_cast<std::size_t>(len));
                    for (std::uint64_t i = 0; i < len; ++i) { std::string s; if (!read_string(s)) return false; v.sa.push_back(std::move(s)); }
                    return true;
                }
                // Non-string array: skip the payload (fixed-width elements), just record the count.
                v.kind = Value::Kind::OtherArray; v.array_len = len;
                const std::size_t w = value_width(elem_type);
                if (w == 0) return false;                       // nested Array-of-Array: not in the spec
                if (!need(w * static_cast<std::size_t>(len))) return false;
                pos_ += w * static_cast<std::size_t>(len);
                return true;
            }
            default: return false;
        }
    }
    static std::size_t value_width(ValueType t) {
        switch (t) {
            case ValueType::UInt8: case ValueType::Int8: case ValueType::Bool: return 1;
            case ValueType::UInt16: case ValueType::Int16: return 2;
            case ValueType::UInt32: case ValueType::Int32: case ValueType::Float32: return 4;
            case ValueType::UInt64: case ValueType::Int64: case ValueType::Float64: return 8;
            default: return 0;   // String/Array: variable width, caller must special-case
        }
    }

    void parse(std::span<const std::uint8_t> data) {
        buf_ = data; pos_ = 0;
        char magic[4];
        if (!need(4)) { err_ = Err::Truncated; return; }
        std::memcpy(magic, buf_.data(), 4); pos_ = 4;
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') { err_ = Err::BadMagic; return; }
        if (!read_pod(version_)) { err_ = Err::Truncated; return; }
        if (version_ != 2 && version_ != 3) { err_ = Err::UnsupportedVersion; return; }

        std::uint64_t tensor_count = 0, kv_count = 0;
        if (!read_pod(tensor_count) || !read_pod(kv_count)) { err_ = Err::Truncated; return; }

        for (std::uint64_t i = 0; i < kv_count; ++i) {
            std::string key;
            std::uint32_t type_raw = 0;
            if (!read_string(key) || !read_pod(type_raw)) { err_ = Err::Truncated; return; }
            Value v;
            if (!read_value(static_cast<ValueType>(type_raw), v)) { err_ = Err::BadValueType; return; }
            meta_.emplace(std::move(key), std::move(v));
        }

        tensors_.reserve(static_cast<std::size_t>(tensor_count));
        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            TensorInfo t;
            std::uint32_t n_dims = 0;
            if (!read_string(t.name) || !read_pod(n_dims)) { err_ = Err::Truncated; return; }
            t.dims.resize(n_dims);
            for (auto& d : t.dims) if (!read_pod(d)) { err_ = Err::Truncated; return; }
            if (!read_pod(t.type_raw) || !read_pod(t.offset)) { err_ = Err::Truncated; return; }
            tensors_.push_back(std::move(t));
        }

        // Data section starts at pos_, rounded up to general.alignment (u32 KV, default 32 when
        // absent -- per the GGUF spec). Every TensorInfo::offset above is relative to THIS point.
        std::uint64_t alignment = 32;
        if (const auto it = meta_.find("general.alignment"); it != meta_.end()) {
            const Value& av = it->second;
            if (av.kind == Value::Kind::UInt) alignment = av.u;
            else if (av.kind == Value::Kind::Int && av.i > 0) alignment = static_cast<std::uint64_t>(av.i);
        }
        if (alignment == 0) alignment = 32;
        data_offset_ = (static_cast<std::uint64_t>(pos_) + alignment - 1) / alignment * alignment;

        err_ = Err::Ok;
    }
};

// --- dequantization (raw tensor bytes -> F32) -----------------------------------------------------

// IEEE-754 binary16 -> binary32. Standard sign/exponent/mantissa bit-manipulation (subnormal- and
// inf/nan-aware) -- ggml's f16 scale factors (Q8_0's block delta) and any F16-stored tensor both need
// this; there is no <stdfloat> std::float16_t support on this toolchain yet.
inline float f16_to_f32(std::uint16_t h) {
    std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    std::uint32_t exp  = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x3FFu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                                    // +-0
        } else {                                             // subnormal half -> normalized float
            int e = -1;
            do { mant <<= 1; ++e; } while (!(mant & 0x400u));
            mant &= 0x3FFu;
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);           // inf / nan (mantissa preserved -> nan payload)
    } else {
        bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// Dequantize a ggml block_q8_0 buffer: each 32-element block is a little-endian f16 scale `d`
// followed by 32 int8 quants `q`; value[i] = q[i] * d. `raw` must be at least
// ceil(n_elements/32)*34 bytes (Reader::tensor_bytes already sizes it exactly this way). Returns
// false only if `raw` is shorter than that -- a truncated/malformed tensor, not decoded partially.
inline bool dequantize_q8_0(std::span<const std::uint8_t> raw, std::uint64_t n_elements,
                            std::vector<float>& out) {
    constexpr std::uint64_t kBlockElems = 32, kBlockBytes = 34;
    const std::uint64_t blocks = (n_elements + kBlockElems - 1) / kBlockElems;
    if (raw.size() < blocks * kBlockBytes) return false;
    out.resize(static_cast<std::size_t>(n_elements));
    std::uint64_t written = 0;
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const std::uint8_t* blk = raw.data() + b * kBlockBytes;
        std::uint16_t d_bits;
        std::memcpy(&d_bits, blk, sizeof d_bits);
        const float d = f16_to_f32(d_bits);
        const auto* q = reinterpret_cast<const std::int8_t*>(blk + 2);
        for (std::uint64_t i = 0; i < kBlockElems && written < n_elements; ++i, ++written)
            out[static_cast<std::size_t>(written)] = static_cast<float>(q[i]) * d;
    }
    return true;
}

// bfloat16 -> f32. Strictly simpler than f16_to_f32 above and NOT a special case of it: bf16 is an
// f32 with its low 16 mantissa bits dropped, so widening is a shift, with no exponent rebias and no
// subnormal renormalization. (Getting these two confused -- reusing the f16 path for a bf16 tensor --
// would decode plausible-magnitude garbage rather than failing, which is why they are separate
// functions rather than one with a flag.)
inline float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// --- K-quant and IQ-quant decoders ----------------------------------------------------------------
//
// Every function below is a transcription of ggml's own `dequantize_row_*` (ggml/src/ggml-quants.c in
// the llama.cpp checkout at D:\Craig\diffusiongemma\llama.cpp), fetched and read rather than recalled,
// per AGENTS.md S5. There is no convention to re-map here the way a ported ALGORITHM needs (the Muon
// axis-order trap): a decoder either reproduces the format's bit layout or it does not. What IS
// re-derived is the buffer contract -- ggml decodes into a caller-sized `float*` and asserts
// `k % QK_K == 0`; these take a bounds-checked span, size `out` themselves, and tolerate a partial
// final block by writing only the elements that exist (exactly dequantize_q8_0's own behaviour above).
//
// The two K-quant families differ in a way worth stating once, since it is the most likely place for a
// transcription slip: Q4_K/Q5_K carry a per-super-block scale AND MIN (`x = d*sc*q - dmin*m`, an affine
// map with 8 sub-blocks of 32 whose 6-bit scale/min pairs are packed 12-bytes-for-16-values by
// get_scale_min_k4), while Q6_K is scale-only and zero-centered (`x = d*sc*(q - 32)`, 16 sub-blocks of
// 16 with plain int8 scales). Mixing those up produces output with a systematic offset, which a
// statistical mean/std check catches and a shape check does not.

// ggml's get_scale_min_k4: unpacks sub-block j's 6-bit scale and 6-bit min out of the 12 packed bytes
// shared by all 8 sub-blocks of a Q4_K/Q5_K super-block.
inline void k_scale_min(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = static_cast<std::uint8_t>((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4));
    }
}

// Shared preamble for every block decoder: validate the raw span against the format's own size rule,
// size `out`, and hand back the block count. Returns 0 (and leaves `out` untouched) on a short buffer.
inline std::uint64_t begin_blocks(TensorType type, std::span<const std::uint8_t> raw,
                                   std::uint64_t n_elements, std::vector<float>& out) {
    const BlockSpec b = block_spec(static_cast<std::uint32_t>(type));
    if (b.elems == 0) return 0;
    const std::uint64_t blocks = (n_elements + b.elems - 1) / b.elems;
    if (raw.size() < blocks * b.bytes) return 0;
    out.assign(static_cast<std::size_t>(n_elements), 0.f);
    return blocks;
}

inline bool dequantize_q4_k(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    const std::uint64_t blocks = begin_blocks(TensorType::Q4_K, raw, n, out);
    if (blocks == 0) return n == 0;
    std::uint64_t w = 0;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* blk = raw.data() + i * 144;
        std::uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        const float d = f16_to_f32(d_bits), dmin = f16_to_f32(dmin_bits);
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* q = blk + 16;
        for (int is = 0; is < 8; is += 2) {
            std::uint8_t sc, m;
            k_scale_min(is, scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            k_scale_min(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && w < n; ++l, ++w) out[static_cast<std::size_t>(w)] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32 && w < n; ++l, ++w) out[static_cast<std::size_t>(w)] = d2 * (q[l] >> 4) - m2;
            q += 32;
        }
    }
    return true;
}

inline bool dequantize_q5_k(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    const std::uint64_t blocks = begin_blocks(TensorType::Q5_K, raw, n, out);
    if (blocks == 0) return n == 0;
    std::uint64_t w = 0;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* blk = raw.data() + i * 176;
        std::uint16_t d_bits, dmin_bits;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        const float d = f16_to_f32(d_bits), dmin = f16_to_f32(dmin_bits);
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* qh = blk + 16;         // 32 bytes: one high bit per element
        const std::uint8_t* ql = blk + 48;         // 128 bytes: two 4-bit low halves per byte
        std::uint8_t u1 = 1, u2 = 2;               // the qh bit pair advanced per 64-element group
        for (int is = 0; is < 8; is += 2) {
            std::uint8_t sc, m;
            k_scale_min(is, scales, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            k_scale_min(is + 1, scales, sc, m);
            const float d2 = d * sc, m2 = dmin * m;
            for (int l = 0; l < 32 && w < n; ++l, ++w)
                out[static_cast<std::size_t>(w)] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32 && w < n; ++l, ++w)
                out[static_cast<std::size_t>(w)] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            ql += 32;
            u1 = static_cast<std::uint8_t>(u1 << 2);
            u2 = static_cast<std::uint8_t>(u2 << 2);
        }
    }
    return true;
}

inline bool dequantize_q6_k(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    const std::uint64_t blocks = begin_blocks(TensorType::Q6_K, raw, n, out);
    if (blocks == 0) return n == 0;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* blk = raw.data() + i * 210;
        const std::uint8_t* ql = blk;              // 128 bytes
        const std::uint8_t* qh = blk + 128;        // 64 bytes
        const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);   // 16 int8 scales
        std::uint16_t d_bits;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = f16_to_f32(d_bits);
        const std::uint64_t base = i * 256;
        // ggml's loop shape kept verbatim: two 128-element halves, each writing four interleaved
        // 32-element strips at +0/+32/+64/+96 rather than sequentially. Flattening it to a linear
        // walk would reorder the elements, which is exactly the sort of "looks fine" change that
        // produces a plausible tensor with the wrong values.
        for (int half = 0; half < 2; ++half) {
            const std::uint64_t y0 = base + static_cast<std::uint64_t>(half) * 128;
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>(static_cast<std::int8_t>((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4))) - 32;
                const int q2 = static_cast<int>(static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4))) - 32;
                const int q3 = static_cast<int>(static_cast<std::int8_t>((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4))) - 32;
                const int q4 = static_cast<int>(static_cast<std::int8_t>((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4))) - 32;
                const std::uint64_t idx[4] = {y0 + l, y0 + l + 32, y0 + l + 64, y0 + l + 96};
                const float v[4] = {d * sc[is + 0] * q1, d * sc[is + 2] * q2,
                                    d * sc[is + 4] * q3, d * sc[is + 6] * q4};
                for (int k = 0; k < 4; ++k)
                    if (idx[k] < n) out[static_cast<std::size_t>(idx[k])] = v[k];
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
    return true;
}

inline bool dequantize_iq4_nl(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    const std::uint64_t blocks = begin_blocks(TensorType::IQ4_NL, raw, n, out);
    if (blocks == 0) return n == 0;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* blk = raw.data() + i * 18;
        std::uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float d = f16_to_f32(d_bits);
        const std::uint8_t* qs = blk + 2;
        const std::uint64_t base = i * 32;
        // NOTE the split, which is NOT the obvious "nibble pairs are adjacent elements": byte j holds
        // element j in its low nibble and element j+16 in its high nibble.
        for (int j = 0; j < 16; ++j) {
            if (base + j      < n) out[static_cast<std::size_t>(base + j)]      = d * KVALUES_IQ4NL[qs[j] & 0xF];
            if (base + j + 16 < n) out[static_cast<std::size_t>(base + j + 16)] = d * KVALUES_IQ4NL[qs[j] >> 4];
        }
    }
    return true;
}

inline bool dequantize_iq2_xxs(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    const std::uint64_t blocks = begin_blocks(TensorType::IQ2_XXS, raw, n, out);
    if (blocks == 0) return n == 0;
    std::uint64_t w = 0;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* blk = raw.data() + i * 66;
        std::uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float d = f16_to_f32(d_bits);
        const std::uint8_t* qs = blk + 2;          // 8 groups of 8 bytes == 2 u32 each
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            std::uint32_t aux[2];
            std::memcpy(aux, qs + 8 * ib32, 8);
            const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux);
            // The scale's four high bits live in aux[1]'s top nibble; the remaining 28 bits are four
            // 7-bit sign-pattern indices, one per 8-element group.
            const float db = d * (0.5f + static_cast<float>(aux[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = IQ2XXS_GRID[aux8[l]];
                const std::uint8_t signs = KSIGNS_IQ2XS[(aux[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8 && w < n; ++j, ++w) {
                    const auto mag = static_cast<std::uint8_t>((grid >> (8 * j)) & 0xFFu);
                    out[static_cast<std::size_t>(w)] =
                        db * static_cast<float>(mag) * ((signs & KMASK_IQ2XS[j]) ? -1.f : 1.f);
                }
            }
        }
    }
    return true;
}

inline bool dequantize_iq1_s(std::span<const std::uint8_t> raw, std::uint64_t n, std::vector<float>& out) {
    const std::uint64_t blocks = begin_blocks(TensorType::IQ1_S, raw, n, out);
    if (blocks == 0) return n == 0;
    std::uint64_t w = 0;
    for (std::uint64_t i = 0; i < blocks; ++i) {
        const std::uint8_t* blk = raw.data() + i * 50;
        std::uint16_t d_bits;
        std::memcpy(&d_bits, blk, 2);
        const float d = f16_to_f32(d_bits);
        const std::uint8_t* qs = blk + 2;          // 32 grid-index low bytes
        for (int ib = 0; ib < 8; ++ib) {
            std::uint16_t qh;
            std::memcpy(&qh, blk + 34 + 2 * ib, 2);
            // qh packs, per 32-element sub-block: three high grid-index bits per 8-element group at
            // [3l+2 : 3l], a 3-bit scale at [14:12], and the delta's SIGN at bit 15.
            const float dl = d * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
            const float delta = (qh & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
            for (int l = 0; l < 4; ++l) {
                const std::uint64_t grid = IQ1S_GRID[qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8 && w < n; ++j, ++w) {
                    const auto q = static_cast<std::int8_t>((grid >> (8 * j)) & 0xFFu);
                    out[static_cast<std::size_t>(w)] = dl * (static_cast<float>(q) + delta);
                }
            }
        }
    }
    return true;
}

// Convert one tensor's raw (still-encoded) bytes to F32, dispatching on its OWN type_raw (never on its
// name -- see TensorType's comment on unsloth's per-layer mixed quantization). Returns false for any
// type this reader does not know -- the caller should treat that as "cannot import this tensor yet",
// not guess.
inline bool to_f32(const TensorInfo& t, std::span<const std::uint8_t> raw, std::vector<float>& out) {
    const std::uint64_t n = t.element_count();
    if (t.is_f32()) {
        if (raw.size() < n * 4) return false;
        out.resize(static_cast<std::size_t>(n));
        std::memcpy(out.data(), raw.data(), static_cast<std::size_t>(n) * 4);
        return true;
    }
    if (t.is_f16() || t.is_bf16()) {
        if (raw.size() < n * 2) return false;
        out.resize(static_cast<std::size_t>(n));
        const bool bf = t.is_bf16();
        for (std::uint64_t i = 0; i < n; ++i) {
            std::uint16_t h;
            std::memcpy(&h, raw.data() + i * 2, sizeof h);
            out[static_cast<std::size_t>(i)] = bf ? bf16_to_f32(h) : f16_to_f32(h);
        }
        return true;
    }
    switch (static_cast<TensorType>(t.type_raw)) {
        case TensorType::Q8_0:    return dequantize_q8_0(raw, n, out);
        case TensorType::Q4_K:    return dequantize_q4_k(raw, n, out);
        case TensorType::Q5_K:    return dequantize_q5_k(raw, n, out);
        case TensorType::Q6_K:    return dequantize_q6_k(raw, n, out);
        case TensorType::IQ2_XXS: return dequantize_iq2_xxs(raw, n, out);
        case TensorType::IQ1_S:   return dequantize_iq1_s(raw, n, out);
        case TensorType::IQ4_NL:  return dequantize_iq4_nl(raw, n, out);
        default:                  return false;
    }
}

// --- import-compatibility report ------------------------------------------------------------------

// The tensor set a plain (non-gated, non-GQA) Llama-1-style decoder needs, in llama.cpp's naming
// convention -- see the module doc comment. `has_gate_up` true means the file ALSO has fused/gated
// FFN tensors (blk.N.ffn_gate/ffn_up), a strong signal the source model is SwiGLU-gated, not the
// plain 2-matrix FFN this engine currently implements -- see gguf-import-feasibility-review.md.
struct CompatReport {
    bool matches_plain_mha  = true;    // attn_k row count == attn_q row count for every layer checked
    bool has_gated_ffn      = false;   // any blk.N.ffn_gate.weight tensor present (SwiGLU signal)
    bool has_attn_bias      = false;   // any blk.N.attn_{q,k,v,output}.bias tensor present
    bool has_ffn_bias       = false;   // any blk.N.ffn_{gate,up,down}.bias tensor present
    bool embedding_present  = false;
    bool output_norm_present = false;
    bool lm_head_present    = false;   // separate "output.weight", vs. a tied embedding (absent)
    bool any_quantized      = false;   // any tensor whose type isn't F32/F16
    std::vector<std::string> missing;  // expected tensor names not found (checked up to n_layers_checked)
    int  n_layers_checked = 0;         // how many blk.N.* layers were actually present to inspect
};

inline CompatReport check_llama_mha_compat(const Reader& r, int expect_layers = -1) {
    CompatReport rep;
    const TensorInfo* emb = r.find_tensor("token_embd.weight");
    rep.embedding_present = (emb != nullptr);
    rep.output_norm_present = (r.find_tensor("output_norm.weight") != nullptr);
    rep.lm_head_present = (r.find_tensor("output.weight") != nullptr);

    for (const auto& t : r.tensors()) if (!t.is_plain_float()) { rep.any_quantized = true; break; }

    // Discover how many blk.N layers actually exist by probing attn_q, rather than trusting a
    // caller-supplied count (a mismatched guess would just report everything "missing").
    int n = 0;
    while (expect_layers < 0 ? true : n < expect_layers) {
        const std::string prefix = "blk." + std::to_string(n) + ".";
        const TensorInfo* q = r.find_tensor(prefix + "attn_q.weight");
        if (!q) { if (expect_layers < 0) break; else rep.missing.push_back(prefix + "attn_q.weight"); }
        const TensorInfo* k = r.find_tensor(prefix + "attn_k.weight");
        const TensorInfo* v = r.find_tensor(prefix + "attn_v.weight");
        const TensorInfo* o = r.find_tensor(prefix + "attn_output.weight");
        const TensorInfo* an = r.find_tensor(prefix + "attn_norm.weight");
        const TensorInfo* fn = r.find_tensor(prefix + "ffn_norm.weight");
        const TensorInfo* fd = r.find_tensor(prefix + "ffn_down.weight");
        for (auto* tp : {k, v, o, an, fn, fd})
            if (!tp) rep.missing.push_back(prefix + "(one of attn_k/attn_v/attn_output/attn_norm/ffn_norm/ffn_down).weight");
        // Only ffn_gate.weight's presence is a genuine gating signal: llama.cpp's naming convention
        // uses ffn_up.weight/ffn_down.weight for BOTH a plain single-branch FFN and the "up" branch
        // of a SwiGLU-gated one, so ffn_up alone does not distinguish them -- only the gate matrix does.
        if (r.find_tensor(prefix + "ffn_gate.weight")) rep.has_gated_ffn = true;
        if (r.find_tensor(prefix + "attn_q.bias")) rep.has_attn_bias = true;
        if (r.find_tensor(prefix + "ffn_down.bias")) rep.has_ffn_bias = true;
        if (q && k && !q->dims.empty() && !k->dims.empty() && q->dims.back() != k->dims.back())
            rep.matches_plain_mha = false;   // attn_k has fewer rows than attn_q -> GQA, not plain MHA
        ++n;
        if (!q && expect_layers < 0) break;
    }
    rep.n_layers_checked = n;
    return rep;
}

}  // namespace sub0::gguf
