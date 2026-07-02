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

// GGML tensor element type IDs this reader recognizes as "importable today" (plain float storage).
// Everything else (the many block-quantized Q4_*/Q5_*/Q8_*/K-quant types) is reported by its raw id
// so a caller can say "quantized, dequantize first" rather than silently misreading bytes.
enum class TensorType : std::uint32_t { F32 = 0, F16 = 1 };

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

    bool is_f32() const { return type_raw == static_cast<std::uint32_t>(TensorType::F32); }
    bool is_f16() const { return type_raw == static_cast<std::uint32_t>(TensorType::F16); }
    bool is_plain_float() const { return is_f32() || is_f16(); }
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

private:
    Err                           err_ = Err::Truncated;
    std::uint32_t                  version_ = 0;
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
        err_ = Err::Ok;   // the (alignment-padded) data section starts at pos_; this reader stops here
    }
};

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
