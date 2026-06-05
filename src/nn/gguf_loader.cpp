#include "sub0llm/nn/gguf_loader.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>

namespace sub0llm::nn {

namespace {

// ── I/O helpers ───────────────────────────────────────────────────────────────

template<typename T>
T read_pod(std::ifstream& f, const std::string& ctx) {
    T val{};
    f.read(reinterpret_cast<char*>(&val), sizeof(T));
    if (!f)
        throw std::runtime_error(std::format("gguf_loader: truncated read in {}", ctx));
    return val;
}

std::string read_gguf_string(std::ifstream& f) {
    const uint64_t len = read_pod<uint64_t>(f, "string length");
    if (len > 128 * 1024 * 1024ULL)
        throw std::runtime_error(std::format(
            "gguf_loader: suspiciously large string length {}", len));
    std::string s(len, '\0');
    if (len > 0) {
        f.read(s.data(), static_cast<std::streamsize>(len));
        if (!f)
            throw std::runtime_error("gguf_loader: truncated string data");
    }
    return s;
}

// ── F16 dequantization ────────────────────────────────────────────────────────

float f16_to_f32(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    if (exp == 0) {
        float val = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -val : val;
    } else if (exp == 31) {
        return sign ? -std::numeric_limits<float>::infinity()
                    :  std::numeric_limits<float>::infinity();
    }
    const uint32_t f32bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    float val;
    std::memcpy(&val, &f32bits, 4);
    return val;
}

// ── GGUF metadata value reader ────────────────────────────────────────────────

void skip_gguf_value(std::ifstream& f, uint32_t type);

void skip_gguf_value(std::ifstream& f, uint32_t type) {
    switch (type) {
        case 0:  read_pod<uint8_t>(f,  "skip u8");   break;
        case 1:  read_pod<int8_t>(f,   "skip i8");   break;
        case 2:  read_pod<uint16_t>(f, "skip u16");  break;
        case 3:  read_pod<int16_t>(f,  "skip i16");  break;
        case 4:  read_pod<uint32_t>(f, "skip u32");  break;
        case 5:  read_pod<int32_t>(f,  "skip i32");  break;
        case 6:  read_pod<float>(f,    "skip f32");  break;
        case 7:  read_pod<uint8_t>(f,  "skip bool"); break;
        case 8:  read_gguf_string(f);                break;
        case 9: {
            const uint32_t elem_type = read_pod<uint32_t>(f, "array elem_type");
            const uint64_t count     = read_pod<uint64_t>(f, "array count");
            for (uint64_t k = 0; k < count; ++k)
                skip_gguf_value(f, elem_type);
            break;
        }
        case 10: read_pod<uint64_t>(f, "skip u64");  break;
        case 11: read_pod<int64_t>(f,  "skip i64");  break;
        case 12: read_pod<double>(f,   "skip f64");  break;
        default:
            throw std::runtime_error(
                std::format("gguf_loader: unknown metadata type {}", type));
    }
}

// Read one metadata kv pair generically: capture the value into typed maps (and a
// stringified dump), populating the vocab for the big token/merge arrays. Config
// interpretation happens later in resolve_config — so it can scope keys to the
// architecture prefix regardless of metadata ordering, and ignore the vision/audio
// sub-model keys that a multimodal GGUF (Gemma 4) interleaves.
void read_kv_pair(std::ifstream& f, GGUFVocab& vocab,
                  std::map<std::string, int64_t>&     ints,
                  std::map<std::string, double>&      floats,
                  std::map<std::string, std::string>& strings,
                  std::vector<std::pair<std::string, std::string>>& meta) {
    const std::string key  = read_gguf_string(f);
    const uint32_t    type = read_pod<uint32_t>(f, "kv type");

    auto put_i = [&](int64_t v) { ints[key] = v;    meta.emplace_back(key, std::to_string(v)); };
    auto put_f = [&](double  v) { floats[key] = v;  meta.emplace_back(key, std::format("{}", v)); };

    switch (type) {
        case 0:  put_i(read_pod<uint8_t>(f,  key)); break;
        case 1:  put_i(read_pod<int8_t>(f,   key)); break;
        case 2:  put_i(read_pod<uint16_t>(f, key)); break;
        case 3:  put_i(read_pod<int16_t>(f,  key)); break;
        case 4:  put_i(read_pod<uint32_t>(f, key)); break;
        case 5:  put_i(read_pod<int32_t>(f,  key)); break;
        case 6:  put_f(read_pod<float>(f,    key)); break;
        case 7:  { const auto v = read_pod<uint8_t>(f, key);
                   ints[key] = v; meta.emplace_back(key, v ? "true" : "false"); break; }
        case 8:  { auto s = read_gguf_string(f); strings[key] = s;
                   if (key == "tokenizer.ggml.model")  vocab.model = s;
                   meta.emplace_back(key, s.size() <= 96 ? s
                                          : std::format("[string len={}]", s.size()));
                   break; }
        case 9:  { const uint32_t et = read_pod<uint32_t>(f, "array elem_type");
                   const uint64_t n  = read_pod<uint64_t>(f, "array count");
                   if (key == "tokenizer.ggml.tokens") {
                       vocab.tokens.reserve(n);
                       for (uint64_t k = 0; k < n; ++k)
                           if (et == 8) vocab.tokens.push_back(read_gguf_string(f));
                           else { skip_gguf_value(f, et); vocab.tokens.emplace_back(); }
                   } else if (key == "tokenizer.ggml.merges") {
                       vocab.merges.reserve(n);
                       for (uint64_t k = 0; k < n; ++k)
                           if (et == 8) vocab.merges.push_back(read_gguf_string(f));
                           else { skip_gguf_value(f, et); vocab.merges.emplace_back(); }
                   } else if (key == "tokenizer.ggml.scores" && et == 6) {
                       vocab.scores.reserve(n);
                       for (uint64_t k = 0; k < n; ++k) vocab.scores.push_back(read_pod<float>(f, key));
                   } else if (key == "tokenizer.ggml.token_type" && et == 5) {
                       vocab.token_types.reserve(n);
                       for (uint64_t k = 0; k < n; ++k) vocab.token_types.push_back(read_pod<int32_t>(f, key));
                   } else {
                       for (uint64_t k = 0; k < n; ++k) skip_gguf_value(f, et);
                   }
                   meta.emplace_back(key, std::format("[array elem_type={} count={}]", et, n));
                   break; }
        case 10: put_i(static_cast<int64_t>(read_pod<uint64_t>(f, key))); break;
        case 11: put_i(read_pod<int64_t>(f, key)); break;
        case 12: put_f(read_pod<double>(f, key)); break;
        default:
            throw std::runtime_error(
                std::format("gguf_loader: unknown metadata type {} for key '{}'", type, key));
    }

    // Tokenizer special-token scalars (just captured into `ints` above).
    if (key == "tokenizer.ggml.bos_token_id")  vocab.bos_id = static_cast<int32_t>(ints[key]);
    else if (key == "tokenizer.ggml.eos_token_id") vocab.eos_id = static_cast<int32_t>(ints[key]);
    else if (key == "tokenizer.ggml.add_bos_token") vocab.add_bos = ints[key] != 0;
}

// Interpret captured metadata into GGUFModelConfig. Architecture-specific keys are
// matched as EXACTLY "<arch><suffix>" (e.g. "gemma4.attention.head_count_kv"), so the
// text-tower config is read and the vision/audio sub-model keys are ignored.
void resolve_config(GGUFModelConfig& cfg,
                    const std::map<std::string, int64_t>&     ints,
                    const std::map<std::string, double>&      floats,
                    const std::map<std::string, std::string>& strings) {
    if (auto it = strings.find("general.architecture"); it != strings.end())
        cfg.arch = it->second;
    const std::string& a = cfg.arch;

    auto geti = [&](const std::string& suffix) -> std::optional<int64_t> {
        if (!a.empty()) { if (auto it = ints.find(a + suffix); it != ints.end()) return it->second; }
        else { // architecture unknown — fall back to legacy suffix match
            for (const auto& [k, v] : ints)
                if (k.size() >= suffix.size() &&
                    k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) return v;
        }
        return std::nullopt;
    };
    auto getf = [&](const std::string& suffix) -> std::optional<double> {
        if (!a.empty()) { if (auto it = floats.find(a + suffix); it != floats.end()) return it->second; }
        else {
            for (const auto& [k, v] : floats)
                if (k.size() >= suffix.size() &&
                    k.compare(k.size() - suffix.size(), suffix.size(), suffix) == 0) return v;
        }
        return std::nullopt;
    };

    if (auto v = geti(".vocab_size"))                       cfg.vocab_size  = *v;
    if (auto v = geti(".embedding_length"))                 cfg.embed_dim   = *v;
    if (auto v = geti(".block_count"))                      cfg.n_layers    = *v;
    if (auto v = geti(".attention.head_count"))             cfg.n_heads     = static_cast<std::size_t>(*v);
    if (auto v = geti(".attention.head_count_kv"))          cfg.n_kv_heads  = static_cast<std::size_t>(*v);
    if (auto v = geti(".attention.key_length"))             cfg.head_dim    = *v;
    if (cfg.head_dim == 0) if (auto v = geti(".attention.value_length")) cfg.head_dim = *v;
    if (auto v = geti(".feed_forward_length"))              cfg.d_ff        = *v;
    if (auto v = geti(".context_length"))                   cfg.context_len = *v;
    if (auto v = geti(".attention.sliding_window"))         cfg.sliding_window = *v;
    if (auto v = getf(".attention.layer_norm_rms_epsilon")) cfg.norm_eps    = static_cast<float>(*v);
    if (auto v = getf(".rope.freq_base"))                   cfg.rope_base   = static_cast<float>(*v);
}

} // anonymous namespace

// ── GGUFReader ────────────────────────────────────────────────────────────────

GGUFReader::GGUFReader(const std::string& path) : path_(path) {
    parse_header();
}

void GGUFReader::parse_header() {
    std::ifstream f(path_, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(
            std::format("GGUFReader: cannot open '{}'", path_));

    const uint32_t magic = read_pod<uint32_t>(f, "magic");
    if (magic != 0x46554747u)
        throw std::runtime_error(std::format(
            "GGUFReader: bad magic 0x{:08X} in '{}' (expected 0x46554747)",
            magic, path_));

    const uint32_t version = read_pod<uint32_t>(f, "version");
    if (version != 2 && version != 3)
        throw std::runtime_error(std::format(
            "GGUFReader: unsupported version {} in '{}'", version, path_));

    const uint64_t tensor_count      = read_pod<uint64_t>(f, "tensor_count");
    const uint64_t metadata_kv_count = read_pod<uint64_t>(f, "metadata_kv_count");

    std::map<std::string, int64_t>     ints;
    std::map<std::string, double>      floats;
    std::map<std::string, std::string> strings;
    metadata_.reserve(metadata_kv_count);
    for (uint64_t i = 0; i < metadata_kv_count; ++i)
        read_kv_pair(f, vocab_, ints, floats, strings, metadata_);
    resolve_config(config_, ints, floats, strings);

    if (config_.vocab_size == 0 && !vocab_.tokens.empty())
        config_.vocab_size = static_cast<int64_t>(vocab_.tokens.size());

    if (config_.n_kv_heads == 0 && config_.n_heads > 0)
        config_.n_kv_heads = config_.n_heads;

    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo ti;
        ti.name = read_gguf_string(f);

        const uint32_t n_dims = read_pod<uint32_t>(f, "n_dims");
        std::vector<int64_t> gguf_dims(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d)
            gguf_dims[d] = read_pod<int64_t>(f, "dim");

        ti.shape.assign(gguf_dims.rbegin(), gguf_dims.rend());

        ti.ggml_type = static_cast<int32_t>(read_pod<uint32_t>(f, "ggml_type"));
        ti.offset    = read_pod<uint64_t>(f, "tensor offset");

        ti.numel = 1;
        for (auto d : ti.shape) ti.numel *= d;

        tensors_[ti.name] = std::move(ti);
    }

    const uint64_t pos = static_cast<uint64_t>(f.tellg());
    data_section_offset_ = (pos + 31u) & ~31u;

    config_.has_separate_lm_head = tensors_.count("output.weight") > 0;
}

bool GGUFReader::has_tensor(const std::string& name) const {
    return tensors_.count(name) > 0;
}

std::vector<float> GGUFReader::dequant_f32(const float* src, int64_t n) {
    return std::vector<float>(src, src + static_cast<std::size_t>(n));
}

std::vector<float> GGUFReader::dequant_f16(const uint16_t* src, int64_t n) {
    std::vector<float> out(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        out[static_cast<std::size_t>(i)] = f16_to_f32(src[static_cast<std::size_t>(i)]);
    return out;
}

std::vector<float> GGUFReader::dequant_q8_0(const uint8_t* src, int64_t n) {
    constexpr int64_t BLOCK       = 32;
    constexpr int64_t BLOCK_BYTES = 34;

    const int64_t n_blocks = (n + BLOCK - 1) / BLOCK;
    std::vector<float> out(static_cast<std::size_t>(n), 0.0f);

    for (int64_t b = 0; b < n_blocks; ++b) {
        const uint8_t* bp = src + b * BLOCK_BYTES;
        uint16_t scale_bits;
        std::memcpy(&scale_bits, bp, 2);
        const float scale = f16_to_f32(scale_bits);

        const int64_t elem_start = b * BLOCK;
        const int64_t elem_end   = std::min(elem_start + BLOCK, n);
        for (int64_t k = elem_start; k < elem_end; ++k) {
            const int8_t ival = static_cast<int8_t>(bp[2 + (k - elem_start)]);
            out[static_cast<std::size_t>(k)] = scale * static_cast<float>(ival);
        }
    }
    return out;
}

int32_t GGUFReader::tensor_type(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error(std::format(
            "GGUFReader: tensor '{}' not found in '{}'", name, path_));
    return it->second.ggml_type;
}

std::vector<backend::cpu::BlockQ8_0> GGUFReader::load_tensor_q8(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error(std::format(
            "GGUFReader: tensor '{}' not found in '{}'", name, path_));
    const GGUFTensorInfo& ti = it->second;
    if (ti.ggml_type != 8)
        throw std::runtime_error(std::format(
            "GGUFReader: tensor '{}' is ggml_type {}, not Q8_0 (8)", name, ti.ggml_type));
    if (ti.numel % 32 != 0)
        throw std::runtime_error(std::format(
            "GGUFReader: Q8_0 tensor '{}' numel {} not a multiple of 32", name, ti.numel));

    const int64_t n_blocks = ti.numel / 32;
    std::vector<backend::cpu::BlockQ8_0> out(static_cast<std::size_t>(n_blocks));
    static_assert(sizeof(backend::cpu::BlockQ8_0) == 34);

    std::ifstream f(path_, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::format("GGUFReader: cannot reopen '{}'", path_));
    f.seekg(static_cast<std::streamoff>(data_section_offset_ + ti.offset), std::ios::beg);
    f.read(reinterpret_cast<char*>(out.data()),
           static_cast<std::streamsize>(n_blocks) * 34);
    if (!f)
        throw std::runtime_error(std::format("GGUFReader: truncated Q8_0 data for '{}'", name));
    return out;
}

std::vector<float> GGUFReader::load_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error(std::format(
            "GGUFReader: tensor '{}' not found in '{}'", name, path_));

    const GGUFTensorInfo& ti = it->second;

    std::ifstream f(path_, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::format(
            "GGUFReader: cannot reopen '{}'", path_));

    const uint64_t abs_offset = data_section_offset_ + ti.offset;
    f.seekg(static_cast<std::streamoff>(abs_offset), std::ios::beg);
    if (!f)
        throw std::runtime_error(std::format(
            "GGUFReader: seek failed for tensor '{}' in '{}'", name, path_));

    switch (ti.ggml_type) {
        case 0: {
            std::vector<float> buf(static_cast<std::size_t>(ti.numel));
            f.read(reinterpret_cast<char*>(buf.data()),
                   static_cast<std::streamsize>(ti.numel) * static_cast<std::streamsize>(sizeof(float)));
            if (!f) throw std::runtime_error(
                std::format("GGUFReader: truncated F32 data for '{}'", name));
            return dequant_f32(buf.data(), ti.numel);
        }
        case 1: {
            std::vector<uint16_t> buf(static_cast<std::size_t>(ti.numel));
            f.read(reinterpret_cast<char*>(buf.data()),
                   static_cast<std::streamsize>(ti.numel * 2));
            if (!f) throw std::runtime_error(
                std::format("GGUFReader: truncated F16 data for '{}'", name));
            return dequant_f16(buf.data(), ti.numel);
        }
        case 8: {
            constexpr int64_t BLOCK_BYTES = 34;
            const int64_t n_blocks = (ti.numel + 31) / 32;
            std::vector<uint8_t> buf(static_cast<std::size_t>(n_blocks * BLOCK_BYTES));
            f.read(reinterpret_cast<char*>(buf.data()),
                   static_cast<std::streamsize>(n_blocks * BLOCK_BYTES));
            if (!f) throw std::runtime_error(
                std::format("GGUFReader: truncated Q8_0 data for '{}'", name));
            return dequant_q8_0(buf.data(), ti.numel);
        }
        default:
            throw std::runtime_error(std::format(
                "GGUFReader: unsupported ggml_type {} for tensor '{}' in '{}'",
                ti.ggml_type, name, path_));
    }
}

// ── load_gguf_model ────────────────────────────────────────────────────────────
//
// parameters() layout (from ModernGPT, ModernTransformerBlock, GroupedQueryAttention):
//
//   Global index:
//   [0]          tok_emb_.weight_  (V, D)
//
//   Per block (block b occupies indices [1 + b*per_block .. 1 + (b+1)*per_block - 1]):
//     per_block = 1 + H*2 + Hkv*2 + 1 + 6
//                 norm1  attn(Q,O interleaved, then K,V)  norm2  ffn(gate W,b + up W,b + down W,b)
//
//     Within block at offset `base`:
//       base+0:           norm1.weight  (D,)
//       base+1+h*2:       W_Q_[h]       (D, Dh)    h in [0, H)
//       base+1+h*2+1:     W_O_[h]       (Dh, D)
//       base+1+H*2+g*2:   W_K_[g]       (D, Dh)    g in [0, Hkv)
//       base+1+H*2+g*2+1: W_V_[g]       (D, Dh)
//       base+1+H*2+Hkv*2: norm2.weight  (D,)
//       base+1+H*2+Hkv*2+1: gate.W      (d_ff, D)
//       base+1+H*2+Hkv*2+2: gate.b      (d_ff,)
//       base+1+H*2+Hkv*2+3: up.W        (d_ff, D)
//       base+1+H*2+Hkv*2+4: up.b        (d_ff,)
//       base+1+H*2+Hkv*2+5: down.W      (D, d_ff)
//       base+1+H*2+Hkv*2+6: down.b      (D,)
//
//   After all blocks:
//       ln_f_.weight_ (D,)    [last parameter, since n_mtp_heads=0]

ModernGPT load_gguf_model(const GGUFReader& reader) {
    const GGUFModelConfig& cfg = reader.config();

    if (cfg.embed_dim == 0 || cfg.n_layers == 0 || cfg.n_heads == 0)
        throw std::runtime_error(std::format(
            "load_gguf_model: incomplete config — embed_dim={} n_layers={} n_heads={}",
            cfg.embed_dim, cfg.n_layers, cfg.n_heads));

    const int64_t     V   = cfg.vocab_size;
    const int64_t     D   = cfg.embed_dim;
    const std::size_t H   = cfg.n_heads;
    const std::size_t Hkv = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : H;
    const int64_t     L   = cfg.n_layers;
    const int64_t     dff = cfg.d_ff;
    // head_dim: explicit from GGUF (Qwen3-4B+) or derived from D/H.
    const std::size_t Dh  = cfg.head_dim > 0
                                ? static_cast<std::size_t>(cfg.head_dim)
                                : static_cast<std::size_t>(D) / H;

    // Qwen3 applies RMSNorm to Q and K per head (before RoPE).  Detect it from the
    // presence of the per-layer norm tensors and build the model with QK-norm so
    // the weights have a home and attention matches the reference implementation.
    const bool use_qk_norm = reader.has_tensor("blk.0.attn_q_norm.weight");

    ModernGPT model(V, D, H, Hkv, L, dff, /*n_mtp=*/0, /*seed=*/42,
                    /*window_size=*/-1, Dh, use_qk_norm);

    // params per block: norm1(1) + attn(H*2 + Hkv*2 + qkn) + norm2(1) + ffn(6).
    // qkn = 2 (q_norm, k_norm) when QK-norm is enabled — appended after W_K/W_V
    // in GroupedQueryAttention::parameters().
    const std::size_t qkn       = use_qk_norm ? 2u : 0u;
    const std::size_t per_block = 1 + H * 2 + Hkv * 2 + qkn + 1 + 6;

    auto block_base = [&](int64_t l) -> std::size_t {
        return 1 + static_cast<std::size_t>(l) * per_block;
    };

    auto copy_data = [&](autograd::Variable* var, const std::vector<float>& data) {
        if (static_cast<std::size_t>(var->data().numel()) != data.size())
            throw std::runtime_error(std::format(
                "load_gguf_model: param size mismatch — var has {} elements, data has {}",
                var->data().numel(), data.size()));
        auto sp = var->data().data_as<float>();
        std::copy(data.begin(), data.end(), sp.begin());
    };

    auto lp = model.parameters();

    // ── token embedding ────────────────────────────────────────────────────────
    if (reader.has_tensor("token_embd.weight"))
        copy_data(lp[0], reader.load_tensor("token_embd.weight"));

    // ── per-layer ─────────────────────────────────────────────────────────────
    for (int64_t l = 0; l < L; ++l) {
        const std::size_t base = block_base(l);

        // norm1.weight
        {
            auto tname = std::format("blk.{}.attn_norm.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[base + 0], reader.load_tensor(tname));
        }

        // W_Q_[h] — GGUF logical shape (H*Dh, D); our W_Q_[h]: (D, Dh)
        // W_Q_[h][d, dh] = gguf_q[(h*Dh + dh) * D + d]
        {
            auto tname = std::format("blk.{}.attn_q.weight", l);
            if (reader.has_tensor(tname)) {
                auto gguf_q = reader.load_tensor(tname);
                for (std::size_t h = 0; h < H; ++h) {
                    auto sp = lp[base + 1 + h * 2]->data().data_as<float>();
                    for (std::size_t d = 0; d < static_cast<std::size_t>(D); ++d)
                        for (std::size_t dh = 0; dh < Dh; ++dh)
                            sp[d * Dh + dh] =
                                gguf_q[(h * Dh + dh) * static_cast<std::size_t>(D) + d];
                }
            }
        }

        // W_O_[h] — GGUF logical shape (D, H*Dh); our W_O_[h]: (Dh, D)
        // W_O_[h][dh, d] = gguf_o[d * (H*Dh) + h*Dh + dh]
        {
            auto tname = std::format("blk.{}.attn_output.weight", l);
            if (reader.has_tensor(tname)) {
                auto gguf_o = reader.load_tensor(tname);
                for (std::size_t h = 0; h < H; ++h) {
                    auto sp = lp[base + 1 + h * 2 + 1]->data().data_as<float>();
                    for (std::size_t dh = 0; dh < Dh; ++dh)
                        for (std::size_t d = 0; d < static_cast<std::size_t>(D); ++d)
                            sp[dh * static_cast<std::size_t>(D) + d] =
                                gguf_o[d * (H * Dh) + h * Dh + dh];
                }
            }
        }

        // W_K_[g] — GGUF logical shape (Hkv*Dh, D); our W_K_[g]: (D, Dh)
        {
            auto tname = std::format("blk.{}.attn_k.weight", l);
            if (reader.has_tensor(tname)) {
                auto gguf_k = reader.load_tensor(tname);
                for (std::size_t g = 0; g < Hkv; ++g) {
                    auto sp = lp[base + 1 + H * 2 + g * 2]->data().data_as<float>();
                    for (std::size_t d = 0; d < static_cast<std::size_t>(D); ++d)
                        for (std::size_t dh = 0; dh < Dh; ++dh)
                            sp[d * Dh + dh] =
                                gguf_k[(g * Dh + dh) * static_cast<std::size_t>(D) + d];
                }
            }
        }

        // W_V_[g] — GGUF logical shape (Hkv*Dh, D); our W_V_[g]: (D, Dh)
        {
            auto tname = std::format("blk.{}.attn_v.weight", l);
            if (reader.has_tensor(tname)) {
                auto gguf_v = reader.load_tensor(tname);
                for (std::size_t g = 0; g < Hkv; ++g) {
                    auto sp = lp[base + 1 + H * 2 + g * 2 + 1]->data().data_as<float>();
                    for (std::size_t d = 0; d < static_cast<std::size_t>(D); ++d)
                        for (std::size_t dh = 0; dh < Dh; ++dh)
                            sp[d * Dh + dh] =
                                gguf_v[(g * Dh + dh) * static_cast<std::size_t>(D) + d];
                }
            }
        }

        // QK-norm weights (Qwen3): q_norm then k_norm, each (head_dim,).
        // Slotted right after W_K/W_V, matching GroupedQueryAttention::parameters().
        if (use_qk_norm) {
            const std::size_t qkb = base + 1 + H * 2 + Hkv * 2;
            auto qn = std::format("blk.{}.attn_q_norm.weight", l);
            auto kn = std::format("blk.{}.attn_k_norm.weight", l);
            if (reader.has_tensor(qn)) copy_data(lp[qkb + 0], reader.load_tensor(qn));
            if (reader.has_tensor(kn)) copy_data(lp[qkb + 1], reader.load_tensor(kn));
        }

        // FFN/norm2 indices shift by qkn to clear the QK-norm params.
        const std::size_t ffn_base = base + 1 + H * 2 + Hkv * 2 + qkn;

        // norm2.weight
        {
            auto tname = std::format("blk.{}.ffn_norm.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[ffn_base], reader.load_tensor(tname));
        }

        // gate.W (d_ff, D)  — Linear stores W as (out_features, in_features) = (d_ff, D) ✓
        {
            auto tname = std::format("blk.{}.ffn_gate.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[ffn_base + 1], reader.load_tensor(tname));
        }
        // gate.b — skip (GGUF has no bias for RMSNorm-style FFN, leave zero-init)

        // up.W (d_ff, D)
        {
            auto tname = std::format("blk.{}.ffn_up.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[ffn_base + 3], reader.load_tensor(tname));
        }
        // up.b — skip

        // down.W (D, d_ff)
        {
            auto tname = std::format("blk.{}.ffn_down.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[ffn_base + 5], reader.load_tensor(tname));
        }
        // down.b — skip
    }

    // ── final norm ─────────────────────────────────────────────────────────────
    // ln_f_ weight is the last parameter (n_mtp_heads=0)
    if (reader.has_tensor("output_norm.weight"))
        copy_data(lp.back(), reader.load_tensor("output_norm.weight"));

    // ── output projection (Qwen2, Mistral, etc.) ───────────────────────────────
    // Models with separate output.weight (not tied to token_embd) store the lm_head
    // separately.  Our ModernGPT::forward() uses tied embeddings (lm_head = tok_emb.T),
    // so we load output.weight into lp[0] (tok_emb_.weight_) to make the tied path
    // use the correct output distribution.  This intentionally overwrites the input
    // embedding — for episodic-memory PoC use, forward perplexity quality matters more
    // than the input embedding lookup.
    if (reader.has_tensor("output.weight"))
        copy_data(lp[0], reader.load_tensor("output.weight"));

    // Note: Qwen2 also has attn_q.bias and attn_k.bias per layer.  Our GQA module
    // does not register bias variables for Q/K projections, so these tensors are
    // silently skipped (zero bias is used instead).  Quality impact: minor for
    // short sequences (Qwen2 bias magnitudes are small relative to weight dot-products).

    return model;
}

ModernGPT load_gguf_model(const std::string& path) {
    GGUFReader reader(path);
    return load_gguf_model(reader);
}

// ── load_gguf_model_q8 — quantize-on-load (Ch27) ─────────────────────────────────
// Builds an inference model with f32 elided (alloc_weights=false) and copies the
// raw Q8_0 blocks straight into the int8 buffers — no f32 is ever materialized, so
// peak RSS stays at the Q8 footprint. GGUF weight layouts are already (out,in)
// row-major, matching our int8 buffers (attention is sliced per head).
ModernGPT load_gguf_model_q8(const GGUFReader& reader) {
    namespace cpu = backend::cpu;
    const GGUFModelConfig& cfg = reader.config();
    if (cfg.embed_dim == 0 || cfg.n_layers == 0 || cfg.n_heads == 0)
        throw std::runtime_error("load_gguf_model_q8: incomplete config");

    const int64_t     V   = cfg.vocab_size;
    const int64_t     D   = cfg.embed_dim;
    const std::size_t H   = cfg.n_heads;
    const std::size_t Hkv = cfg.n_kv_heads > 0 ? cfg.n_kv_heads : H;
    const int64_t     L   = cfg.n_layers;
    const int64_t     dff = cfg.d_ff;
    const std::size_t Dh  = cfg.head_dim > 0 ? static_cast<std::size_t>(cfg.head_dim)
                                             : static_cast<std::size_t>(D) / H;
    if (D % cpu::QK8_0 != 0 || (Dh % cpu::QK8_0) != 0)
        throw std::runtime_error(std::format(
            "load_gguf_model_q8: embed_dim={} and head_dim={} must be multiples of 32",
            D, Dh));
    const bool use_qk_norm = reader.has_tensor("blk.0.attn_q_norm.weight");

    ModernGPT model(V, D, H, Hkv, L, dff, /*n_mtp=*/0, /*seed=*/42,
                    /*window=*/-1, Dh, use_qk_norm, /*alloc_weights=*/false);

    auto set_norm = [&](RMSNorm& nrm, const std::string& name) {
        if (!reader.has_tensor(name)) return;
        const auto f = reader.load_tensor(name);
        auto d = nrm.weight_.data().data_as<float>();
        if (d.size() != f.size())
            throw std::runtime_error(std::format("load_gguf_model_q8: norm size mismatch {}", name));
        std::copy(f.begin(), f.end(), d.begin());
    };

    const int64_t nbD   = D / cpu::QK8_0;                              // blocks per D-row
    const int64_t nbDh  = static_cast<int64_t>(Dh) / cpu::QK8_0;      // blocks per head slice
    const int64_t nbHDh = static_cast<int64_t>(H * Dh) / cpu::QK8_0;  // attn_output row length

    for (int64_t l = 0; l < L; ++l) {
        auto& blk = model.blocks_[static_cast<std::size_t>(l)];
        auto& att = blk.attn_;

        set_norm(blk.norm1_, std::format("blk.{}.attn_norm.weight", l));
        set_norm(blk.norm2_, std::format("blk.{}.ffn_norm.weight", l));
        if (use_qk_norm) {
            if (att.q_norm_) set_norm(*att.q_norm_, std::format("blk.{}.attn_q_norm.weight", l));
            if (att.k_norm_) set_norm(*att.k_norm_, std::format("blk.{}.attn_k_norm.weight", l));
        }

        // Q/K/V: GGUF (heads·Dh, D) row-major → per-head contiguous (Dh, D) slices.
        const auto qraw = reader.load_tensor_q8(std::format("blk.{}.attn_q.weight", l));
        const auto kraw = reader.load_tensor_q8(std::format("blk.{}.attn_k.weight", l));
        const auto vraw = reader.load_tensor_q8(std::format("blk.{}.attn_v.weight", l));
        const std::size_t head_blocks = static_cast<std::size_t>(Dh) * static_cast<std::size_t>(nbD);
        const auto hb = static_cast<std::ptrdiff_t>(head_blocks);
        att.wq_q8_.resize(H);
        for (std::size_t h = 0; h < H; ++h) {
            const auto off = static_cast<std::ptrdiff_t>(h) * hb;
            att.wq_q8_[h].assign(qraw.begin() + off, qraw.begin() + off + hb);
        }
        att.wk_q8_.resize(Hkv);
        att.wv_q8_.resize(Hkv);
        for (std::size_t g = 0; g < Hkv; ++g) {
            const auto off = static_cast<std::ptrdiff_t>(g) * hb;
            att.wk_q8_[g].assign(kraw.begin() + off, kraw.begin() + off + hb);
            att.wv_q8_[g].assign(vraw.begin() + off, vraw.begin() + off + hb);
        }

        // O: GGUF (D, H·Dh) row-major → per-head (D, Dh): row d, blocks [h·nbDh : (h+1)·nbDh].
        const auto oraw = reader.load_tensor_q8(std::format("blk.{}.attn_output.weight", l));
        att.wo_q8_.resize(H);
        for (std::size_t h = 0; h < H; ++h) {
            att.wo_q8_[h].resize(static_cast<std::size_t>(D) * static_cast<std::size_t>(nbDh));
            for (int64_t d = 0; d < D; ++d) {
                const std::size_t src = static_cast<std::size_t>(d) * static_cast<std::size_t>(nbHDh)
                                      + h * static_cast<std::size_t>(nbDh);
                const std::size_t dst = static_cast<std::size_t>(d) * static_cast<std::size_t>(nbDh);
                for (int64_t b = 0; b < nbDh; ++b)
                    att.wo_q8_[h][dst + static_cast<std::size_t>(b)] = oraw[src + static_cast<std::size_t>(b)];
            }
        }
        att.q8_ = true;

        // FFN: GGUF (out, in) row-major → directly our int8 layout.
        blk.ffn_.gate_.wq8_ = reader.load_tensor_q8(std::format("blk.{}.ffn_gate.weight", l));
        blk.ffn_.up_.wq8_   = reader.load_tensor_q8(std::format("blk.{}.ffn_up.weight", l));
        blk.ffn_.down_.wq8_ = reader.load_tensor_q8(std::format("blk.{}.ffn_down.weight", l));
    }

    set_norm(model.ln_f_, "output_norm.weight");

    // LM head / tied embedding: raw Q8; the embedding lookup dequantizes a row from it.
    const std::string emb = reader.has_tensor("output.weight") ? "output.weight"
                                                               : "token_embd.weight";
    model.lm_head_q8_  = reader.load_tensor_q8(emb);
    model.emb_from_q8_ = true;
    return model;
}

} // namespace sub0llm::nn
