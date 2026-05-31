#include "sub0llm/nn/gguf_loader.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
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

void read_kv_pair(std::ifstream& f, GGUFModelConfig& cfg, GGUFVocab& vocab) {
    const std::string key  = read_gguf_string(f);
    const uint32_t    type = read_pod<uint32_t>(f, "kv type");

    auto read_int_val = [&]() -> int64_t {
        switch (type) {
            case 0:  return static_cast<int64_t>(read_pod<uint8_t>(f,  key));
            case 1:  return static_cast<int64_t>(read_pod<int8_t>(f,   key));
            case 2:  return static_cast<int64_t>(read_pod<uint16_t>(f, key));
            case 3:  return static_cast<int64_t>(read_pod<int16_t>(f,  key));
            case 4:  return static_cast<int64_t>(read_pod<uint32_t>(f, key));
            case 5:  return static_cast<int64_t>(read_pod<int32_t>(f,  key));
            case 10: return static_cast<int64_t>(read_pod<uint64_t>(f, key));
            case 11: return read_pod<int64_t>(f, key);
            default:
                skip_gguf_value(f, type);
                return 0;
        }
    };

    if (key == "general.architecture") {
        if (type == 8) cfg.arch = read_gguf_string(f);
        else skip_gguf_value(f, type);
        return;
    }
    if (key == "tokenizer.ggml.model") {
        if (type == 8) vocab.model = read_gguf_string(f);
        else skip_gguf_value(f, type);
        return;
    }
    if (key == "tokenizer.ggml.tokens") {
        if (type == 9) {
            const uint32_t elem_type = read_pod<uint32_t>(f, "tokens elem_type");
            const uint64_t count     = read_pod<uint64_t>(f, "tokens count");
            vocab.tokens.reserve(count);
            for (uint64_t k = 0; k < count; ++k) {
                if (elem_type == 8) vocab.tokens.push_back(read_gguf_string(f));
                else { skip_gguf_value(f, elem_type); vocab.tokens.emplace_back(); }
            }
        } else { skip_gguf_value(f, type); }
        return;
    }
    if (key == "tokenizer.ggml.merges") {
        if (type == 9) {
            const uint32_t elem_type = read_pod<uint32_t>(f, "merges elem_type");
            const uint64_t count     = read_pod<uint64_t>(f, "merges count");
            vocab.merges.reserve(count);
            for (uint64_t k = 0; k < count; ++k) {
                if (elem_type == 8) vocab.merges.push_back(read_gguf_string(f));
                else { skip_gguf_value(f, elem_type); vocab.merges.emplace_back(); }
            }
        } else { skip_gguf_value(f, type); }
        return;
    }

    auto ends_with = [&](const std::string& suffix) {
        return key.size() >= suffix.size() &&
               key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    if (ends_with(".embedding_length"))          { cfg.embed_dim   = read_int_val(); return; }
    if (ends_with(".block_count"))               { cfg.n_layers    = read_int_val(); return; }
    if (ends_with(".attention.head_count"))      { cfg.n_heads     = static_cast<std::size_t>(read_int_val()); return; }
    if (ends_with(".attention.head_count_kv"))   { cfg.n_kv_heads  = static_cast<std::size_t>(read_int_val()); return; }
    if (ends_with(".feed_forward_length"))       { cfg.d_ff        = read_int_val(); return; }
    if (ends_with(".context_length"))            { cfg.context_len = read_int_val(); return; }
    if (ends_with(".rope.freq_base")) {
        if (type == 6) cfg.rope_base = read_pod<float>(f, key);
        else skip_gguf_value(f, type);
        return;
    }

    skip_gguf_value(f, type);
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

    for (uint64_t i = 0; i < metadata_kv_count; ++i)
        read_kv_pair(f, config_, vocab_);

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
                   static_cast<std::streamsize>(ti.numel * sizeof(float)));
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

    ModernGPT model(V, D, H, Hkv, L, dff, /*n_mtp=*/0, /*seed=*/42);

    const std::size_t Dh = static_cast<std::size_t>(D) / H;

    // params per block: norm1(1) + attn(H*2 + Hkv*2) + norm2(1) + ffn(6) = H*2+Hkv*2+8
    const std::size_t per_block = 1 + H * 2 + Hkv * 2 + 1 + 6;

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

        // norm2.weight
        {
            auto tname = std::format("blk.{}.ffn_norm.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[base + 1 + H * 2 + Hkv * 2], reader.load_tensor(tname));
        }

        // gate.W (d_ff, D)  — Linear stores W as (out_features, in_features) = (d_ff, D) ✓
        {
            auto tname = std::format("blk.{}.ffn_gate.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[base + 1 + H * 2 + Hkv * 2 + 1], reader.load_tensor(tname));
        }
        // gate.b — skip (GGUF has no bias for RMSNorm-style FFN, leave zero-init)

        // up.W (d_ff, D)
        {
            auto tname = std::format("blk.{}.ffn_up.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[base + 1 + H * 2 + Hkv * 2 + 3], reader.load_tensor(tname));
        }
        // up.b — skip

        // down.W (D, d_ff)
        {
            auto tname = std::format("blk.{}.ffn_down.weight", l);
            if (reader.has_tensor(tname))
                copy_data(lp[base + 1 + H * 2 + Hkv * 2 + 5], reader.load_tensor(tname));
        }
        // down.b — skip
    }

    // ── final norm ─────────────────────────────────────────────────────────────
    // ln_f_ weight is the last parameter (n_mtp_heads=0)
    if (reader.has_tensor("output_norm.weight"))
        copy_data(lp.back(), reader.load_tensor("output_norm.weight"));

    return model;
}

ModernGPT load_gguf_model(const std::string& path) {
    GGUFReader reader(path);
    return load_gguf_model(reader);
}

} // namespace sub0llm::nn
