#pragma once

// gemma.hpp (Ch27) — a faithful, self-contained Gemma 4 (12B) text forward.
//
// Gemma 4 is per-layer HETEROGENEOUS and cannot be expressed by the uniform
// ModernGPT: head_dim, kv-head count, RoPE base, and attention window all vary by
// layer (local vs global). This model is therefore its own type, built directly
// from the GGUF's per-layer tensor shapes (the authoritative source — the HF config
// and even some GGUF metadata disagree with the file).
//
// Architecture (resolved from tensor shapes + llama.cpp's gemma4.cpp, the reference
// interpreter of THIS file):
//   - 48 layers, D=3840, V=262144, d_ff=15360, eps 1e-6, tied embeddings.
//   - LOCAL layers (index%6 != 5): head_dim 256, 16 q / 8 kv heads, sliding window
//     1024, RoPE base 1e4, full rotary.
//   - GLOBAL layers (index%6 == 5): head_dim 512, 16 q / 1 kv head (MQA), full
//     attention, RoPE base 1e6, rope_freqs kills pairs 64..255 (only 64 pairs rotate).
//   - Block: cur = rmsnorm(inpL, attn_norm); Q/K get rmsnorm(q/k_norm)+RoPE per head;
//     V gets plain rmsnorm (NO weight); attn scale = 1.0 (NOT 1/sqrt(dh)); wo·;
//     cur = rmsnorm(cur, post_attn_norm); attn_out = cur + inpL;
//     ffn = down·(gelu_tanh(gate·)⊙(up·)); cur = rmsnorm(ffn, post_ffw_norm);
//     inpL = (cur + attn_out) * layer_output_scale[l].
//   - Final: rmsnorm(output_norm); logits = token_embd·(tied); 30·tanh(logits/30).
//   - Embedding: dequant(token_embd[id]) * sqrt(D).
// (1+w) RMSNorm is baked into the GGUF weights at conversion, so norms are a plain
// x_normed*w. Weights are Q8_0, loaded raw in (out,in) layout; norms are f32.

#include "sub0llm/backends/cpu/quant.hpp"
#include "sub0llm/nn/gguf_loader.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// Thread count for the Q8 GEMVs in the Gemma forward. 0 = auto (hardware_concurrency,
// capped at 32). Set to 1 for a single-core baseline. Applies to subsequent forwards.
void set_gemma_threads(int n);

// Toggle Q/K/V and gate/up GEMV fusion (one barrier per fused group). On by default;
// off dispatches each output separately — for drift-free A/B of the optimization.
void set_gemma_fuse(bool on);

// Per-layer weights + shape. Q8 weights are stored as the GGUF gives them: row-major
// (out_features, in_features), so backend::cpu::matvec_q8_0_q8_0 consumes them with
// M=out, K=in directly (q/k/v are sliced per head AFTER projection, in f32).
struct GemmaLayer {
    int64_t head_dim   = 0;   // 256 local / 512 global
    int64_t n_head     = 0;   // 16
    int64_t n_kv_head  = 0;   // 8 local / 1 global
    bool    is_global  = false;
    bool    has_wv     = true; // global layers omit attn_v → V is the raw K projection
    float   rope_base  = 1e4f;
    int64_t window     = 0;   // sliding window (local); 0 = full attention (global)
    float   out_scale  = 1.0f;

    // f32 norms.
    std::vector<float> attn_norm;        // (D)
    std::vector<float> post_attn_norm;   // (D)
    std::vector<float> ffn_norm;         // (D)
    std::vector<float> post_ffw_norm;    // (D)
    std::vector<float> q_norm;           // (head_dim)
    std::vector<float> k_norm;           // (head_dim)

    // Q8 projections (out, in).
    std::vector<backend::cpu::BlockQ8_0> wq;   // (n_head*head_dim, D)
    std::vector<backend::cpu::BlockQ8_0> wk;   // (n_kv*head_dim, D)
    std::vector<backend::cpu::BlockQ8_0> wv;   // (n_kv*head_dim, D)
    std::vector<backend::cpu::BlockQ8_0> wo;   // (D, n_head*head_dim)
    std::vector<backend::cpu::BlockQ8_0> gate; // (d_ff, D)
    std::vector<backend::cpu::BlockQ8_0> up;   // (d_ff, D)
    std::vector<backend::cpu::BlockQ8_0> down; // (D, d_ff)
};

// KV cache: per layer, the (already RoPE'd K / RMS-normed V) per kv-head per position,
// stored as f32 contiguous [kv_head][pos][head_dim].
struct GemmaKVCache {
    struct Layer {
        int64_t n_kv_head = 0;
        int64_t head_dim  = 0;
        std::vector<float> k;   // flat: kv_head * (max_pos*head_dim) + pos*head_dim + d
        std::vector<float> v;
    };
    std::vector<Layer> layers;
    int64_t max_pos = 0;
    int64_t len     = 0;   // positions filled so far
};

class GemmaModel {
public:
    // Build from a Q8_0 Gemma 4 GGUF (quantize-on-load: no f32 weights materialized).
    [[nodiscard]] static GemmaModel load_q8(const GGUFReader& reader);

    [[nodiscard]] GemmaKVCache make_cache(int64_t max_pos) const;

    // One decode step: append token at position `pos` to the cache, return logits (V).
    // `apply_softcap` controls the final logit soft-cap (30·tanh(z/30)). It is
    // order-preserving, so GREEDY decoding can pass false (identical argmax) and skip
    // 262 K tanh/token; pass true when the magnitudes matter (sampling, logit/nll parity).
    [[nodiscard]] std::vector<float> forward_one(int32_t token, int64_t pos,
                                                 GemmaKVCache& kv,
                                                 bool apply_softcap = true) const;

    [[nodiscard]] int64_t vocab_size()  const noexcept { return V_; }
    [[nodiscard]] int64_t embed_dim()   const noexcept { return D_; }
    [[nodiscard]] int64_t n_layers()    const noexcept { return static_cast<int64_t>(layers_.size()); }
    [[nodiscard]] int64_t n_params()    const noexcept { return n_params_; }

private:
    int64_t V_ = 0, D_ = 0, d_ff_ = 0;
    float   eps_ = 1e-6f;
    float   embed_scale_ = 1.0f;
    float   final_softcap_ = 30.0f;
    int64_t n_params_ = 0;

    std::vector<GemmaLayer> layers_;
    std::vector<float>      rope_freqs_;   // global freq_factors (head_dim_global/2)
    std::vector<float>      output_norm_;  // (D)
    std::vector<backend::cpu::BlockQ8_0> token_embd_;  // (V, D) — embedding + tied LM head
};

} // namespace sub0llm::nn
