#pragma once

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/gpt.hpp"       // reuses Linear from Ch08
#include "sub0llm/nn/kv_cache.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace sub0llm::nn {

// ── RoPE scaling (Ch25) ───────────────────────────────────────────────────────
//
// NTK-aware scaling extends the effective RoPE context beyond the training
// length without fine-tuning.  The base frequency is scaled so that high-
// frequency dimensions (which encode fine-grained position) are distorted less
// than low-frequency ones.  Formula (Chen et al. 2023):
//
//   base_scaled = base × alpha^(head_dim / (head_dim − 2))
//   alpha       = target_len / train_len
//
// Set scale_factor = 1.0 (default) to disable scaling.
struct RopeScalingConfig {
    float scale_factor = 1.0f; // > 1 extends context; 1 = no scaling (default)
};

// ── RMSNorm ───────────────────────────────────────────────────────────────────
//
// Replaces LayerNorm: no mean subtraction, no bias, cheaper to compute.
//   y[t,d] = weight[d] * x[t,d] / RMS(x[t,:])
//   weight (gamma): (D,), initialised to 1.
class RMSNorm {
public:
    explicit RMSNorm(int64_t D, float eps = 1e-6f);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;

    // Inference-only: pure Tensor path, no autograd. x: (1, D) → (1, D).
    [[nodiscard]] Tensor apply_one(const Tensor& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t dim() const noexcept;

private:
    autograd::Variable weight_;
    float              eps_;

    friend ModernGPT load_gguf_model_q8(const GGUFReader&);
};

// ── SwiGLUFeedForward ─────────────────────────────────────────────────────────
//
// Replaces GELU FFN.  Three projections:
//   gate(x) = SiLU(W_gate · x)    (D → d_ff)
//   up(x)   = W_up   · x          (D → d_ff)
//   down(x) = W_down · (gate ⊙ up) (d_ff → D)
//
// d_ff defaults to 8*D/3 (rounded to nearest multiple of 64) to match the
// parameter count of the 2-layer GELU FFN.
class SwiGLUFeedForward {
public:
    // d_ff = 0 → auto-compute 8*D/3 rounded up to next 64.
    SwiGLUFeedForward(int64_t D, int64_t d_ff = 0, std::uint64_t seed = 42,
                      bool alloc_weights = true);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;

    // Inference-only: pure Tensor path. x: (1, D) → (1, D).
    [[nodiscard]] Tensor apply_one(const Tensor& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    // Attach low-rank adapters to all three projections (base frozen).
    void enable_lora(int64_t rank, float alpha, std::uint64_t seed);

    // Quantize all three projections to Q8_0 for fast int8 inference (Ch27).
    void quantize_weights();
    void free_f32_weights();   // drop f32 after quantizing (inference-only RAM reclaim)

private:
    Linear gate_;
    Linear up_;
    Linear down_;

    friend ModernGPT load_gguf_model_q8(const GGUFReader&);
};

// ── GroupedQueryAttention ─────────────────────────────────────────────────────
//
// Generalises MHA with n_kv_heads ≤ n_heads shared KV heads (GQA).
// Special cases: n_kv_heads == n_heads → standard MHA;
//                n_kv_heads == 1       → MQA (Multi-Query Attention).
//
// RoPE positional embeddings are applied to Q and K inside forward().
// Frequencies are recomputed each call (O(T·head_dim) trig ops — acceptable
// for education; a real implementation would cache them).
//
// Weight layout:
//   W_Q[h]:  (embed_dim, head_dim)   — n_heads  matrices
//   W_K[g]:  (embed_dim, head_dim)   — n_kv_heads matrices
//   W_V[g]:  (embed_dim, head_dim)   — n_kv_heads matrices
//   W_O[h]:  (head_dim,  embed_dim)  — n_heads  matrices
class GroupedQueryAttention {
public:
    // embed_dim must be divisible by n_heads; n_heads must be divisible by n_kv_heads.
    // window_size: sliding-window context (tokens attend only to the last W tokens).
    //              -1 = full causal attention (default).
    GroupedQueryAttention(std::size_t   embed_dim,
                          std::size_t   n_heads,
                          std::size_t   n_kv_heads,
                          float         rope_base    = 10000.0f,
                          std::uint64_t seed         = 42,
                          int64_t       window_size  = -1,
                          std::size_t   head_dim     = 0,
                          bool          use_qk_norm  = false,
                          bool          alloc_weights = true);

    // x: (T, embed_dim); causal = apply lower-triangular (or sliding-window) mask.
    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x,
                                              bool causal = true) const;

    // Inference-only single-token forward with KV cache.
    // x_new: (1, embed_dim). pos: absolute position of this token.
    // Appends K,V to cache[layer_idx] and returns output (1, embed_dim).
    [[nodiscard]] Tensor forward_one(const Tensor& x_new,
                                      int64_t       pos,
                                      KVCache&      cache,
                                      std::size_t   layer_idx,
                                      float         rope_base_override = 0.0f) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    // Quantize Q/K/V/O projections to Q8_0 (transposed to row-major out×in) for the
    // int8 forward_one() fast path (Ch27). No-op if dims aren't multiples of 32.
    void quantize_weights();
    void free_f32_weights();   // drop f32 Q/K/V/O after quantizing (RAM reclaim)

    [[nodiscard]] std::size_t n_heads()     const noexcept { return n_heads_; }
    [[nodiscard]] std::size_t n_kv_heads()  const noexcept { return n_kv_heads_; }
    [[nodiscard]] std::size_t head_dim()    const noexcept { return head_dim_; }
    [[nodiscard]] int64_t     window_size() const noexcept { return window_size_; }

private:
    std::size_t embed_dim_, n_heads_, n_kv_heads_, head_dim_;
    float       rope_base_;
    int64_t     window_size_;  // -1 = full attention
    std::vector<autograd::Variable> W_Q_, W_O_;  // n_heads each
    std::vector<autograd::Variable> W_K_, W_V_;  // n_kv_heads each
    // Q8 inference weights (transposed to out×in row-major), opt-in via
    // quantize_weights(). Empty = f32 path. WQ/WK/WV: (Dh × D/32); WO: (D × Dh/32).
    std::vector<std::vector<backend::cpu::BlockQ8_0>> wq_q8_, wk_q8_, wv_q8_, wo_q8_;
    bool                            q8_ = false;
    // Qwen3 QK-normalisation: RMSNorm over each head's head_dim slice of Q and K,
    // applied before RoPE.  Shared across heads (one weight of dim head_dim each).
    // Absent (nullopt) for architectures without QK-norm (LLaMA, Qwen2, …).
    bool                            use_qk_norm_ = false;
    std::optional<RMSNorm>          q_norm_;
    std::optional<RMSNorm>          k_norm_;
    // Cached RoPE frequencies — recomputed only when T changes.
    mutable int64_t        rope_cache_T_  = -1;
    mutable Tensor         rope_cache_cos_;
    mutable Tensor         rope_cache_sin_;
    // Cached causal/sliding-window mask — recomputed only when T or window_size changes.
    mutable int64_t        mask_cache_T_  = -1;
    mutable Tensor         mask_cache_;

    friend ModernGPT load_gguf_model_q8(const GGUFReader&);
};

// ── ModernTransformerBlock ────────────────────────────────────────────────────
//
// Pre-norm GPT block using modern components:
//   h = x + GQA_RoPE(RMSNorm(x))     (residual #1)
//   y = h + SwiGLU(RMSNorm(h))        (residual #2)
class ModernTransformerBlock {
public:
    ModernTransformerBlock(int64_t       D,
                           std::size_t   n_heads,
                           std::size_t   n_kv_heads,
                           int64_t       d_ff         = 0,
                           std::uint64_t seed         = 42,
                           int64_t       window_size  = -1,
                           std::size_t   head_dim     = 0,
                           bool          use_qk_norm  = false,
                           bool          alloc_weights = true);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;

    // Inference-only: single token, KV-cached. Returns (1, D).
    [[nodiscard]] Tensor forward_one(const Tensor& x,
                                      int64_t       pos,
                                      KVCache&      cache,
                                      std::size_t   layer_idx,
                                      float         rope_base_override = 0.0f) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    // Attach low-rank adapters to this block's feed-forward network (base frozen).
    void enable_lora(int64_t rank, float alpha, std::uint64_t seed);

    // Quantize the feed-forward projections to Q8_0 for fast int8 inference (Ch27).
    void quantize_weights();
    void free_f32_weights();   // drop f32 attn + ffn weights after quantizing

private:
    RMSNorm               norm1_;
    GroupedQueryAttention  attn_;
    RMSNorm               norm2_;
    SwiGLUFeedForward     ffn_;

    friend ModernGPT load_gguf_model_q8(const GGUFReader&);
};

// ── ModernGPT ─────────────────────────────────────────────────────────────────
//
// LLaMA / Gemma-style GPT:
//   • Token embedding (weight-tied with LM head)
//   • No learned positional embedding (RoPE applied inside each attention block)
//   • N × ModernTransformerBlock
//   • Final RMSNorm
//   • LM head = transpose(tok_emb weight)  (weight tying)
//
// Optional MTP (Multi-Token Prediction) heads (n_mtp_heads > 0):
//   Extra linear projections from final hidden state, each predicting
//   the token at offset +k (k = 1 … n_mtp_heads).  Only used at
//   training time via forward_mtp(); forward() uses the tied head only.
class ModernGPT {
public:
    ModernGPT(int64_t       vocab_size,
              int64_t       embed_dim,
              std::size_t   n_heads,
              std::size_t   n_kv_heads,
              int64_t       n_layers,
              int64_t       d_ff          = 0,
              int64_t       n_mtp_heads   = 0,
              std::uint64_t seed          = 42,
              int64_t       window_size   = -1,
              std::size_t   head_dim      = 0,
              bool          use_qk_norm   = false,
              bool          alloc_weights = true);

    // Returns logits (T, vocab_size) — training path via autograd.
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // MTP training: returns per-head logits including the weight-tied head.
    [[nodiscard]] std::vector<autograd::Variable> forward_mtp(
        const Tensor& token_ids) const;

    // ── KV-cached inference (Ch25) ─────────────────────────────────────────────

    // Allocate a KV cache for up to max_seq tokens.
    [[nodiscard]] KVCache make_kv_cache(int64_t max_seq) const;

    // Single-token forward with KV cache. Returns logits (1, vocab_size).
    // Appends this token's K,V to cache and increments cache.filled.
    // rope_scaling: NTK scale factor (>1 extends context; 1.0 = disabled).
    [[nodiscard]] Tensor forward_one(int32_t  token_id,
                                      KVCache& cache,
                                      float    rope_scaling = 1.0f) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    // Restrict gradient updates to the last `n` transformer blocks (+ final norm),
    // freezing the embedding and earlier blocks. Used for cheap, localised
    // fine-tuning / episodic writes: backprop skips the frozen subgraph.
    //   n < 0  → last ceil(num_layers/2) blocks (half)
    //   n == 0 → all parameters trainable (full model)
    //   n > 0  → last min(n, num_layers) blocks
    void set_trainable_last_layers(int n);

    // Freeze ALL base weights and attach trainable low-rank (LoRA) adapters to the
    // feed-forward networks of the last `last_layers` blocks (-1 = last half). The
    // adapters become the only trainable parameters, so an episodic write is
    // confined to a small, base-frozen, low-rank delta — specific by construction
    // and cheap in RAM/compute. Call before make_episodic_state so the adapter
    // weights are included in the parameter/delta layout.
    void enable_episodic_lora(int last_layers, int64_t rank, float alpha = -1.0f);

    // Quantize the weight-heavy inference GEMVs (attention Q/K/V/O, FFN projections,
    // and the LM head) to Q8_0 so forward_one() runs the int8 fast path (Ch27).
    //
    // drop_f32: after quantizing, free the f32 attention + FFN weights (the Q8 path
    // no longer needs them) — reclaims ~3.76× their RAM. The tied embedding/LM-head
    // table stays f32 (still used for the per-token embedding lookup). Inference-only:
    // the autograd/training path is unusable after dropping f32.
    void quantize_for_inference(bool drop_f32 = false);

    [[nodiscard]] int64_t     vocab_size()   const noexcept;
    [[nodiscard]] int64_t     embed_dim()    const noexcept;
    [[nodiscard]] std::size_t num_layers()   const noexcept;
    [[nodiscard]] int64_t     n_mtp_heads()  const noexcept;
    [[nodiscard]] std::size_t n_kv_heads()   const noexcept;
    [[nodiscard]] std::size_t head_dim()     const noexcept;

private:
    Embedding                           tok_emb_;
    std::vector<ModernTransformerBlock>  blocks_;
    RMSNorm                             ln_f_;
    std::vector<Linear>                 mtp_heads_;  // empty if n_mtp_heads == 0
    std::size_t                         n_kv_heads_ = 0;
    std::size_t                         head_dim_   = 0;
    std::vector<backend::cpu::BlockQ8_0> lm_head_q8_;  // Q8 LM head (V × D/32), opt-in
    int64_t                             cached_vocab_ = 0;  // survive tok_emb f32 free
    int64_t                             cached_embed_ = 0;
    bool                                emb_from_q8_  = false;  // dequant embed from lm_head_q8_

    friend ModernGPT load_gguf_model_q8(const GGUFReader&);
};

} // namespace sub0llm::nn
