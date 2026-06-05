#pragma once

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/backends/cpu/quant.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/attention.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/positional_encoding.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// Quantize-on-load (Ch27): a friend loader fills Q8 buffers directly from a GGUF,
// never materializing f32. Forward-declared so the nn headers needn't include the
// gguf loader (which depends on them).
class GGUFReader;
class ModernGPT;
[[nodiscard]] ModernGPT load_gguf_model_q8(const GGUFReader& reader);

// Linear projection: y = x @ W^T + b
//   x: (T, in_features) → (T, out_features)
//   W: (out_features, in_features), Xavier uniform init
//   b: (out_features,), zero init
class Linear {
public:
    // alloc_weights=false elides the (out×in) f32 weight (placeholder only); the
    // Q8 buffer is filled by the quantize-on-load path. Bias is always allocated.
    Linear(int64_t in_features, int64_t out_features, std::uint64_t seed = 42,
           bool alloc_weights = true);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;

    // Inference-only: x @ W^T + b using pure Tensor ops (no autograd graph).
    // x: (*, in_features) → (*, out_features).
    [[nodiscard]] Tensor apply_one(const Tensor& x) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    // Attach a trainable low-rank adapter and FREEZE the base weight/bias:
    //   y = x @ Wᵀ + b + (alpha/rank) · (x @ A) @ B
    //   A: (in, rank) small-random, B: (rank, out) zeros → zero initial delta.
    // Once enabled, parameters() also returns A,B (the only trainable params).
    // Used by episodic memory to confine the write to a low-rank, base-frozen
    // delta (specificity by construction).
    void enable_lora(int64_t rank, float alpha, std::uint64_t seed);
    [[nodiscard]] bool lora_enabled() const noexcept { return lora_scaling_ != 0.0f; }

    // Quantize the weight to Q8_0 for fast int8 single-token inference (Ch27).
    // After this, apply_one() on a single row uses the quantized matmul (≈1.4–1.7×
    // faster, ~0.4% relRMS). No-op when in_features is not a multiple of 32. The
    // f32 weight is retained for the autograd/batched path.
    void quantize_weights();
    [[nodiscard]] bool q8_enabled() const noexcept { return !wq8_.empty(); }

    // Pure-inference RAM reclaim: drop the f32 weight after quantizing (the Q8 path
    // no longer needs it). Reclaims ~3.76× the weight's RAM. The autograd/batched
    // path is unusable afterward (asserts off here) — inference only.
    void free_f32_weights();

private:
    autograd::Variable W_;
    autograd::Variable b_;
    autograd::Variable lora_A_;            // (in, rank)  — trainable when enabled
    autograd::Variable lora_B_;            // (rank, out) — trainable when enabled
    float              lora_scaling_ = 0.0f;  // 0 = no adapter
    int64_t            in_features_ = 0;   // cached so the Q8 path survives free_f32
    int64_t            out_features_ = 0;
    std::vector<backend::cpu::BlockQ8_0> wq8_;  // Q8 weight (out × in/32 blocks), opt-in

    friend ModernGPT load_gguf_model_q8(const GGUFReader&);
};

// Layer normalisation with learnable affine parameters.
//   weight (gamma): (D,), initialised to 1
//   bias   (beta):  (D,), initialised to 0
class LayerNorm {
public:
    explicit LayerNorm(int64_t D, float eps = 1e-5f);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    autograd::Variable weight_;
    autograd::Variable bias_;
    float              eps_;
};

// Position-wise feed-forward network.
//   FFN(x) = gelu(x @ W1^T + b1) @ W2^T + b2
//   Expansion factor 4: W1 (4D, D), W2 (D, 4D)
class FeedForward {
public:
    FeedForward(int64_t D, std::uint64_t seed = 42);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    Linear fc1_;
    Linear fc2_;
};

// GPT-style transformer block with pre-norm.
//   x = x + attn(norm1(x))   (causal self-attention)
//   x = x + ffn(norm2(x))    (feed-forward network)
class TransformerBlock {
public:
    TransformerBlock(int64_t D, int64_t num_heads, std::uint64_t seed = 42);

    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x) const;
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    LayerNorm              norm1_;
    MultiHeadSelfAttention attn_;
    LayerNorm              norm2_;
    FeedForward            ffn_;
};

// Minimal GPT model: token embedding + positional encoding + N transformer
// blocks + final layer norm + weight-tied language model head.
//
//   forward(ids) : (T,) int32 → (T, vocab_size) logits
//   logits = ln_f(x) @ W_emb^T   (weight tying with tok_emb)
class GPT {
public:
    GPT(int64_t vocab_size, int64_t embed_dim, int64_t num_heads,
        int64_t num_layers, int64_t max_seq_len, std::uint64_t seed = 42);

    // Returns logits of shape (T, vocab_size).
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // All trainable parameters (tok_emb weight appears once; it is shared with
    // the LM head via weight tying).
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t vocab_size()  const noexcept { return tok_emb_.vocab_size(); }
    [[nodiscard]] int64_t embed_dim()   const noexcept { return tok_emb_.embed_dim(); }
    [[nodiscard]] std::size_t num_layers() const noexcept { return blocks_.size(); }

private:
    Embedding              tok_emb_;
    LearnedPositionalEncoding pos_emb_;
    std::vector<TransformerBlock> blocks_;
    LayerNorm              ln_f_;
};

} // namespace sub0llm::nn
