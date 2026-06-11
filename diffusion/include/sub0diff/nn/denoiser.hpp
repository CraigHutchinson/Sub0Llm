#pragma once

// denoiser.hpp (Ch28) — the diffusion model: a BIDIRECTIONAL transformer that, given a
// corrupted sequence and its noise level, predicts the clean token at every position.
//
// It reuses sub0llm's transformer building blocks unchanged. The one essential difference
// from an autoregressive GPT is the attention mask: a denoiser must let every position see
// every other position (the future is not "cheating" — there is no left-to-right order to
// protect), so we call GroupedQueryAttention::forward(x, /*causal=*/false). Because the
// Ch10 ModernTransformerBlock hardcodes causal attention, we compose our own bidirectional
// block here from the public RMSNorm / GroupedQueryAttention / SwiGLUFeedForward parts —
// additive, with zero changes to the shared autoregressive code.

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <vector>

namespace sub0diff::nn {

// Pre-norm transformer block with FULL (bidirectional) self-attention.
//   h = x + Attn_noncausal(RMSNorm(x))
//   y = h + SwiGLU(RMSNorm(h))
class BidirectionalBlock {
public:
    BidirectionalBlock(std::int64_t D, std::size_t n_heads, std::size_t n_kv_heads,
                       std::int64_t d_ff = 0, std::uint64_t seed = 42, float rope_base = 10000.0f);

    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::autograd::Variable& x) const;
    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters();

private:
    sub0llm::nn::RMSNorm                norm1_;
    sub0llm::nn::GroupedQueryAttention  attn_;
    sub0llm::nn::RMSNorm                norm2_;
    sub0llm::nn::SwiGLUFeedForward      ffn_;
};

// The denoiser. vocab_size is the REAL vocabulary; the model is built with vocab_size+1 rows
// so the [MASK] absorbing-state token (id == vocab_size) has its own embedding.
class Denoiser {
public:
    Denoiser(std::int64_t vocab_size, std::int64_t embed_dim,
             std::size_t n_heads, std::size_t n_kv_heads, std::int64_t n_layers,
             std::int64_t d_ff = 0, std::uint64_t seed = 42);

    // token_ids: (T,) int32 (may contain the mask id). noise_level ∈ [0,1].
    // Returns logits (T, vocab_size+1) over the clean token at every position.
    [[nodiscard]] sub0llm::autograd::Variable forward(const sub0llm::Tensor& token_ids,
                                                      float noise_level) const;

    [[nodiscard]] std::vector<sub0llm::autograd::Variable*> parameters();

    [[nodiscard]] std::int32_t mask_id()       const noexcept { return static_cast<std::int32_t>(real_vocab_); }
    [[nodiscard]] std::int64_t real_vocab()    const noexcept { return real_vocab_; }
    [[nodiscard]] std::int64_t model_vocab()   const noexcept { return real_vocab_ + 1; }
    [[nodiscard]] std::int64_t embed_dim()     const noexcept { return embed_dim_; }

private:
    std::int64_t                    real_vocab_;
    std::int64_t                    embed_dim_;
    sub0llm::nn::Embedding          tok_emb_;
    std::vector<BidirectionalBlock> blocks_;
    sub0llm::nn::RMSNorm            ln_f_;
};

} // namespace sub0diff::nn
