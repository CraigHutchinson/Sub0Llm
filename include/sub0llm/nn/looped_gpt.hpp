#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/modern_gpt.hpp"

#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// ── LoopedGPT ─────────────────────────────────────────────────────────────────
//
// Universal-Transformer variant: a single ModernTransformerBlock run K times
// per forward pass.  Every loop shares the same weights, so the model has
// dramatically fewer parameters than a K-layer ModernGPT while using the same
// amount of compute.
//
// Architecture (per forward call):
//   x = token_embedding(token_ids)          (T, D)
//   for k in 0 … n_loops-1:
//       x = block.forward(x)                (residual + GQA-RoPE + SwiGLU)
//   x = final_norm(x)
//   logits = x · tok_emb_weight^T           (T, V)  weight-tied LM head
//
// n_loops is the training-time loop count; forward_k() allows a different
// count at inference time, making K a runtime reasoning budget.
class LoopedGPT {
public:
    // vocab_size  : number of tokens
    // embed_dim   : hidden dimension D
    // n_heads     : query heads
    // n_kv_heads  : key/value heads (1 = MQA, n_heads = MHA)
    // n_loops     : K — how many times the block is applied each forward pass
    // d_ff        : FFN hidden dim (0 = auto: 8*D/3 rounded up to 64)
    // seed        : RNG seed for weight initialisation
    LoopedGPT(int64_t     vocab_size,
              int64_t     embed_dim,
              std::size_t n_heads,
              std::size_t n_kv_heads,
              int64_t     n_loops,
              int64_t     d_ff  = 0,
              std::uint64_t seed = 42);

    // Standard forward: run block n_loops_ times.
    // token_ids: 1-D (T,) Int32 tensor.
    // Returns logits (T, vocab_size).
    [[nodiscard]] autograd::Variable forward(const Tensor& token_ids) const;

    // Inference-time forward with an explicit loop count (reasoning budget).
    // k must be >= 1.
    [[nodiscard]] autograd::Variable forward_k(const Tensor& token_ids,
                                                int64_t k) const;

    [[nodiscard]] std::vector<autograd::Variable*> parameters();

    [[nodiscard]] int64_t     vocab_size() const noexcept;
    [[nodiscard]] int64_t     embed_dim()  const noexcept;
    [[nodiscard]] int64_t     n_loops()    const noexcept;

private:
    autograd::Variable forward_impl(const Tensor& token_ids, int64_t k) const;

    Embedding              tok_emb_;
    ModernTransformerBlock block_;
    RMSNorm                ln_f_;
    int64_t                n_loops_;
};

} // namespace sub0llm::nn
