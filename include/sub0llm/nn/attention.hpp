#pragma once

#include "sub0llm/autograd/variable.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace sub0llm::nn {

// Multi-head causal self-attention.
//
// Each head h independently computes:
//   Q_h = x @ W_Q[h]                         (T, head_dim)
//   K_h = x @ W_K[h]                         (T, head_dim)
//   V_h = x @ W_V[h]                         (T, head_dim)
//   A_h = softmax(Q_h @ K_h^T / sqrt(D_h) + mask)
//   ctx_h = A_h @ V_h                         (T, head_dim)
//
// Outputs are summed over the head-output projections:
//   output = sum_h  ctx_h @ W_O[h]           (T, embed_dim)
//
// This is mathematically equivalent to the standard concat-then-project
// formulation and avoids needing a concat autograd op.
class MultiHeadSelfAttention {
public:
    // embed_dim must be divisible by num_heads.
    MultiHeadSelfAttention(std::size_t embed_dim, std::size_t num_heads,
                           std::uint64_t seed = 42);

    // x: Variable (T, embed_dim).
    // causal=true applies a lower-triangular mask (future tokens → −∞).
    // Returns Variable (T, embed_dim).
    [[nodiscard]] autograd::Variable forward(const autograd::Variable& x,
                                             bool causal = true) const;

    [[nodiscard]] std::size_t embed_dim() const noexcept { return embed_dim_; }
    [[nodiscard]] std::size_t num_heads() const noexcept { return num_heads_; }
    [[nodiscard]] std::size_t head_dim()  const noexcept { return head_dim_; }

    // Non-const: returns mutable pointers so optimisers can update weights.
    [[nodiscard]] std::vector<autograd::Variable*> parameters();

private:
    std::size_t embed_dim_, num_heads_, head_dim_;
    // Per-head weights: W_Q[h], W_K[h], W_V[h]: (embed_dim, head_dim)
    //                   W_O[h]:                  (head_dim,  embed_dim)
    std::vector<autograd::Variable> W_Q_, W_K_, W_V_, W_O_;
};

} // namespace sub0llm::nn
