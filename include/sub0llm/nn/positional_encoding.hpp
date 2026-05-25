#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/embedding.hpp"

#include <cstdint>

namespace sub0llm::nn {

// Fixed sinusoidal position encoding (Vaswani et al., 2017).
//
//   PE(pos, 2i)   = sin(pos / 10000^(2i / embed_dim))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i / embed_dim))
//
// Returns a Tensor (seq_len, embed_dim) with no gradient.
[[nodiscard]] Tensor sinusoidal_encoding(std::int64_t seq_len,
                                         std::int64_t embed_dim);

// Learned positional encoding: a trainable embedding table indexed by
// position [0, max_seq_len).  Equivalent to a standard Embedding over
// the range [0, seq_len) generated at call time.
class LearnedPositionalEncoding {
public:
    LearnedPositionalEncoding(std::int64_t max_seq_len, std::int64_t embed_dim,
                              std::uint64_t seed = 1);

    // Returns Variable (seq_len, embed_dim) for positions [0, seq_len).
    [[nodiscard]] autograd::Variable forward(std::int64_t seq_len) const;

    [[nodiscard]] autograd::Variable&       weight()       noexcept { return emb_.weight(); }
    [[nodiscard]] const autograd::Variable& weight() const noexcept { return emb_.weight(); }

    [[nodiscard]] std::int64_t max_seq_len() const noexcept;
    [[nodiscard]] std::int64_t embed_dim()   const noexcept;

private:
    Embedding    emb_;
};

// Apply Rotary Position Encoding (RoPE) to a query or key tensor.
//
//   x   : Tensor (T, D)  where D must be even  (single head, no batch)
//   base: frequency base (default 10000, as in LLaMA)
//
// Returns a new Tensor (T, D) with rotated embeddings.
// Note: RoPE is non-differentiable here; Ch07 makes it differentiable inside
// the attention forward pass.
[[nodiscard]] Tensor apply_rope(const Tensor& x, float base = 10000.0f);

} // namespace sub0llm::nn
