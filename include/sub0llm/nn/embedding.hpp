#pragma once

#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/tensor.hpp"

#include <cstdint>

namespace sub0llm::nn {

// A differentiable lookup table: maps integer token ids to dense vectors.
//
// The weight matrix W is (vocab_size, embed_dim); forward(indices) returns
// W[indices[i,j], :] for each position, and backward accumulates
// dL/dW[tok, :] += upstream[i, j, :] for each occurrence of tok.
class Embedding {
public:
    // alloc_weights=false elides the (V×D) f32 table (placeholder only) for a
    // pure-Q8 inference load that never materializes f32 (Ch27 quantize-on-load).
    Embedding(std::int64_t vocab_size, std::int64_t embed_dim,
              std::uint64_t seed = 42, bool alloc_weights = true);

    // Forward pass.
    // indices: (T,) or (B, T) int32 → output: (T, D) or (B, T, D)
    [[nodiscard]] autograd::Variable forward(const Tensor& indices) const;

    [[nodiscard]] autograd::Variable&       weight()       noexcept { return weight_; }
    [[nodiscard]] const autograd::Variable& weight() const noexcept { return weight_; }

    [[nodiscard]] std::int64_t vocab_size()  const noexcept;
    [[nodiscard]] std::int64_t embed_dim()   const noexcept;

private:
    autograd::Variable weight_;
    std::int64_t       vocab_size_ = 0;   // cached so dims survive an elided table
    std::int64_t       embed_dim_  = 0;
};

} // namespace sub0llm::nn
