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
    Embedding(std::int64_t vocab_size, std::int64_t embed_dim,
              std::uint64_t seed = 42);

    // Forward pass.
    // indices: (T,) or (B, T) int32 → output: (T, D) or (B, T, D)
    [[nodiscard]] autograd::Variable forward(const Tensor& indices) const;

    [[nodiscard]] autograd::Variable&       weight()       noexcept { return weight_; }
    [[nodiscard]] const autograd::Variable& weight() const noexcept { return weight_; }

    [[nodiscard]] std::int64_t vocab_size()  const noexcept;
    [[nodiscard]] std::int64_t embed_dim()   const noexcept;

private:
    autograd::Variable weight_;
};

} // namespace sub0llm::nn
