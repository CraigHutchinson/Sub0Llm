#pragma once

#include "sub0llm/autograd/variable.hpp"

namespace sub0llm::autograd {

// Gather rows from a weight matrix by integer token indices.
//
//   weight : Variable (V, D),  requires_grad may be true
//   indices: Tensor   (T,) or (B, T)  of int32 values in [0, V)
//   output : Variable (T, D) or (B, T, D)
//
// Backward: sparse scatter-add — for each position i, j:
//   weight.grad[indices[i,j], :] += upstream[i, j, :]
// Same-token appearances accumulate (correct for repeated tokens).
[[nodiscard]] Variable embedding_lookup(const Variable& weight,
                                        const Tensor&   indices);

} // namespace sub0llm::autograd
