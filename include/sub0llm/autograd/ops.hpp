#pragma once

#include "sub0llm/autograd/variable.hpp"

namespace sub0llm::autograd {

// ── Element-wise binary ops ────────────────────────────────────────────────────

[[nodiscard]] Variable add(const Variable& a, const Variable& b);
[[nodiscard]] Variable sub(const Variable& a, const Variable& b);
[[nodiscard]] Variable mul(const Variable& a, const Variable& b);  // Hadamard

// ── Matrix ops ────────────────────────────────────────────────────────────────

[[nodiscard]] Variable matmul(const Variable& a, const Variable& b);

// ── Reductions ────────────────────────────────────────────────────────────────

// Sum all elements; result has shape {1} and numel()==1.
[[nodiscard]] Variable sum(const Variable& x);

// ── Activations ───────────────────────────────────────────────────────────────

[[nodiscard]] Variable relu(const Variable& x);

// Numerically stable log-softmax over the last axis (dim = -1 / last dim).
// Input shape: (N, C) or (C,); output: same shape.
[[nodiscard]] Variable log_softmax(const Variable& x);

// ── Losses ────────────────────────────────────────────────────────────────────

// Cross-entropy loss for multiclass classification.
//   logits : Variable (N, C)
//   targets: Tensor   (N,) of int32 class indices in [0, C)
// Returns a scalar Variable (mean NLL over the batch).
[[nodiscard]] Variable cross_entropy(const Variable& logits, const Tensor& targets);

// Add a bias vector to every row of a 2D matrix.
//   x: (N, C),  b: (C,) or (1, C)  →  output: (N, C)
// Backward: grad_x = upstream, grad_b = sum(upstream over rows).
[[nodiscard]] Variable bias_add(const Variable& x, const Variable& b);

// ── Operator overloads (thin wrappers) ────────────────────────────────────────

inline Variable operator+(const Variable& a, const Variable& b) { return add(a, b); }
inline Variable operator-(const Variable& a, const Variable& b) { return sub(a, b); }
inline Variable operator*(const Variable& a, const Variable& b) { return mul(a, b); }

} // namespace sub0llm::autograd
