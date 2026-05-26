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

// ── Activations (Ch08) ────────────────────────────────────────────────────────

// Gaussian Error Linear Unit (tanh approximation, same as GPT-2).
[[nodiscard]] Variable gelu(const Variable& x);

// ── Activations / norms (Ch10) ────────────────────────────────────────────────

// SiLU activation: y = x * sigmoid(x).  Used as the gate in SwiGLU FFNs.
[[nodiscard]] Variable silu(const Variable& x);

// RMS Normalisation — no mean subtraction, no bias (LLaMA/Gemma/Qwen style).
//   y[t,d] = weight[d] * x[t,d] / RMS(x[t,:])
//   RMS(v) = sqrt((1/D) * Σ v_i² + eps)
//   x: (T, D),  weight: (D,).
[[nodiscard]] Variable rms_norm(const Variable& x, const Variable& weight,
                                float eps = 1e-6f);

// Rotary Position Embedding (RoPE) — applied to one head's Q or K.
//   x:         (T, Dh)      input head tensor
//   cos_freqs: (T, Dh/2)   cos(pos * θ_i),  θ_i = base^(-2i/Dh)
//   sin_freqs: (T, Dh/2)   sin(pos * θ_i)
// For each position t and pair i:
//   out[t, 2i]   = x[t,2i]*cos[t,i] − x[t,2i+1]*sin[t,i]
//   out[t, 2i+1] = x[t,2i]*sin[t,i] + x[t,2i+1]*cos[t,i]
[[nodiscard]] Variable rope(const Variable& x,
                             const Tensor& cos_freqs,
                             const Tensor& sin_freqs);

// Layer normalisation with affine transform.
//   x      : (T, D)   — input to normalise
//   weight : (D,)     — learnable scale (gamma)
//   bias   : (D,)     — learnable shift (beta)
//   eps    : stability constant (default 1e-5)
// Returns Variable (T, D).
[[nodiscard]] Variable layer_norm(const Variable& x,
                                  const Variable& weight,
                                  const Variable& bias,
                                  float eps = 1e-5f);

// ── Attention ops ─────────────────────────────────────────────────────────────

// Row-wise softmax over the last dimension.  Input must be 2D (N, C).
// Backward: grad_x[i] = y[i] * (g[i] - dot(g[i], y[i])) per row.
[[nodiscard]] Variable softmax(const Variable& x);

// Swap dimensions 0 and 1 of a 2D matrix: (M, N) → (N, M).
[[nodiscard]] Variable transpose2d(const Variable& x);

// Multiply all elements by a scalar constant (no learnable parameter).
[[nodiscard]] Variable scale(const Variable& x, float alpha);

// ── Indexing / slicing (Ch11) ─────────────────────────────────────────────────

// Extract rows [start, start+length) along dim 0 of a 2-D or 1-D Variable.
// Backward scatters upstream gradient back into the original row range.
[[nodiscard]] Variable narrow(const Variable& x, int64_t start, int64_t length);

// ── Operator overloads (thin wrappers) ────────────────────────────────────────

inline Variable operator+(const Variable& a, const Variable& b) { return add(a, b); }
inline Variable operator-(const Variable& a, const Variable& b) { return sub(a, b); }
inline Variable operator*(const Variable& a, const Variable& b) { return mul(a, b); }

} // namespace sub0llm::autograd
