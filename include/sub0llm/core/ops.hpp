#pragma once
#include "tensor.hpp"

// ── Ch01 / Ch02 naive CPU ops — educational baseline ─────────────────────────
// SIMD-accelerated versions live in src/backends/cpu_simd.cpp (Ch02).
// CUDA kernels live in src/backends/cuda/ (Ch02).
//
// These ops are intentionally simple: loop-based, float32, no broadcasting.
// Broadcasting, mixed dtypes, and GPU dispatch are added in later chapters.

namespace sub0llm::ops {

// ── Element-wise arithmetic ───────────────────────────────────────────────────
[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor sub(const Tensor& a, const Tensor& b);
[[nodiscard]] Tensor mul(const Tensor& a, const Tensor& b);   // element-wise
[[nodiscard]] Tensor div(const Tensor& a, const Tensor& b);   // element-wise

// Scalar broadcast versions.
[[nodiscard]] Tensor add(const Tensor& a, float scalar);
[[nodiscard]] Tensor mul(const Tensor& a, float scalar);

// ── Reduction ops ─────────────────────────────────────────────────────────────
[[nodiscard]] float  sum(const Tensor& t);
[[nodiscard]] float  mean(const Tensor& t);
[[nodiscard]] float  max(const Tensor& t);
[[nodiscard]] float  min(const Tensor& t);

// ── Activations ───────────────────────────────────────────────────────────────
[[nodiscard]] Tensor relu(const Tensor& t);
[[nodiscard]] Tensor gelu(const Tensor& t);   // added in Ch08 (GPT uses GELU)
[[nodiscard]] Tensor silu(const Tensor& t);   // added in Ch10 (SwiGLU uses SiLU)
[[nodiscard]] Tensor softmax(const Tensor& t, int dim = -1);
[[nodiscard]] Tensor sigmoid(const Tensor& t);

// ── Linear algebra ────────────────────────────────────────────────────────────
// Matrix multiply: (M, K) × (K, N) → (M, N).
// Batched: (..., M, K) × (..., K, N) → (..., M, N).
[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b);

// ── Unary ─────────────────────────────────────────────────────────────────────
[[nodiscard]] Tensor exp(const Tensor& t);
[[nodiscard]] Tensor log(const Tensor& t);
[[nodiscard]] Tensor sqrt(const Tensor& t);
[[nodiscard]] Tensor abs(const Tensor& t);
[[nodiscard]] Tensor neg(const Tensor& t);

// ── Norms ─────────────────────────────────────────────────────────────────────
[[nodiscard]] float  norm(const Tensor& t);        // L2 norm

} // namespace sub0llm::ops
