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

// Fused A·Bᵀ: (M, K) × (N, K)ᵀ → (M, N), 2D only. B stays row-major — no transposed
// copy is materialized. The natural orientation for a weight-tied LM head (x · Wᵀ).
[[nodiscard]] Tensor matmul_bt(const Tensor& a, const Tensor& b);

// Fused Aᵀ·B: (M, K)ᵀ × (M, N) → (K, N), 2D only. A stays row-major (rank-1 row
// updates) — the matmul-backward workhorse (dL/dB = Aᵀ·g without a transpose copy).
[[nodiscard]] Tensor matmul_tb(const Tensor& a, const Tensor& b);

// Zero-allocation matmul into a PREALLOCATED contiguous output (CPU/f32, 2D). The
// caller owns c, sized (M,N) for matmul_into / (M,N) for matmul_bt_into. The building
// block for the zero-alloc training hot path and for threaded GEMM (each worker writes
// a disjoint M-row block of the same c). No graph, no autograd — raw kernel dispatch.
void matmul_into(const Tensor& a, const Tensor& b, Tensor& c);      // C = A·B,  B (K,N)  → C(M,N)
void matmul_bt_into(const Tensor& a, const Tensor& b, Tensor& c);   // C = A·Bᵀ, B (N,K)  → C(M,N)
void matmul_tb_into(const Tensor& a, const Tensor& b, Tensor& c);   // C = Aᵀ·B, A(M,K),B(M,N) → C(K,N)

// ── Unary ─────────────────────────────────────────────────────────────────────
[[nodiscard]] Tensor exp(const Tensor& t);
[[nodiscard]] Tensor log(const Tensor& t);
[[nodiscard]] Tensor sqrt(const Tensor& t);
[[nodiscard]] Tensor abs(const Tensor& t);
[[nodiscard]] Tensor neg(const Tensor& t);

// ── Norms ─────────────────────────────────────────────────────────────────────
[[nodiscard]] float  norm(const Tensor& t);        // L2 norm

// ── Indexing / slicing (Ch11) ─────────────────────────────────────────────────
// Extract a contiguous slice along dim 0.
//   t:      (T, ...) input
//   start:  first row index (0-indexed)
//   length: number of rows to extract
// Returns tensor of shape (length, ...).
[[nodiscard]] Tensor narrow(const Tensor& t, int64_t start, int64_t length);

} // namespace sub0llm::ops
