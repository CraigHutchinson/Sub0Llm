#pragma once
#include <cstddef>

// Internal CPU kernel API — selected at compile time by CMake feature flags.
// ops.cpp calls these; the implementation file picks the best available path:
//   SUB0LLM_AVX512 → 16-wide float ops
//   SUB0LLM_AVX2   →  8-wide float ops  (default on modern x86-64)
//   else           → portable scalar fallback

namespace sub0llm::backend::cpu {

// ── Element-wise binary ───────────────────────────────────────────────────────
void add_f32 (const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept;
void sub_f32 (const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept;
void mul_f32 (const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept;
void div_f32 (const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept;

// ── Scalar broadcast ─────────────────────────────────────────────────────────
void add_scalar_f32(const float* __restrict__ a, float s, float* __restrict__ out, std::size_t n) noexcept;
void mul_scalar_f32(const float* __restrict__ a, float s, float* __restrict__ out, std::size_t n) noexcept;

// ── Activations ──────────────────────────────────────────────────────────────
void relu_f32   (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
void neg_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
void exp_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
void log_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
void sqrt_f32   (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
void abs_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
void sigmoid_f32(const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
// SiLU: out = in * sigmoid(in).  Uses fast polynomial exp — same accuracy as sigmoid_f32.
void silu_f32   (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;
// GELU: out = in * sigmoid(2k), k = sqrt(2/π)*(in + 0.044715*in³). Tanh-approx via sigmoid identity.
void gelu_f32   (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept;

// ── Activation backward kernels ──────────────────────────────────────────────
// grad_in[i] = grad_out[i] * local_derivative(x[i])
void silu_backward_f32(const float* __restrict__ grad_out, const float* __restrict__ x,
                       float* __restrict__ grad_in, std::size_t n) noexcept;
void gelu_backward_f32(const float* __restrict__ grad_out, const float* __restrict__ x,
                       float* __restrict__ grad_in, std::size_t n) noexcept;

// ── Softmax ───────────────────────────────────────────────────────────────────
// Numerically-stable row-wise softmax: exp(in[r,c] - max_r) / sum.
// Uses fast polynomial exp for throughput; accurate enough for attention and logits.
void softmax_rows_f32(const float* __restrict__ in, float* __restrict__ out,
                      std::size_t rows, std::size_t cols) noexcept;

// ── Reduction ─────────────────────────────────────────────────────────────────
float sum_f32 (const float* __restrict__ in, std::size_t n) noexcept;
float max_f32 (const float* __restrict__ in, std::size_t n) noexcept;
float min_f32 (const float* __restrict__ in, std::size_t n) noexcept;
float norm_f32(const float* __restrict__ in, std::size_t n) noexcept;

// ── Matrix multiply ───────────────────────────────────────────────────────────
// C = A × B   where A(M,K), B(K,N), C(M,N) — row-major, C is zeroed inside.
void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t N, std::size_t K) noexcept;

} // namespace sub0llm::backend::cpu
