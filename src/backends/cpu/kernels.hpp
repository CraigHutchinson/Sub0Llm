#pragma once
#include <cstddef>

// Internal CPU kernel API — selected at compile time by CMake feature flags.
// ops.cpp calls these; the implementation file picks the best available path:
//   SUB0LLM_AVX512 → 16-wide float ops
//   SUB0LLM_AVX2   →  8-wide float ops  (default on modern x86-64)
//   else           → portable scalar fallback

namespace sub0llm::backend::cpu {

// ── Element-wise binary ───────────────────────────────────────────────────────
void add_f32 (const float* a, const float* b, float* out, std::size_t n) noexcept;
void sub_f32 (const float* a, const float* b, float* out, std::size_t n) noexcept;
void mul_f32 (const float* a, const float* b, float* out, std::size_t n) noexcept;
void div_f32 (const float* a, const float* b, float* out, std::size_t n) noexcept;

// ── Scalar broadcast ─────────────────────────────────────────────────────────
void add_scalar_f32(const float* a, float s, float* out, std::size_t n) noexcept;
void mul_scalar_f32(const float* a, float s, float* out, std::size_t n) noexcept;

// ── Activations ──────────────────────────────────────────────────────────────
void relu_f32   (const float* in, float* out, std::size_t n) noexcept;
void neg_f32    (const float* in, float* out, std::size_t n) noexcept;
void exp_f32    (const float* in, float* out, std::size_t n) noexcept;
void log_f32    (const float* in, float* out, std::size_t n) noexcept;
void sqrt_f32   (const float* in, float* out, std::size_t n) noexcept;
void abs_f32    (const float* in, float* out, std::size_t n) noexcept;
void sigmoid_f32(const float* in, float* out, std::size_t n) noexcept;

// ── Reduction ─────────────────────────────────────────────────────────────────
float sum_f32 (const float* in, std::size_t n) noexcept;
float max_f32 (const float* in, std::size_t n) noexcept;
float min_f32 (const float* in, std::size_t n) noexcept;
float norm_f32(const float* in, std::size_t n) noexcept;

// ── Matrix multiply ───────────────────────────────────────────────────────────
// C = A × B   where A(M,K), B(K,N), C(M,N) — row-major, C is zeroed inside.
void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t N, std::size_t K) noexcept;

} // namespace sub0llm::backend::cpu
