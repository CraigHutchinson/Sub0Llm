#pragma once
#include <cstddef>
#include <cstdint>


// Define RESTRICT for different compilers
#if defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#elif defined(_MSC_VER)
    #define RESTRICT __restrict
#else
    #define RESTRICT /* no restrict */
#endif

// Internal CPU kernel API — selected at compile time by CMake feature flags.
// ops.cpp calls these; the implementation file picks the best available path:
//   SUB0LLM_AVX512 → 16-wide float ops
//   SUB0LLM_AVX2   →  8-wide float ops  (default on modern x86-64)
//   else           → portable scalar fallback

namespace sub0llm::backend::cpu {

// ── Element-wise binary ───────────────────────────────────────────────────────
void add_f32 (const float* RESTRICT a, const float* RESTRICT b, float* RESTRICT out, std::size_t n) noexcept;
void sub_f32 (const float* RESTRICT a, const float* RESTRICT b, float* RESTRICT out, std::size_t n) noexcept;
void mul_f32 (const float* RESTRICT a, const float* RESTRICT b, float* RESTRICT out, std::size_t n) noexcept;
void div_f32 (const float* RESTRICT a, const float* RESTRICT b, float* RESTRICT out, std::size_t n) noexcept;

// ── Scalar broadcast ─────────────────────────────────────────────────────────
void add_scalar_f32(const float* RESTRICT a, float s, float* RESTRICT out, std::size_t n) noexcept;
void mul_scalar_f32(const float* RESTRICT a, float s, float* RESTRICT out, std::size_t n) noexcept;

// ── Activations ──────────────────────────────────────────────────────────────
void relu_f32   (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
void neg_f32    (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
void exp_f32    (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
void log_f32    (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
void sqrt_f32   (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
void abs_f32    (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
void sigmoid_f32(const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
// SiLU: out = in * sigmoid(in).  Uses fast polynomial exp — same accuracy as sigmoid_f32.
void silu_f32   (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;
// GELU: out = in * sigmoid(2k), k = sqrt(2/π)*(in + 0.044715*in³). Tanh-approx via sigmoid identity.
void gelu_f32   (const float* RESTRICT in, float* RESTRICT out, std::size_t n) noexcept;

// ── Activation backward kernels ──────────────────────────────────────────────
// grad_in[i] = grad_out[i] * local_derivative(x[i])
void silu_backward_f32(const float* RESTRICT grad_out, const float* RESTRICT x,
                       float* RESTRICT grad_in, std::size_t n) noexcept;
void gelu_backward_f32(const float* RESTRICT grad_out, const float* RESTRICT x,
                       float* RESTRICT grad_in, std::size_t n) noexcept;

// ── Softmax ───────────────────────────────────────────────────────────────────
// Numerically-stable row-wise softmax: exp(in[r,c] - max_r) / sum.
// Uses fast polynomial exp for throughput; accurate enough for attention and logits.
void softmax_rows_f32(const float* RESTRICT in, float* RESTRICT out,
                      std::size_t rows, std::size_t cols) noexcept;

// ── RMSNorm ───────────────────────────────────────────────────────────────────
// Forward: fills x_norm[T,D], inv_rms[T], out[T,D].
// out[t,d] = w[d] * (x[t,d] / rms[t]),  inv_rms[t] = 1/sqrt(mean(x[t]²)+eps).
void rms_norm_fwd_f32(const float* RESTRICT x,  const float* RESTRICT w,
                      float* RESTRICT x_norm, float* RESTRICT inv_rms,
                      float* RESTRICT out,
                      std::size_t T, std::size_t D, float eps) noexcept;

// Backward for x: gx[t,d] = inv_rms[t]*(w[d]*g[t,d] − (σ/D)*x_norm[t,d])
//                where σ = dot(w*g[t,:], x_norm[t,:]).
void rms_norm_bwd_x_f32(const float* RESTRICT g,
                         const float* RESTRICT x_norm,
                         const float* RESTRICT inv_rms,
                         const float* RESTRICT w,
                         float* RESTRICT gx,
                         std::size_t T, std::size_t D) noexcept;

// Backward for weight: gw[d] += Σ_t g[t,d]*x_norm[t,d].  Pure accumulator — caller
// must provide a zeroed gw (e.g. zeros({D})) before the first call.
// gw is read-modified-write so RESTRICT is omitted on that parameter.
void rms_norm_bwd_w_f32(const float* RESTRICT g,
                         const float* RESTRICT x_norm,
                         float* gw,
                         std::size_t T, std::size_t D) noexcept;

// ── Embedding ─────────────────────────────────────────────────────────────────
// Backward scatter-add: g_w[idx[i], :] += g_out[i, :]  for i in [0,N).
void embed_bwd_f32(const float* RESTRICT g_out, const int32_t* RESTRICT idx,
                   float* RESTRICT g_w, std::size_t N, std::size_t D) noexcept;

// ── 2-D strided copy ─────────────────────────────────────────────────────────
// dst[i,j] = src[i*rs + j*cs]  for all (i,j) in [0,rows) × [0,cols).
// dst is always written contiguously (row-major, stride = cols).
// rs and cs are in float elements (not bytes).
// Covers general non-contiguous slices and the common A.T.contiguous() transpose.
void copy_strided_2d_f32(const float* RESTRICT src,
                          std::size_t rs, std::size_t cs,
                          float* RESTRICT dst,
                          std::size_t rows, std::size_t cols) noexcept;

// ── Reduction ─────────────────────────────────────────────────────────────────
float sum_f32 (const float* RESTRICT in, std::size_t n) noexcept;
float max_f32 (const float* RESTRICT in, std::size_t n) noexcept;
float min_f32 (const float* RESTRICT in, std::size_t n) noexcept;
float norm_f32(const float* RESTRICT in, std::size_t n) noexcept;

// ── Matrix multiply ───────────────────────────────────────────────────────────
// C = A × B   where A(M,K), B(K,N), C(M,N) — row-major, C is zeroed inside.
void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t N, std::size_t K) noexcept;

// C = A × Bᵀ  where A(M,K), B(N,K), C(M,N) — row-major. Both operands stream
// contiguously (row·row dots), so no transposed copy of B is materialized.
void matmul_bt_f32(const float* A, const float* B, float* C,
                   std::size_t M, std::size_t N, std::size_t K) noexcept;

} // namespace sub0llm::backend::cpu
