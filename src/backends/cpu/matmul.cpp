#include "kernels.hpp"

#include <algorithm>
#include <cstring>

#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
#  include <immintrin.h>
#endif

// ── Cache-blocked matmul ──────────────────────────────────────────────────────
// The naive O(M·N·K) triple loop has terrible cache behaviour for large matrices
// because the inner loop strides across B with step N (cache-unfriendly).
//
// Blocking keeps all three working sets in L1/L2 by tiling:
//   • ii, kk, jj tiles of size BLOCK
//   • Inside each tile, the inner k loop reads a contiguous row of B → cache hit
//
// BLOCK = 64 floats = 256 bytes ≈ 4 cache lines — chosen empirically for
// a typical 256 KB L2 cache.  Ch09 will auto-tune this at build time.
//
// With AVX2 enabled, the innermost j loop is vectorised with FMA:
//   C[i,j:j+8] += A[i,k] * B[k,j:j+8]

namespace sub0llm::backend::cpu {

namespace {

constexpr std::size_t BLOCK = 64;

#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)

void matmul_tile_avx2(const float* __restrict__ A,
                      const float* __restrict__ B,
                      float*       __restrict__ C,
                      std::size_t M, std::size_t N, std::size_t K) noexcept {
    std::memset(C, 0, M * N * sizeof(float));

    for (std::size_t ii = 0; ii < M; ii += BLOCK) {
    for (std::size_t kk = 0; kk < K; kk += BLOCK) {
    for (std::size_t jj = 0; jj < N; jj += BLOCK) {
        const std::size_t imax = std::min(ii + BLOCK, M);
        const std::size_t kmax = std::min(kk + BLOCK, K);
        const std::size_t jmax = std::min(jj + BLOCK, N);

        for (std::size_t i = ii; i < imax; ++i) {
        for (std::size_t k = kk; k < kmax; ++k) {
            const float a_ik = A[i * K + k];
#if defined(SUB0LLM_AVX512)
            const __m512 va = _mm512_set1_ps(a_ik);
            std::size_t j = jj;
            for (; j + 16 <= jmax; j += 16) {
                __m512 vc = _mm512_loadu_ps(C + i*N + j);
                __m512 vb = _mm512_loadu_ps(B + k*N + j);
                _mm512_storeu_ps(C + i*N + j, _mm512_fmadd_ps(va, vb, vc));
            }
            for (; j < jmax; ++j)
                C[i*N + j] += a_ik * B[k*N + j];
#else
            const __m256 va = _mm256_set1_ps(a_ik);
            std::size_t j = jj;
            for (; j + 8 <= jmax; j += 8) {
                __m256 vc = _mm256_loadu_ps(C + i*N + j);
                __m256 vb = _mm256_loadu_ps(B + k*N + j);
                _mm256_storeu_ps(C + i*N + j, _mm256_fmadd_ps(va, vb, vc));
            }
            for (; j < jmax; ++j)
                C[i*N + j] += a_ik * B[k*N + j];
#endif
        }}
    }}}
}

#else

void matmul_scalar_blocked(const float* A, const float* B, float* C,
                           std::size_t M, std::size_t N, std::size_t K) noexcept {
    std::memset(C, 0, M * N * sizeof(float));
    for (std::size_t ii = 0; ii < M; ii += BLOCK)
    for (std::size_t kk = 0; kk < K; kk += BLOCK)
    for (std::size_t jj = 0; jj < N; jj += BLOCK) {
        const std::size_t imax = std::min(ii + BLOCK, M);
        const std::size_t kmax = std::min(kk + BLOCK, K);
        const std::size_t jmax = std::min(jj + BLOCK, N);
        for (std::size_t i = ii; i < imax; ++i)
        for (std::size_t k = kk; k < kmax; ++k) {
            const float a_ik = A[i * K + k];
            for (std::size_t j = jj; j < jmax; ++j)
                C[i*N + j] += a_ik * B[k*N + j];
        }
    }
}

#endif

} // anonymous namespace

void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t N, std::size_t K) noexcept {
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    matmul_tile_avx2(A, B, C, M, N, K);
#else
    matmul_scalar_blocked(A, B, C, M, N, K);
#endif
}

} // namespace sub0llm::backend::cpu
