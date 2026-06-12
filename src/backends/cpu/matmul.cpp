#include "kernels.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
#  include <immintrin.h>
#endif

#if defined(SUB0LLM_BLAS)
// System BLAS (OpenBLAS, MKL, Apple Accelerate, …) via find_package(BLAS)
// Use the system header so parameter types (e.g. MKL_INT vs int) are correct.
#  if __has_include(<cblas.h>)
#    include <cblas.h>
#  else
extern "C" {
void cblas_sgemm(int Order, int TransA, int TransB,
                 int M, int N, int K,
                 float alpha, const float* A, int lda,
                 const float* B, int ldb,
                 float beta, float* C, int ldc);
}
constexpr int CblasRowMajor = 101, CblasNoTrans = 111, CblasTrans = 112;
#  endif
#elif defined(SUB0LLM_EIGEN)
// Eigen3 — header-only, fetched via CPM; no CBLAS/Fortran dependency
#  include <Eigen/Core>
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
            for (; j + 64 <= jmax; j += 64) {
                __m512 vc0 = _mm512_loadu_ps(C + i*N + j);
                __m512 vc1 = _mm512_loadu_ps(C + i*N + j + 16);
                __m512 vc2 = _mm512_loadu_ps(C + i*N + j + 32);
                __m512 vc3 = _mm512_loadu_ps(C + i*N + j + 48);
                _mm512_storeu_ps(C + i*N + j,      _mm512_fmadd_ps(va, _mm512_loadu_ps(B + k*N + j),      vc0));
                _mm512_storeu_ps(C + i*N + j + 16, _mm512_fmadd_ps(va, _mm512_loadu_ps(B + k*N + j + 16), vc1));
                _mm512_storeu_ps(C + i*N + j + 32, _mm512_fmadd_ps(va, _mm512_loadu_ps(B + k*N + j + 32), vc2));
                _mm512_storeu_ps(C + i*N + j + 48, _mm512_fmadd_ps(va, _mm512_loadu_ps(B + k*N + j + 48), vc3));
            }
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
            for (; j + 32 <= jmax; j += 32) {
                __m256 vc0 = _mm256_loadu_ps(C + i*N + j);
                __m256 vc1 = _mm256_loadu_ps(C + i*N + j + 8);
                __m256 vc2 = _mm256_loadu_ps(C + i*N + j + 16);
                __m256 vc3 = _mm256_loadu_ps(C + i*N + j + 24);
                _mm256_storeu_ps(C + i*N + j,      _mm256_fmadd_ps(va, _mm256_loadu_ps(B + k*N + j),      vc0));
                _mm256_storeu_ps(C + i*N + j + 8,  _mm256_fmadd_ps(va, _mm256_loadu_ps(B + k*N + j + 8),  vc1));
                _mm256_storeu_ps(C + i*N + j + 16, _mm256_fmadd_ps(va, _mm256_loadu_ps(B + k*N + j + 16), vc2));
                _mm256_storeu_ps(C + i*N + j + 24, _mm256_fmadd_ps(va, _mm256_loadu_ps(B + k*N + j + 24), vc3));
            }
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

#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)

// C = A·Bᵀ with B stored row-major (N, K): C[i,j] = dot(A row i, B row j).
// Both operands stream contiguously, so no transpose materialization is needed —
// this is the natural orientation for a weight-tied LM head (logits = x · Wᵀ).
void matmul_bt_tile_avx2(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         std::size_t M, std::size_t N, std::size_t K) noexcept {
#if defined(SUB0LLM_AVX512)
    constexpr std::size_t W = 16;
    auto hsum = [](__m512 v) noexcept { return _mm512_reduce_add_ps(v); };
    auto load = [](const float* p) noexcept { return _mm512_loadu_ps(p); };
    auto fma  = [](__m512 a, __m512 b, __m512 c) noexcept { return _mm512_fmadd_ps(a, b, c); };
    auto zero = []() noexcept { return _mm512_setzero_ps(); };
#else
    constexpr std::size_t W = 8;
    auto hsum = [](__m256 v) noexcept {
        __m128 lo  = _mm256_castps256_ps128(v);
        __m128 hi  = _mm256_extractf128_ps(v, 1);
        __m128 s   = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_movehdup_ps(s));
        return _mm_cvtss_f32(s);
    };
    auto load = [](const float* p) noexcept { return _mm256_loadu_ps(p); };
    auto fma  = [](__m256 a, __m256 b, __m256 c) noexcept { return _mm256_fmadd_ps(a, b, c); };
    auto zero = []() noexcept { return _mm256_setzero_ps(); };
#endif
    for (std::size_t i = 0; i < M; ++i) {
        const float* a = A + i * K;
        for (std::size_t j = 0; j < N; ++j) {
            const float* b = B + j * K;
            auto acc0 = zero(), acc1 = zero();
            std::size_t k = 0;
            for (; k + 2 * W <= K; k += 2 * W) {
                acc0 = fma(load(a + k),     load(b + k),     acc0);
                acc1 = fma(load(a + k + W), load(b + k + W), acc1);
            }
            for (; k + W <= K; k += W)
                acc0 = fma(load(a + k), load(b + k), acc0);
            float s = hsum(acc0) + hsum(acc1);
            for (; k < K; ++k) s += a[k] * b[k];
            C[i * N + j] = s;
        }
    }
}

#else

void matmul_bt_scalar(const float* A, const float* B, float* C,
                      std::size_t M, std::size_t N, std::size_t K) noexcept {
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            float s = 0.0f;
            for (std::size_t k = 0; k < K; ++k) s += A[i * K + k] * B[j * K + k];
            C[i * N + j] = s;
        }
}

#endif

// C = Aᵀ·B with A(M,K), B(M,N) stored row-major: C(K,N) accumulated as M rank-1
// updates C[k,:] += A[m,k] * B[m,:]. The inner loop streams a contiguous row of B
// and a contiguous row of C — no transposed copy of A is ever materialized.
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)

void matmul_tb_tile_avx2(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float*       __restrict__ C,
                         std::size_t M, std::size_t N, std::size_t K) noexcept {
    std::memset(C, 0, K * N * sizeof(float));
    for (std::size_t m = 0; m < M; ++m) {
        const float* b = B + m * N;
        for (std::size_t k = 0; k < K; ++k) {
            const float a_mk = A[m * K + k];
            float* c = C + k * N;
            std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
            const __m512 va = _mm512_set1_ps(a_mk);
            for (; j + 16 <= N; j += 16)
                _mm512_storeu_ps(c + j, _mm512_fmadd_ps(va, _mm512_loadu_ps(b + j),
                                                        _mm512_loadu_ps(c + j)));
#else
            const __m256 va = _mm256_set1_ps(a_mk);
            for (; j + 8 <= N; j += 8)
                _mm256_storeu_ps(c + j, _mm256_fmadd_ps(va, _mm256_loadu_ps(b + j),
                                                        _mm256_loadu_ps(c + j)));
#endif
            for (; j < N; ++j) c[j] += a_mk * b[j];
        }
    }
}

#else

void matmul_tb_scalar(const float* A, const float* B, float* C,
                      std::size_t M, std::size_t N, std::size_t K) noexcept {
    std::memset(C, 0, K * N * sizeof(float));
    for (std::size_t m = 0; m < M; ++m)
        for (std::size_t k = 0; k < K; ++k) {
            const float a_mk = A[m * K + k];
            for (std::size_t j = 0; j < N; ++j)
                C[k * N + j] += a_mk * B[m * N + j];
        }
}

#endif

} // anonymous namespace

// Dispatch strategy (priority: BLAS > Eigen > AVX2 > scalar):
//   K < 64  → AVX2/scalar  (matrix fits in L1/L2; BLAS call overhead dominates)
//   K >= 64 → BLAS or Eigen (optimal for larger D values used in future chapters)
void matmul_f32(const float* A, const float* B, float* C,
                std::size_t M, std::size_t N, std::size_t K) noexcept {
#if defined(SUB0LLM_BLAS)
    if (K >= 64) {
        // BLAS dimensions are int; assert before narrowing cast (educational sizes are
        // always far below INT_MAX, but the contract must hold for future larger models).
        assert(M <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        assert(N <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        assert(K <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0f, A, static_cast<int>(K),
                          B, static_cast<int>(N),
                    0.0f, C, static_cast<int>(N));
        return;
    }
#elif defined(SUB0LLM_EIGEN)
    if (K >= 64) {
        using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMat> Am(A, static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(K));
        Eigen::Map<const RowMat> Bm(B, static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(N));
        Eigen::Map<RowMat>       Cm(C, static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(N));
        // operator* may allocate a heap temporary; catch any exception (bad_alloc) so
        // the noexcept contract is honoured — fall through to the blocked path on OOM.
        try {
            Cm.noalias() = Am * Bm;
            return;
        } catch (...) {}
    }
#endif
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    matmul_tile_avx2(A, B, C, M, N, K);
#else
    matmul_scalar_blocked(A, B, C, M, N, K);
#endif
}

// C = A·Bᵀ, A(M,K) and B(N,K) row-major. Same dispatch priority as matmul_f32;
// the fallback kernels read both operands contiguously (row·row dot products),
// so unlike matmul_f32 no transposed copy of B is ever materialized.
void matmul_bt_f32(const float* A, const float* B, float* C,
                   std::size_t M, std::size_t N, std::size_t K) noexcept {
#if defined(SUB0LLM_BLAS)
    if (K >= 64) {
        assert(M <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        assert(N <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        assert(K <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                    1.0f, A, static_cast<int>(K),
                          B, static_cast<int>(K),
                    0.0f, C, static_cast<int>(N));
        return;
    }
#elif defined(SUB0LLM_EIGEN)
    if (K >= 64) {
        using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMat> Am(A, static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(K));
        Eigen::Map<const RowMat> Bm(B, static_cast<Eigen::Index>(N), static_cast<Eigen::Index>(K));
        Eigen::Map<RowMat>       Cm(C, static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(N));
        try {
            Cm.noalias() = Am * Bm.transpose();
            return;
        } catch (...) {}
    }
#endif
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    matmul_bt_tile_avx2(A, B, C, M, N, K);
#else
    matmul_bt_scalar(A, B, C, M, N, K);
#endif
}

// C = Aᵀ × B, A(M,K) and B(M,N) row-major, C(K,N). Same dispatch priority; the
// fallback kernel uses rank-1 row updates so no transposed copy is materialized.
void matmul_tb_f32(const float* A, const float* B, float* C,
                   std::size_t M, std::size_t N, std::size_t K) noexcept {
#if defined(SUB0LLM_BLAS)
    if (M >= 64) {
        assert(M <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        assert(N <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        assert(K <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    static_cast<int>(K), static_cast<int>(N), static_cast<int>(M),
                    1.0f, A, static_cast<int>(K),
                          B, static_cast<int>(N),
                    0.0f, C, static_cast<int>(N));
        return;
    }
#elif defined(SUB0LLM_EIGEN)
    if (M >= 64) {
        using RowMat = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RowMat> Am(A, static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(K));
        Eigen::Map<const RowMat> Bm(B, static_cast<Eigen::Index>(M), static_cast<Eigen::Index>(N));
        Eigen::Map<RowMat>       Cm(C, static_cast<Eigen::Index>(K), static_cast<Eigen::Index>(N));
        try {
            Cm.noalias() = Am.transpose() * Bm;
            return;
        } catch (...) {}
    }
#endif
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    matmul_tb_tile_avx2(A, B, C, M, N, K);
#else
    matmul_tb_scalar(A, B, C, M, N, K);
#endif
}

} // namespace sub0llm::backend::cpu
