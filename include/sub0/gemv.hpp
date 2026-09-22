// sub0/gemv.hpp -- the one GEMV primitive for this project's [rows=in, cols=out] weight convention:
//
//     y[o] = sum_{i in [0,in)} x[i] * W[i*out + o]          (INPUT-major, "axpy" form)
//
// WHY ONE PRIMITIVE. Until O2 this loop was hand-written at every projection site (decode.cpp's
// linear_row, qsa::linear_row, gdn::forward's in/out projections, moe::expert_ffn_row, the router), none
// of them explicitly vectorized -- the bf16 weights are read through the Bf16CPtr widening proxy -- and
// none threaded. Decode's bf16 backbone runs ~70% of every token through this shape on ONE core, which
// cannot exceed ~30 GB/s against a machine that sustains ~79 GB/s on its P-cores
// (docs/optimization/profile_post_o1.md). Fixing it once, here, fixes every site.
//
// BIT-EXACT BY CONSTRUCTION, at every thread count and on every ISA path. Each output o is the same
// sequential sum over i, in the same order, as the scalar loops it replaces: AVX2 lanes and threads both
// split ONLY across o, never across i. The scalar form `y[o] += x[i]*w` contracts to an FMA under this
// build's flags, and the vector path uses _mm256_fmadd_ps -- the same single rounding per term. bf16 is
// widened by zero-extend + 16-bit shift, which IS bf16_widen (bf16.hpp). So the forward/forward_one
// parity check, which compares two callers of this one function, cannot drift because of it.
//
// ZERO-SKIP: no. Some replaced sites skipped `x[i] == 0`; for the finite weights a checkpoint holds,
// `y + 0*w == y`, so skipping changes no nonzero result -- and a branch per input row costs more than it
// saves once the inner loop is vector-wide.
//
// THREADS is a compile-time template argument (default 1 = serial, the pre-O2 behaviour). Only decode
// passes more (DECODE_GEMV_THREADS, `sub0llm-configure --decode-gemv-threads`); the batched forward()
// and every test keep 1. A call made from inside an existing OpenMP team runs serially rather than nest.

#pragma once

#include "sub0/bf16.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define SUB0_GEMV_AVX2 1
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sub0::gemv {

namespace detail {

/// y[o] = sum_i x[i]*W[i*out+o] for o in [o0, o1): the scalar reference every other path must equal.
template <class WP>
inline void axpy_range_scalar(const float* x, WP W, int in, int out, int o0, int o1, float* y) noexcept {
    for (int o = o0; o < o1; ++o) y[o] = 0.f;
    for (int i = 0; i < in; ++i) {
        const float xi = x[i];
        const auto wr = W + static_cast<std::size_t>(i) * static_cast<std::size_t>(out);
        for (int o = o0; o < o1; ++o) y[o] += xi * wr[o];
    }
}

#if defined(SUB0_GEMV_AVX2)
/// Eight bf16 -> eight f32: zero-extend each 16-bit pattern and shift it into the high half (bf16_widen).
[[nodiscard]] inline __m256 widen8(const bf16* p) noexcept {
    const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
}
[[nodiscard]] inline __m256 load8(const float* p) noexcept { return _mm256_loadu_ps(p); }
[[nodiscard]] inline const float* raw(const float* w) noexcept { return w; }
[[nodiscard]] inline const bf16* raw(Bf16CPtr w) noexcept { return w.p; }

[[nodiscard]] inline float widen1(bf16 v) noexcept { return bf16_widen(v.bits); }
[[nodiscard]] inline float widen1(float v) noexcept { return v; }

/** The vector path for f32 or bf16 weights, shaped for a MEMORY-bound GEMV.
 *
 * Weight rows are STREAMED contiguously, four at a time: for each 8-output lane group, y is loaded once,
 * receives the four rows' FMAs in i order (i, i+1, i+2, i+3), and is stored once. That cuts y's L1
 * traffic 4x against a one-row loop while each of the four rows stays one contiguous read -- what the
 * hardware prefetcher and the TLB want. (The first cut held a 32-output tile in registers across ALL
 * rows instead, which touched one 64-byte line per row at a stride of `out*2` bytes: a fresh 4 KiB page,
 * and so a likely TLB miss, on every access. Right for a compute-bound kernel, wrong for this one.)
 *
 * Order per output is still i = 0, 1, 2, ... exactly, one FMA per term -- bit-identical to the scalar
 * reference at any o-range split.
 */
/// acc[0, o1-o0) = x . W[:, o0:o1). The one inner kernel; `acc` is either y itself or a private tile.
template <class Raw>
inline void accumulate_avx2(const float* x, const Raw* W, int in, int out, int o0, int o1, float* acc) noexcept {
    const auto stride = static_cast<std::size_t>(out);
    const auto ld = [](const Raw* p) {
        if constexpr (std::is_same_v<Raw, bf16>) return widen8(p);
        else                                     return load8(p);
    };
    const int n = o1 - o0;
    const int n_vec = n / 8 * 8;                 // [0, n_vec) in 8-lane groups, [n_vec, n) scalar
    for (int o = 0; o < n; ++o) acc[o] = 0.f;
    const Raw* base = W + o0;
    int i = 0;
    for (; i + 4 <= in; i += 4) {
        const __m256 x0 = _mm256_set1_ps(x[i]), x1 = _mm256_set1_ps(x[i + 1]);
        const __m256 x2 = _mm256_set1_ps(x[i + 2]), x3 = _mm256_set1_ps(x[i + 3]);
        const Raw* w0 = base + static_cast<std::size_t>(i) * stride;
        const Raw* w1 = w0 + stride;
        const Raw* w2 = w1 + stride;
        const Raw* w3 = w2 + stride;
        for (int o = 0; o < n_vec; o += 8) {
            __m256 a = _mm256_loadu_ps(acc + o);
            a = _mm256_fmadd_ps(x0, ld(w0 + o), a);
            a = _mm256_fmadd_ps(x1, ld(w1 + o), a);
            a = _mm256_fmadd_ps(x2, ld(w2 + o), a);
            a = _mm256_fmadd_ps(x3, ld(w3 + o), a);
            _mm256_storeu_ps(acc + o, a);
        }
        for (int o = n_vec; o < n; ++o) {
            float a = acc[o];
            a = std::fma(x[i], widen1(w0[o]), a);
            a = std::fma(x[i + 1], widen1(w1[o]), a);
            a = std::fma(x[i + 2], widen1(w2[o]), a);
            a = std::fma(x[i + 3], widen1(w3[o]), a);
            acc[o] = a;
        }
    }
    for (; i < in; ++i) {                       // < 4 trailing rows, same order
        const __m256 xi = _mm256_set1_ps(x[i]);
        const Raw* w = base + static_cast<std::size_t>(i) * stride;
        for (int o = 0; o < n_vec; o += 8) _mm256_storeu_ps(acc + o, _mm256_fmadd_ps(xi, ld(w + o), _mm256_loadu_ps(acc + o)));
        for (int o = n_vec; o < n; ++o) acc[o] = std::fma(x[i], widen1(w[o]), acc[o]);
    }
}

/// Outputs at or below this span accumulate in a private stack tile and write `y` once (see axpy).
inline constexpr int kLocalTile = 128;

template <class Raw>
inline void axpy_range_avx2(const float* x, const Raw* W, int in, int out, int o0, int o1, float* y) noexcept {
    if (o1 - o0 <= kLocalTile) {
        alignas(32) float tile[kLocalTile];
        accumulate_avx2(x, W, in, out, o0, o1, tile);
        std::copy_n(tile, o1 - o0, y + o0);
    } else {
        accumulate_avx2(x, W, in, out, o0, o1, y + o0);
    }
}
#endif

/// One thread's share: the vector path where the weight type has one, else the scalar reference.
template <class WP>
inline void axpy_range(const float* x, WP W, int in, int out, int o0, int o1, float* y) noexcept {
#if defined(SUB0_GEMV_AVX2)
    if constexpr (std::is_same_v<WP, const float*> || std::is_same_v<WP, float*> ||
                  std::is_same_v<WP, Bf16CPtr>) {
        axpy_range_avx2(x, raw(W), in, out, o0, o1, y);
        return;
    }
#endif
    axpy_range_scalar(x, W, in, out, o0, o1, y);
}

}  // namespace detail

/** y[0,out) = x[0,in) . W[in,out], split across `Threads` threads by output column.
 *
 * @tparam Threads  compile-time fan-out (1 = serial).
 *
 * Chunking depends on how many outputs each thread gets. WIDE (> kLocalTile per thread): multiples of
 * 32 outputs, whole cache lines of f32 y, so no two threads ever write the same line while accumulating
 * in place. NARROW: multiples of 8, so all threads get work (Gated Residual's 10240 -> 320 down
 * projection used only 5 of 8 threads at 32-granularity, ~28 GB/s); each thread then accumulates in a
 * private stack tile and writes y once, so the finer split cannot false-share.
 * @note `y` must not alias `x` or `W`. No heap, no locks (AGENTS.md S1).
 */
template <int Threads = 1, class WP>
inline void axpy(const float* x, WP W, int in, int out, float* y) noexcept {
    static_assert(Threads >= 1, "Threads must be >= 1");
    if constexpr (Threads == 1) {
        detail::axpy_range(x, W, in, out, 0, out, y);
    } else {
#if defined(_OPENMP)
        if (!omp_in_parallel()) {
            const int per = (out + Threads - 1) / Threads;
            const int chunk = per > detail::kLocalTile ? (per + 31) / 32 * 32 : (per + 7) / 8 * 8;
            #pragma omp parallel for num_threads(Threads) schedule(static)
            for (int t = 0; t < Threads; ++t) {
                const int o0 = std::min(out, t * chunk), o1 = std::min(out, o0 + chunk);
                if (o0 < o1) detail::axpy_range(x, W, in, out, o0, o1, y);
            }
            return;
        }
#endif
        detail::axpy_range(x, W, in, out, 0, out, y);
    }
}

}  // namespace sub0::gemv
