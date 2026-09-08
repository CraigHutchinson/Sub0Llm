// sub0/muon.hpp — pure Newton-Schulz "zeropower" orthogonalization for the Muon optimizer.
//
// Muon (Keller Jordan et al., https://github.com/KellerJordan/Muon) replaces AdamW's per-element
// update for HIDDEN 2D weight matrices with an orthogonalized momentum update: it drives the
// momentum matrix toward the nearest matrix with all singular values ~1 (approximately UV^T from
// its SVD USV^T) via a few Newton-Schulz quintic iterations -- without ever forming the SVD, which
// would be far too slow to run every optimizer step. Embeddings, the output head, and 1D params
// (norm gains, biases) stay on AdamW; only the hidden GEMM weight matrices (Wq/Wk/Wv/Wo/W1/W2/Wg)
// route through this. See src/backend_cpu.cpp's AdamW::step() for the hybrid dispatch.
//
// The algorithm, coefficients (3.4445, -4.7750, 2.0315), default step count (5), Frobenius
// normalization, transpose-for-compute-efficiency trick, and the post-orthogonalization scale
// factor are all verified against the reference implementation (2026-07-04), not re-derived from
// memory -- this is precise numerical code where a wrong constant would silently produce a worse
// optimizer, not a crash.
//
// Engine-free (no generated config, no engine link) so it's unit-testable standalone, matching
// memplan.hpp/config_util.hpp.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

namespace sub0::muon {

/** Returns the scratch capacity needed for a matrix and its smaller Gram matrix.
 * @param[in] matrix_floats rows*cols, or the maximum over all matrices to be processed.
 * @param[in] gram_floats min(rows,cols)^2, or the maximum over those matrices.
 * @return Float count for two matrix buffers and two Gram buffers.
 * @pre The resulting float count must be representable in size_t.
 */
[[nodiscard]] constexpr std::size_t scratch_floats(std::size_t matrix_floats,
                                                  std::size_t gram_floats) noexcept {
    return 2 * matrix_floats + 2 * gram_floats;
}

/** Orthogonalizes a row-major matrix using allocation-free Newton–Schulz iterations.
 *
 * Works with the smaller dimension first, bounding each iteration by
 * min(rows,cols)^2 * max(rows,cols); tall matrices are transposed back on output.
 * @param[in] in Input matrix, containing rows*cols floats.
 * @param[in] rows Positive input row count.
 * @param[in] cols Positive input column count.
 * @param[out] out Output matrix, containing rows*cols floats; may equal in.
 * @param[in,out] scratch Caller-owned storage, at least scratch_floats(rows*cols,
 *                       min(rows,cols)^2) floats; must not overlap in or out.
 * @param[in] steps Nonnegative iteration count; five is the reference default.
 * @note Concurrent calls require disjoint scratch/output storage. No storage is retained.
 */
inline void newton_schulz5(const float* in, int rows, int cols, float* out,
                         std::span<float> scratch, int steps = 5) {
    assert(rows > 0 && cols > 0 && steps >= 0);
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const bool transposed = rows > cols;
    const int m = transposed ? cols : rows;   // working shape [m, n], always m <= n
    const int n = transposed ? rows : cols;
    const auto mn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const auto mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);

    assert(scratch.size() >= scratch_floats(mn, mm));
    float* __restrict Xp = scratch.data();
    float* __restrict Ap = Xp + mn;
    float* __restrict AAp = Ap + mm;
    float* __restrict BXp = AAp + mm;

    if (transposed) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                Xp[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)] =
                    in[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j)];
    } else {
        std::copy(in, in + mn, Xp);
    }

    // Frobenius-norm normalize (accumulate in double -- these are large reductions over up to a
    // few million elements for this project's biggest FFN matrices).
    double ss = 0.0;
    #pragma omp simd reduction(+ : ss)
    for (std::size_t i = 0; i < mn; ++i) ss += static_cast<double>(Xp[i]) * Xp[i];
    const float norm = static_cast<float>(std::sqrt(ss)) + 1e-7f;

    #pragma omp simd
    for (std::size_t i = 0; i < mn; ++i) Xp[i] /= norm;

    for (int it = 0; it < steps; ++it) {
        // A = X @ X^T is symmetric: compute each dot product once and mirror it.
        for (int i = 0; i < m; ++i) {
            const float* __restrict xi = Xp + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
            for (int j = i; j < m; ++j) {
                const float* __restrict xj = Xp + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
                double s = 0.0;
                #pragma omp simd reduction(+ : s)
                for (int k = 0; k < n; ++k) s += static_cast<double>(xi[k]) * xj[k];
                Ap[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = static_cast<float>(s);
                Ap[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) + static_cast<std::size_t>(i)] = static_cast<float>(s);
            }
        }
        // A's symmetry lets A @ A read two contiguous rows instead of striding a column.
        for (int i = 0; i < m; ++i) {
            const float* __restrict ai = Ap + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int j = 0; j < m; ++j) {
                const float* __restrict aj = Ap + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
                double s = 0.0;
                #pragma omp simd reduction(+ : s)
                for (int k = 0; k < m; ++k) s += static_cast<double>(ai[k]) * aj[k];
                AAp[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = static_cast<float>(s);
            }
        }
        // B = b*A + c*AA, written back into A (elementwise -- AA is already fully computed, so
        // aliasing A's own storage here is safe).
        #pragma omp simd
        for (std::size_t i = 0; i < mm; ++i) Ap[i] = b * Ap[i] + c * AAp[i];
        // BX = B @ X  [m,n] (B now lives in A's buffer)
        for (int i = 0; i < m; ++i) {
            const float* __restrict bi = Ap + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            float* __restrict bxr = BXp + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
            for (int j = 0; j < n; ++j) bxr[j] = 0.f;
            for (int k = 0; k < m; ++k) {
                const float bik = bi[k];
                if (bik == 0.f) continue;
                const float* __restrict xk = Xp + static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
                #pragma omp simd
                for (int j = 0; j < n; ++j) bxr[j] += bik * xk[j];
            }
        }
        // X = a*X + BX
        #pragma omp simd
        for (std::size_t i = 0; i < mn; ++i) Xp[i] = a * Xp[i] + BXp[i];
    }

    if (transposed) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                out[static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j)] =
                    Xp[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
    } else {
        std::copy(Xp, Xp + mn, out);
    }
}

// Post-orthogonalization scale factor: sqrt(max(1, fan_out/fan_in)). This project's weight-matrix
// convention is [rows=in_features, cols=out_features] (op_linear computes y = x @ W with W stored
// [in,out]) -- the OPPOSITE axis order from the PyTorch nn.Linear convention ([out,in]) the
// reference implementation is written against -- so this is cols/rows here, not the naive rows/cols
// a direct port would use.
inline float scale_factor(int rows, int cols) {
    return std::sqrt(std::max(1.0f, static_cast<float>(cols) / static_cast<float>(rows)));
}

}  // namespace sub0::muon
