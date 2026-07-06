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
#include <cmath>
#include <cstddef>
#include <vector>

namespace sub0::muon {

// Orthogonalizes an [rows x cols] row-major matrix `in` via `steps` (default 5) Newton-Schulz
// quintic iterations, writing the result to `out` (`out == in` is safe -- `in` is fully consumed
// into an internal working buffer before anything is written to `out`).
//
// Internally works in whichever orientation has the SMALLER dimension first (transposing if
// rows > cols, transposing back before returning) purely to bound the per-iteration cost by
// min(rows,cols)^2 * max(rows,cols) rather than the larger dimension squared -- the mathematical
// result is identical either way, this is a compute-only optimization (this project's FFN weight
// matrices are frequently non-square, e.g. D_MODEL x D_FF, so it matters here).
//
// ZERO heap allocation once warmed up (AGENTS.md #1): the working buffers are `static thread_local`
// and only ever GROW (resize() is a no-op once already large enough), so after the first call at
// each thread's largest-seen matrix size, every subsequent call -- including every optimizer step
// for the rest of training -- allocates nothing. Every loop below is explicitly bounded by the
// CURRENT call's mn/mm/m/n, never by the (possibly larger, stale-tailed) buffer's own .size() --
// that distinction matters once the buffers are reused across differently-shaped calls, see
// tests/muon_tests.cpp's "buffer reuse across shapes" case for what this specifically guards against.
inline void newton_schulz5(const float* in, int rows, int cols, float* out, int steps = 5) {
    constexpr float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    const bool transposed = rows > cols;
    const int m = transposed ? cols : rows;   // working shape [m, n], always m <= n
    const int n = transposed ? rows : cols;
    const auto mn = static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
    const auto mm = static_cast<std::size_t>(m) * static_cast<std::size_t>(m);

    //TODO: Allocations inside hot loop need avoiding, violates ##1 project rule, the buffer size/limit should be determinable upfront and allocated ahead of time
    static thread_local std::vector<float> X, A, AA, BX;
    if (X.size() < mn) X.resize(mn);
    if (A.size() < mm) A.resize(mm);
    if (AA.size() < mm) AA.resize(mm);
    if (BX.size() < mn) BX.resize(mn);
    float* __restrict Xp = X.data();
    float* __restrict Ap = A.data();
    float* __restrict AAp = AA.data();
    float* __restrict BXp = BX.data();

    //TODO: Avoid unecessary transpose route - design for the ideal case where the input is already in the correct orientation, and avoid the transpose if possible
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

    //TODO: division is slower than multiplication, so we should consider using multiplication with the reciprocal of norm instead of division. This may improve performance, especially for large matrices.
    #pragma omp simd
    for (std::size_t i = 0; i < mn; ++i) Xp[i] /= norm;

    for (int it = 0; it < steps; ++it) {
        // A = X @ X^T  [m,m]: inner loop over `k` is contiguous in both operands (dot-product
        // reduction), matches op_linear's backward dX pattern.
        for (int i = 0; i < m; ++i) {
            const float* __restrict xi = Xp + static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
            for (int j = 0; j < m; ++j) {
                const float* __restrict xj = Xp + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
                double s = 0.0;
                #pragma omp simd reduction(+ : s)
                for (int k = 0; k < n; ++k) s += static_cast<double>(xi[k]) * xj[k];
                Ap[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)] = static_cast<float>(s);
            }
        }
        // AA = A @ A  [m,m]
        for (int i = 0; i < m; ++i) {
            const float* __restrict ai = Ap + static_cast<std::size_t>(i) * static_cast<std::size_t>(m);
            for (int j = 0; j < m; ++j) {
                double s = 0.0;
                #pragma omp simd reduction(+ : s)
                for (int k = 0; k < m; ++k) s += static_cast<double>(ai[k]) * Ap[static_cast<std::size_t>(k) * static_cast<std::size_t>(m) + static_cast<std::size_t>(j)];
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

    // TODO: As earlier, avoid unecessary transpose route - design for the ideal case where the input is already in the correct orientation, and avoid the transpose if possible
    // this may also avoid the temporary buffer allocation for the transpose, and avoid the copy back to the output buffer if the input is already in the correct orientation
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
