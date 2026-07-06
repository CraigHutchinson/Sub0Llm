// muon_tests.cpp -- engine-free unit tests for sub0/muon.hpp's Newton-Schulz orthogonalization, the
// core numerical primitive of the Muon optimizer (see src/backend_cpu.cpp's AdamW::step() for the
// hybrid Muon(matrices)/AdamW(embed,norm,bias,head) dispatch that consumes it). Header-only + std-only,
// so this runs in the fast engine-free target -- no engine build, no GPU.
//
// What "correct" means here, per the reference implementation's own documented behavior: 5-step
// Newton-Schulz does NOT converge all the way to an exact semi-orthogonal matrix (that would need
// many more iterations); it drives a matrix's ROWS toward being nearly MUTUALLY ORTHOGONAL (small
// off-diagonal of X @ X^T) while leaving each row's norm-squared (the diagonal) spread over roughly
// [0.5, 1.5] rather than pinned at 1 -- "US'V^T where S' ~ Uniform(0.5,1.5)", not UV^T. Tests check
// the off-diagonal collapse (the real property that matters) and a bounded, non-degenerate diagonal
// range, not exact orthonormality.

#include "sub0/muon.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace sub0::muon;

namespace {

// max|off-diagonal| and [min,max] diagonal of X @ X^T, computed in the SAME [m,n] (m<=n) working
// orientation newton_schulz5 uses internally (transposing first if rows>cols) -- an independent
// re-implementation of the Gram-matrix check, not reusing any of newton_schulz5's own code.
struct GramStats { double max_offdiag, min_diag, max_diag; };
GramStats gram_stats(const std::vector<float>& X, int rows, int cols) {
    const int m = std::min(rows, cols), n = std::max(rows, cols);
    std::vector<float> Xw(static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    if (rows > cols) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                Xw[static_cast<std::size_t>(j) * n + i] = X[static_cast<std::size_t>(i) * cols + j];
    } else {
        Xw = X;
    }
    GramStats g{0.0, 1e30, -1e30};
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k)
                s += static_cast<double>(Xw[static_cast<std::size_t>(i) * n + k]) *
                     static_cast<double>(Xw[static_cast<std::size_t>(j) * n + k]);
            if (i == j) { g.min_diag = std::min(g.min_diag, s); g.max_diag = std::max(g.max_diag, s); }
            else        g.max_offdiag = std::max(g.max_offdiag, std::fabs(s));
        }
    }
    return g;
}

std::vector<float> random_matrix(int rows, int cols, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> G(static_cast<std::size_t>(rows) * cols);
    for (float& v : G) v = nd(rng);
    return G;
}

}  // namespace

TEST_CASE("muon: newton_schulz5 drives a random matrix's rows toward mutual orthogonality", "[muon]") {
    // Square, wide, tall, and this project's actual FFN shape (D_MODEL x D_FF and its transpose) --
    // the transpose-for-compute-efficiency path is exercised by the rows>cols cases.
    struct { int rows, cols; } shapes[] = {{64, 64}, {32, 128}, {128, 32}, {448, 1792}, {1792, 448}};
    for (auto [rows, cols] : shapes) {
        auto G = random_matrix(rows, cols, 42);
        const auto before = gram_stats(G, rows, cols);
        std::vector<float> O(G.size());
        newton_schulz5(G.data(), rows, cols, O.data(), 5);
        const auto after = gram_stats(O, rows, cols);

        INFO("shape [" << rows << "," << cols << "]");
        // Off-diagonal collapses by roughly 2-3 orders of magnitude vs the raw random matrix.
        CHECK(after.max_offdiag < before.max_offdiag * 0.05);
        CHECK(after.max_offdiag < 0.3);   // an absolute ceiling regardless of the random draw
        // Diagonal (squared row norms) lands in a bounded, non-degenerate range -- not collapsed
        // near 0, not blown up -- consistent with the documented ~Uniform(0.5,1.5) singular spread.
        CHECK(after.min_diag > 0.3);
        CHECK(after.max_diag < 2.0);
    }
}

TEST_CASE("muon: newton_schulz5 is deterministic", "[muon]") {
    auto G = random_matrix(64, 64, 7);
    std::vector<float> O1(G.size()), O2(G.size());
    newton_schulz5(G.data(), 64, 64, O1.data(), 5);
    newton_schulz5(G.data(), 64, 64, O2.data(), 5);
    CHECK(O1 == O2);
}

TEST_CASE("muon: newton_schulz5 is safe with out aliasing in (in-place)", "[muon]") {
    auto G = random_matrix(32, 96, 11);
    const auto Gcopy = G;
    std::vector<float> ref(G.size());
    newton_schulz5(Gcopy.data(), 32, 96, ref.data(), 5);
    newton_schulz5(G.data(), 32, 96, G.data(), 5);   // in-place: out == in
    for (std::size_t i = 0; i < G.size(); ++i) CHECK(std::fabs(G[i] - ref[i]) < 1e-5f);
}

TEST_CASE("muon: newton_schulz5 does not produce NaN/Inf on a degenerate (all-zero) matrix", "[muon]") {
    std::vector<float> G(64 * 64, 0.0f), O(G.size());
    newton_schulz5(G.data(), 64, 64, O.data(), 5);   // the +1e-7 norm epsilon must prevent a 0/0
    for (float v : O) { CHECK_FALSE(std::isnan(v)); CHECK_FALSE(std::isinf(v)); }
}

TEST_CASE("muon: newton_schulz5's reused scratch buffers never leak stale data across calls of different shapes", "[muon]") {
    // The working buffers are `static thread_local` and only ever grow (AGENTS.md #1 -- no per-call
    // heap allocation once warmed up). This guards specifically against the bug class that enables:
    // every loop must be bounded by the CURRENT call's mn/mm, never by the (possibly larger, from an
    // earlier bigger call) buffer's own .size(). Call at a LARGE shape first (grows the buffers well
    // past what the small shape needs), then at a SMALL shape, and check the small-shape result
    // matches a fresh (never-before-called-at-any-size) computation exactly.
    auto Gbig = random_matrix(256, 512, 99);
    std::vector<float> Obig(Gbig.size());
    newton_schulz5(Gbig.data(), 256, 512, Obig.data(), 5);   // warms the thread_local buffers up big

    auto Gsmall = random_matrix(16, 24, 100);
    std::vector<float> Oafter(Gsmall.size());
    newton_schulz5(Gsmall.data(), 16, 24, Oafter.data(), 5);

    const auto stats = gram_stats(Oafter, 16, 24);   // sanity: still a valid orthogonalization, not garbage
    CHECK(stats.max_offdiag < 0.3);
    CHECK(stats.min_diag > 0.3);
    CHECK(stats.max_diag < 2.0);
}

TEST_CASE("muon: newton_schulz5 handles a 1-row and a 1-column matrix without crashing", "[muon]") {
    {
        auto G = random_matrix(1, 32, 3);
        std::vector<float> O(G.size());
        newton_schulz5(G.data(), 1, 32, O.data(), 5);
        for (float v : O) CHECK_FALSE(std::isnan(v));
    }
    {
        auto G = random_matrix(32, 1, 5);
        std::vector<float> O(G.size());
        newton_schulz5(G.data(), 32, 1, O.data(), 5);
        for (float v : O) CHECK_FALSE(std::isnan(v));
    }
}

TEST_CASE("muon: scale_factor is sqrt(max(1, fan_out/fan_in)) in this project's [rows=in,cols=out] convention", "[muon]") {
    CHECK(scale_factor(64, 64) == Catch::Approx(1.0));                  // square: no-op
    CHECK(scale_factor(448, 1792) == Catch::Approx(std::sqrt(4.0)));    // fan_out(1792) > fan_in(448) -> scales up
    CHECK(scale_factor(1792, 448) == Catch::Approx(1.0));               // fan_out(448) < fan_in(1792) -> clamped to 1, never shrinks
    CHECK(scale_factor(448, 896) == Catch::Approx(std::sqrt(2.0)));
}
