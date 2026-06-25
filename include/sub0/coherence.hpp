// coherence.hpp -- pure helper math for the `autotemp` coherence tuner.
//
// Like tune.hpp, this is deliberately engine-free (std-only): it knows nothing about
// the model or sampling, only how to (a) measure repetition structure in a token stream
// and (b) recover the temperature at which a monotone statistic crosses a target. Keeping
// it pure makes it unit testable in isolation (see tests/coherence_tests.cpp) and keeps
// the stage code -- which wires these against real generations -- thin.
//
// Background: `autotemp` picks the sampling temperature whose GENERATIONS are as
// in-distribution as real held-out text, judged by the model itself. It sweeps a grid of
// temperatures measuring two statistics -- generation self-perplexity (rises with temp)
// and n-gram repeat rate (falls with temp) -- and interpolates where each crosses the
// value real text exhibits. These two functions are that measurement/interpolation core.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace sub0::coherence {

// Fraction of k-grams in a token stream that have already appeared earlier in the same
// stream -- a direct, granularity-correct degeneration signal ("box box box" and other
// loops repeat n-grams even when no two adjacent tokens are identical). Token ids must fit
// in 12 bits (VOCAB <= 4096) so a 4-gram packs losslessly into one 64-bit key. Returns 0
// for a stream shorter than k (no k-gram to repeat).
inline double ngram_repeat(const int* ids, int n, int k = 4) {
    if (n < k || k <= 0) return 0.0;
    std::unordered_set<std::uint64_t> seen;
    long total = 0, repeats = 0;
    for (int i = k - 1; i < n; ++i) {
        std::uint64_t key = 0;
        for (int j = 0; j < k; ++j) key = (key << 12) | static_cast<std::uint64_t>(ids[i - j] & 0xFFF);
        ++total;
        if (!seen.insert(key).second) ++repeats;
    }
    return total > 0 ? static_cast<double>(repeats) / static_cast<double>(total) : 0.0;
}

// The temperature at which a monotone series y(temp) crosses `target`, by linear
// interpolation between the two bracketing grid points. `increasing` states the direction
// (generation perplexity rises with temperature; repeat rate falls). If `target` lies
// outside the measured range the nearer endpoint is returned with hit=false, so the caller
// can flag that the crossing was off-grid (clamped) rather than interpolated. `temp` and
// `y` must be the same length, with `temp` ascending.
struct Crossing { float temp = 0.f; bool hit = false; };

inline Crossing interp_cross(const std::vector<float>& temp, const std::vector<double>& y,
                             double target, bool increasing) {
    if (temp.empty() || temp.size() != y.size()) return {};
    for (std::size_t i = 0; i + 1 < temp.size(); ++i) {
        const double a = y[i], b = y[i + 1];
        const bool between = increasing ? (a <= target && target <= b)
                                        : (a >= target && target >= b);
        if (between) {
            const double denom = b - a;
            const double f = std::abs(denom) > 1e-12 ? (target - a) / denom : 0.0;
            return { temp[i] + static_cast<float>(f) * (temp[i + 1] - temp[i]), true };
        }
    }
    // Out of range: clamp to the endpoint on the side the target overshoots.
    if (increasing) return (target < y.front()) ? Crossing{temp.front(), false} : Crossing{temp.back(), false};
    return (target > y.front()) ? Crossing{temp.front(), false} : Crossing{temp.back(), false};
}

}  // namespace sub0::coherence
