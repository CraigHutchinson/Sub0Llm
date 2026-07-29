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

// Plateau detector for a (noisily) decreasing validation series -- e.g. val NELBO sampled every few
// steps. Fits a least-squares line to the LAST `window`+1 points and asks how much that BEST-FIT
// trend implies the series fell ACROSS the window, relative to the current level. Using the fitted
// gradient -- not endpoint differences or a sign count -- is the robust choice: a few upward blips on
// a strongly descending curve barely move the regression line, so a genuinely-improving run is not
// stopped early (the failure the sign test had); a series truly bouncing around a floor has ~zero
// slope and IS stopped. Returns true (plateaued) when the fitted relative drop over the window is
// below `min_rel`. Needs `window`+1 samples; returns false until then (keep training). `min_rel` is
// a fraction (e.g. 0.02 = the fit must still be shedding >=2% of the current level per window).
inline bool trend_plateaued(const std::vector<double>& series, int window, double min_rel) {
    const int m = window + 1;
    if (window < 1 || static_cast<int>(series.size()) < m) return false;
    const std::size_t n = series.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < m; ++i) {
        const double x = i;                          // eval index within the window (0..window)
        const double y = series[n - m + i];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double denom = m * sxx - sx * sx;
    const double slope = denom != 0.0 ? (m * sxy - sx * sy) / denom : 0.0;   // per-eval change (<0 = improving)
    const double level = series[n - 1];
    const double rel_trend = level > 0.0 ? (-slope * window) / level : 0.0;  // fitted drop over window (>0 = improving)
    return rel_trend < min_rel;
}

// Did the series reach a NEW GLOBAL MINIMUM within its last `patience` entries?
//
// The classic early-stopping patience criterion, and the guard trend_plateaued alone cannot provide.
// A trend fit answers "how fast is this falling", which a threshold then judges -- and any threshold
// can be too strict for a run whose real improvement rate is simply slower than the threshold assumes.
// This answers a different, THRESHOLD-FREE question: is the run still producing the best model it has
// ever produced? A series doing that is not plateaued, whatever its slope looks like.
//
// The failure that motivated it (2026-07-28): two LoopSplit arms stopped at the same eval marked
// "plateaued" while val_nelbo was setting a new best at EVERY eval -- 2.1435, 2.1385, 2.1332,
// monotonically down. The fitted drop (~1.4% per window) merely sat under the effective threshold
// (~1.64%, loosened because the run was near its expected-plateau hint). Nothing had plateaued; the
// runs were truncated, and the results are matched-budget rather than converged.
//
// Returns true when there is not yet enough history to judge -- too little evidence must mean
// "keep training", never "stop".
inline bool improved_recently(const std::vector<double>& series, int patience) {
    const int n = static_cast<int>(series.size());
    if (patience < 1 || n <= patience) return true;   // not enough history to say otherwise
    double best_before = series[0];
    for (int i = 1; i < n - patience; ++i) best_before = std::min(best_before, series[i]);
    for (int i = n - patience; i < n; ++i) if (series[i] < best_before) return true;
    return false;
}

}  // namespace sub0::coherence
