// tune.hpp -- a small, self-contained black-box tuning core.
//
// This is the reusable search engine behind `sub0llm tune`. It is deliberately
// independent of the model, the bench harness and OpenMP: it knows nothing about
// *what* it is optimising, only how to search a discrete space of "knobs" to
// MAXIMISE a caller-supplied objective. That keeps it pure (std-only), unit
// testable in isolation, and shareable -- the bench wires throughput (window/s)
// as the objective today, and a future `train` tuner can wire negative
// validation loss into the exact same search with no changes here.
//
// --- Why this and not a library --------------------------------------------
// Prior art surveyed: the mature auto-tuning stacks (Optuna, NNI, scikit-optimize,
// Vizier, FLAML) are Python. The C++ options are biteopt (header-only derivative-
// free *global* optimiser), CMA-ES, NLopt (DIRECT) and dlib's find_max_global.
// They are built for continuous, higher-dimensional, expensive landscapes. Our
// space is tiny, discrete and low-cardinality (a thread count, a short batch-
// granularity ladder), where a coarse-to-fine grid search is both near-optimal and
// -- crucially -- deterministic, which the stochastic population optimisers are
// not (and determinism is what makes this unit testable). If the space ever grows
// continuous / high-dimensional, biteopt (gh:avaneev/biteopt, header-only) is the
// drop-in escalation: implement Objective on top of it behind the same API.
//
// --- Algorithm --------------------------------------------------------------
// Coarse global grid + top-K basin refinement -- a deterministic divide-and-conquer
// global search:
//   1. COARSE GRID: evaluate the Cartesian product of a strided sample of every
//      axis. Sampling the *joint* space (not one axis at a time) is what makes the
//      search robust to non-separable, multimodal landscapes where the best value of
//      one knob depends on another and a good basin sits off the centre lines -- a
//      per-axis sweep would never see it. Every basin is hit within one stride.
//   2. PICK BASINS: take the top-K coarse points (K>1 keeps an alternative trough in
//      play, not just the first one found).
//   3. REFINE: from each basin, coordinate-descent zoom within a +/-stride window that
//      recenters each pass, walking to the exact local optimum. Refinement stops when
//      a pass no longer improves the basin's best beyond a relative tolerance ("run to
//      convergence"). The global best across all basins wins.
//   4. CONFIRM: every measurement is kept as a sample and points are ranked by the
//      MEDIAN of their samples. Once the search converges, the leading finalists and
//      the immediate neighbours of the winner are re-measured several more times -- so
//      a noisy objective (thermal drift, scheduler jitter) cannot let a single lucky
//      reading crown a fluke; the winner is the one that holds up under re-measurement.
//      Finalists that fall clearly below the leader (by more than the noise floor) are
//      pruned each round, so the measurement budget concentrates on real contenders.
// The coarse grid is exponential in the knob count, which is fine for the low-
// dimensional discrete spaces this targets; once a space grows continuous or high-
// dimensional, biteopt (gh:avaneev/biteopt, header-only) is the drop-in escalation:
// implement Objective on top of it behind this same API.

#ifndef SUB0_TUNE_HPP
#define SUB0_TUNE_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace sub0::tune {

// One tunable dimension: a human name and an ASCENDING list of discrete candidate
// values. Continuous controls (e.g. a learning rate) are represented by pre-sampling
// them onto a ladder via the *_steps helpers below, which keeps the space finite and
// the chosen optimum exactly reproducible. Add a dimension by appending a Knob.
struct Knob {
    std::string         name;
    std::vector<double> values;   // size >= 1, ascending
};

using Space      = std::vector<Knob>;
using Assignment = std::vector<double>;   // one chosen value per knob, in Space order

// Maps a concrete assignment to a score the search MAXIMISES. Throughput (window/s)
// is passed through as-is; a loss objective passes -loss. The objective may be NOISY
// (thermal drift, scheduler jitter): the search re-samples and aggregates by median,
// and confirms the finalists, so a single lucky reading does not win.
using Objective = std::function<double(const Assignment&)>;

// The search proceeds through three phases of increasing precision. The caller is
// notified at each transition (Options::on_phase) so it can raise its measurement
// effort as the search narrows -- e.g. a throughput objective lengthens each timed
// run during Refine/Confirm so sustained thermal throttling expresses itself, which
// is exactly the regime a long training run lives in.
enum class Phase { Explore, Refine, Confirm };

struct Options {
    int    coarse_points   = 6;      // strided samples per axis in the coarse pass
    int    top_basins      = 2;      // basins refined per axis (>1 escapes alt. troughs)
    double tol             = 0.005;  // a pass improving the best by < tol (relative) = converged
    int    max_passes      = 8;      // safety cap on coordinate-descent sweeps
    // Noise confirmation: after the search converges, the leading finalists and the
    // immediate neighbours of the winner are re-measured `confirm_rounds` extra times
    // and ranked by the median, so thermal drift / sampling noise cannot crown a fluke.
    // Set confirm_rounds = 0 to disable (e.g. for a known-deterministic objective).
    int    confirm_top     = 4;      // number of leading finalists to re-measure
    int    confirm_rounds  = 4;      // extra samples taken per finalist during confirmation
    double noise_floor     = 0.02;   // relative band (fraction of the leader's median); during
                                     // confirmation a finalist more than this below the leader is
                                     // clearly worse and is dropped -- no point re-measuring losers
    // Optional cool-down hook, invoked before each refinement pass and confirmation
    // round. The coarse sweep runs the machine hot; a short settle here lets thermals
    // recover so the narrowing-down measurements are steadier. Kept as a callback so
    // the header stays pure (no sleeping / timing) -- the caller supplies the pause.
    std::function<void()> settle = nullptr;
    // Optional phase-transition hook. The caller typically raises its measurement
    // effort (longer timed runs) on Refine/Confirm so the precise readings reflect
    // sustained, throttled throughput rather than a brief un-throttled burst.
    std::function<void(Phase)> on_phase = nullptr;
    // Optional hard stop (a wall-clock time BUDGET): checked before each coarse point,
    // each refinement pass and each confirmation round. When it returns true the search
    // stops EARLY but gracefully -- it still returns the best point measured so far. This
    // is what bounds total tune time: the coarse grid lays down the shape first (cheap,
    // noisy), then refinement/confirmation narrow into the troughs for only as long as the
    // budget allows. Null = run to convergence (the deterministic default).
    std::function<bool()> should_stop = nullptr;
    // Coarse-pass strategy. false (default) = the joint CARTESIAN grid: robust to non-separable /
    // cross-interfering axes (where the best value of one knob depends on another) but coarse_points^K
    // evals. true = COORDINATE DESCENT: sweep one axis at a time holding the others at their running
    // best -- K * coarse_points evals, correct and far cheaper WHEN the axes are separable (the
    // empirically-observed case for threads / windows-per-thread / batch). The tuner sets this true by
    // default and false under --thorough (where the joint grid earns its cost).
    bool separable = false;
};

// --- Budgeted-sweep schedule (shared by every caller, unit-tested) ---------
// The effort knobs + per-phase measurement budget for a time-bounded sweep. Both the CPU
// (throughput) and GPU (device-step) tuners want the same shape -- a cheap/noisy coarse pass
// then progressively longer narrowing runs, with a fast vs `thorough` level -- so the numbers
// live here ONCE (not copy-pasted per sweep) and can be asserted in isolation. Pure: no I/O,
// no engine, no clock; `apply` installs it onto Options, the caller supplies the deadline and
// any logging. `gpu` selects the device-step budgets (steps cost ~seconds, so longer per phase
// and a lighter confirm than the CPU sweep).
struct Schedule {
    int    coarse_points = 5, max_passes = 4, confirm_top = 3, confirm_rounds = 2;
    double explore_ms = 120.0, refine_ms = 400.0, confirm_ms = 800.0;  // measurement budget per phase
    double phase_ms(Phase p) const {
        return p == Phase::Explore ? explore_ms : (p == Phase::Refine ? refine_ms : confirm_ms);
    }
};

// The fast (default) and `thorough` schedules for a CPU vs GPU sweep -- the single source of
// truth for those magic numbers. Fast keeps the coarse shape cheap and the confirm tail short
// (the wall-clock deadline is the real cap); thorough widens the grid and re-measures more for
// a tighter, lower-noise optimum when time is not the constraint.
inline Schedule schedule_for(bool thorough, bool gpu) {
    Schedule s;
    s.coarse_points  = thorough ? 8 : 5;
    s.max_passes     = thorough ? 8 : 4;
    s.confirm_top    = thorough ? 4 : 3;
    s.confirm_rounds = gpu ? (thorough ? 3 : 2)   // GPU steps cost seconds: confirm lighter
                           : (thorough ? 4 : 2);
    if (gpu) { s.explore_ms = thorough ? 500.0 : 300.0; s.refine_ms = thorough ? 800.0 : 600.0; s.confirm_ms = thorough ? 1200.0 : 900.0; }
    else     { s.explore_ms = thorough ? 250.0 : 120.0; s.refine_ms = thorough ? 600.0 : 400.0; s.confirm_ms = thorough ? 1200.0 : 800.0; }
    return s;
}

// Wire a Schedule + a wall-clock `deadline` predicate into Options: copies the effort knobs,
// installs should_stop, and builds an on_phase that publishes the live per-phase measurement
// budget into *budget_ms (the caller's objective reads it) before invoking `also` (logging).
inline void apply(Options& opt, const Schedule& s, std::function<bool()> deadline,
                  double* budget_ms, std::function<void(Phase)> also = nullptr) {
    opt.coarse_points  = s.coarse_points;
    opt.max_passes     = s.max_passes;
    opt.confirm_top    = s.confirm_top;
    opt.confirm_rounds = s.confirm_rounds;
    opt.should_stop    = std::move(deadline);
    opt.on_phase = [s, budget_ms, also = std::move(also)](Phase p) {
        if (budget_ms) *budget_ms = s.phase_ms(p);
        if (also) also(p);
    };
}

// A single objective measurement, recorded in call order for inspection/tests.
struct Eval {
    std::vector<int> index;        // candidate index per knob
    Assignment       values;       // the assignment evaluated
    double           score = 0.0;  // this individual measurement (not the aggregate)
};

struct Result {
    std::vector<int> best_index;                                  // winning candidate index per knob
    Assignment       best;                                        // winning value per knob
    double           best_score = -std::numeric_limits<double>::infinity();  // median over its samples
    int              best_samples = 0;                            // measurements behind the winner
    std::vector<Eval> trace;                                      // every measurement, in order
    int              evaluations = 0;                             // == trace.size()
};

// Build a discrete ladder for a continuous knob -- linear or geometric (log) spacing.
// `geometric_steps` is the natural choice for scale-free controls like a learning rate.
inline std::vector<double> linear_steps(double lo, double hi, int count) {
    std::vector<double> v;
    if (count <= 1) { v.push_back(lo); return v; }
    v.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i)
        v.push_back(lo + (hi - lo) * (static_cast<double>(i) / (count - 1)));
    return v;
}
inline std::vector<double> geometric_steps(double lo, double hi, int count) {
    std::vector<double> v;
    if (count <= 1 || lo <= 0.0 || hi <= 0.0) { v.push_back(lo); return v; }
    v.reserve(static_cast<std::size_t>(count));
    const double lr = std::log(lo), hr = std::log(hi);
    for (int i = 0; i < count; ++i)
        v.push_back(std::exp(lr + (hr - lr) * (static_cast<double>(i) / (count - 1))));
    return v;
}

// Search `space` to maximise `objective`. Robust to a noisy objective: every point
// keeps its samples and is ranked by their median, and the finalists are re-measured
// (the confirmation phase) before a winner is declared. Pure and deterministic for a
// deterministic objective -- repeated samples are identical, so the median is a no-op.
inline Result maximize(const Space& space, const Objective& objective, const Options& opt = {}) {
    Result R;
    const int K = static_cast<int>(space.size());
    if (K == 0) return R;
    const int coarse_points = std::max(2, opt.coarse_points);
    const int top_basins    = std::max(1, opt.top_basins);
    auto stop = [&] { return opt.should_stop && opt.should_stop(); };   // time-budget exhausted?

    std::map<std::vector<int>, std::vector<double>> samples;   // index -> all measurements

    auto assignment_of = [&](const std::vector<int>& idx) {
        Assignment a(static_cast<std::size_t>(K));
        for (int k = 0; k < K; ++k)
            a[static_cast<std::size_t>(k)] =
                space[static_cast<std::size_t>(k)].values[static_cast<std::size_t>(idx[static_cast<std::size_t>(k)])];
        return a;
    };

    // Median of a point's samples -- the robust score used for all ranking decisions.
    auto aggregate_of = [&](const std::vector<int>& idx) -> double {
        std::vector<double> v = samples.at(idx);
        std::sort(v.begin(), v.end());
        const std::size_t n = v.size();
        return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    };

    // Take one fresh measurement of a point (always calls the objective).
    auto sample = [&](const std::vector<int>& idx) -> double {
        Assignment a = assignment_of(idx);
        const double s = objective(a);
        samples[idx].push_back(s);
        R.trace.push_back({idx, a, s});
        ++R.evaluations;
        return aggregate_of(idx);
    };

    // Return a point's aggregate, sampling it once if it has never been measured.
    auto eval_at = [&](const std::vector<int>& idx) -> double {
        if (samples.find(idx) == samples.end()) return sample(idx);
        return aggregate_of(idx);
    };

    // Per-axis strided coarse sample indices (endpoints always included) and stride.
    std::vector<std::vector<int>> axis_samples(static_cast<std::size_t>(K));
    std::vector<int> stride(static_cast<std::size_t>(K));
    for (int k = 0; k < K; ++k) {
        const int n = static_cast<int>(space[static_cast<std::size_t>(k)].values.size());
        const int st = std::max(1, (n - 1) / std::max(1, coarse_points - 1));
        stride[static_cast<std::size_t>(k)] = st;
        std::vector<int>& s = axis_samples[static_cast<std::size_t>(k)];
        for (int i = 0; i < n; i += st) s.push_back(i);
        if (s.empty() || s.back() != n - 1) s.push_back(n - 1);   // anchor the top end
    }

    // 1. Coarse pass: lay down the basin seed(s) the refinement then narrows.
    if (opt.on_phase) opt.on_phase(Phase::Explore);
    std::vector<std::pair<double, std::vector<int>>> coarse;
    if (opt.separable) {
        // COORDINATE DESCENT (prior art: coordinate search / line search on a separable objective):
        // start each axis at its middle coarse sample, then sweep ONE axis at a time over its coarse
        // samples holding the others at their running best. K * coarse_points evals instead of the
        // product -- e.g. threads(6) + wpt(5) = 11, not 30 -- and exact when the axes don't
        // cross-interfere. A single seed results; the ternary refinement below fine-tunes it.
        std::vector<int> cur(static_cast<std::size_t>(K));
        for (int k = 0; k < K; ++k) {
            const std::vector<int>& s = axis_samples[static_cast<std::size_t>(k)];
            cur[static_cast<std::size_t>(k)] = s[s.size() / 2];               // middle coarse sample
        }
        for (int k = 0; k < K && !stop(); ++k) {
            const std::vector<int>& s = axis_samples[static_cast<std::size_t>(k)];
            std::vector<int> probe = cur;
            double best = -std::numeric_limits<double>::infinity();
            int best_i = cur[static_cast<std::size_t>(k)];
            for (const int idx : s) {
                probe[static_cast<std::size_t>(k)] = idx;
                const double v = eval_at(probe);
                if (v > best) { best = v; best_i = idx; }
            }
            cur[static_cast<std::size_t>(k)] = best_i;
        }
        coarse.push_back({eval_at(cur), cur});
    } else {
        // JOINT CARTESIAN grid (--thorough): the coarse_points^K product, robust to cross-interference.
        std::vector<int> pick(static_cast<std::size_t>(K), 0);   // mixed-radix index into axis_samples
        for (;;) {
            // Budget guard: once exhausted, stop laying down coarse points but keep at least
            // one so a basin seed (and a winner) always exists.
            if (!coarse.empty() && stop()) break;
            std::vector<int> point(static_cast<std::size_t>(K));
            for (int k = 0; k < K; ++k)
                point[static_cast<std::size_t>(k)] =
                    axis_samples[static_cast<std::size_t>(k)][static_cast<std::size_t>(pick[static_cast<std::size_t>(k)])];
            coarse.push_back({eval_at(point), point});
            int k = 0;
            for (; k < K; ++k) {
                if (++pick[static_cast<std::size_t>(k)] < static_cast<int>(axis_samples[static_cast<std::size_t>(k)].size())) break;
                pick[static_cast<std::size_t>(k)] = 0;
            }
            if (k == K) break;
        }
    }

    // 2. Pick the top-K coarse points as basin seeds (>1 keeps an alternative trough).
    std::stable_sort(coarse.begin(), coarse.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    const int basins = std::min(static_cast<int>(coarse.size()), top_basins);

    // 3. Refine each basin by coordinate descent, narrowing each axis with a TERNARY CHOP rather
    //    than scanning every point in the +/-stride window. Within a basin (the coarse grid already
    //    localized it) each axis is ~unimodal, so probing two interior thirds and discarding the
    //    worse outer third converges in O(log range) evals -- the "binary chop driven by the delta
    //    between checks" -- and a monotonic axis (throughput rising with threads/batch to the core
    //    count) collapses straight to its top end instead of re-scanning the whole window every pass.
    //    Every probe is cached, so the final winner is still the global max over ALL points measured
    //    (chop probes included), and the confirm phase re-measures finalists -- so the unimodal
    //    assumption only steers WHERE to look, it cannot crown an unmeasured point.
    if (opt.on_phase) opt.on_phase(Phase::Refine);
    for (int b = 0; b < basins; ++b) {
        if (stop()) break;                  // budget exhausted -> skip remaining basins
        std::vector<int> cur = coarse[static_cast<std::size_t>(b)].second;
        double basin_best = eval_at(cur);
        for (int pass = 0; pass < std::max(1, opt.max_passes); ++pass) {
            if (stop()) break;
            if (opt.settle) opt.settle();   // cool-down before narrowing down -- steadier readings
            const double prev = basin_best;
            for (int k = 0; k < K; ++k) {
                const int n = static_cast<int>(space[static_cast<std::size_t>(k)].values.size());
                int lo = std::max(0, cur[static_cast<std::size_t>(k)] - stride[static_cast<std::size_t>(k)]);
                int hi = std::min(n - 1, cur[static_cast<std::size_t>(k)] + stride[static_cast<std::size_t>(k)]);
                std::vector<int> probe = cur;
                auto at = [&](int i) { probe[static_cast<std::size_t>(k)] = i; return eval_at(probe); };
                while (hi - lo > 2) {                   // ternary chop: discard the worse outer third
                    const int m1 = lo + (hi - lo) / 3;
                    const int m2 = hi - (hi - lo) / 3;
                    if (at(m1) < at(m2)) lo = m1 + 1; else hi = m2 - 1;
                }
                int best_i = cur[static_cast<std::size_t>(k)];
                for (int i = lo; i <= hi; ++i) {        // pin the exact max in the collapsed bracket
                    const double s = at(i);
                    if (s > basin_best) { basin_best = s; best_i = i; }
                }
                cur[static_cast<std::size_t>(k)] = best_i;
            }
            const double denom = std::max(1e-12, std::fabs(prev));
            if ((basin_best - prev) <= opt.tol * denom) break;   // refinement stalled -> converged
        }
    }

    // Current winner = highest median over all sampled points.
    auto best_so_far = [&]() -> std::vector<int> {
        std::vector<int> bi;
        double bs = -std::numeric_limits<double>::infinity();
        for (const auto& [idx, _] : samples) {
            const double s = aggregate_of(idx);
            if (s > bs) { bs = s; bi = idx; }
        }
        return bi;
    };

    // 4. Confirm: re-measure the leading finalists plus the winner's immediate neighbours
    //    so sampling noise / thermal drift cannot crown a fluke. Ranking by the median
    //    over the extra rounds is what makes the reported optimum reproducible.
    if (opt.confirm_rounds > 0 && opt.confirm_top > 0 && !samples.empty()) {
        if (opt.on_phase) opt.on_phase(Phase::Confirm);
        std::vector<std::vector<int>> pts;
        pts.reserve(samples.size());
        for (const auto& [idx, _] : samples) pts.push_back(idx);
        std::stable_sort(pts.begin(), pts.end(),
                         [&](const auto& A, const auto& B) { return aggregate_of(A) > aggregate_of(B); });
        const int top = std::min<int>(opt.confirm_top, static_cast<int>(pts.size()));
        std::vector<std::vector<int>> finalists(pts.begin(), pts.begin() + top);

        // Add the +/-1 neighbours of the winner -- "expand to confirm" a steadier neighbour.
        const std::vector<int> win = best_so_far();
        for (int k = 0; k < K && !win.empty(); ++k) {
            const int n = static_cast<int>(space[static_cast<std::size_t>(k)].values.size());
            for (int d = -1; d <= 1; d += 2) {
                std::vector<int> nb = win;
                const int j = nb[static_cast<std::size_t>(k)] + d;
                if (j < 0 || j >= n) continue;
                nb[static_cast<std::size_t>(k)] = j;
                if (std::find(finalists.begin(), finalists.end(), nb) == finalists.end())
                    finalists.push_back(nb);
            }
        }

        for (int round = 0; round < opt.confirm_rounds; ++round) {
            if (stop()) break;              // budget exhausted -> stop re-measuring, rank what we have
            if (opt.settle) opt.settle();   // cool-down before each confirmation round
            for (const auto& idx : finalists) sample(idx);

            // Prune finalists that are clearly below the leader -- a config more than the
            // noise floor under the best can't be the true optimum within measurement
            // noise, so we stop spending long runs on it and let the contenders fight on.
            if (finalists.size() > 1) {
                std::vector<int> leader;
                double lead_med = -std::numeric_limits<double>::infinity();
                for (const auto& idx : finalists) {
                    const double m = aggregate_of(idx);
                    if (m > lead_med) { lead_med = m; leader = idx; }
                }
                const std::vector<double>& ls = samples.at(leader);
                const double spread = ls.empty() ? 0.0
                    : (*std::max_element(ls.begin(), ls.end()) - *std::min_element(ls.begin(), ls.end()));
                const double band = std::max(opt.noise_floor * std::fabs(lead_med), spread);
                std::vector<std::vector<int>> kept;
                for (const auto& idx : finalists)
                    if (lead_med - aggregate_of(idx) <= band) kept.push_back(idx);
                finalists.swap(kept);   // losers drop out; survivors keep being confirmed
            }
        }
    }

    // Finalise: winner = highest median over all (now re-measured) points.
    if (const std::vector<int> bi = best_so_far(); !bi.empty()) {
        R.best_index   = bi;
        R.best         = assignment_of(bi);
        R.best_score   = aggregate_of(bi);
        R.best_samples = static_cast<int>(samples.at(bi).size());
    }
    return R;
}

}  // namespace sub0::tune

#endif  // SUB0_TUNE_HPP
