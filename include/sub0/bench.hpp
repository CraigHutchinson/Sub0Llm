// bench.hpp — shared adaptive measurement primitive for the autotuner / benchmarks.
//
// One honest way to time a workload whose per-step cost varies by orders of magnitude across
// the knob space (a batch-64 GPU step vs a batch-1024 one; a 1-thread CPU step vs 24-thread):
// size the timed run to a wall-time BUDGET rather than a fixed iteration count. Otherwise the
// profiling time balloons with the workload -- exactly the failure the GPU batch sweep hit,
// where each larger batch took proportionally longer to profile and the whole sweep ran away.
//
// This mirrors how Google Benchmark and friends pick their iteration count: warm up, probe once
// to estimate the per-step cost, then run clamp(budget/est, min, max) timed steps. Every
// measurement then costs ~the same wall time regardless of how heavy a single step is, while the
// min/max bounds keep tiny steps statistically stable and huge steps from overrunning.
//
// Timing-source-agnostic by design: the caller supplies `run_timed(n) -> elapsed_ms`, so the CPU
// path can use a steady_clock and the GPU path CUDA events, both driving the identical adaption.

#pragma once

#include <algorithm>

namespace sub0::bench {

// How long each measurement should take and the iteration bounds that keep it honest.
struct Budget {
    double budget_ms = 1200.0;   // target wall-time per measurement (the run is sized to hit this)
    int    warmup    = 2;        // untimed steps before probing: clock ramp / cache warm / graph capture
    int    min_iters = 3;        // floor so run-to-run jitter averages out even on slow steps
    int    max_iters = 60;       // ceiling so a fast step doesn't spin for the whole budget
};

// Result of one adaptive measurement.
struct Timing {
    double per_step_ms = 0.0;    // mean ms per timed step (the metric callers turn into tok/s, window/s, ...)
    double total_ms    = 0.0;    // wall time actually spent timing (probe + sized run), for live duration output
    int    iters       = 0;      // timed steps run (excludes warmup + probe)
};

// Adaptive, budget-sized measurement. `step()` runs one step untimed (warmup); `run_timed(n)`
// runs n steps and returns their elapsed milliseconds. We warm up, take one timed probe to
// estimate the per-step cost, size the run to the budget within [min,max], then time it.
template <class Step, class RunTimed>
Timing adaptive_time(Step&& step, RunTimed&& run_timed, const Budget& b = {}) {
    for (int w = 0; w < b.warmup; ++w) step();

    const double budget = b.budget_ms > 0.0 ? b.budget_ms : Budget{}.budget_ms;
    const double probe  = run_timed(1);                       // one timed probe (doubles as extra warmup)
    const double est    = probe > 1e-3 ? probe : 1e-3;        // guard against a zero/garbage probe
    const int    sized  = static_cast<int>(budget / est);
    const int    n      = std::clamp(sized, b.min_iters, b.max_iters);

    const double timed = run_timed(n);
    return Timing{ .per_step_ms = timed / n, .total_ms = probe + timed, .iters = n };
}

}  // namespace sub0::bench
