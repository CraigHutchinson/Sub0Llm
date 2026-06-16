#pragma once

// schedule.hpp — corpus-scaled training schedule in ONE tested place.
//
// THE COVERAGE RULE (a Ch28 lesson): a held-out eval is only a meaningful decision
// signal once the model has trained over a significant fraction of the corpus SINCE the
// last eval. Eval more often than that and consecutive evals differ only by noise, so
// early-stopping patience trips on noise rather than real plateau. We measure coverage
// in NON-OVERLAPPING windows (train_tokens / seq_len = one epoch) and require at least
// `min_coverage` of an epoch between evals (hard floor 0.5 = 50%).
//
// A data-parallel step trains `threads` windows, so the STEP-denominated cadence folds
// the thread count in HERE, once — fixing the prior bug where §2b printed the pre-÷threads
// value as the step cadence. Every corpus-scaled knob (eval cadence, safety bound, eval
// sample sizes) is derived here so the logic is shared, overridable, and unit-tested.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace sub0diff::train {

struct ScheduleConfig {
    double        eval_factor     = 1.0;   // epochs of train coverage per eval (1.0 = once per full epoch)
    double        min_coverage    = 0.5;   // HARD floor — never eval on < 50% epoch coverage
    double        coverage_epochs = 50.0;  // safety bound (early stopping ends the run sooner)
    std::uint64_t min_eval_steps  = 100;   // never eval more often than this (tiny corpora)
    // Held-out eval sample sizes: clamp(sliding_eval / div, lo, hi).
    std::size_t   nelbo_div = 500,  nelbo_lo = 64,   nelbo_hi = 512;
    std::size_t   recall_div = 20,  recall_lo = 2000, recall_hi = 16000;
};

struct Schedule {
    std::uint64_t epoch_windows = 0;      // non-overlapping windows = 1 epoch (coverage unit)
    std::uint64_t sliding_train = 0;      // overlapping sliding positions (train)
    std::uint64_t sliding_eval  = 0;      // overlapping sliding positions (eval)
    std::uint64_t eval_every    = 0;      // STEPS between evals (thread count folded in)
    double        coverage_per_eval = 0;  // fraction of an epoch trained between evals (≥ min_coverage)
    std::uint64_t steps_bound   = 0;      // safety bound (steps)
    std::size_t   eval_nelbo_windows = 0;
    std::size_t   recall_windows = 0;
};

// Sliding (overlapping) window count for a token stream.
[[nodiscard]] constexpr std::uint64_t sliding_positions(std::uint64_t tokens,
                                                        std::int64_t seq_len) noexcept {
    const auto T = static_cast<std::uint64_t>(seq_len);
    return (seq_len > 0 && tokens >= T) ? tokens - T + 1 : 0;
}

// Sentinel for "derive this value" on the override parameters.
inline constexpr std::uint64_t kDerive   = 0;
inline constexpr std::size_t   kDeriveSz = std::numeric_limits<std::size_t>::max();

// Derive the whole schedule. `masked_per_window` = E[t]·seq_len (the diffusion masked-
// token rate). Overrides: eval_every/steps == kDerive → derive; recall == kDeriveSz →
// derive (any other value, including 0 for "exhaustive", is passed through unchanged).
// NOTE on `windows_per_step`: a data-parallel optimizer step trains the EFFECTIVE BATCH B windows
// (= --batch, split B/W across W workers), so the step-denominated cadences fold B in here — NOT
// the worker count W. (Before the B⊥W decoupling, B always equalled W=threads; passing threads here
// was correct then and silently wrong once B>W — it made evals/curriculum decisions B/W× too rare.)
[[nodiscard]] inline Schedule make_schedule(std::uint64_t train_tokens, std::uint64_t eval_tokens,
                                            std::int64_t seq_len, std::size_t windows_per_step,
                                            double masked_per_window,
                                            const ScheduleConfig& cfg = {},
                                            std::uint64_t eval_every_override = kDerive,
                                            std::uint64_t steps_override = kDerive,
                                            std::size_t recall_override = kDeriveSz) {
    Schedule s;
    const auto T = static_cast<std::uint64_t>(seq_len);
    const std::size_t B = std::max<std::size_t>(1, windows_per_step);
    s.epoch_windows = (seq_len > 0) ? train_tokens / T : 0;
    s.sliding_train = sliding_positions(train_tokens, seq_len);
    s.sliding_eval  = sliding_positions(eval_tokens, seq_len);

    // Eval cadence: ≥ min_coverage of an epoch (in windows) between evals, converted to
    // STEPS (each step trains B windows), floored at min_eval_steps.
    if (eval_every_override != kDerive) {
        s.eval_every = eval_every_override;
    } else {
        const double cov = std::max(cfg.eval_factor, cfg.min_coverage);
        const double windows_per_eval = cov * static_cast<double>(s.epoch_windows);
        // Round UP so the realised coverage is GUARANTEED ≥ cov (never just under 50%).
        s.eval_every = std::max(cfg.min_eval_steps,
                                static_cast<std::uint64_t>(std::ceil(windows_per_eval / static_cast<double>(B))));
    }
    s.coverage_per_eval = s.epoch_windows
        ? static_cast<double>(s.eval_every) * static_cast<double>(B) / static_cast<double>(s.epoch_windows)
        : 0.0;

    // Safety bound: coverage_epochs over the train tokens, in masked-token terms.
    if (steps_override != kDerive) {
        s.steps_bound = steps_override;
    } else {
        const double masked_per_step = masked_per_window * static_cast<double>(B);
        s.steps_bound = (masked_per_step > 0.0)
            ? static_cast<std::uint64_t>(cfg.coverage_epochs * static_cast<double>(train_tokens) / masked_per_step)
            : 0;
    }

    const std::size_t se = static_cast<std::size_t>(s.sliding_eval);
    s.eval_nelbo_windows = std::clamp(se / cfg.nelbo_div, cfg.nelbo_lo, cfg.nelbo_hi);
    s.recall_windows = (recall_override == kDeriveSz)
        ? std::clamp(se / cfg.recall_div, cfg.recall_lo, cfg.recall_hi)
        : recall_override;
    return s;
}

} // namespace sub0diff::train
