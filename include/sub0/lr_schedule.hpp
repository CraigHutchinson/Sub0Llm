// sub0/lr_schedule.hpp — the training learning-rate schedule (Warmup-Stable-Decay).
//
// Header-only and engine-free so it is testable on its own: this is ~10 lines of pure math whose
// boundary conditions (end of warmup, entry to cooldown, end of cooldown, the floor) silently ruin
// every training run if they are wrong, and it previously lived as a file-local `inline` inside
// train_stage.cpp where nothing could reach it. See tests/lr_schedule_tests.cpp.
//
// Full rationale in docs/LR_SCHEDULE.md.

#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

namespace sub0::lr {

// --- WSD cooldown (replaces the 2017 inverse-sqrt / Noam decay) -------------------------------
// The schedule is Warmup-Stable-Decay: linear warmup to peak, HOLD at peak, then a cosine cooldown to
// FLOOR_FRACTION of peak once the plateau detector fires.
//
// Why this replaced inverse-sqrt:
//   1. Because this project stops on a PLATEAU DETECTOR rather than a fixed horizon, the old schedule
//      halted at an arbitrary point on a slowly-decaying tail and **never ran a cooldown at all**. The
//      literature finds a disproportionate share of the final loss drop happens during that anneal, so
//      these models were systematically UNDER-FINISHED. The gap was never "our decay has the wrong
//      shape"; it was "we omit the anneal".
//   2. A plateau cannot be cleanly detected under a decaying LR -- flattening is confounded with the
//      decay. Measured on arm A: improvement fell to 0.35x while LR fell to 0.66x, so part of the
//      flattening was step size, not saturation. Worse, under inverse-sqrt the cumulative learning
//      distance integral(lr dt) ~ sqrt(t) DIVERGES, so loss creeps down forever and there is no true
//      flat line to detect -- which is why every magnitude threshold felt arbitrary: it was. A constant
//      stable phase makes "stopped improving" a property of model+data, not a schedule artifact.
// That is the real reason WSD and plateau-stopping pair naturally -- a better argument than the
// cooldown-loss one, because it makes the DETECTOR well-posed rather than merely adding an anneal.
//
// DeepSeek-V4 (arXiv 2606.19348) is the reference for the shape: cosine to 10% of peak. NOT linear, and
// NOT to zero -- the 10% floor is what Chinchilla used too, and it stops the schedule strangling
// learning before the real flat line. Caveat kept honest: V4 reports Muon REDUCES sensitivity to
// schedule hyperparameters, and this project already runs Muon on the matrices, so the SHAPE should
// matter less here than the headline suggests. That is an argument to measure, not to skip -- reduced
// sensitivity to the shape of a decay says nothing about OMITTING the decay entirely.
inline constexpr double FLOOR_FRACTION = 0.10;   // cosine cooldown target, as a fraction of peak

// Cooldown LENGTH as a fraction of the steps already trained when the plateau fired. Derived from the
// run's own scale rather than a fixed step count (this project's standing "derive bounds from scale,
// not fixed thresholds" rule): a plateau at 30k steps earns a proportionally longer anneal than one at
// 3k, and no constant here can be wrong for a corpus size it was never calibrated on. 10% matches
// common WSD practice.
inline constexpr double COOLDOWN_FRACTION = 0.10;
inline constexpr long   COOLDOWN_MIN_STEPS = 50;   // floor, so a very early plateau still anneals

// Cooldown length for a plateau detected at `plateau_step`. Pure, so a resume reconstructs it exactly
// from the single persisted field (cooldown_start) -- one source of truth, nothing to drift.
constexpr long cooldown_steps_for(long plateau_step) {
    const long derived = static_cast<long>(COOLDOWN_FRACTION * static_cast<double>(plateau_step));
    return derived > COOLDOWN_MIN_STEPS ? derived : COOLDOWN_MIN_STEPS;
}

// Per-step learning rate:
//   step <  warmup                LINEAR warmup, 0 -> peak. Avoids early high-lr instability.
//   warmup <= step, no cooldown   CONSTANT at peak -- the stable phase.
//   cooldown_start >= 0           COSINE from peak down to FLOOR_FRACTION * peak over cooldown_steps,
//                                 then held at the floor.
//
// `cooldown_start` is RUN STATE, not a function of step, so unlike the old schedule this is not pure in
// `step` alone -- it must be persisted or a resumed mid-cooldown run silently reverts to constant LR
// and finishes un-annealed. It rides the checkpoint (CKPT_VERSION 6) alongside `step` itself, the
// natural home for exact-resume state.
//
// That bump STRANDS every existing checkpoint, including the parked LoopSplit arms', and it is the sole
// cause of that -- an earlier version of this comment claimed schemeV5's vocabulary change had already
// invalidated them via the header's nfloat guard, so the bump "cost nothing". That was asserted, not
// traced, and it is wrong: schemeV5 changes WHICH strings occupy the vocabulary, not how many, so
// PARAM_FLOATS need not move at all. The bump is taken deliberately -- the user waived backward compat
// for this window and the arms are being retrained from scratch anyway -- and NOT because it was free.
// (AGENTS.md 3 rule 2 would otherwise prefer an additive trailing field with a short-read default of
// -1, which every pre-WSD checkpoint could have satisfied, since none was ever mid-cooldown.)
//
// -1 in `cooldown_start` means "not cooling down yet".
inline float lr_at(long step, float peak, long warmup, long cooldown_start, long cooldown_steps) {
    if (warmup < 1) warmup = 1;
    if (step < warmup)                                              // linear warmup
        return peak * (static_cast<float>(step) / static_cast<float>(warmup));
    if (cooldown_start < 0 || step < cooldown_start) return peak;   // stable phase, held at peak
    if (cooldown_steps < 1) cooldown_steps = 1;
    const double t = std::clamp(static_cast<double>(step - cooldown_start) /
                                static_cast<double>(cooldown_steps), 0.0, 1.0);
    // Cosine from 1.0 to FLOOR_FRACTION -- HALF a cosine period, so the rate of change is zero at BOTH
    // ends: no discontinuity entering the anneal from the stable phase, and a gentle landing on the
    // floor. t is clamped, so past the end this correctly holds at the floor rather than turning back up.
    const double shape = 0.5 * (1.0 + std::cos(std::numbers::pi * t));
    return static_cast<float>(static_cast<double>(peak) * (FLOOR_FRACTION + (1.0 - FLOOR_FRACTION) * shape));
}

}  // namespace sub0::lr
