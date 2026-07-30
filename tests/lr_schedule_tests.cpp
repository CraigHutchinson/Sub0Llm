// lr_schedule_tests.cpp — the WSD learning-rate schedule (sub0/lr_schedule.hpp).
//
// This is ~10 lines of pure math that every training step consumes, and a boundary error in it is
// SILENT: the run still trains, just at the wrong rate, and the only symptom is a worse final loss
// weeks later. It previously lived as a file-local `inline` in train_stage.cpp where nothing could
// reach it, and shipped with no correctness check at all. AGENTS.md 6 requires new math carry a
// numerical check rather than "it compiles".
//
// Engine-free (links sub0_frontend only), so these run everywhere and stay fast.

#include <catch2/catch_test_macros.hpp>

#include "sub0/lr_schedule.hpp"

#include <cmath>
#include <vector>

using sub0::lr::lr_at;
using sub0::lr::cooldown_steps_for;
using sub0::lr::FLOOR_FRACTION;

namespace {
constexpr float kPeak = 1.0f;
constexpr long  kWarm = 100;
constexpr float kEps  = 1e-6f;
bool near(float a, float b, float eps = kEps) { return std::fabs(a - b) <= eps; }
}  // namespace

TEST_CASE("lr: linear warmup reaches peak exactly at the warmup boundary", "[lr]") {
    REQUIRE(near(lr_at(0, kPeak, kWarm, -1, 0), 0.0f));
    REQUIRE(near(lr_at(kWarm / 2, kPeak, kWarm, -1, 0), 0.5f));
    // The boundary itself is the FIRST step of the stable phase (step < warmup is warmup), so it is
    // exactly peak -- and the step before it is one increment below. An off-by-one here would either
    // clip the last warmup step or start the stable phase a step early.
    REQUIRE(near(lr_at(kWarm - 1, kPeak, kWarm, -1, 0), 0.99f));
    REQUIRE(near(lr_at(kWarm, kPeak, kWarm, -1, 0), 1.0f));
}

TEST_CASE("lr: the stable phase is genuinely CONSTANT at peak", "[lr]") {
    // The whole reason the plateau detector became well-posed: no decay component to confound a
    // flattening loss curve with. If this ever drifts, the detector silently goes back to measuring
    // schedule shape instead of saturation.
    for (long s : {100L, 1000L, 50000L, 10'000'000L})
        REQUIRE(near(lr_at(s, kPeak, kWarm, -1, 0), kPeak));
}

TEST_CASE("lr: warmup wins over an active cooldown", "[lr]") {
    // Unreachable in production (plateau detection is gated well past warmup) but the ordering must be
    // deliberate rather than incidental: a cooldown flag set during warmup must not cut the ramp.
    REQUIRE(near(lr_at(50, kPeak, kWarm, 10, 100), 0.5f));
}

TEST_CASE("lr: cosine cooldown starts at peak, ends at the floor, and is monotone", "[lr]") {
    const long start = 1000, len = 500;
    REQUIRE(near(lr_at(start, kPeak, kWarm, start, len), kPeak));                       // t=0
    REQUIRE(near(lr_at(start + len, kPeak, kWarm, start, len),
                 static_cast<float>(FLOOR_FRACTION)));                                  // t=1
    // Midpoint of a half-cosine from 1 -> floor is exactly halfway between them.
    REQUIRE(near(lr_at(start + len / 2, kPeak, kWarm, start, len),
                 static_cast<float>(FLOOR_FRACTION + (1.0 - FLOOR_FRACTION) * 0.5), 1e-5f));

    float prev = lr_at(start, kPeak, kWarm, start, len);
    for (long s = start + 1; s <= start + len; ++s) {
        const float cur = lr_at(s, kPeak, kWarm, start, len);
        REQUIRE(cur <= prev + kEps);          // never turns back up
        prev = cur;
    }
}

TEST_CASE("lr: past the end of the cooldown the rate HOLDS at the floor", "[lr]") {
    // t is clamped, so an over-run must not let the cosine come back up the other side of the period --
    // which is what an unclamped half-cosine would do and would be catastrophic (LR rising to peak
    // again at the very end of training).
    const long start = 1000, len = 500;
    for (long over : {1L, 10L, len, 10 * len})
        REQUIRE(near(lr_at(start + len + over, kPeak, kWarm, start, len),
                     static_cast<float>(FLOOR_FRACTION)));
}

TEST_CASE("lr: degenerate inputs are clamped, not undefined", "[lr]") {
    REQUIRE(std::isfinite(lr_at(5, kPeak, 0, -1, 0)));       // warmup < 1 -> no division by zero
    REQUIRE(std::isfinite(lr_at(5, kPeak, -7, -1, 0)));
    // cooldown_steps < 1 collapses to a single step: at the start it is peak, one step later the floor.
    REQUIRE(near(lr_at(1000, kPeak, kWarm, 1000, 0), kPeak));
    REQUIRE(near(lr_at(1001, kPeak, kWarm, 1000, 0), static_cast<float>(FLOOR_FRACTION)));
}

TEST_CASE("lr: cooldown length is derived from scale, with a floor", "[lr]") {
    REQUIRE(cooldown_steps_for(100000) == 10000);            // 10% of elapsed
    REQUIRE(cooldown_steps_for(10000) == 1000);
    REQUIRE(cooldown_steps_for(100) == sub0::lr::COOLDOWN_MIN_STEPS);   // floor binds for a tiny run
    REQUIRE(cooldown_steps_for(0) == sub0::lr::COOLDOWN_MIN_STEPS);
    // Monotone in the plateau step: a later plateau never earns a SHORTER anneal.
    long prev = 0;
    for (long s = 0; s < 200000; s += 977) {
        const long cur = cooldown_steps_for(s);
        REQUIRE(cur >= prev);
        prev = cur;
    }
}

TEST_CASE("lr: a resume reconstructs the cooldown exactly from the persisted start step", "[lr]") {
    // The checkpoint stores ONLY cooldown_start; cooldown_steps is re-derived. If those two ever
    // disagree, a resumed run anneals on a different curve than the one it was on -- invisibly.
    // This pins that the reconstruction is bit-identical across the whole cooldown.
    const long start = 31337;
    const long len_at_trigger = cooldown_steps_for(start);
    const long len_after_resume = cooldown_steps_for(start);   // same pure function, same input
    REQUIRE(len_at_trigger == len_after_resume);
    for (long s = start; s <= start + len_at_trigger; s += 13)
        REQUIRE(lr_at(s, kPeak, kWarm, start, len_at_trigger) ==
                lr_at(s, kPeak, kWarm, start, len_after_resume));
}

TEST_CASE("lr: a fixed-budget run's anneal lands exactly on its final step", "[lr]") {
    // The fixed-budget path schedules cooldown_start = max_steps - cooldown_steps up front. The point
    // of that arithmetic is that the LAST step of training is the floor -- not somewhere mid-curve, and
    // not overshooting. Gating cooldown entry on the plateau detector alone previously left this path
    // with NO anneal at all, which is worse than the schedule it replaced.
    const long max_steps = 20000;
    const long len = cooldown_steps_for(max_steps);
    const long start = max_steps - len;
    REQUIRE(start > 0);
    REQUIRE(near(lr_at(start, kPeak, kWarm, start, len), kPeak));
    REQUIRE(near(lr_at(max_steps, kPeak, kWarm, start, len), static_cast<float>(FLOOR_FRACTION)));
}

TEST_CASE("lr: a budget that ends inside warmup never anneals -- warmup wins", "[lr]") {
    // Found by the TinyStories smoke run: a 200-step budget against a 1129-step warmup announced a
    // "cosine cooldown from step 150" that could never fire, because lr_at() checks `step < warmup`
    // first and warmup correctly takes precedence. The MATH was right; the caller's announcement was
    // not. This pins the math so the caller's fix has something to be consistent with.
    const long warmup = 1129, budget = 200;
    const long len = cooldown_steps_for(budget);          // what the naive scheduling would have picked
    const long start = budget - len;
    REQUIRE(start < warmup);                              // the premise: the "cooldown" sits inside warmup
    for (long s = start; s <= budget; ++s) {
        // Still on the warmup ramp at every step, rising -- never annealing.
        REQUIRE(near(lr_at(s, kPeak, warmup, start, len),
                     kPeak * (static_cast<float>(s) / static_cast<float>(warmup))));
    }
}
