#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0diff/train/schedule.hpp"

using Catch::Matchers::WithinAbs;
using namespace sub0diff::train;

// completeb-scale numbers: 2.19M train tokens, ~132K eval tokens, seq_len 64.
static constexpr std::uint64_t kTrain = 2'190'949, kEval = 131'702;
static constexpr std::int64_t  kT     = 64;
static constexpr double        kMaskPerWin = 0.51 * 64.0;   // E[t]·T, t∈(0.02,1]

TEST_CASE("schedule: epoch = non-overlapping windows", "[schedule]") {
    auto s = make_schedule(kTrain, kEval, kT, /*threads=*/4, kMaskPerWin);
    REQUIRE(s.epoch_windows == kTrain / 64);                 // 34233
    REQUIRE(s.sliding_train == kTrain - 64 + 1);
    REQUIRE(s.sliding_eval  == kEval  - 64 + 1);
}

TEST_CASE("schedule: eval cadence enforces >=50% epoch coverage", "[schedule]") {
    for (std::size_t threads : {1u, 2u, 4u, 8u}) {
        auto s = make_schedule(kTrain, kEval, kT, threads, kMaskPerWin);
        // Coverage = windows trained between evals / epoch. Must be >= the 50% floor.
        REQUIRE(s.coverage_per_eval >= 0.5 - 1e-6);
        // And it folds the thread count in: windows/eval = eval_every * threads.
        const double windows_per_eval = static_cast<double>(s.eval_every) * threads;
        REQUIRE_THAT(windows_per_eval / s.epoch_windows, WithinAbs(s.coverage_per_eval, 1e-9));
    }
}

TEST_CASE("schedule: cadence in STEPS scales 1/threads (same window coverage)", "[schedule]") {
    auto s1 = make_schedule(kTrain, kEval, kT, /*threads=*/1, kMaskPerWin);
    auto s4 = make_schedule(kTrain, kEval, kT, /*threads=*/4, kMaskPerWin);
    // Same ~50% window coverage, but 4 threads ⇒ ~4× fewer STEPS between evals.
    REQUIRE_THAT(s1.coverage_per_eval, WithinAbs(s4.coverage_per_eval, 0.01));
    REQUIRE(s4.eval_every < s1.eval_every);
    REQUIRE_THAT(static_cast<double>(s1.eval_every) / static_cast<double>(s4.eval_every),
                 WithinAbs(4.0, 0.3));
}

TEST_CASE("schedule: eval_factor below floor is clamped to 50%", "[schedule]") {
    ScheduleConfig low; low.eval_factor = 0.1;               // below the 0.5 floor
    auto s = make_schedule(kTrain, kEval, kT, 4, kMaskPerWin, low);
    REQUIRE(s.coverage_per_eval >= 0.5 - 1e-6);              // clamped up, not 10%
}

TEST_CASE("schedule: larger eval_factor means less frequent evals, more coverage", "[schedule]") {
    ScheduleConfig half;  half.eval_factor = 0.5;
    ScheduleConfig full;  full.eval_factor = 1.0;
    auto sh = make_schedule(kTrain, kEval, kT, 4, kMaskPerWin, half);
    auto sf = make_schedule(kTrain, kEval, kT, 4, kMaskPerWin, full);
    REQUIRE(sf.eval_every > sh.eval_every);
    REQUIRE(sf.coverage_per_eval > sh.coverage_per_eval);
    REQUIRE_THAT(sf.coverage_per_eval, WithinAbs(1.0, 0.02));
}

TEST_CASE("schedule: steps bound = coverage_epochs of masked-token epochs", "[schedule]") {
    ScheduleConfig c; c.coverage_epochs = 50.0;
    auto s = make_schedule(kTrain, kEval, kT, /*threads=*/4, kMaskPerWin, c);
    // steps_bound * masked_per_step ≈ coverage_epochs * train_tokens.
    const double masked_per_step = kMaskPerWin * 4.0;
    REQUIRE_THAT(static_cast<double>(s.steps_bound) * masked_per_step
                     / static_cast<double>(kTrain),
                 WithinAbs(50.0, 0.5));
}

TEST_CASE("schedule: eval-sample sizes clamp to the configured band", "[schedule]") {
    auto s = make_schedule(kTrain, kEval, kT, 4, kMaskPerWin);
    REQUIRE(s.eval_nelbo_windows >= 64);
    REQUIRE(s.eval_nelbo_windows <= 512);
    REQUIRE(s.recall_windows >= 2000);
    REQUIRE(s.recall_windows <= 16000);
}

TEST_CASE("schedule: overrides bypass derivation", "[schedule]") {
    auto s = make_schedule(kTrain, kEval, kT, 4, kMaskPerWin, ScheduleConfig{},
                           /*eval_every=*/777, /*steps=*/12345, /*recall=*/4242);
    REQUIRE(s.eval_every == 777);
    REQUIRE(s.steps_bound == 12345);
    REQUIRE(s.recall_windows == 4242);
    // recall override of 0 (exhaustive) passes through unchanged.
    auto e = make_schedule(kTrain, kEval, kT, 4, kMaskPerWin, ScheduleConfig{},
                           kDerive, kDerive, /*recall=*/0);
    REQUIRE(e.recall_windows == 0);
}

TEST_CASE("schedule: tiny corpus floors eval cadence (never too frequent)", "[schedule]") {
    // 2000 tokens / 64 = 31 epoch windows; 50% = ~15 windows / 4 threads ≈ 4 steps,
    // but min_eval_steps=100 floors it ⇒ coverage well above 50%.
    auto s = make_schedule(2000, 500, kT, /*threads=*/4, kMaskPerWin);
    REQUIRE(s.eval_every >= 100);
    REQUIRE(s.coverage_per_eval >= 0.5);
}
