// tune_tests.cpp -- unit tests for the sub0::tune search core (include/sub0/tune.hpp).
//
// The search is pure and engine-independent, so these tests drive it with synthetic
// objectives and assert the optimum is found. They cover the properties that matter
// for the real tuner: it finds a clear maximum, it escapes a local optimum to reach a
// taller alternative basin, it handles multi-knob interaction, and it is deterministic.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <map>

#include "sub0/tune.hpp"
#include "sub0/memplan.hpp"

using sub0::tune::Knob;
using sub0::tune::Space;
using sub0::tune::Assignment;
using sub0::tune::maximize;

TEST_CASE("unimodal 1-D search finds the single peak", "[tune]") {
    // Discrete parabola peaking at value 7 over the ladder 0..20.
    Space space = {{"x", sub0::tune::linear_steps(0, 20, 21)}};
    auto obj = [](const Assignment& a) { return -std::pow(a[0] - 7.0, 2.0); };

    auto r = maximize(space, obj);

    REQUIRE(r.best.size() == 1);
    CHECK(r.best[0] == 7.0);
    CHECK(r.best_score == Catch::Approx(0.0));
}

TEST_CASE("bimodal search escapes a local peak to the taller basin", "[tune]") {
    // Two basins along one axis: a SHORTER local peak near the midpoint (index 6,
    // where the search starts) and a TALLER global peak out near the edge (index 18).
    // A naive local search seeded at the middle would settle on the short peak; the
    // coarse + top-K basin refinement must reach the taller one.
    Space space = {{"x", sub0::tune::linear_steps(0, 20, 21)}};
    auto obj = [](const Assignment& a) {
        const double x = a[0];
        const double local  = 1.0 * std::exp(-std::pow(x - 6.0,  2.0) / 4.0);   // height 1.0
        const double global = 2.0 * std::exp(-std::pow(x - 18.0, 2.0) / 4.0);   // height 2.0
        return local + global;
    };

    auto r = maximize(space, obj);

    CHECK(r.best[0] == 18.0);
}

TEST_CASE("2-knob interaction converges to the joint optimum", "[tune]") {
    // Peak at (threads=5, wpt=8) -- the maximum requires the right value on BOTH axes.
    Space space = {
        {"threads", sub0::tune::linear_steps(1, 12, 12)},
        {"wpt",     {1, 2, 4, 8, 16, 32}},
    };
    auto obj = [](const Assignment& a) {
        return -std::pow(a[0] - 5.0, 2.0) - std::pow(a[1] - 8.0, 2.0);
    };

    auto r = maximize(space, obj);

    REQUIRE(r.best.size() == 2);
    CHECK(r.best[0] == 5.0);
    CHECK(r.best[1] == 8.0);
}

TEST_CASE("bimodal 2-knob landscape with cross-interference reaches the global basin", "[tune]") {
    // The hard case the real tuner must survive: TWO separate optima on a 2-D grid,
    // each requiring a coordinated (x, y) pair, plus cross-interference so the axes
    // are NOT separable -- the best y depends on which x-basin you are in. A decoy
    // basin near the centre (where the search seeds) is tall enough to trap a greedy
    // coordinate descent; the true global sits in the opposite corner.
    //
    //   decoy peak  : (x=8,  y=8)  height 1.6   <- near the midpoint start
    //   global peak : (x=26, y=24) height 2.0   <- far corner, different basin
    // The cross term -0.02*(x-y)^2 ties the axes together so fixing one axis at the
    // decoy's value misleads the other -- only escaping the whole basin recovers it.
    Space space = {
        {"x", sub0::tune::linear_steps(0, 30, 31)},
        {"y", sub0::tune::linear_steps(0, 30, 31)},
    };
    auto obj = [](const Assignment& a) {
        const double x = a[0], y = a[1];
        const double decoy  = 1.6 * std::exp(-(std::pow(x -  8.0, 2.0) + std::pow(y -  8.0, 2.0)) / 8.0);
        const double global = 2.0 * std::exp(-(std::pow(x - 26.0, 2.0) + std::pow(y - 24.0, 2.0)) / 8.0);
        const double coupling = -0.02 * std::pow(x - y, 2.0);   // non-separable cross-interference
        return decoy + global + coupling;
    };

    auto r = maximize(space, obj);

    REQUIRE(r.best.size() == 2);
    CHECK(r.best[0] == 26.0);
    CHECK(r.best[1] == 24.0);
}

TEST_CASE("confirmation phase rejects a noisy one-off spike", "[tune]") {
    // Simulates thermal/scheduler noise: a sub-optimal config reads abnormally HIGH on
    // its first measurement (a lucky spike), while the true optimum reads steadily. A
    // single-sample search would be fooled by the spike; the confirmation phase re-
    // measures the finalists and ranks by the median, so the steady true optimum wins.
    //
    //   index 3 ("decoy") : first read 200, every later read 100  -> median ~100
    //   index 7 ("true")  : always 150                            -> median  150
    // Counters make the noise deterministic (so the test is reproducible) while still
    // exercising the "re-measure to confirm" path.
    Space space = {{"x", sub0::tune::linear_steps(0, 10, 11)}};
    std::map<int, int> reads;                       // per-index measurement count
    auto obj = [&](const Assignment& a) -> double {
        const int x = static_cast<int>(std::lround(a[0]));
        const int n = reads[x]++;
        if (x == 3) return n == 0 ? 200.0 : 100.0;  // one-off high spike, then its true level
        if (x == 7) return 150.0;                   // the steady true optimum
        return 50.0 + x;                            // bland background
    };

    sub0::tune::Options opt;                         // default confirm_rounds > 0
    auto r = maximize(space, obj, opt);

    // The spike must NOT win; the steady higher-median config must.
    CHECK(r.best[0] == 7.0);
    CHECK(r.best_score == Catch::Approx(150.0));
    CHECK(r.best_samples > 1);                       // winner was actually re-measured
}

TEST_CASE("confirmation stops re-measuring finalists that are clear losers", "[tune]") {
    // A clear loser (well below the leader by more than the noise floor) should be
    // dropped from confirmation after the first round, so it is not re-measured the
    // full confirm_rounds times -- the budget goes to genuine contenders instead.
    Space space = {{"x", sub0::tune::linear_steps(0, 6, 7)}};
    std::map<int, int> reads;
    auto obj = [&](const Assignment& a) -> double {
        const int x = static_cast<int>(std::lround(a[0]));
        ++reads[x];
        return x == 5 ? 1000.0 : 100.0;   // x=5 is the clear, steady leader; the rest far below
    };

    sub0::tune::Options opt;
    opt.confirm_rounds = 6;
    auto r = maximize(space, obj, opt);

    CHECK(r.best[0] == 5.0);
    // The leader keeps being measured across rounds; a clear loser is pruned early and
    // so accrues far fewer measurements than the leader.
    CHECK(reads[5] > reads[0]);
    CHECK(reads[0] <= 3);                  // loser dropped well before the full 6 rounds
}

TEST_CASE("settle hook fires during refinement and confirmation", "[tune]") {
    // The cool-down callback must be invoked (refinement passes + confirmation rounds)
    // so callers can rely on it to pace measurements; a no-op default must also be safe.
    Space space = {{"x", sub0::tune::linear_steps(0, 12, 13)}};
    auto obj = [](const Assignment& a) { return -std::pow(a[0] - 5.0, 2.0); };

    int settles = 0;
    sub0::tune::Options opt;
    opt.settle = [&] { ++settles; };
    auto r = maximize(space, obj, opt);

    CHECK(settles > 0);
    CHECK(r.best[0] == 5.0);
}

TEST_CASE("on_phase hook reports Explore, Refine and Confirm", "[tune]") {
    // The phase hook lets the caller raise measurement effort as the search narrows
    // (e.g. longer throttled timing runs). All three phases must be signalled, in order.
    Space space = {{"x", sub0::tune::linear_steps(0, 12, 13)}};
    auto obj = [](const Assignment& a) { return -std::pow(a[0] - 5.0, 2.0); };

    std::vector<sub0::tune::Phase> phases;
    sub0::tune::Options opt;
    opt.on_phase = [&](sub0::tune::Phase p) { phases.push_back(p); };
    auto r = maximize(space, obj, opt);

    REQUIRE(phases.size() == 3);
    CHECK(phases[0] == sub0::tune::Phase::Explore);
    CHECK(phases[1] == sub0::tune::Phase::Refine);
    CHECK(phases[2] == sub0::tune::Phase::Confirm);
    CHECK(r.best[0] == 5.0);
}

TEST_CASE("search is deterministic across runs", "[tune]") {
    Space space = {
        {"a", sub0::tune::linear_steps(0, 30, 31)},
        {"b", sub0::tune::linear_steps(0, 30, 31)},
    };
    auto obj = [](const Assignment& a) {
        return std::sin(a[0] * 0.3) + std::cos(a[1] * 0.2) - 0.01 * a[0];
    };

    auto r1 = maximize(space, obj);
    auto r2 = maximize(space, obj);

    CHECK(r1.best_index == r2.best_index);
    CHECK(r1.best_score == r2.best_score);
    CHECK(r1.evaluations == r2.evaluations);
}

TEST_CASE("a single-candidate knob is handled", "[tune]") {
    Space space = {
        {"fixed", {4.0}},
        {"x",     sub0::tune::linear_steps(0, 10, 11)},
    };
    auto obj = [](const Assignment& a) { return -std::pow(a[1] - 3.0, 2.0) + a[0]; };

    auto r = maximize(space, obj);

    CHECK(r.best[0] == 4.0);
    CHECK(r.best[1] == 3.0);
}

// memplan::max_batch_for_vram backs the tuner's runtime batch ceiling: it must return the LARGEST
// batch that still fits the VRAM budget (and that batch must actually fit, the next one must not),
// stay monotonic, honour the hard cap, and degrade sensibly when VRAM is unknown or tiny.
TEST_CASE("max_batch_for_vram returns the largest fitting batch", "[tune][memplan]") {
    using namespace sub0::memplan;
    constexpr Dims d{ 448, 11, 7, 1792, 256, 2048 };   // a representative model
    constexpr int cap = 4096;

    SECTION("the result fits and the next step does not") {
        const int vram = 8151, act = 2;                 // 8 GB, bf16 activations
        const int b = max_batch_for_vram(d, vram, cap, act);
        REQUIRE(b > 0);
        REQUIRE(b < cap);
        CHECK(train_resident_mb(d, b, act) <= vram);
        CHECK(train_resident_mb(d, b + 1, act) > vram);
    }
    SECTION("more VRAM never lowers the ceiling (monotonic)") {
        CHECK(max_batch_for_vram(d, 16000, cap, 2) >= max_batch_for_vram(d, 8151, cap, 2));
        CHECK(max_batch_for_vram(d, 8151, cap, 2)  >= max_batch_for_vram(d, 4000, cap, 2));
    }
    SECTION("bf16 fits a bigger batch than f32") {
        CHECK(max_batch_for_vram(d, 8151, cap, 2) >= max_batch_for_vram(d, 8151, cap, 4));
    }
    SECTION("unknown VRAM defers to the hard cap; a tiny budget yields none") {
        CHECK(max_batch_for_vram(d, 0, cap, 2) == cap);
        CHECK(max_batch_for_vram(d, 1, cap, 2) == 0);
    }
}

