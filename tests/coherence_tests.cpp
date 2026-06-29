// coherence_tests.cpp -- unit tests for the pure autotemp math (sub0::coherence).
//
// These exercise the engine-free helpers behind `sub0llm autotemp`: the n-gram repeat
// metric and the monotone-crossing interpolation. Keeping them pure (coherence.hpp) is
// what lets us pin their edge cases here without running a model.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/coherence.hpp"

#include <vector>

using sub0::coherence::ngram_repeat;
using sub0::coherence::interp_cross;
using Catch::Approx;

TEST_CASE("ngram_repeat: no repeats in a strictly distinct stream", "[coherence]") {
    const std::vector<int> ids = {1, 2, 3, 4, 5, 6, 7, 8};   // every 4-gram unique
    REQUIRE(ngram_repeat(ids.data(), (int)ids.size(), 4) == Approx(0.0));
}

TEST_CASE("ngram_repeat: a tight loop repeats almost every 4-gram", "[coherence]") {
    // "a b a b a b ..." -> 4-grams (abab / babа) recur after the first two.
    std::vector<int> ids;
    for (int i = 0; i < 20; ++i) ids.push_back(i % 2);
    const double r = ngram_repeat(ids.data(), (int)ids.size(), 4);
    REQUIRE(r > 0.8);   // 15/17 four-grams are repeats
}

TEST_CASE("ngram_repeat: streams shorter than k have no k-gram", "[coherence]") {
    const std::vector<int> ids = {1, 2, 3};
    REQUIRE(ngram_repeat(ids.data(), (int)ids.size(), 4) == Approx(0.0));
    REQUIRE(ngram_repeat(ids.data(), 0, 4) == Approx(0.0));
}

TEST_CASE("ngram_repeat: a single repeated 4-gram is counted once", "[coherence]") {
    // 1 2 3 4 5 1 2 3 4 -> the second "1 2 3 4" is the only repeat among 6 four-grams.
    const std::vector<int> ids = {1, 2, 3, 4, 5, 1, 2, 3, 4};
    REQUIRE(ngram_repeat(ids.data(), (int)ids.size(), 4) == Approx(1.0 / 6.0));
}

TEST_CASE("interp_cross: increasing series interpolates linearly", "[coherence]") {
    const std::vector<float>  t = {0.2f, 0.4f, 0.6f, 0.8f};
    const std::vector<double> y = {1.0, 2.0, 3.0, 4.0};      // y rises with t
    const auto c = interp_cross(t, y, 2.5, /*increasing=*/true);
    REQUIRE(c.hit);
    REQUIRE(c.temp == Approx(0.5f));   // halfway between t=0.4 (2.0) and t=0.6 (3.0)
}

TEST_CASE("interp_cross: decreasing series interpolates linearly", "[coherence]") {
    const std::vector<float>  t = {0.2f, 0.4f, 0.6f, 0.8f};
    const std::vector<double> y = {10.0, 8.0, 6.0, 4.0};     // y falls with t
    const auto c = interp_cross(t, y, 7.0, /*increasing=*/false);
    REQUIRE(c.hit);
    REQUIRE(c.temp == Approx(0.5f));   // halfway between t=0.4 (8.0) and t=0.6 (6.0)
}

TEST_CASE("interp_cross: target above an increasing range clamps to the top", "[coherence]") {
    const std::vector<float>  t = {0.2f, 0.4f, 0.6f};
    const std::vector<double> y = {1.0, 2.0, 3.0};
    const auto c = interp_cross(t, y, 9.0, /*increasing=*/true);
    REQUIRE_FALSE(c.hit);
    REQUIRE(c.temp == Approx(0.6f));   // never reaches 9.0 -> clamp to hottest grid point
}

TEST_CASE("interp_cross: target below an increasing range clamps to the bottom", "[coherence]") {
    const std::vector<float>  t = {0.2f, 0.4f, 0.6f};
    const std::vector<double> y = {1.0, 2.0, 3.0};
    const auto c = interp_cross(t, y, 0.5, /*increasing=*/true);
    REQUIRE_FALSE(c.hit);
    REQUIRE(c.temp == Approx(0.2f));   // already exceeds 0.5 at the coolest grid point
}

TEST_CASE("interp_cross: decreasing range clamps on the correct side", "[coherence]") {
    const std::vector<float>  t = {0.2f, 0.4f, 0.6f};
    const std::vector<double> y = {10.0, 8.0, 6.0};
    // Target hotter (higher repeat) than even the coolest point -> clamp to coolest temp.
    const auto hi = interp_cross(t, y, 12.0, /*increasing=*/false);
    REQUIRE_FALSE(hi.hit);
    REQUIRE(hi.temp == Approx(0.2f));
    // Target below the hottest point's value -> clamp to hottest temp.
    const auto lo = interp_cross(t, y, 3.0, /*increasing=*/false);
    REQUIRE_FALSE(lo.hit);
    REQUIRE(lo.temp == Approx(0.6f));
}

TEST_CASE("interp_cross: degenerate inputs are handled, not crashed", "[coherence]") {
    REQUIRE(interp_cross({}, {}, 1.0, true).hit == false);
    // Mismatched lengths -> empty Crossing (hit=false, temp 0).
    const auto c = interp_cross({0.2f, 0.4f}, {1.0}, 0.5, true);
    REQUIRE_FALSE(c.hit);
}

// --- trend_plateaued: validated against REAL observed val-NELBO series from this build ----------
// The data below are the actual eval sequences logged while training tinystories models on the GPU.
// They pin the exact failure the old sign test had (false-stopping a noisy-but-descending run) and
// confirm the best-fit gradient test gets both cases right.
using sub0::coherence::trend_plateaued;

TEST_CASE("trend_plateaued: needs a full window before it can fire", "[coherence]") {
    const std::vector<double> few = {3.4, 3.3, 3.2};       // < window+1 points
    REQUIRE_FALSE(trend_plateaued(few, 6, 0.005));
}

TEST_CASE("trend_plateaued: a strongly-but-noisily descending run does NOT stop", "[coherence]") {
    // REAL d128 L5 H2 val NELBO, evals ending at the step where the OLD sign test false-plateaued
    // (step 72): the curve fell 3.381 -> 3.174 over the window (~6%) despite two upticks. The
    // best-fit gradient must keep it training.
    const std::vector<double> d128 = {3.381, 3.444, 3.305, 3.311, 3.233, 3.165, 3.174};
    REQUIRE_FALSE(trend_plateaued(d128, 6, 0.005));
}

TEST_CASE("trend_plateaued: a still-slowly-improving tail does NOT stop at the tight floor", "[coherence]") {
    // REAL d96 L5 H4 tail near where it was STOPPED (steps ~340-364, val ~1.985). Best-fit shows it
    // was still shedding ~1.7% per window -- NOT a true floor. The old 2% threshold called this a
    // plateau and quit early; the tightened 0.5% floor correctly keeps training toward lower NELBO.
    const std::vector<double> d96_tail = {2.0097, 2.0160, 1.9938, 1.9972, 1.9880, 1.9803, 1.9850};
    REQUIRE_FALSE(trend_plateaued(d96_tail, 6, 0.005));
}

TEST_CASE("trend_plateaued: an early fast-descending run does NOT stop", "[coherence]") {
    // REAL d128 warmup-to-early descent (steps ~20-44): a clean steep drop, obviously not a plateau.
    const std::vector<double> early = {3.9700, 3.9495, 3.8576, 3.7367, 3.5716, 3.4508, 3.3882};
    REQUIRE_FALSE(trend_plateaued(early, 6, 0.005));
}

TEST_CASE("trend_plateaued: a genuine floor stops; a real slope does not", "[coherence]") {
    const std::vector<double> flat = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    REQUIRE(trend_plateaued(flat, 6, 0.005));
    // Barely-improving tail at a true floor: ~0.05%/window net -> below 0.5% -> stop.
    const std::vector<double> floor = {1.9010, 1.9008, 1.9006, 1.9007, 1.9005, 1.9004, 1.9005};
    REQUIRE(trend_plateaued(floor, 6, 0.005));
    // Steady -0.2%/eval => -1.2% over a 6-window, past the 0.5% floor: keep training.
    std::vector<double> slope;
    for (int i = 0; i < 7; ++i) slope.push_back(2.0 * (1.0 - 0.002 * i));
    REQUIRE_FALSE(trend_plateaued(slope, 6, 0.005));
}

