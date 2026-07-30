// win_loss_tests.cpp -- the PER-WINDOW loss readout (train_batch's out_win_loss; the CUDA path's is
// gated alongside the other device parity cases in cuda_tests.cpp).
//
// This is the measurement docs/TUTOR.md's mastery surface is built from, and the whole argument for it
// being affordable is that it is a READOUT of a value the loss already computes -- not a second forward
// pass. So the property under test is an identity, not an approximation:
//
//     mean over b of out_win_loss[b]  ==  the batch mean train_batch already returned
//
// If that ever drifts, the per-entry signal and the scalar the trainer reports are no longer the same
// quantity, and every velocity computed from the surface is measuring something the training loop is
// not optimizing. The three cases below are the three shapes a real training batch takes:
//   1. dense (every window full width, every position graded),
//   2. ragged (short documents via `lengths` -- each window's mean is over ITS length, so a plain mean
//      over windows is correct and a token-weighted one would NOT be; this pins which),
//   3. loss-masked (LOSS_IGNORE_INDEX interior positions -- the normalizer is the ACTIVE count).
// Case 4 pins the other half of the contract: passing nullptr changes nothing about the scalar.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"

#include <numeric>
#include <random>
#include <vector>

namespace {

// A token stream long enough to host `batch` windows of width T at stride T+1, plus the shifted target.
std::vector<int> make_stream(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> data(n);
    for (int& v : data) v = tok(rng);
    return data;
}

double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

}  // namespace

TEST_CASE("win_loss: per-window means average to the batch mean (dense)", "[engine][winloss]") {
    sub0::build_model();
    constexpr int batch = 6, T = 16;
    const std::vector<int> data = make_stream(static_cast<std::size_t>(batch) * (T + 1) + 1, 21);
    std::vector<std::size_t> starts(batch);
    for (int b = 0; b < batch; ++b) starts[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b) * (T + 1);

    std::vector<double> win(batch, -1.0);
    const float mean = sub0::train_batch(data.data(), starts.data(), batch, T,
                                         nullptr, nullptr, nullptr, nullptr, nullptr, win.data());

    for (double w : win) REQUIRE(w > 0.0);           // every window was actually written
    REQUIRE(mean_of(win) == Catch::Approx(static_cast<double>(mean)).epsilon(1e-6));
}

TEST_CASE("win_loss: ragged windows average UNWEIGHTED, matching the returned mean", "[engine][winloss]") {
    sub0::build_model();
    constexpr int batch = 5, T = 16;
    const std::vector<int> data = make_stream(static_cast<std::size_t>(batch) * (T + 1) + 1, 22);
    std::vector<std::size_t> starts(batch);
    std::vector<int>         lens(batch);
    for (int b = 0; b < batch; ++b) {
        starts[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b) * (T + 1);
        lens[static_cast<std::size_t>(b)]   = T - 2 * b;      // 16, 14, 12, 10, 8 -- deliberately unequal
    }
    REQUIRE(lens.back() >= 2);

    std::vector<double> win(batch, -1.0);
    const float mean = sub0::train_batch(data.data(), starts.data(), batch, T,
                                         lens.data(), nullptr, nullptr, nullptr, nullptr, win.data());

    // The unweighted mean is the one that matches. A token-weighted mean would NOT here (the lengths
    // differ by 2x end to end), which is exactly why this case exists -- it pins the convention the
    // mastery surface must use when it aggregates windows into a per-document figure.
    REQUIRE(mean_of(win) == Catch::Approx(static_cast<double>(mean)).epsilon(1e-6));
}

TEST_CASE("win_loss: masked positions normalize over the ACTIVE count", "[engine][winloss][mask]") {
    sub0::build_model();
    constexpr int batch = 4, T = 16;
    const std::size_t n = static_cast<std::size_t>(batch) * (T + 1) + 1;
    const std::vector<int> data = make_stream(n, 23);
    std::vector<std::size_t> starts(batch);
    for (int b = 0; b < batch; ++b) starts[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b) * (T + 1);

    // Mask every third position. Window 0 additionally keeps all of its positions, so the batch mixes
    // masked and unmasked windows -- the case where a wrong denominator would still look plausible.
    std::vector<std::uint8_t> mask(n, 1);
    for (std::size_t p = static_cast<std::size_t>(T + 1); p < n; p += 3) mask[p] = 0;

    std::vector<double> win(batch, -1.0);
    const float mean = sub0::train_batch(data.data(), starts.data(), batch, T,
                                         nullptr, mask.data(), nullptr, nullptr, nullptr, win.data());

    for (double w : win) REQUIRE(w > 0.0);
    REQUIRE(mean_of(win) == Catch::Approx(static_cast<double>(mean)).epsilon(1e-6));
}

TEST_CASE("win_loss: the readout does not perturb the loss it reads", "[engine][winloss]") {
    sub0::build_model();
    constexpr int batch = 4, T = 16;
    const std::vector<int> data = make_stream(static_cast<std::size_t>(batch) * (T + 1) + 1, 24);
    std::vector<std::size_t> starts(batch);
    for (int b = 0; b < batch; ++b) starts[static_cast<std::size_t>(b)] = static_cast<std::size_t>(b) * (T + 1);

    // Same batch twice from the same parameters, once with the readout and once without. The scalar must
    // be BIT-identical: this is the claim that makes the surface free to collect on the training path,
    // and the claim that lets a Tutor arm be compared against a run that never collected it at all.
    const float without = sub0::train_batch(data.data(), starts.data(), batch, T);
    std::vector<double> win(batch, -1.0);
    const float with_   = sub0::train_batch(data.data(), starts.data(), batch, T,
                                            nullptr, nullptr, nullptr, nullptr, nullptr, win.data());
    REQUIRE(without == with_);
}
