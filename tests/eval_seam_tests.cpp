// eval_seam_tests.cpp -- end-to-end integration tests for the held-out NELBO evaluation, driven
// through a MOCK device backend (tests/mock_device_backend.cpp) so the whole device dispatch runs on
// any machine, GPU or not.
//
// The property under test is the one that matters and the one nothing previously checked: routing an
// evaluation through the device seam must produce the SAME NUMBER as running it on the CPU. That is
// what makes it safe to compare a device-scored model against a CPU-scored one -- which every A/B in
// this project now does. A mean NELBO is exactly the kind of averaged scalar where a systematic
// difference hides: a target shifted by one token, or a ragged final batch weighted wrongly, both
// yield a perfectly plausible-looking number that is simply not the number it claims to be.
//
// The mock computes its answer with the real CPU engine over the ids/targets it is handed, so it
// agrees with the CPU route if and only if the plumbing between them is right. It says nothing about
// whether the CUDA kernels are right -- that is cuda_tests.cpp's "forward_loss matches the CPU
// evaluate" case, which needs real hardware. Neither test subsumes the other.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"
#include "sub0/eval.hpp"
#include "sub0/tokmap.hpp"

#include <cmath>
#include <random>
#include <vector>

#if !defined(SUB0_BUILD_MOCK_DEVICE)
#error "eval_seam_tests.cpp must be built against the mock device backend"
#endif

extern "C" int  sub0_mock_call_count();
extern "C" int  sub0_mock_batch_at(int i);
extern "C" void sub0_mock_reset_log();

namespace {

// A deterministic pseudo-corpus of in-range ids, long enough that plan() produces many windows.
std::vector<int> make_corpus(std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> v(n);
    for (auto& t : v) t = tok(rng);
    return v;
}

constexpr int kThreads = 2;   // fixed, so the CPU reduction order is stable across cases

}  // namespace

TEST_CASE("window planning is stable and stays in bounds", "[eval]") {
    const std::vector<int> corpus = make_corpus(static_cast<std::size_t>(SEQ_LEN) * 40 + 7, 11);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());

    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0);
    REQUIRE(ws.count > 0);
    REQUIRE(ws.count <= sub0::eval::WINDOWS_MAX);

    // Every window must have a full SEQ_LEN+1 tokens available -- the target of the last position is
    // one token past the window, so an off-by-one here reads past the end of the corpus.
    for (int w = 0; w < ws.count; ++w) {
        const std::size_t s = ws.start_of(w);
        REQUIRE(s + static_cast<std::size_t>(SEQ_LEN) + 1 <= corpus.size());
    }
    // Monotone and clamped: the set is "fixed, evenly spaced", which is what makes two evals of the
    // same model on the same data comparable rather than a resampling of different text.
    for (int w = 1; w < ws.count; ++w) REQUIRE(ws.start_of(w) >= ws.start_of(w - 1));
    REQUIRE(sub0::eval::plan(data, 0).start_of(0) == ws.start_of(0));

    // Boundary: SEQ_LEN+1 tokens is exactly one evaluable window (SEQ_LEN inputs + the final target).
    // One token less has no window at all -- and is where the size_t arithmetic would underflow if the
    // guard were missing, yielding a huge `last` and an out-of-bounds read rather than an empty plan.
    const sub0::eval::WindowSet one = sub0::eval::plan(data.first(static_cast<std::size_t>(SEQ_LEN) + 1), 0);
    REQUIRE(one.count == 1);
    REQUIRE(one.start_of(0) == 0);
    REQUIRE(sub0::eval::plan(data.first(static_cast<std::size_t>(SEQ_LEN)), 0).empty());
    REQUIRE(sub0::eval::plan(data.first(0), 0).empty());
}

TEST_CASE("device eval reproduces the CPU eval exactly", "[eval][seam]") {
    sub0::build_model();
    const std::vector<int> corpus = make_corpus(static_cast<std::size_t>(SEQ_LEN) * 40 + 7, 21);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0);
    REQUIRE(ws.count > 1);

    const sub0::eval::Session s(/*allow=*/true);
    REQUIRE(s.use_device);                     // the mock backend must actually have been selected

    // The headline property. Both routes score the SAME window set at the SAME width; they may not
    // differ by more than float reduction order.
    const double cpu = sub0::eval::nelbo_cpu(data, ws, SEQ_LEN, kThreads);
    const double dev = sub0::eval::nelbo_device(data, ws, SEQ_LEN);
    REQUIRE(std::isfinite(cpu));
    REQUIRE(dev == Catch::Approx(cpu).epsilon(1e-5));

    // ...and at a SHORT width too: the context-length curve scores several widths off one plan, and a
    // width-dependent indexing bug would show up here and not at full width.
    const double cpu64 = sub0::eval::nelbo_cpu(data, ws, 64, kThreads);
    const double dev64 = sub0::eval::nelbo_device(data, ws, 64);
    REQUIRE(dev64 == Catch::Approx(cpu64).epsilon(1e-5));
}

TEST_CASE("a shifted target pairing is detectable, not silently plausible", "[eval][seam]") {
    // Guards the guard: confirms the comparison above would actually FAIL on the off-by-one it exists
    // to catch, rather than being satisfied by any two numbers of roughly the right magnitude. Scores
    // the same windows against deliberately mispaired targets and requires a visible difference.
    sub0::build_model();
    const std::vector<int> corpus = make_corpus(static_cast<std::size_t>(SEQ_LEN) * 8, 31);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0);
    REQUIRE(ws.count > 0);

    const sub0::eval::Session s(/*allow=*/true);
    REQUIRE(s.use_device);
    const double correct = sub0::eval::nelbo_device(data, ws, SEQ_LEN);

    // Same windows, targets taken from the window's own positions instead of the next one.
    const int ctx = SEQ_LEN;
    std::vector<int> ids(static_cast<std::size_t>(ctx)), tgt(static_cast<std::size_t>(ctx));
    double shifted_total = 0.0;
    for (int w = 0; w < ws.count; ++w) {
        std::vector<int> win(static_cast<std::size_t>(ctx) + 1);
        data.copy_to(ws.start_of(w), static_cast<std::size_t>(ctx) + 1, win.data());
        for (int t = 0; t < ctx; ++t) { ids[t] = win[t]; tgt[t] = win[t]; }   // WRONG on purpose
        double one = 0.0;
        REQUIRE(sub0_dev_forward_loss(ids.data(), tgt.data(), 1, ctx, &one, nullptr) == 0);
        shifted_total += one;
    }
    const double shifted = shifted_total / ws.count;
    REQUIRE(std::abs(shifted - correct) > 1e-3);
}

TEST_CASE("ragged final group is weighted by window count", "[eval][seam]") {
    // The recombination bug this pins: with nw windows split into groups, each call returns the mean
    // over ITS group, so a plain mean-of-means over-weights a short final group. It is invisible in
    // any single number -- only a comparison against the ungrouped CPU value exposes it.
    sub0::build_model();
    const std::vector<int> corpus = make_corpus(static_cast<std::size_t>(SEQ_LEN) * 40 + 7, 41);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());

    // A window count that cannot divide evenly into the device batch, so a ragged tail is guaranteed.
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0, 17);
    REQUIRE(ws.count == 17);

    const sub0::eval::Session s(/*allow=*/true);
    REQUIRE(s.use_device);

    sub0_mock_reset_log();
    const double dev = sub0::eval::nelbo_device(data, ws, 64);
    const double cpu = sub0::eval::nelbo_cpu(data, ws, 64, kThreads);
    REQUIRE(dev == Catch::Approx(cpu).epsilon(1e-5));

    // Whatever grouping was chosen, the submitted batches must sum to exactly the planned windows --
    // never fewer (a dropped tail) and never more (a padded group silently graded as real text).
    const int calls = sub0_mock_call_count();
    REQUIRE(calls >= 1);
    int total_rows = 0;
    for (int i = 0; i < calls; ++i) {
        REQUIRE(sub0_mock_batch_at(i) > 0);
        total_rows += sub0_mock_batch_at(i);
    }
    REQUIRE(total_rows == ws.count);
}

TEST_CASE("device batch sizing respects its buffer budget and the window count", "[eval]") {
    // The batch is derived from the logits-buffer budget rather than fixed, so it must scale the right
    // way with context width and never exceed what there is to evaluate.
    const int wide   = sub0::eval::device_batch(SEQ_LEN, 128);
    const int narrow = sub0::eval::device_batch(64, 128);
    REQUIRE(wide >= 1);
    REQUIRE(narrow >= wide);                   // less context per window => more windows per call
    REQUIRE(sub0::eval::device_batch(SEQ_LEN, 3) <= 3);
    REQUIRE(sub0::eval::device_batch(SEQ_LEN, 1) == 1);

    // The chosen batch must actually fit the budget it was derived from.
    const std::size_t bytes = static_cast<std::size_t>(wide) * SEQ_LEN * VOCAB * sizeof(float);
    REQUIRE(bytes <= sub0::eval::DEVICE_LOGITS_BUDGET_BYTES);
}

TEST_CASE("a declined session falls back to the CPU rather than failing", "[eval][seam]") {
    // How report keeps a running trainer's machine to itself: allow = false must yield a correct
    // number on the CPU, not a NaN and not a refusal.
    sub0::build_model();
    const std::vector<int> corpus = make_corpus(static_cast<std::size_t>(SEQ_LEN) * 8, 51);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0);

    const sub0::eval::Session declined(/*allow=*/false);
    REQUIRE_FALSE(declined.use_device);

    sub0_mock_reset_log();
    const double v = sub0::eval::nelbo(data, ws, SEQ_LEN, declined, kThreads);
    REQUIRE(std::isfinite(v));
    REQUIRE(v == Catch::Approx(sub0::eval::nelbo_cpu(data, ws, SEQ_LEN, kThreads)).epsilon(1e-5));
    REQUIRE(sub0_mock_call_count() == 0);      // it really did not touch the device
}

TEST_CASE("an unbrought-up device is refused rather than scored wrongly", "[eval][seam]") {
    // The seam's fail-fast contract: calling forward_loss without an init/upload must return nonzero,
    // so nelbo_device yields NaN and nelbo() falls back -- rather than returning a stale-parameter
    // number that looks like a measurement.
    sub0::build_model();
    const std::vector<int> corpus = make_corpus(static_cast<std::size_t>(SEQ_LEN) * 8, 61);
    const sub0::TokView data = sub0::TokView::over_int32(corpus.data(), corpus.size());
    const sub0::eval::WindowSet ws = sub0::eval::plan(data, 0);

    sub0_dev_shutdown();                       // ensure nothing is up
    const double d = sub0::eval::nelbo_device(data, ws, SEQ_LEN);
    REQUIRE(std::isnan(d));

    // ...and the dispatching entry still returns a real number by taking the CPU route.
    const sub0::eval::Session s(/*allow=*/true);
    REQUIRE(s.use_device);
    REQUIRE(std::isfinite(sub0::eval::nelbo(data, ws, SEQ_LEN, s, kThreads)));
}
