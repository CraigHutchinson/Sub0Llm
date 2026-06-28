// engine_tests.cpp — fast, training-free regression tests for the engine core.
//
// These pin the differentiable substrate (forward / cross_entropy / backward),
// the optimizer and serialization so the hot loops can be optimized with
// confidence: a broken gradient, a lost causal mask or a serialization slip is
// caught in milliseconds instead of after a training run. The centerpiece is a
// finite-difference gradient check -- it exercises every op's backward path and is
// the direct guard for any reassociation/vectorization of op_linear & friends.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>

namespace {

// Loss of one (ids,targets) window at the current parameters. Self-contained so a
// finite-difference probe can call it repeatedly between graph resets.
float window_loss(const std::vector<int>& ids, const std::vector<int>& tgt) {
    const int T = static_cast<int>(ids.size());
    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(ids.data(), T);
    sub0::Node* loss   = sub0::cross_entropy(logits, tgt.data());
    const float v = loss->data[0];
    return v;
}

// A small deterministic window of in-range token ids.
void make_window(std::vector<int>& ids, std::vector<int>& tgt, int T, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    ids.resize(T); tgt.resize(T);
    for (int i = 0; i < T; ++i) { ids[i] = tok(rng); tgt[i] = tok(rng); }
}

}  // namespace

TEST_CASE("forward has the right shape and is deterministic", "[engine]") {
    sub0::build_model();
    std::vector<int> ids, tgt;
    make_window(ids, tgt, 12, 1);

    sub0::graph_reset();
    sub0::Node* a = sub0::forward(ids.data(), 12);
    REQUIRE(a->rows == 12);
    REQUIRE(a->cols == VOCAB);
    std::vector<float> first(a->data.begin(), a->data.end());

    sub0::graph_reset();
    sub0::Node* b = sub0::forward(ids.data(), 12);
    for (size_t i = 0; i < first.size(); ++i) REQUIRE(b->data[i] == first[i]);  // bit-exact repeat
}

TEST_CASE("untrained cross-entropy is near ln(VOCAB)", "[engine]") {
    sub0::build_model();
    std::vector<int> ids, tgt;
    make_window(ids, tgt, 16, 2);
    const float loss = window_loss(ids, tgt);
    const float uniform = std::log(static_cast<float>(VOCAB));   // ~7.62 for VOCAB=2048
    // Random init gives near-uniform logits, so loss sits close to the uniform bound.
    REQUIRE(loss > 0.5f * uniform);
    REQUIRE(loss < 1.5f * uniform);
}

TEST_CASE("variable-length training windows run across a range of T", "[engine]") {
    sub0::build_model();
    // The training loop draws a per-step window length T in [MIN_TRAIN_SEQ, SEQ_LEN]; every such
    // length must run cleanly through forward -> loss -> backward and yield a finite, non-trivial
    // loss (a fixed-SEQ_LEN assumption anywhere in the stack would crash or misindex on short T).
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    for (int T : {8, 17, 32, SEQ_LEN}) {
        std::vector<int> data(static_cast<std::size_t>(T) + 1);
        for (int& v : data) v = tok(rng);
        const std::vector<std::size_t> starts(1, std::size_t{0});
        const float loss = sub0::train_batch(data.data(), starts.data(), 1, T);
        INFO("T = " << T << "  loss = " << loss);
        REQUIRE(loss > 0.0f);          // > 0 also rejects NaN (NaN compares false)
        REQUIRE(loss < 100.0f);        // finite and near the untrained ln(VOCAB) scale
    }
}

TEST_CASE("attention is causal: future tokens cannot change past logits", "[engine]") {
    sub0::build_model();
    std::vector<int> ids, tgt;
    make_window(ids, tgt, 16, 3);

    sub0::graph_reset();
    sub0::Node* base = sub0::forward(ids.data(), 16);
    std::vector<float> row0(base->data.begin(), base->data.begin() + VOCAB);  // logits at position 0

    ids.back() = (ids.back() + 7) % VOCAB;            // perturb the LAST token only
    sub0::graph_reset();
    sub0::Node* perturbed = sub0::forward(ids.data(), 16);
    for (int j = 0; j < VOCAB; ++j) REQUIRE(perturbed->data[j] == row0[j]);  // position 0 unchanged
}

TEST_CASE("analytic gradients match finite differences", "[engine][grad]") {
    sub0::build_model();
    std::vector<int> ids, tgt;
    make_window(ids, tgt, 8, 4);

    // Take a few steps off the flat random-init point, where every gradient is
    // ~0 and a finite-difference check is vacuous (both sides near zero).
    sub0::AdamW opt(0.01f);
    for (int s = 0; s < 5; ++s) {
        opt.zero_grad();
        sub0::graph_reset();
        sub0::Node* lg = sub0::forward(ids.data(), 8);
        sub0::backward(sub0::cross_entropy(lg, tgt.data()), 1.f);
        sub0::reduce_gradients();
        opt.step();
    }

    // Analytic gradient of the whole forward->loss->backward stack at this point.
    opt.zero_grad();
    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(ids.data(), 8);
    sub0::backward(sub0::cross_entropy(logits, tgt.data()), 1.f);
    sub0::reduce_gradients();   // publish accumulator -> the gradient grad_ptr() exposes
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> g(sub0::grad_ptr(), sub0::grad_ptr() + n);
    std::vector<float> base(sub0::params_ptr(), sub0::params_ptr() + n);

    double gnorm2 = 0.0;
    for (float v : g) gnorm2 += static_cast<double>(v) * v;
    const double gnorm = std::sqrt(gnorm2);
    REQUIRE(gnorm > 1e-2);     // a real gradient exists to test against

    // Directional check: the central difference along the unit gradient direction
    // must equal ||g|| (the directional derivative along g). This aggregates every
    // parameter into one large-signal comparison that no single op can fake.
    const float eps = 1e-2f;
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = base[i] + eps * static_cast<float>(g[i] / gnorm);
    const double lp = window_loss(ids, tgt);
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = base[i] - eps * static_cast<float>(g[i] / gnorm);
    const double lm = window_loss(ids, tgt);
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = base[i];
    const double directional = (lp - lm) / (2.0 * eps);
    INFO("directional numeric=" << directional << " ||g||=" << gnorm);
    REQUIRE(directional == Catch::Approx(gnorm).epsilon(0.05));

    // Per-parameter spot checks localize a fault to a specific op. ||g|| spreads
    // thinly over ~740k params (~0.004 RMS each), so probe the few largest-magnitude
    // gradients -- these carry enough signal to clear float finite-difference noise.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::partial_sort(order.begin(), order.begin() + 6, order.end(),
                      [&](std::size_t a, std::size_t b) { return std::abs(g[a]) > std::abs(g[b]); });
    REQUIRE(std::abs(g[order[0]]) > 0.02f);   // a genuinely non-trivial gradient exists
    for (int k = 0; k < 6; ++k) {
        const std::size_t idx = order[k];
        const float orig = base[idx];
        sub0::params_ptr()[idx] = orig + eps; const double a = window_loss(ids, tgt);
        sub0::params_ptr()[idx] = orig - eps; const double b = window_loss(ids, tgt);
        sub0::params_ptr()[idx] = orig;
        const double num = (a - b) / (2.0 * eps);
        INFO("idx=" << idx << " numeric=" << num << " analytic=" << g[idx]);
        REQUIRE(std::abs(num - g[idx]) <= 0.10 * std::max<double>(std::abs(num), std::abs(g[idx])) + 5e-3);
    }
}

TEST_CASE("train_batch gradient matches sequential accumulation", "[engine][grad]") {
    sub0::build_model();
    const int T = 8, batch = 5;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> data(2000);
    for (int& v : data) v = tok(rng);
    std::uniform_int_distribution<std::size_t> pick(0, data.size() - T - 2);
    std::vector<std::size_t> starts(batch);
    for (auto& s : starts) s = pick(rng);

    const std::size_t n = sub0::trainable_floats();

    // Sequential reference: accumulate every window's 1/batch-scaled grad on one thread.
    sub0::AdamW opt(0.01f);
    opt.zero_grad();
    double seq_loss = 0.0;
    for (int b = 0; b < batch; ++b) {
        sub0::graph_reset();
        sub0::Node* lg = sub0::forward(data.data() + starts[b], T);
        sub0::Node* loss = sub0::cross_entropy(lg, data.data() + starts[b] + 1);
        seq_loss += loss->data[0];
        sub0::backward(loss, 1.f / batch);
    }
    sub0::reduce_gradients();
    std::vector<float> seq(sub0::grad_ptr(), sub0::grad_ptr() + n);   // reduced single-thread grad

    // Data-parallel: same windows, split across threads, summed in train_batch.
    const float par_loss = sub0::train_batch(data.data(), starts.data(), batch, T);
    std::vector<float> par(sub0::grad_ptr(), sub0::grad_ptr() + n);   // reduced multi-thread grad

    REQUIRE(par_loss == Catch::Approx(seq_loss / batch).epsilon(1e-4));
    double max_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) max_abs = std::max(max_abs, std::abs((double)seq[i] - par[i]));
    INFO("max abs grad diff = " << max_abs);
    REQUIRE(max_abs < 1e-4);
}

TEST_CASE("one window can be overfit: AdamW drives the loss down", "[engine]") {
    sub0::build_model();
    std::vector<int> ids, tgt;
    make_window(ids, tgt, 16, 5);

    const float l0 = window_loss(ids, tgt);
    sub0::AdamW opt(0.01f);
    for (int s = 0; s < 60; ++s) {
        opt.zero_grad();
        sub0::graph_reset();
        sub0::Node* logits = sub0::forward(ids.data(), 16);
        sub0::Node* loss   = sub0::cross_entropy(logits, tgt.data());
        sub0::backward(loss, 1.f);
        sub0::reduce_gradients();
        opt.step();
    }
    const float l1 = window_loss(ids, tgt);
    REQUIRE(l1 < l0 * 0.5f);    // a single window should overfit substantially
}

TEST_CASE("model save/load round-trips the parameters exactly", "[engine]") {
    sub0::build_model();
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> before(sub0::params_ptr(), sub0::params_ptr() + n);

    const auto path = (std::filesystem::temp_directory_path() / "sub0_test_model.bin").string();
    sub0::save_model(path.c_str());

    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = -1234.5f;  // clobber
    REQUIRE(sub0::load_model(path.c_str()));
    for (std::size_t i = 0; i < n; ++i) REQUIRE(sub0::params_ptr()[i] == before[i]);
    std::filesystem::remove(path);
}

TEST_CASE("CPU param-sync hooks are no-ops that preserve the parameters", "[engine]") {
    sub0::build_model();
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> before(sub0::params_ptr(), sub0::params_ptr() + n);
    sub0::sync_params_to_host();     // CPU: params already live in host memory
    sub0::sync_params_to_device();   // CPU: no device copy to push to
    for (std::size_t i = 0; i < n; ++i) REQUIRE(sub0::params_ptr()[i] == before[i]);
}

TEST_CASE("load_model rejects files that are not a matching model", "[engine]") {
    sub0::build_model();
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> before(sub0::params_ptr(), sub0::params_ptr() + n);

    // A short junk file: the header magic/config check must reject it and leave the
    // in-memory parameters untouched (a stale/foreign file never trains the wrong thing).
    const auto path = (std::filesystem::temp_directory_path() / "sub0_bad_model.bin").string();
    { std::ofstream os(path, std::ios::binary); const char junk[8] = {1, 2, 3, 4, 5, 6, 7, 8}; os.write(junk, sizeof junk); }
    REQUIRE_FALSE(sub0::load_model(path.c_str()));
    for (std::size_t i = 0; i < n; ++i) REQUIRE(sub0::params_ptr()[i] == before[i]);
    std::filesystem::remove(path);

    REQUIRE_FALSE(sub0::load_model("sub0_definitely_missing_model.bin"));  // absent file -> false
}

TEST_CASE("AdamW step counter survives get/set", "[engine]") {
    sub0::AdamW opt(0.001f);
    REQUIRE(opt.step_count() == 0);
    opt.set_step_count(123);
    REQUIRE(opt.step_count() == 123);
}
