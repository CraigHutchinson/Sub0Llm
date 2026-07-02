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
#include "sub0/window.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <span>
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

TEST_CASE("window sampler keeps each training window inside one document", "[window]") {
    // Documents occupy [0,100), [100,160), [160,400) of 400 trainable tokens. For any sampled
    // window, the inputs [start,start+len) and the last shifted target at start+len must stay within
    // the single document that contains `start`. A document too short for the full width T is no
    // longer skipped -- it yields a shorter window (len < T) that the caller pads; verify that path
    // is exercised (the 60-token middle document at the larger widths).
    const std::vector<std::uint64_t> docs = {0u, 100u, 160u};
    const std::size_t train_tok = 400;
    std::mt19937 rng(123);
    bool saw_short = false;
    for (int T : {8, 33, 64, 200}) {
        for (int it = 0; it < 5000; ++it) {
            const sub0::Window w = sub0::sample_window(rng, T, train_tok,
                                                       std::span<const std::uint64_t>(docs));
            REQUIRE(w.len >= 1);
            REQUIRE(w.len <= T);
            REQUIRE(w.start + static_cast<std::size_t>(w.len) < train_tok);   // last target in range
            std::size_t ds = 0, de = train_tok;                              // document containing start
            for (std::size_t k = 0; k < docs.size(); ++k)
                if (docs[k] <= w.start) { ds = docs[k]; de = (k + 1 < docs.size()) ? docs[k + 1] : train_tok; }
            REQUIRE(w.start >= ds);
            REQUIRE(w.start + static_cast<std::size_t>(w.len) < de);          // window + target in-doc
            if (w.len < T) saw_short = true;
        }
    }
    REQUIRE(saw_short);   // short documents are trained (as padded windows), not dropped
}

TEST_CASE("window sampler with no document index gives a valid full-width window", "[window]") {
    std::mt19937 rng(7);
    const std::size_t train_tok = 200;
    const int T = 16;
    for (int it = 0; it < 2000; ++it) {
        const sub0::Window w = sub0::sample_window(rng, T, train_tok, {});
        REQUIRE(w.len == T);
        REQUIRE(w.start + static_cast<std::size_t>(T) < train_tok);
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

// The KV-cache incremental decode (forward_one) must reproduce the full forward's per-position
// logits: for a sequence, forward_one(id_t, t) after prefilling 0..t-1 attends the same causal
// context as forward()'s row t, so their logits agree to fast-math reduction tolerance. This is the
// correctness gate for the O(T)-per-token gen fast path. (Dense builds only; ternary skips the path.)
TEST_CASE("forward_one (KV-cache) matches the full forward per position", "[engine]") {
    if constexpr (!USE_TERNARY) {
        sub0::build_model();
        const int L = std::min(24, SEQ_LEN);
        std::vector<int> ids(L);
        std::mt19937 rng(99);
        for (int& x : ids) x = static_cast<int>(rng() % VOCAB);

        sub0::graph_reset();
        sub0::Node* full = sub0::forward(ids.data(), L);
        const std::vector<float> ref(full->data.begin(), full->data.begin() + static_cast<size_t>(L) * VOCAB);

        sub0::kv_reset();
        double worst = 0.0;
        for (int pos = 0; pos < L; ++pos) {
            const float* one = sub0::forward_one(ids[pos], pos);
            const float* rr  = ref.data() + static_cast<size_t>(pos) * VOCAB;
            double maxabs = 0.0, maxmag = 1e-30;
            for (int j = 0; j < VOCAB; ++j) {
                maxabs = std::max(maxabs, std::fabs(static_cast<double>(one[j]) - rr[j]));
                maxmag = std::max(maxmag, std::fabs(static_cast<double>(rr[j])));
            }
            worst = std::max(worst, maxabs / maxmag);
        }
        INFO("worst per-position rel diff = " << worst);
        REQUIRE(worst < 1e-3);
    }
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

// gen's stopping condition (gen_stage.cpp: `if (next == eos_id) break;`, identical across all three
// decode loops -- GPU KV-cache, CPU KV-cache, full-forward) is a one-line check once `next` can
// equal eos_id; the mechanism worth actually testing is that half, at the token-id level -- not by
// inferring a stop from shorter-than-expected generated text, which an undertrained model can produce
// for unrelated reasons (see the memory notes on this: length is not evidence). This uses the exact
// two calls gen_stage.cpp makes -- sub0::eos_token_id() and sub0::sample_token() -- so it is a real
// mechanism test of the primitives gen relies on, not a re-implementation of them.
TEST_CASE("gen's EOS stop condition: sample_token returns eos_id when the model favors it", "[engine][gen]") {
    REQUIRE(sub0::load_tokenizer(sub0::default_tokenizer()));
    const int eos_id = sub0::eos_token_id();
    REQUIRE(eos_id >= 0);              // the build's tokenizer carries the fixed EOS marker
    REQUIRE(eos_id < VOCAB);

    std::vector<float> logits(VOCAB, -10.f);
    logits[static_cast<std::size_t>(eos_id)] = 10.f;   // overwhelmingly the argmax
    std::mt19937 rng(1);
    for (int trial = 0; trial < 20; ++trial) {
        const int next = sub0::sample_token(logits.data(), /*temp=*/0.8f, /*topk=*/1, rng);
        REQUIRE(next == eos_id);       // top-1 sampling must return it -- gen's `next == eos_id` fires
    }
}
