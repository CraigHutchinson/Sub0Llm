// loss_mask_tests.cpp -- the ignore-index cross-entropy primitive (LOSS_IGNORE_INDEX; core.hpp +
// backend_cpu.cpp). Loss-masking is the foundation for the combine/uncombine spike's injected spans
// and future instruction-tuning's assistant-span masking, so it gets its own correctness gate:
//   1. masked forward loss == the hand-computed mean over ACTIVE positions (masking + normalization),
//   2. an all-masked window is a well-defined no-op (0 loss, no crash),
//   3. the masked forward and masked backward are mutually consistent (finite-difference directional
//      check -- the same gold-standard check engine_tests uses, but with masked targets).

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "sub0/core.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace {

// Mean over active (target >= 0) positions of -log softmax(logits[t])[target[t]], from public logits.
double manual_active_ce(const sub0::Node* logits, const std::vector<int>& tgt) {
    const int T = logits->rows, V = logits->cols;
    double total = 0.0; int active = 0;
    for (int t = 0; t < T; ++t) {
        if (tgt[static_cast<std::size_t>(t)] < 0) continue;
        const float* lr = logits->data.data() + static_cast<std::size_t>(t) * V;
        float mx = -1e30f;
        for (int j = 0; j < V; ++j) mx = std::max(mx, lr[j]);
        double Z = 0.0;
        for (int j = 0; j < V; ++j) Z += std::exp(static_cast<double>(lr[j] - mx));
        const double p = std::exp(static_cast<double>(lr[tgt[static_cast<std::size_t>(t)]] - mx)) / Z;
        total += -std::log(std::max(1e-9, p));
        ++active;
    }
    return active ? total / active : 0.0;
}

double masked_loss(const std::vector<int>& ids, const std::vector<int>& tgt) {
    sub0::graph_reset();
    sub0::Node* lg = sub0::forward(ids.data(), static_cast<int>(ids.size()));
    return sub0::cross_entropy(lg, tgt.data())->data[0];
}

}  // namespace

TEST_CASE("loss mask: masked forward loss is the mean over active positions", "[engine][mask]") {
    sub0::build_model();
    constexpr int T = 8;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> ids(T), tgt(T);
    for (int i = 0; i < T; ++i) { ids[i] = tok(rng); tgt[i] = tok(rng); }
    // Mask a few interior positions.
    tgt[1] = sub0::LOSS_IGNORE_INDEX;
    tgt[4] = sub0::LOSS_IGNORE_INDEX;
    tgt[5] = sub0::LOSS_IGNORE_INDEX;

    sub0::graph_reset();
    sub0::Node* logits = sub0::forward(ids.data(), T);
    const double engine = sub0::cross_entropy(logits, tgt.data())->data[0];
    const double manual = manual_active_ce(logits, tgt);
    REQUIRE(engine == Catch::Approx(manual).epsilon(1e-4));

    // Sanity: it differs from the UNMASKED loss (masking actually changed the normalization/sum).
    std::vector<int> full(tgt);
    full[1] = tok(rng); full[4] = tok(rng); full[5] = tok(rng);
    sub0::graph_reset();
    sub0::Node* lg2 = sub0::forward(ids.data(), T);
    const double unmasked = sub0::cross_entropy(lg2, full.data())->data[0];
    REQUIRE(engine != Catch::Approx(unmasked));
}

TEST_CASE("loss mask: an all-masked window is a 0-loss no-op", "[engine][mask]") {
    sub0::build_model();
    constexpr int T = 6;
    std::vector<int> ids(T, 3), tgt(T, sub0::LOSS_IGNORE_INDEX);
    sub0::graph_reset();
    sub0::Node* lg = sub0::forward(ids.data(), T);
    sub0::Node* loss = sub0::cross_entropy(lg, tgt.data());
    REQUIRE(loss->data[0] == Catch::Approx(0.0));
    sub0::backward(loss, 1.f);          // must not divide by zero / crash
    sub0::reduce_gradients();
    const std::size_t n = sub0::trainable_floats();
    for (std::size_t i = 0; i < n; ++i) REQUIRE(std::isfinite(sub0::grad_ptr()[i]));
}

TEST_CASE("loss mask: masked backward matches finite differences", "[engine][mask][grad]") {
    sub0::build_model();
    constexpr int T = 8;
    std::mt19937 rng(11);
    std::uniform_int_distribution<int> tok(0, VOCAB - 1);
    std::vector<int> ids(T), tgt(T);
    for (int i = 0; i < T; ++i) { ids[i] = tok(rng); tgt[i] = tok(rng); }
    tgt[2] = sub0::LOSS_IGNORE_INDEX;    // interior masked positions
    tgt[6] = sub0::LOSS_IGNORE_INDEX;

    // Move off the flat init so a real gradient exists (mirrors engine_tests' fd check).
    sub0::AdamW opt(0.01f);
    for (int s = 0; s < 5; ++s) {
        opt.zero_grad();
        sub0::graph_reset();
        sub0::backward(sub0::cross_entropy(sub0::forward(ids.data(), T), tgt.data()), 1.f);
        sub0::reduce_gradients();
        opt.step();
    }

    opt.zero_grad();
    sub0::graph_reset();
    sub0::backward(sub0::cross_entropy(sub0::forward(ids.data(), T), tgt.data()), 1.f);
    sub0::reduce_gradients();
    const std::size_t n = sub0::trainable_floats();
    std::vector<float> g(sub0::grad_ptr(), sub0::grad_ptr() + n);
    std::vector<float> base(sub0::params_ptr(), sub0::params_ptr() + n);

    double gnorm2 = 0.0; for (float v : g) gnorm2 += static_cast<double>(v) * v;
    const double gnorm = std::sqrt(gnorm2);
    REQUIRE(gnorm > 1e-2);               // a real masked gradient to test against

    const float eps = 1e-2f;
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = base[i] + eps * static_cast<float>(g[i] / gnorm);
    const double lp = masked_loss(ids, tgt);
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = base[i] - eps * static_cast<float>(g[i] / gnorm);
    const double lm = masked_loss(ids, tgt);
    for (std::size_t i = 0; i < n; ++i) sub0::params_ptr()[i] = base[i];
    const double directional = (lp - lm) / (2.0 * eps);
    INFO("masked directional numeric=" << directional << " ||g||=" << gnorm);
    REQUIRE(directional == Catch::Approx(gnorm).epsilon(0.05));
}
