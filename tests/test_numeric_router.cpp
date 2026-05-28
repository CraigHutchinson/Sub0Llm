#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/autograd/ops.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static Variable make_input(int64_t T, int64_t D) {
    Tensor t({T, D}, DType::Float32);
    auto sp = t.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i)
        sp[i] = static_cast<float>(i % 11) * 0.1f - 0.5f;
    return Variable(t, /*requires_grad=*/true);
}

// ── RouteType values ──────────────────────────────────────────────────────────

TEST_CASE("RouteType values") {
    REQUIRE(static_cast<int>(RouteType::FFN)     == 0);
    REQUIRE(static_cast<int>(RouteType::Add)     == 1);
    REQUIRE(static_cast<int>(RouteType::Sub)     == 2);
    REQUIRE(static_cast<int>(RouteType::Mul)     == 3);
    REQUIRE(static_cast<int>(RouteType::Div)     == 4);
    REQUIRE(static_cast<int>(RouteType::IsLessThan) == 5);
    REQUIRE(kNumRouteTypes == 6);
}

// ── NumericRouter: output shapes ─────────────────────────────────────────────

TEST_CASE("NumericRouter: output shapes") {
    const int64_t D = 16, T = 8;
    NumericRouter router(D, /*seed=*/42);

    auto x = make_input(T, D);
    auto [soft_probs, hard_mask] = router.forward(x);

    REQUIRE(soft_probs.data().shape(0) == T);
    REQUIRE(soft_probs.data().shape(1) == static_cast<int64_t>(kNumRouteTypes));
    REQUIRE(hard_mask.shape(0) == T);
    REQUIRE(hard_mask.shape(1) == static_cast<int64_t>(kNumRouteTypes));
}

// ── NumericRouter: hard_mask is one-hot ──────────────────────────────────────

TEST_CASE("NumericRouter: hard_mask is one-hot") {
    const int64_t D = 16, T = 8;
    NumericRouter router(D, /*seed=*/7);

    auto x = make_input(T, D);
    auto [soft_probs, hard_mask] = router.forward(x);

    const int64_t K = static_cast<int64_t>(kNumRouteTypes);
    auto mask_sp = hard_mask.data_as<float>();

    for (int64_t t = 0; t < T; ++t) {
        float row_sum = 0.f;
        int   active  = 0;
        for (int64_t k = 0; k < K; ++k) {
            float v = mask_sp[static_cast<std::size_t>(t * K + k)];
            REQUIRE((v == 0.f || v == 1.f));
            row_sum += v;
            if (v == 1.f) ++active;
        }
        REQUIRE(row_sum == 1.f);
        REQUIRE(active  == 1);
    }
}

// ── NumericRouter: soft_probs sum to 1 ───────────────────────────────────────

TEST_CASE("NumericRouter: soft_probs sum to 1") {
    const int64_t D = 16, T = 8;
    NumericRouter router(D, /*seed=*/3);

    auto x = make_input(T, D);
    auto [soft_probs, hard_mask] = router.forward(x);

    const int64_t K = static_cast<int64_t>(kNumRouteTypes);
    auto sp = soft_probs.data().data_as<float>();

    for (int64_t t = 0; t < T; ++t) {
        float row_sum = 0.f;
        for (int64_t k = 0; k < K; ++k)
            row_sum += sp[static_cast<std::size_t>(t * K + k)];
        REQUIRE(std::abs(row_sum - 1.0f) < 1e-5f);
    }
}

// ── NumericRouter: route_hard returns T items ─────────────────────────────────

TEST_CASE("NumericRouter: route_hard returns T items") {
    const int64_t D = 16, T = 8;
    NumericRouter router(D, /*seed=*/13);

    auto x = make_input(T, D);
    auto routes = router.route_hard(x);

    REQUIRE(static_cast<int64_t>(routes.size()) == T);
}

// ── NumericRouter: route_hard values in range ─────────────────────────────────

TEST_CASE("NumericRouter: route_hard values in range") {
    const int64_t D = 16, T = 8;
    NumericRouter router(D, /*seed=*/17);

    auto x = make_input(T, D);
    auto routes = router.route_hard(x);

    for (auto r : routes) {
        int val = static_cast<int>(r);
        REQUIRE(val >= 0);
        REQUIRE(val < kNumRouteTypes);
    }
}

// ── NumericRouter: parameters non-empty ──────────────────────────────────────

TEST_CASE("NumericRouter: parameters non-empty") {
    NumericRouter router(32, /*seed=*/42);
    auto params = router.parameters();
    REQUIRE_FALSE(params.empty());

    int64_t total = 0;
    for (auto* p : params) total += static_cast<int64_t>(p->data().numel());
    REQUIRE(total > 0);
}

// ── NumericRouter: gradient flows ────────────────────────────────────────────

TEST_CASE("NumericRouter: gradient flows through soft_probs") {
    const int64_t D = 8, T = 4;
    NumericRouter router(D, /*seed=*/99);

    auto x = make_input(T, D);
    auto [soft_probs, hard_mask] = router.forward(x);

    // Backpropagate through soft_probs
    auto loss = sum(soft_probs);
    loss.backward();

    bool any_grad = false;
    for (auto* p : router.parameters()) {
        if (p->grad().numel() == 0) continue;
        for (auto f : p->grad().data_as<float>())
            if (f != 0.f) { any_grad = true; break; }
        if (any_grad) break;
    }
    REQUIRE(any_grad);
}
