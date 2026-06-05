#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Variable leaf(std::vector<float> vals, bool rg = true) {
    Tensor t = zeros({static_cast<int64_t>(vals.size())});
    auto   s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

static Variable leaf2d(std::vector<float> vals, int64_t rows, int64_t cols,
                       bool rg = true) {
    Tensor t = zeros({rows, cols});
    auto   s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

// Returns max absolute error between analytical and numerical gradients.
// Preserves the shape of x when creating perturbed copies.
static float grad_check(std::function<Variable(const Variable&)> f,
                        const Variable& x,
                        float eps = 1e-3f) {
    const auto xs    = x.data().data_as<float>();
    const auto shape = x.data().shape();
    const std::size_t n = static_cast<std::size_t>(x.data().numel());

    Variable y = f(x);
    y.backward();
    const auto analytic = x.grad().data_as<float>();

    float max_err = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        auto perturb = [&](float delta) {
            Tensor t = zeros(shape);
            auto   s = t.data_as<float>();
            for (std::size_t j = 0; j < n; ++j) s[j] = xs[j];
            s[i] += delta;
            Variable xv(std::move(t), false);
            return f(xv).data().data_as<float>()[0];
        };
        const float numerical = (perturb(+eps) - perturb(-eps)) / (2.0f * eps);
        max_err = std::max(max_err, std::abs(analytic[i] - numerical));
    }
    return max_err;
}

// ── Variable basics ───────────────────────────────────────────────────────────

TEST_CASE("Variable - leaf properties", "[autograd]") {
    auto x = leaf({1.0f, 2.0f, 3.0f});
    REQUIRE(x.requires_grad());
    REQUIRE(x.is_leaf());
    REQUIRE(x.defined());
    REQUIRE(x.data().numel() == 3);
}

TEST_CASE("Variable - default constructed is undefined", "[autograd]") {
    Variable v;
    REQUIRE_FALSE(v.defined());
    REQUIRE_FALSE(v.requires_grad());
}

TEST_CASE("Variable - no_grad leaf does not track", "[autograd]") {
    auto x = leaf({1.0f, 2.0f}, false);
    REQUIRE_FALSE(x.requires_grad());
}

TEST_CASE("Variable - zero_grad resets gradient", "[autograd]") {
    auto x = leaf({1.0f, 2.0f});
    auto y = sum(x);
    y.backward();
    REQUIRE(x.grad().numel() == 2);
    x.zero_grad();
    REQUIRE_THAT(x.grad().data_as<float>()[0], WithinAbs(0.0f, 1e-9f));
}

// ── backward guards ───────────────────────────────────────────────────────────

TEST_CASE("backward - non-scalar without upstream_grad throws", "[autograd]") {
    auto x = leaf({1.0f, 2.0f});
    auto y = add(x, x);
    REQUIRE_THROWS_AS(y.backward(), std::runtime_error);
}

// ── add ───────────────────────────────────────────────────────────────────────

TEST_CASE("add - gradient check", "[autograd][grad_check]") {
    auto x  = leaf({0.5f, -1.0f, 2.0f});
    auto cx = leaf({1.0f,  1.0f, 1.0f}, false);
    auto f  = [&](const Variable& v) { return sum(add(v, cx)); };
    REQUIRE(grad_check(f, x) < 1e-3f);
}

TEST_CASE("add - gradient is ones for both inputs", "[autograd]") {
    auto a = leaf({1.0f, 2.0f});
    auto b = leaf({3.0f, 4.0f});
    auto c = sum(add(a, b));
    c.backward();
    for (float v : a.grad().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(1.0f, 1e-5f));
    for (float v : b.grad().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(1.0f, 1e-5f));
}

// ── sub ───────────────────────────────────────────────────────────────────────

TEST_CASE("sub - gradient check", "[autograd][grad_check]") {
    auto x = leaf({1.0f, -0.5f, 3.0f});
    auto y = leaf({0.5f,  1.0f, 1.0f}, false);
    auto f = [&](const Variable& v) { return sum(sub(v, y)); };
    REQUIRE(grad_check(f, x) < 1e-3f);
}

TEST_CASE("sub - grad_b is negated", "[autograd]") {
    auto a = leaf({2.0f});
    auto b = leaf({1.0f});
    auto c = sum(sub(a, b));
    c.backward();
    REQUIRE_THAT(a.grad().data_as<float>()[0], WithinAbs( 1.0f, 1e-5f));
    REQUIRE_THAT(b.grad().data_as<float>()[0], WithinAbs(-1.0f, 1e-5f));
}

// ── mul ───────────────────────────────────────────────────────────────────────

TEST_CASE("mul - gradient check", "[autograd][grad_check]") {
    auto x = leaf({0.5f, 2.0f, -1.0f});
    auto y = leaf({1.0f, 3.0f,  2.0f}, false);
    auto f = [&](const Variable& v) { return sum(mul(v, y)); };
    REQUIRE(grad_check(f, x) < 1e-3f);
}

TEST_CASE("mul - grad_a = upstream * b_data", "[autograd]") {
    auto a = leaf({3.0f, 4.0f});
    auto b = leaf({2.0f, 5.0f});
    auto c = sum(mul(a, b));
    c.backward();
    const auto ga = a.grad().data_as<float>();
    REQUIRE_THAT(ga[0], WithinAbs(2.0f, 1e-5f));
    REQUIRE_THAT(ga[1], WithinAbs(5.0f, 1e-5f));
}

// ── matmul ────────────────────────────────────────────────────────────────────

TEST_CASE("matmul - gradient check (A input)", "[autograd][grad_check]") {
    // Use small values to keep output magnitudes low for float32 finite-diff.
    auto B = leaf2d({0.7f,0.8f, 0.9f,1.0f, 1.1f,1.2f}, 3, 2, false);
    auto A = leaf2d({0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f}, 2, 3);
    auto f = [&](const Variable& v) { return sum(matmul(v, B)); };
    REQUIRE(grad_check(f, A) < 1e-2f);
}

TEST_CASE("matmul - gradient check (B input)", "[autograd][grad_check]") {
    auto A = leaf2d({0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f}, 2, 3, false);
    auto B = leaf2d({0.7f,0.8f, 0.9f,1.0f, 1.1f,1.2f}, 3, 2);
    auto f = [&](const Variable& v) { return sum(matmul(A, v)); };
    REQUIRE(grad_check(f, B) < 1e-2f);
}

// ── sum ───────────────────────────────────────────────────────────────────────

TEST_CASE("sum - gradient is broadcast ones", "[autograd]") {
    auto x = leaf({1.0f, 2.0f, 3.0f, 4.0f});
    auto s = sum(x);
    s.backward();
    for (float v : x.grad().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(1.0f, 1e-5f));
}

// ── relu ──────────────────────────────────────────────────────────────────────

TEST_CASE("relu - gradient check", "[autograd][grad_check]") {
    auto x = leaf({0.5f, 2.0f, -1.0f, -0.5f, 3.0f});
    auto f = [](const Variable& v) { return sum(relu(v)); };
    REQUIRE(grad_check(f, x) < 1e-3f);
}

TEST_CASE("relu - zero gradient at negative inputs", "[autograd]") {
    auto x = leaf({-2.0f, 1.0f, -0.5f, 3.0f});
    auto y = sum(relu(x));
    y.backward();
    const auto g = x.grad().data_as<float>();
    REQUIRE_THAT(g[0], WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(g[1], WithinAbs(1.0f, 1e-5f));
    REQUIRE_THAT(g[2], WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(g[3], WithinAbs(1.0f, 1e-5f));
}

// ── log_softmax ───────────────────────────────────────────────────────────────

TEST_CASE("log_softmax - gradient check 1D", "[autograd][grad_check]") {
    auto x = leaf({1.0f, 2.0f, 0.5f, -1.0f});
    auto f = [](const Variable& v) { return sum(log_softmax(v)); };
    REQUIRE(grad_check(f, x) < 1e-2f);
}

TEST_CASE("log_softmax - exp of output sums to 1", "[autograd]") {
    auto x  = leaf({2.0f, 1.0f, 0.1f}, false);
    auto lp = log_softmax(x);
    float sum_probs = 0.0f;
    for (float v : lp.data().data_as<float>()) sum_probs += std::exp(v);
    REQUIRE_THAT(sum_probs, WithinAbs(1.0f, 1e-5f));
}

// ── cross_entropy ─────────────────────────────────────────────────────────────

TEST_CASE("cross_entropy - loss is positive scalar", "[autograd]") {
    auto logits = leaf2d({1.f,2.f,3.f,4.f, 5.f,6.f,7.f,8.f, 1.f,1.f,1.f,1.f}, 3, 4);
    Tensor targets = zeros({3}, DType::Int32);
    targets.data_as<int32_t>()[0] = 0;
    targets.data_as<int32_t>()[1] = 2;
    targets.data_as<int32_t>()[2] = 3;
    auto loss = cross_entropy(logits, targets);
    REQUIRE(loss.data().numel() == 1);
    REQUIRE(loss.data().data_as<float>()[0] > 0.0f);
}

TEST_CASE("cross_entropy - perfect prediction gives near-zero loss", "[autograd]") {
    auto logits = leaf2d({100.f,0.f,0.f, 0.f,100.f,0.f}, 2, 3, false);
    Tensor targets = zeros({2}, DType::Int32);
    targets.data_as<int32_t>()[0] = 0;
    targets.data_as<int32_t>()[1] = 1;
    auto loss = cross_entropy(logits, targets);
    REQUIRE_THAT(loss.data().data_as<float>()[0], WithinAbs(0.0f, 1e-3f));
}

TEST_CASE("cross_entropy - per-row gradient sums to zero", "[autograd]") {
    auto logits = leaf2d({1.f,2.f,3.f, 4.f,5.f,6.f}, 2, 3);
    Tensor targets = zeros({2}, DType::Int32);
    targets.data_as<int32_t>()[0] = 1;
    targets.data_as<int32_t>()[1] = 0;
    auto loss = cross_entropy(logits, targets);
    loss.backward();
    const auto g = logits.grad().data_as<float>();
    REQUIRE_THAT(g[0] + g[1] + g[2], WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(g[3] + g[4] + g[5], WithinAbs(0.0f, 1e-5f));
}

// ── Chain rule & gradient accumulation ───────────────────────────────────────

TEST_CASE("chain rule - relu(add(x, x)) gradient check", "[autograd][grad_check]") {
    auto x = leaf({0.5f, -0.2f, 1.5f});
    auto f = [](const Variable& v) { return sum(relu(add(v, v))); };
    REQUIRE(grad_check(f, x) < 1e-3f);
}

TEST_CASE("gradient accumulation - x + x gives grad = 2", "[autograd]") {
    auto x = leaf({1.0f, 2.0f, 3.0f});
    auto y = sum(add(x, x));
    y.backward();
    for (float v : x.grad().data_as<float>())
        REQUIRE_THAT(v, WithinAbs(2.0f, 1e-5f));
}

// ── bias_add ──────────────────────────────────────────────────────────────────

TEST_CASE("bias_add - forward adds bias to each row", "[autograd]") {
    auto x = leaf2d({1.f,2.f, 3.f,4.f}, 2, 2, false);
    auto b = leaf({10.f, 20.f}, false);
    auto y = bias_add(x, b);
    const auto yd = y.data().data_as<float>();
    REQUIRE_THAT(yd[0], WithinAbs(11.0f, 1e-5f));
    REQUIRE_THAT(yd[1], WithinAbs(22.0f, 1e-5f));
    REQUIRE_THAT(yd[2], WithinAbs(13.0f, 1e-5f));
    REQUIRE_THAT(yd[3], WithinAbs(24.0f, 1e-5f));
}

TEST_CASE("bias_add - bias grad is column sum of upstream", "[autograd]") {
    auto x = leaf2d({1.f,2.f, 3.f,4.f, 5.f,6.f}, 3, 2, false);
    auto b = leaf({0.f, 0.f});
    auto s = sum(bias_add(x, b));
    s.backward();
    // upstream ones(3,2); column sum: col0=3, col1=3
    const auto bg = b.grad().data_as<float>();
    REQUIRE_THAT(bg[0], WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(bg[1], WithinAbs(3.0f, 1e-5f));
}

TEST_CASE("bias_add - gradient check for bias", "[autograd][grad_check]") {
    auto x = leaf2d({0.5f,1.0f, 1.5f,2.0f, 2.5f,3.0f}, 3, 2, false);
    auto b = leaf({0.1f, 0.2f});
    auto f = [&](const Variable& v) { return sum(bias_add(x, v)); };
    REQUIRE(grad_check(f, b) < 1e-3f);
}

// ── cross_entropy grad_check ──────────────────────────────────────────────────

TEST_CASE("cross_entropy - gradient check via finite differences", "[autograd][grad_check]") {
    // Small (2, 3) logits so the numerical grad is tractable.
    Tensor targets = zeros({2}, DType::Int32);
    targets.data_as<int32_t>()[0] = 1;
    targets.data_as<int32_t>()[1] = 0;

    auto logits = leaf2d({0.5f,1.0f,0.3f, 1.2f,0.4f,0.8f}, 2, 3);
    auto f = [&targets](const Variable& v) {
        return cross_entropy(v, targets);
    };
    REQUIRE(grad_check(f, logits) < 1e-2f);
}

// ── weighted_cross_entropy ─────────────────────────────────────────────────────

TEST_CASE("weighted_cross_entropy - uniform weights equal cross_entropy", "[autograd]") {
    auto logits = leaf2d({0.5f,1.0f,0.3f, 1.2f,0.4f,0.8f}, 2, 3, false);
    Tensor targets = zeros({2}, DType::Int32);
    targets.data_as<int32_t>()[0] = 1;
    targets.data_as<int32_t>()[1] = 0;
    Tensor w = zeros({2});
    for (auto& v : w.data_as<float>()) v = 1.0f;

    const float a = cross_entropy(logits, targets).data().data_as<float>()[0];
    const float b = weighted_cross_entropy(logits, targets, w).data().data_as<float>()[0];
    REQUIRE_THAT(b, WithinAbs(a, 1e-5f));
}

TEST_CASE("weighted_cross_entropy - gradient check via finite differences",
          "[autograd][grad_check]") {
    Tensor targets = zeros({2}, DType::Int32);
    targets.data_as<int32_t>()[0] = 1;
    targets.data_as<int32_t>()[1] = 0;
    Tensor w = zeros({2});
    w.data_as<float>()[0] = 0.5f;
    w.data_as<float>()[1] = 2.0f;

    auto logits = leaf2d({0.5f,1.0f,0.3f, 1.2f,0.4f,0.8f}, 2, 3);
    auto f = [&](const Variable& v) { return weighted_cross_entropy(v, targets, w); };
    REQUIRE(grad_check(f, logits) < 1e-2f);
}

// ── detach ────────────────────────────────────────────────────────────────────

TEST_CASE("detach - result does not require grad", "[autograd]") {
    auto x = leaf({1.0f, 2.0f});
    auto d = detach(x);
    REQUIRE_FALSE(d.requires_grad());
}
