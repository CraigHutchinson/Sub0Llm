#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <stdexcept>

#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"

using namespace sub0llm;
using namespace sub0llm::autograd;
using Catch::Matchers::WithinAbs;

// ─── helpers ─────────────────────────────────────────────────────────────────

static Variable leaf1d(std::initializer_list<float> vals) {
    const auto n = static_cast<int64_t>(vals.size());
    Tensor t = zeros({n});
    auto d = t.data_as<float>();
    std::size_t i = 0;
    for (float v : vals) d[i++] = v;
    return Variable(std::move(t), true);
}

static Tensor make_grad(std::initializer_list<float> vals) {
    const auto n = static_cast<int64_t>(vals.size());
    Tensor t = zeros({n});
    auto d = t.data_as<float>();
    std::size_t i = 0;
    for (float v : vals) d[i++] = v;
    return t;
}

// ─── clip_grad_norm ───────────────────────────────────────────────────────────

TEST_CASE("clip_grad_norm: throws on non-positive max_norm", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({1.0f});
    std::vector<Variable*> params = {&p};
    REQUIRE_THROWS_AS(nn::clip_grad_norm(params, 0.0f),  std::runtime_error);
    REQUIRE_THROWS_AS(nn::clip_grad_norm(params, -1.0f), std::runtime_error);
}

TEST_CASE("clip_grad_norm: no clipping when norm <= max_norm", "[optimizer]") {
    // p grad = [3, 4] → norm = 5; max_norm = 10 → no clip
    Variable p = leaf1d({3.0f, 4.0f});
    p.grad() = make_grad({3.0f, 4.0f});

    std::vector<Variable*> params = {&p};
    const float norm = nn::clip_grad_norm(params, 10.0f);

    REQUIRE_THAT(norm, WithinAbs(5.0f, 1e-5f));
    const auto gd = p.grad().data_as<float>();
    REQUIRE_THAT(gd[0], WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(gd[1], WithinAbs(4.0f, 1e-5f));
}

TEST_CASE("clip_grad_norm: clips when norm > max_norm", "[optimizer]") {
    // p grad = [3, 4] → norm = 5; max_norm = 2.5 → scale = 0.5
    Variable p = leaf1d({3.0f, 4.0f});
    p.grad() = make_grad({3.0f, 4.0f});

    std::vector<Variable*> params = {&p};
    const float norm = nn::clip_grad_norm(params, 2.5f);

    REQUIRE_THAT(norm, WithinAbs(5.0f, 1e-5f));
    const auto gd = p.grad().data_as<float>();
    REQUIRE_THAT(gd[0], WithinAbs(1.5f, 1e-5f));
    REQUIRE_THAT(gd[1], WithinAbs(2.0f, 1e-5f));
}

TEST_CASE("clip_grad_norm: skips params without grad", "[optimizer]") {
    // p1 has no grad (numel==0); p2 grad = [6, 8] → norm = 10
    Variable p1 = leaf1d({0.0f, 0.0f});  // no grad set → numel==0
    Variable p2 = leaf1d({0.0f, 0.0f});
    p2.grad() = make_grad({6.0f, 8.0f});

    std::vector<Variable*> params = {&p1, &p2};
    const float norm = nn::clip_grad_norm(params, 5.0f);

    // Only p2 contributes: norm = 10, max = 5 → scale = 0.5
    REQUIRE_THAT(norm, WithinAbs(10.0f, 1e-5f));
    const auto gd2 = p2.grad().data_as<float>();
    REQUIRE_THAT(gd2[0], WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(gd2[1], WithinAbs(4.0f, 1e-5f));
}

TEST_CASE("clip_grad_norm: multi-param global norm", "[optimizer]") {
    // p1 grad = [3], p2 grad = [4] → global norm = 5; max = 5 → no clip
    Variable p1 = leaf1d({1.0f});
    Variable p2 = leaf1d({2.0f});
    p1.grad() = make_grad({3.0f});
    p2.grad() = make_grad({4.0f});

    std::vector<Variable*> params = {&p1, &p2};
    const float norm = nn::clip_grad_norm(params, 5.0f);
    REQUIRE_THAT(norm, WithinAbs(5.0f, 1e-5f));
    // No clip — gradients unchanged
    REQUIRE_THAT(p1.grad().data_as<float>()[0], WithinAbs(3.0f, 1e-5f));
    REQUIRE_THAT(p2.grad().data_as<float>()[0], WithinAbs(4.0f, 1e-5f));
}

// ─── SGD ─────────────────────────────────────────────────────────────────────

TEST_CASE("SGD: single step without momentum", "[optimizer]") {
    // p = 1.0, grad = 0.5, lr = 0.1
    // v = 0*0 - 0.1*0.5 = -0.05; p_new = 1.0 + (-0.05) = 0.95
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({0.5f});

    nn::SGD sgd({&p}, 0.1f);
    sgd.step();

    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(0.95f, 1e-6f));
}

TEST_CASE("SGD: momentum accumulates velocity", "[optimizer]") {
    // lr=0.1, momentum=0.9, grad=1.0 each step
    // step1: v = 0.9*0 - 0.1*1 = -0.1; p = 1 + (-0.1) = 0.9
    // step2: v = 0.9*(-0.1) - 0.1*1 = -0.19; p = 0.9 + (-0.19) = 0.71
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({1.0f});

    nn::SGD sgd({&p}, 0.1f, 0.9f);
    sgd.step();

    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(0.9f, 1e-6f));

    p.grad() = make_grad({1.0f});
    sgd.step();

    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(0.71f, 1e-6f));
}

TEST_CASE("SGD: zero_grad zeroes gradients", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({1.0f});

    nn::SGD sgd({&p}, 0.01f);
    sgd.zero_grad();

    // zero_grad fills with zeros, matching param shape
    REQUIRE(p.grad().numel() == 1);
    REQUIRE_THAT(p.grad().data_as<float>()[0], WithinAbs(0.0f, 1e-9f));
}

TEST_CASE("SGD: skips params with no grad", "[optimizer]") {
    Variable p = leaf1d({5.0f});  // no grad set → numel==0
    nn::SGD sgd({&p}, 0.1f);
    sgd.step();
    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(5.0f, 1e-6f));
}

// ─── Adam ─────────────────────────────────────────────────────────────────────

TEST_CASE("Adam: single step matches reference formula", "[optimizer]") {
    // p=1, g=1, lr=0.1, b1=0.9, b2=0.999, eps=1e-8
    // t=1: m=0.1, v=0.001
    // bc1=0.1, bc2=0.001
    // m_hat=1.0, v_hat=1.0
    // p_new = 1 - 0.1 * 1.0 / (1.0 + 1e-8) ≈ 0.9
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({1.0f});

    nn::Adam adam({&p}, 0.1f, 0.9f, 0.999f, 1e-8f);
    adam.step();

    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(0.9f, 1e-4f));
}

TEST_CASE("Adam: loss decreases over multiple steps on quadratic", "[optimizer]") {
    // Minimise f(p) = 0.5 * p^2 → grad = p. Expect p → 0.
    Variable p = leaf1d({2.0f});
    nn::Adam adam({&p}, 0.1f);

    for (int step = 0; step < 50; ++step) {
        const float pv = p.data().data_as<float>()[0];
        p.grad() = make_grad({pv});
        adam.step();
    }
    REQUIRE(std::abs(p.data().data_as<float>()[0]) < 0.1f);
}

TEST_CASE("Adam: zero_grad zeroes gradients", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({1.0f});

    nn::Adam adam({&p}, 0.01f);
    adam.zero_grad();

    // zero_grad fills with zeros, matching param shape
    REQUIRE(p.grad().numel() == 1);
    REQUIRE_THAT(p.grad().data_as<float>()[0], WithinAbs(0.0f, 1e-9f));
}

TEST_CASE("Adam: skips params with no grad", "[optimizer]") {
    Variable p = leaf1d({3.0f});  // no grad set → numel==0
    nn::Adam adam({&p}, 0.1f);
    adam.step();
    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(3.0f, 1e-6f));
}

TEST_CASE("Adam: constructor throws on invalid b1", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    REQUIRE_THROWS_AS((nn::Adam({&p}, 1e-3f, 1.0f, 0.999f, 1e-8f)), std::runtime_error);
    REQUIRE_THROWS_AS((nn::Adam({&p}, 1e-3f, -0.1f, 0.999f, 1e-8f)), std::runtime_error);
}

TEST_CASE("Adam: constructor throws on invalid b2", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    REQUIRE_THROWS_AS((nn::Adam({&p}, 1e-3f, 0.9f, 1.0f, 1e-8f)), std::runtime_error);
}

TEST_CASE("Adam: constructor throws on non-positive eps", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    REQUIRE_THROWS_AS((nn::Adam({&p}, 1e-3f, 0.9f, 0.999f, 0.0f)), std::runtime_error);
    REQUIRE_THROWS_AS((nn::Adam({&p}, 1e-3f, 0.9f, 0.999f, -1e-8f)), std::runtime_error);
}

TEST_CASE("Adam: multiple params updated independently", "[optimizer]") {
    // p1=2, g1=1; p2=4, g2=2 — both should decrease
    Variable p1 = leaf1d({2.0f});
    Variable p2 = leaf1d({4.0f});
    p1.grad() = make_grad({1.0f});
    p2.grad() = make_grad({2.0f});

    nn::Adam adam({&p1, &p2}, 0.1f);
    adam.step();

    REQUIRE(p1.data().data_as<float>()[0] < 2.0f);
    REQUIRE(p2.data().data_as<float>()[0] < 4.0f);
}
