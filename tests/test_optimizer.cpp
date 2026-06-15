#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
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

// ─── Newton-Schulz orthogonalisation (Muon's core kernel) ──────────────────────

TEST_CASE("newton_schulz_orthogonalize: drives singular values to ~1", "[optimizer][muon]") {
    // A FULL-RANK, well-conditioned matrix with a spread of singular values → after NS the
    // singular values collapse toward 1, so ‖O‖_F ≈ sqrt(min(rows,cols)) and OᵀO ≈ I.
    // (NS can only equalise NONZERO singular values — a rank-deficient input stays so.)
    const int64_t R = 6, C = 3;
    // Independent column directions (the first 3 rows are I₃ ⇒ full column rank), then more
    // rows with mixed directions and per-row scaling to spread the singular values.
    const float base[6][3] = {{1,0,0},{0,1,0},{0,0,1},{1,1,0},{0,1,1},{1,0,1}};
    Tensor G = zeros({R, C});
    auto g = G.data_as<float>();
    for (int64_t i = 0; i < R; ++i)
        for (int64_t j = 0; j < C; ++j)
            g[i * C + j] = base[i][j] * (1.0f + 0.5f * static_cast<float>(i));

    const Tensor O = nn::newton_schulz_orthogonalize(G, 5);
    REQUIRE(O.shape()[0] == R);
    REQUIRE(O.shape()[1] == C);
    // ‖O‖_F ≈ sqrt(min(R,C)) = sqrt(3) ≈ 1.732 (all 3 singular values ≈ 1). NS5 is
    // approximate, so allow a generous band.
    REQUIRE_THAT(ops::norm(O), WithinAbs(std::sqrt(3.0f), 0.5f));

    // OᵀO ≈ I_C: diagonals near 1, off-diagonals near 0.
    const Tensor gram = ops::matmul_tb(O, O);     // (C×C) = Oᵀ·O
    const auto gd = gram.data_as<float>();
    for (int64_t i = 0; i < C; ++i) {
        REQUIRE_THAT(gd[i * C + i], WithinAbs(1.0f, 0.45f));         // diagonal ~1
        for (int64_t j = 0; j < C; ++j)
            if (i != j) REQUIRE(std::abs(gd[i * C + j]) < 0.45f);    // off-diag ~0
    }
}

// ─── optimizer factory ─────────────────────────────────────────────────────────

TEST_CASE("make_optimizer: names resolve, unknown throws", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    auto mk = [&](std::string_view n) {
        std::vector<Variable*> ps = {&p};
        return nn::make_optimizer(n, ps, 1e-3f);
    };
    REQUIRE(mk("adam")  != nullptr);
    REQUIRE(mk("adamw") != nullptr);
    REQUIRE(mk("muon")  != nullptr);
    std::vector<Variable*> ps = {&p};
    REQUIRE_THROWS_AS(nn::make_optimizer("nope", ps, 1e-3f), std::runtime_error);
}

TEST_CASE("AdamW: decoupled weight decay shrinks a zero-gradient weight", "[optimizer]") {
    // grad = 0 → plain Adam leaves the weight unchanged; AdamW shrinks it by (1 - lr*wd).
    Variable p = leaf1d({1.0f});
    p.grad() = make_grad({0.0f});
    nn::Adam adamw({&p}, /*lr=*/0.1f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.5f);
    adamw.step();
    // Expected: 1 * (1 - 0.1*0.5) = 0.95 (the Adam term is 0 since grad=0).
    REQUIRE_THAT(p.data().data_as<float>()[0], WithinAbs(0.95f, 1e-4f));
}

TEST_CASE("Muon: reduces loss on a 2D linear-regression problem", "[optimizer][muon]") {
    // Learn W (4×3) so that mean((W·x - y)²) → 0 for a fixed (x, y). W is 2D ⇒ Muon path.
    Tensor x = zeros({3, 1});
    for (int i = 0; i < 3; ++i) x.data_as<float>()[i] = static_cast<float>(i + 1);
    Tensor ytgt = zeros({4, 1});
    for (int i = 0; i < 4; ++i) ytgt.data_as<float>()[i] = static_cast<float>(4 - i);

    Variable W(zeros({4, 3}), true);
    for (float& w : W.data().data_as<float>()) w = 0.05f;
    Variable xv(x, false), yv(ytgt, false);

    nn::Muon opt({&W}, /*lr=*/0.05f);
    auto loss_of = [&] {
        Variable pred = matmul(W, xv);             // (4×1)
        Variable diff = sub(pred, yv);
        return sum(mul(diff, diff)).data().data_as<float>()[0];
    };
    const float l0 = loss_of();
    for (int it = 0; it < 50; ++it) {
        opt.zero_grad();
        Variable pred = matmul(W, xv);
        Variable diff = sub(pred, yv);
        Variable loss = sum(mul(diff, diff));
        loss.backward();
        opt.step();
    }
    const float l1 = loss_of();
    REQUIRE(l1 < l0 * 0.5f);   // Muon makes clear progress on a convex problem
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
    // No clip - gradients unchanged
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

TEST_CASE("SGD: constructor throws on null or duplicate params", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    Variable q = leaf1d({2.0f});
    // null pointer
    std::vector<Variable*> null_params = {nullptr};
    REQUIRE_THROWS_AS(nn::SGD(null_params, 0.1f), std::runtime_error);
    // duplicate pointer
    std::vector<Variable*> dup_params = {&p, &p};
    REQUIRE_THROWS_AS(nn::SGD(dup_params, 0.1f), std::runtime_error);
    // valid
    REQUIRE_NOTHROW(nn::SGD({&p, &q}, 0.1f));
}

TEST_CASE("Adam: constructor throws on null or duplicate params", "[optimizer]") {
    Variable p = leaf1d({1.0f});
    Variable q = leaf1d({2.0f});
    std::vector<Variable*> null_params = {nullptr};
    REQUIRE_THROWS_AS(nn::Adam(null_params, 0.001f), std::runtime_error);
    std::vector<Variable*> dup_params = {&p, &p};
    REQUIRE_THROWS_AS(nn::Adam(dup_params, 0.001f), std::runtime_error);
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
    // p1=2, g1=1; p2=4, g2=2 - both should decrease
    Variable p1 = leaf1d({2.0f});
    Variable p2 = leaf1d({4.0f});
    p1.grad() = make_grad({1.0f});
    p2.grad() = make_grad({2.0f});

    nn::Adam adam({&p1, &p2}, 0.1f);
    adam.step();

    REQUIRE(p1.data().data_as<float>()[0] < 2.0f);
    REQUIRE(p2.data().data_as<float>()[0] < 4.0f);
}
