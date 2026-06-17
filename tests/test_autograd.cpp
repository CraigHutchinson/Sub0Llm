#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/core/ops.hpp"
#include "backends/cuda/backend.hpp"   // rope_bwd kernel parity (Stage 4 Phase 4)

#include <cmath>
#include <stdexcept>
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

TEST_CASE("matmul_bt - gradient check (A input)", "[autograd][grad_check]") {
    auto B = leaf2d({0.7f,0.9f,1.1f, 0.8f,1.0f,1.2f}, 2, 3, false);   // (N=2, K=3)
    auto A = leaf2d({0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f}, 2, 3);          // (M=2, K=3)
    auto f = [&](const Variable& v) { return sum(matmul_bt(v, B)); };
    REQUIRE(grad_check(f, A) < 1e-2f);
}

TEST_CASE("matmul_bt - gradient check (B input)", "[autograd][grad_check]") {
    auto A = leaf2d({0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f}, 2, 3, false);
    auto B = leaf2d({0.7f,0.9f,1.1f, 0.8f,1.0f,1.2f}, 2, 3);
    auto f = [&](const Variable& v) { return sum(matmul_bt(A, v)); };
    REQUIRE(grad_check(f, B) < 1e-2f);
}

TEST_CASE("matmul_bt - matches matmul(transpose2d) forward and backward", "[autograd]") {
    auto A1 = leaf2d({0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f}, 2, 3);
    auto B1 = leaf2d({0.7f,0.9f,1.1f, 0.8f,1.0f,1.2f}, 2, 3);
    auto A2 = leaf2d({0.1f,0.2f,0.3f, 0.4f,0.5f,0.6f}, 2, 3);
    auto B2 = leaf2d({0.7f,0.9f,1.1f, 0.8f,1.0f,1.2f}, 2, 3);

    auto fused = sum(matmul_bt(A1, B1));
    auto ref   = sum(matmul(A2, transpose2d(B2)));
    fused.backward();
    ref.backward();

    REQUIRE_THAT(fused.data().data_as<float>()[0],
                 WithinAbs(ref.data().data_as<float>()[0], 1e-5f));
    auto ga1 = A1.grad().data_as<float>(); auto ga2 = A2.grad().data_as<float>();
    auto gb1 = B1.grad().data_as<float>(); auto gb2 = B2.grad().data_as<float>();
    for (std::size_t i = 0; i < ga1.size(); ++i) REQUIRE_THAT(ga1[i], WithinAbs(ga2[i], 1e-5f));
    for (std::size_t i = 0; i < gb1.size(); ++i) REQUIRE_THAT(gb1[i], WithinAbs(gb2[i], 1e-5f));
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

// ── Batched matmul gradients (Ch29 re-architecture) ────────────────────────────
// The batched core ops reuse the 2D VJPs (matmul_bt/matmul_tb), so batched
// gradients must equal stacking the per-slice 2D gradients. This guards that the
// batch dimension threads correctly through backward.
namespace {
Variable leaf3d(const Tensor& t, bool rg = true) { return Variable(t, rg); }
Tensor   slice2d_ag(const Tensor& t, int64_t bi) {
    const int64_t R = t.shape(1), C = t.shape(2);
    Tensor out({R, C});
    auto s = t.data_as<float>(); auto d = out.data_as<float>();
    for (int64_t i = 0; i < R * C; ++i) d[static_cast<std::size_t>(i)] =
        s[static_cast<std::size_t>(bi * R * C + i)];
    return out;
}
} // namespace

TEST_CASE("batched matmul gradients equal stacked per-slice 2D gradients", "[autograd][batched]") {
    const int64_t B = 3, M = 4, K = 5, N = 2;
    Tensor at = randn({B, M, K}), bt = randn({B, K, N});

    auto a = leaf3d(at), b = leaf3d(bt);
    sum(autograd::matmul(a, b)).backward();   // sum reduces the (B,M,N) output to a scalar
    auto ga = a.grad().data_as<float>();
    auto gb = b.grad().data_as<float>();

    for (int64_t bi = 0; bi < B; ++bi) {
        auto a2 = Variable(slice2d_ag(at, bi), true);
        auto b2 = Variable(slice2d_ag(bt, bi), true);
        sum(autograd::matmul(a2, b2)).backward();
        auto ga2 = a2.grad().data_as<float>();
        auto gb2 = b2.grad().data_as<float>();
        for (int64_t i = 0; i < M * K; ++i)
            REQUIRE_THAT(ga[static_cast<std::size_t>(bi * M * K + i)],
                         WithinAbs(ga2[static_cast<std::size_t>(i)], 1e-4f));
        for (int64_t i = 0; i < K * N; ++i)
            REQUIRE_THAT(gb[static_cast<std::size_t>(bi * K * N + i)],
                         WithinAbs(gb2[static_cast<std::size_t>(i)], 1e-4f));
    }
}

// ── Device plumbing (Stage 4 Phase 0) ─────────────────────────────────────────
//
// Variable::to moves a parameter's storage in place, keeping Node identity. The
// CPU cases run on every build; the CUDA cases (gated) exercise the real H2D/D2H
// path and the grad-moving branch. The backward closures' device-correctness
// (grads land on the input's device) is validated by these round-trips plus the
// full CPU suite staying green — an on-device gradcheck arrives with the Phase 1
// kernels, since forward is still CPU-pointer math here.

TEST_CASE("Variable::to - same device is an identity no-op", "[autograd][device]") {
    auto x = leaf2d({1, 2, 3, 4, 5, 6}, 2, 3, /*rg=*/true);
    Variable& ret = x.to(Device::cpu());
    REQUIRE(&ret == &x);                 // returns *this for chaining
    REQUIRE(x.data().device().is_cpu());
    REQUIRE(x.requires_grad());
    REQUIRE(x.is_leaf());
    auto d = x.data().data_as<float>();
    REQUIRE(d[0] == 1.0f);
    REQUIRE(d[5] == 6.0f);
}

TEST_CASE("Variable::to - preserves an accumulated grad", "[autograd][device]") {
    auto x = leaf2d({1, -2, 3, -4}, 2, 2, /*rg=*/true);
    sum(relu(x)).backward();             // populates x.grad() on CPU
    REQUIRE(x.grad().numel() == 4);
    const float g0 = x.grad().data_as<float>()[0];

    x.to(Device::cpu());                 // no-op must not disturb the grad
    REQUIRE(x.grad().numel() == 4);
    REQUIRE(x.grad().device().is_cpu());
    REQUIRE(x.grad().data_as<float>()[0] == g0);
}

TEST_CASE("Variable::to - undefined variable is a safe no-op", "[autograd][device]") {
    Variable v;                          // default-constructed, no Node
    REQUIRE_NOTHROW(v.to(Device::cpu()));
    REQUIRE_FALSE(v.defined());
}

TEST_CASE("Variable::to - GPU round-trip moves data and grad", "[autograd][device]") {
    auto x = leaf2d({1, 2, 3, 4}, 2, 2, /*rg=*/true);
    sum(relu(x)).backward();
    const float g0 = x.grad().data_as<float>()[0];
#ifdef SUB0LLM_CUDA
    x.to(Device::cuda());
    REQUIRE(x.data().device().is_cuda());
    REQUIRE(x.grad().device().is_cuda());   // grad-moving branch exercised
    x.to(Device::cpu());
    REQUIRE(x.data().device().is_cpu());
    REQUIRE(x.data().data_as<float>()[3] == 4.0f);
    REQUIRE(x.grad().device().is_cpu());
    REQUIRE(x.grad().data_as<float>()[0] == g0);
#else
    REQUIRE_THROWS_AS(x.to(Device::cuda()), std::runtime_error);
#endif
}

// The component test for the backward-closure device plumbing — runs WITHOUT a GPU.
//
// A {CPU, index=1} device is host-backed (the Tensor allocator dispatches on
// is_cpu(), ignoring index) yet compares unequal to Device::cpu(). The forward
// ops (unary_cpu / softmax / narrow) preserve the input's device value, so a real
// CPU forward+backward on a {CPU,1} input must yield grads tagged {CPU,1}. Before
// the fix the closures allocated grads with the default {CPU,0}, so each REQUIRE
// below would fail — i.e. this genuinely exercises the change, not a tautology.
TEST_CASE("backward closures allocate grads on the input's device value",
          "[autograd][device]") {
    const Device alt{DeviceType::CPU, 1};
    REQUIRE_FALSE(alt == Device::cpu());

    auto mk2d = [&](std::vector<float> v, int64_t r, int64_t c) {
        Tensor t({r, c}, DType::Float32, alt);
        auto s = t.data_as<float>();
        for (std::size_t i = 0; i < v.size(); ++i) s[i] = v[i];
        return Variable(std::move(t), true);
    };
    auto mk1d = [&](std::vector<float> v) {
        Tensor t({static_cast<int64_t>(v.size())}, DType::Float32, alt);
        auto s = t.data_as<float>();
        for (std::size_t i = 0; i < v.size(); ++i) s[i] = v[i];
        return Variable(std::move(t), true);
    };

    SECTION("rms_norm — x and weight grads") {
        auto x = mk2d({1, 2, 3, 4, 5, 6}, 2, 3);
        auto w = mk1d({1, 1, 1});
        sum(rms_norm(x, w, 1e-5f)).backward();
        REQUIRE(x.grad().device() == alt);
        REQUIRE(w.grad().device() == alt);
    }
    SECTION("layer_norm — x, weight and bias grads") {
        auto x = mk2d({1, 2, 3, 4, 5, 6}, 2, 3);
        auto w = mk1d({1, 1, 1});
        auto b = mk1d({0, 0, 0});
        sum(layer_norm(x, w, b, 1e-5f)).backward();
        REQUIRE(x.grad().device() == alt);
        REQUIRE(w.grad().device() == alt);
        REQUIRE(b.grad().device() == alt);
    }
    SECTION("rope — x grad") {
        auto x = mk2d({1, 2, 3, 4, 5, 6, 7, 8}, 2, 4);   // T=2, Dh=4
        Tensor cosf({2, 2}, DType::Float32, alt);
        Tensor sinf({2, 2}, DType::Float32, alt);
        auto cs = cosf.data_as<float>();
        auto ss = sinf.data_as<float>();
        for (std::size_t i = 0; i < 4; ++i) { cs[i] = 0.5f; ss[i] = 0.5f; }
        sum(rope(x, cosf, sinf)).backward();
        REQUIRE(x.grad().device() == alt);
    }
    SECTION("narrow — x grad") {
        auto x = mk2d({1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);
        sum(narrow(x, 1, 2)).backward();
        REQUIRE(x.grad().device() == alt);
    }
    SECTION("row_scale — x and v grads") {
        auto x = mk2d({1, 2, 3, 4}, 2, 2);
        auto v = mk2d({2, 3}, 2, 1);
        sum(row_scale(x, v)).backward();
        REQUIRE(x.grad().device() == alt);
        REQUIRE(v.grad().device() == alt);
    }
    SECTION("log_softmax — x grad") {
        auto x = mk2d({1, 2, 3, 4, 5, 6}, 2, 3);
        sum(log_softmax(x)).backward();
        REQUIRE(x.grad().device() == alt);
    }
    SECTION("log_sigmoid — x grad") {
        auto x = mk2d({0.5f, -0.5f, 1.0f, -1.0f}, 2, 2);
        sum(log_sigmoid(x)).backward();
        REQUIRE(x.grad().device() == alt);
    }
}

// Stage 4 Phase 2: autograd::rms_norm forward must dispatch to the CUDA kernel and produce the
// same output as the CPU path (end-to-end through the public autograd API, like the softmax
// forward). Gated on the CUDA build; backward-on-CUDA end-to-end waits on Phase 7 (device copy()).
TEST_CASE("rms_norm forward dispatches to CUDA and matches CPU", "[autograd][device]") {
    const int64_t T = 4, D = 6;
    std::vector<float> xv(static_cast<std::size_t>(T * D));
    for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = 0.3f * static_cast<float>(i) - 1.1f;
    std::vector<float> wv = {1.0f, 0.5f, -0.7f, 2.0f, 0.1f, -1.3f};

    auto x_cpu = leaf2d(xv, T, D, /*rg=*/false);
    auto w_cpu = leaf(wv, /*rg=*/false);
    const Tensor y_cpu = rms_norm(x_cpu, w_cpu, 1e-5f).data();
#ifdef SUB0LLM_CUDA
    auto x_gpu = leaf2d(xv, T, D, /*rg=*/false);  x_gpu.to(Device::cuda());
    auto w_gpu = leaf(wv, /*rg=*/false);          w_gpu.to(Device::cuda());
    const Variable y_gpu = rms_norm(x_gpu, w_gpu, 1e-5f);
    REQUIRE(y_gpu.data().device().is_cuda());
    const Tensor y_back = y_gpu.data().to(Device::cpu());

    const auto yc = y_cpu.data_as<float>();
    const auto yb = y_back.data_as<float>();
    for (int64_t i = 0; i < T * D; ++i)
        REQUIRE_THAT(yb[static_cast<std::size_t>(i)],
                     WithinAbs(yc[static_cast<std::size_t>(i)], 1e-4f));
#else
    REQUIRE(y_cpu.numel() == T * D);   // CPU path still exercised; CUDA dispatch on the cuda preset
#endif
}

// Stage 4 Phase 4: rope (half-split). Forward dispatches to CUDA end-to-end; the backward kernel
// is validated against the autograd CPU gradient (the trusted reference — no math duplication).
TEST_CASE("rope forward + backward match CPU on CUDA", "[autograd][device]") {
    const int64_t T = 5, Dh = 8, D2 = Dh / 2;
    std::vector<float> xv(static_cast<std::size_t>(T * Dh)), gv(static_cast<std::size_t>(T * Dh));
    for (std::size_t i = 0; i < xv.size(); ++i) {
        xv[i] = 0.2f * static_cast<float>(i) - 0.9f;
        gv[i] = 0.13f * static_cast<float>(i) - 0.5f;
    }
    Tensor cosT({T, D2}, DType::Float32), sinT({T, D2}, DType::Float32);
    {   // arbitrary but valid rotation angles
        auto cp = cosT.data_as<float>();  auto sp = sinT.data_as<float>();
        for (int64_t k = 0; k < T * D2; ++k) {
            const float th = 0.05f * static_cast<float>(k);
            cp[static_cast<std::size_t>(k)] = std::cos(th);
            sp[static_cast<std::size_t>(k)] = std::sin(th);
        }
    }
    Tensor up({T, Dh}, DType::Float32);
    { auto u = up.data_as<float>(); for (std::size_t i = 0; i < gv.size(); ++i) u[i] = gv[i]; }

    // CPU reference: forward output and the autograd backward gradient.
    const Variable y_cpu = rope(leaf2d(xv, T, Dh, /*rg=*/false), cosT, sinT);
    Variable x_for_grad = leaf2d(xv, T, Dh, /*rg=*/true);
    rope(x_for_grad, cosT, sinT).backward(copy(up));
    const Tensor gx_cpu = x_for_grad.grad();
#ifdef SUB0LLM_CUDA
    // Forward parity (end-to-end through autograd::rope dispatch).
    auto x_gpu = leaf2d(xv, T, Dh, /*rg=*/false);  x_gpu.to(Device::cuda());
    const Tensor cosD = cosT.to(Device::cuda()), sinD = sinT.to(Device::cuda());
    const Variable y_gpu = rope(x_gpu, cosD, sinD);
    REQUIRE(y_gpu.data().device().is_cuda());
    const Tensor y_back = y_gpu.data().to(Device::cpu());
    const auto yc = y_cpu.data().data_as<float>();
    const auto yb = y_back.data_as<float>();
    for (int64_t i = 0; i < T * Dh; ++i)
        REQUIRE_THAT(yb[static_cast<std::size_t>(i)], WithinAbs(yc[static_cast<std::size_t>(i)], 1e-4f));

    // Backward kernel parity vs the autograd CPU gradient.
    const Tensor gD = up.to(Device::cuda());
    Tensor gx_d = zeros({T, Dh}, DType::Float32, Device::cuda());
    backend::cuda::rope_bwd(
        reinterpret_cast<const float*>(gD.raw_ptr()),
        reinterpret_cast<const float*>(cosD.raw_ptr()),
        reinterpret_cast<const float*>(sinD.raw_ptr()),
        reinterpret_cast<float*>(gx_d.raw_ptr()),
        static_cast<int>(T), static_cast<int>(Dh));
    const Tensor gx_gpu = gx_d.to(Device::cpu());
    const auto gc = gx_cpu.data_as<float>();
    const auto gg = gx_gpu.data_as<float>();
    for (int64_t i = 0; i < T * Dh; ++i)
        REQUIRE_THAT(gg[static_cast<std::size_t>(i)], WithinAbs(gc[static_cast<std::size_t>(i)], 1e-4f));
#else
    REQUIRE(y_cpu.data().numel() == T * Dh);
    REQUIRE(gx_cpu.numel() == T * Dh);
#endif
}
