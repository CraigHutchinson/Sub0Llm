#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/attention.hpp"

#include <cmath>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace sub0llm;
using namespace sub0llm::autograd;
using namespace sub0llm::nn;

// ── Helpers ───────────────────────────────────────────────────────────────────

static Variable leaf2d(std::vector<float> vals, int64_t rows, int64_t cols,
                       bool rg = true) {
    Tensor t = zeros({rows, cols});
    auto s = t.data_as<float>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return Variable(std::move(t), rg);
}

// ── scale ─────────────────────────────────────────────────────────────────────

TEST_CASE("scale — forward multiplies all elements by alpha", "[attn]") {
    auto x   = leaf2d({1.f, 2.f, 3.f, 4.f}, 2, 2);
    auto y   = scale(x, 2.0f);
    const auto od = y.data().data_as<float>();
    REQUIRE_THAT(od[0], WithinAbs(2.0f, 1e-5f));
    REQUIRE_THAT(od[3], WithinAbs(8.0f, 1e-5f));
}

TEST_CASE("scale — backward propagates alpha * upstream", "[attn]") {
    auto x = leaf2d({1.f, -1.f, 2.f, -2.f}, 2, 2);
    sum(scale(x, 3.0f)).backward();
    const auto gd = x.grad().data_as<float>();
    for (std::size_t i = 0; i < 4u; ++i)
        REQUIRE_THAT(gd[i], WithinAbs(3.0f, 1e-5f));
}

TEST_CASE("scale — alpha=0 gives zero output and zero grad", "[attn]") {
    auto x = leaf2d({5.f, -3.f}, 1, 2);
    sum(scale(x, 0.0f)).backward();
    for (std::size_t i = 0; i < 2u; ++i) {
        REQUIRE_THAT(x.grad().data_as<float>()[i], WithinAbs(0.0f, 1e-5f));
    }
}

// ── transpose2d ───────────────────────────────────────────────────────────────

TEST_CASE("transpose2d — output shape is (N, M) for (M, N) input", "[attn]") {
    auto x   = leaf2d(std::vector<float>(3 * 4, 0.f), 3, 4);
    auto xt  = transpose2d(x);
    REQUIRE(xt.data().shape()[0] == 4);
    REQUIRE(xt.data().shape()[1] == 3);
}

TEST_CASE("transpose2d — values are correctly transposed", "[attn]") {
    // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
    auto x  = leaf2d({1.f,2.f,3.f, 4.f,5.f,6.f}, 2, 3);
    auto xt = transpose2d(x);
    const auto td = xt.data().data_as<float>();
    REQUIRE_THAT(td[0], WithinAbs(1.0f, 1e-5f));  // [0,0]
    REQUIRE_THAT(td[1], WithinAbs(4.0f, 1e-5f));  // [0,1]
    REQUIRE_THAT(td[2], WithinAbs(2.0f, 1e-5f));  // [1,0]
    REQUIRE_THAT(td[5], WithinAbs(6.0f, 1e-5f));  // [2,1]
}

TEST_CASE("transpose2d — backward gives transposed upstream", "[attn]") {
    // sum(x^T) = sum(x), so grad_x = ones
    auto x  = leaf2d({1.f,2.f, 3.f,4.f}, 2, 2);
    sum(transpose2d(x)).backward();
    const auto gd = x.grad().data_as<float>();
    for (std::size_t i = 0; i < 4u; ++i)
        REQUIRE_THAT(gd[i], WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("transpose2d — non-2D input throws", "[attn]") {
    Tensor t3 = zeros({2, 3, 4});
    Variable v3(std::move(t3), true);
    REQUIRE_THROWS_AS(transpose2d(v3), std::runtime_error);
}

// ── softmax ───────────────────────────────────────────────────────────────────

TEST_CASE("softmax — each row sums to 1", "[attn]") {
    auto x = leaf2d({1.f,2.f,3.f, 0.f,-1.f,2.f}, 2, 3);
    auto y = softmax(x);
    const auto yd = y.data().data_as<float>();
    float row0 = yd[0] + yd[1] + yd[2];
    float row1 = yd[3] + yd[4] + yd[5];
    REQUIRE_THAT(row0, WithinAbs(1.0f, 1e-5f));
    REQUIRE_THAT(row1, WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("softmax — masked position (-inf) gets zero weight", "[attn]") {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto x = leaf2d({0.5f, neg_inf, neg_inf,
                     0.5f, 0.3f,    neg_inf}, 2, 3, false);
    auto y = softmax(x);
    const auto yd = y.data().data_as<float>();
    REQUIRE_THAT(yd[0], WithinAbs(1.0f, 1e-5f));  // row 0: only pos 0
    REQUIRE_THAT(yd[1], WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(yd[2], WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(yd[5], WithinAbs(0.0f, 1e-5f));  // row 1: pos 2 masked
}

TEST_CASE("softmax — backward grad check", "[attn]") {
    // f = sum(softmax(x) * w) with non-uniform w — non-constant, non-trivial grad.
    const float eps = 1e-3f;

    // Fixed input and weight tensors.
    Tensor xt = zeros({2, 3});
    { auto s = xt.data_as<float>();
      s[0]=0.3f; s[1]=-0.1f; s[2]=0.8f;
      s[3]=1.2f; s[4]= 0.0f; s[5]=-0.5f; }

    Tensor wt = zeros({2, 3});
    { auto s = wt.data_as<float>();
      s[0]=2.f; s[1]=0.f; s[2]=1.f;   // row 0: non-uniform
      s[3]=0.f; s[4]=3.f; s[5]=1.f; } // row 1: non-uniform

    // Analytical gradient.
    Variable x(xt, true);
    Variable w(wt, false);
    sum(mul(softmax(x), w)).backward();
    const auto ag = x.grad().data_as<float>();

    // Numerical gradient via central differences.
    const auto xd = x.data().data_as<float>();
    const auto wd = wt.data_as<float>();
    for (std::size_t k = 0; k < 6u; ++k) {
        auto eval = [&](float delta) {
            Tensor xk = zeros({2, 3});
            auto s = xk.data_as<float>();
            for (std::size_t i = 0; i < 6u; ++i) s[i] = xd[i];
            s[k] += delta;
            Variable vk(std::move(xk), false);
            auto yk = softmax(vk);  // named: avoids dangling span
            float val = 0.0f;
            const auto ykd = yk.data().data_as<float>();
            for (std::size_t i = 0; i < 6u; ++i) val += ykd[i] * wd[i];
            return val;
        };
        const float ng = (eval(eps) - eval(-eps)) / (2.0f * eps);
        REQUIRE_THAT(ag[k], WithinAbs(ng, 2e-3f));
    }
}

TEST_CASE("softmax — non-2D input throws", "[attn]") {
    Variable v(zeros({4}), true);
    REQUIRE_THROWS_AS(softmax(v), std::runtime_error);
}

// ── MultiHeadSelfAttention ────────────────────────────────────────────────────

TEST_CASE("MHA — output shape is (T, D)", "[attn][mha]") {
    MultiHeadSelfAttention mha(8, 2);
    auto x   = leaf2d(std::vector<float>(4 * 8, 0.1f), 4, 8);
    auto out = mha.forward(x);
    REQUIRE(out.data().shape()[0] == 4);
    REQUIRE(out.data().shape()[1] == 8);
}

TEST_CASE("MHA — single head, output shape is (T, D)", "[attn][mha]") {
    MultiHeadSelfAttention mha(6, 1);
    auto x   = leaf2d(std::vector<float>(3 * 6, 0.2f), 3, 6);
    auto out = mha.forward(x);
    REQUIRE(out.data().shape()[0] == 3);
    REQUIRE(out.data().shape()[1] == 6);
}

TEST_CASE("MHA — invalid embed_dim/num_heads throws", "[attn][mha]") {
    REQUIRE_THROWS_AS(MultiHeadSelfAttention(0, 2),  std::runtime_error);
    REQUIRE_THROWS_AS(MultiHeadSelfAttention(6, 0),  std::runtime_error);
    REQUIRE_THROWS_AS(MultiHeadSelfAttention(6, 4),  std::runtime_error);  // not divisible
}

TEST_CASE("MHA — gradient flows to all weight matrices", "[attn][mha]") {
    MultiHeadSelfAttention mha(4, 2, /*seed=*/0);
    Tensor xt = randn({3, 4});
    Variable x(std::move(xt), true);
    auto out = mha.forward(x);
    sum(out).backward();

    for (auto* p : mha.parameters()) {
        REQUIRE(p->grad().numel() > 0);
        // At least one grad element is non-zero.
        bool any_nonzero = false;
        for (float v : p->grad().data_as<float>())
            if (v != 0.0f) { any_nonzero = true; break; }
        REQUIRE(any_nonzero);
    }
}

TEST_CASE("MHA — causal: position 0 output depends only on token 0", "[attn][mha]") {
    // With causal masking, output[0] must not change when tokens 1+ change.
    MultiHeadSelfAttention mha(4, 1, /*seed=*/7);

    Tensor t1 = zeros({3, 4}); auto s1 = t1.data_as<float>();
    Tensor t2 = zeros({3, 4}); auto s2 = t2.data_as<float>();
    // Same token 0, different tokens 1 and 2.
    for (std::size_t j = 0; j < 4u; ++j) { s1[j] = s2[j] = 0.5f; }  // row 0
    for (std::size_t j = 0; j < 4u; ++j) { s1[4 + j] = 1.0f; s2[4 + j] = -1.0f; }
    for (std::size_t j = 0; j < 4u; ++j) { s1[8 + j] = 0.3f; s2[8 + j] =  2.0f; }

    Variable x1(std::move(t1), false), x2(std::move(t2), false);
    auto out1 = mha.forward(x1, /*causal=*/true);
    auto out2 = mha.forward(x2, /*causal=*/true);

    const auto o1 = out1.data().data_as<float>();
    const auto o2 = out2.data().data_as<float>();
    for (std::size_t j = 0; j < 4u; ++j)
        REQUIRE_THAT(o1[j], WithinAbs(o2[j], 1e-5f));
}

TEST_CASE("MHA — non-causal differs from causal for T>1", "[attn][mha]") {
    MultiHeadSelfAttention mha(4, 1, /*seed=*/3);
    Tensor xt = randn({3, 4});
    Variable x1(xt, false), x2(xt, false);
    auto causal_out    = mha.forward(x1, /*causal=*/true);
    auto noncausal_out = mha.forward(x2, /*causal=*/false);

    // Outputs should differ (causal mask changes the attention pattern).
    bool any_diff = false;
    const auto c = causal_out.data().data_as<float>();
    const auto n = noncausal_out.data().data_as<float>();
    for (std::size_t i = 0; i < 12u; ++i)
        if (std::abs(c[i] - n[i]) > 1e-6f) { any_diff = true; break; }
    REQUIRE(any_diff);
}

TEST_CASE("MHA — parameters() returns 4 * num_heads pointers", "[attn][mha]") {
    MultiHeadSelfAttention mha(8, 4);
    REQUIRE(mha.parameters().size() == 16u);
}
