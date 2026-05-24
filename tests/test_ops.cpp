#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/ops.hpp"

using namespace sub0llm;
using namespace sub0llm::ops;
using Catch::Matchers::WithinAbs;

static constexpr float kEps = 1e-5f;

// Helper: compare two tensors element-wise within tolerance.
static void require_near(const Tensor& a, const Tensor& b, float eps = kEps) {
    REQUIRE(a.shape() == b.shape());
    auto sa = a.data_as<float>();
    auto sb = b.data_as<float>();
    for (std::size_t i = 0; i < sa.size(); ++i)
        REQUIRE_THAT(sa[i], WithinAbs(sb[i], eps));
}

// ── Element-wise ──────────────────────────────────────────────────────────────

TEST_CASE("add tensors", "[ops]") {
    Tensor a = arange(4);           // [0,1,2,3]
    Tensor b = ones({4});           // [1,1,1,1]
    Tensor c = add(a, b);           // [1,2,3,4]
    auto sp = c.data_as<float>();
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(sp[i], WithinAbs(static_cast<float>(i) + 1.0f, kEps));
}

TEST_CASE("sub tensors", "[ops]") {
    Tensor a = ones({3});
    Tensor b = ones({3});
    Tensor c = sub(a, b);
    for (float v : c.data_as<float>()) REQUIRE_THAT(v, WithinAbs(0.0f, kEps));
}

TEST_CASE("mul tensors", "[ops]") {
    Tensor a = arange(4);
    Tensor b = arange(4);
    Tensor c = mul(a, b);           // [0,1,4,9]
    auto sp = c.data_as<float>();
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE_THAT(sp[i], WithinAbs(static_cast<float>(i) * static_cast<float>(i), kEps));
}

TEST_CASE("scalar add", "[ops]") {
    Tensor a = zeros({4});
    Tensor b = add(a, 5.0f);
    for (float v : b.data_as<float>()) REQUIRE_THAT(v, WithinAbs(5.0f, kEps));
}

TEST_CASE("scalar mul", "[ops]") {
    Tensor a = ones({4});
    Tensor b = mul(a, 3.0f);
    for (float v : b.data_as<float>()) REQUIRE_THAT(v, WithinAbs(3.0f, kEps));
}

// ── Reductions ────────────────────────────────────────────────────────────────

TEST_CASE("sum", "[ops]") {
    Tensor a = ones({5});
    REQUIRE_THAT(sum(a), WithinAbs(5.0f, kEps));
}

TEST_CASE("mean", "[ops]") {
    Tensor a = arange(5);           // [0,1,2,3,4] → mean = 2.0
    REQUIRE_THAT(mean(a), WithinAbs(2.0f, kEps));
}

TEST_CASE("max", "[ops]") {
    Tensor a = arange(6);
    REQUIRE_THAT(max(a), WithinAbs(5.0f, kEps));
}

TEST_CASE("min", "[ops]") {
    Tensor a = arange(6);
    REQUIRE_THAT(min(a), WithinAbs(0.0f, kEps));
}

// ── Activations ───────────────────────────────────────────────────────────────

TEST_CASE("relu — positive stays, negative zeros", "[ops]") {
    Tensor t({4}, DType::Float32);
    {
        auto sp = t.data_as<float>();
        sp[0] = -2.0f; sp[1] = -0.5f; sp[2] = 0.0f; sp[3] = 3.0f;
    }
    Tensor r = relu(t);
    auto sp = r.data_as<float>();
    REQUIRE_THAT(sp[0], WithinAbs(0.0f,  kEps));
    REQUIRE_THAT(sp[1], WithinAbs(0.0f,  kEps));
    REQUIRE_THAT(sp[2], WithinAbs(0.0f,  kEps));
    REQUIRE_THAT(sp[3], WithinAbs(3.0f,  kEps));
}

TEST_CASE("softmax sums to 1", "[ops]") {
    Tensor logits({1, 4}, DType::Float32);
    {
        auto sp = logits.data_as<float>();
        sp[0] = 1.0f; sp[1] = 2.0f; sp[2] = 3.0f; sp[3] = 4.0f;
    }
    Tensor probs = softmax(logits);
    REQUIRE_THAT(sum(probs), WithinAbs(1.0f, kEps));
    for (float v : probs.data_as<float>())
        REQUIRE(v > 0.0f);
}

TEST_CASE("sigmoid output in (0,1)", "[ops]") {
    Tensor t({5}, DType::Float32);
    {
        auto sp = t.data_as<float>();
        sp[0] = -100.0f; sp[1] = -1.0f; sp[2] = 0.0f; sp[3] = 1.0f; sp[4] = 100.0f;
    }
    Tensor s = sigmoid(t);
    for (float v : s.data_as<float>()) {
        REQUIRE(v >= 0.0f);
        REQUIRE(v <= 1.0f);
    }
    // sigmoid(0) ≈ 0.5
    REQUIRE_THAT(s.data_as<float>()[2], WithinAbs(0.5f, kEps));
    // sigmoid(1) > sigmoid(-1): monotonically increasing
    REQUIRE(s.data_as<float>()[3] > s.data_as<float>()[1]);
}

TEST_CASE("gelu: gelu(0) == 0", "[ops]") {
    Tensor t = zeros({1});
    Tensor g = gelu(t);
    REQUIRE_THAT(g.data_as<float>()[0], WithinAbs(0.0f, kEps));
}

// ── Matmul ────────────────────────────────────────────────────────────────────

TEST_CASE("matmul identity", "[ops]") {
    // I @ A == A
    Tensor I = zeros({3, 3});
    I.data_as<float>()[0] = 1.0f;
    I.data_as<float>()[4] = 1.0f;
    I.data_as<float>()[8] = 1.0f;

    Tensor A = randn({3, 3});
    Tensor B = matmul(I, A);

    require_near(A, B, 1e-5f);
}

TEST_CASE("matmul (2,3)x(3,2) shape", "[ops]") {
    Tensor a = ones({2, 3});
    Tensor b = ones({3, 2});
    Tensor c = matmul(a, b);

    REQUIRE(c.shape(0) == 2);
    REQUIRE(c.shape(1) == 2);
    // ones × ones = 3 in each cell (K=3 additions of 1×1)
    for (float v : c.data_as<float>())
        REQUIRE_THAT(v, WithinAbs(3.0f, kEps));
}

TEST_CASE("matmul dimension mismatch throws", "[ops]") {
    Tensor a = ones({2, 3});
    Tensor b = ones({4, 2});
    REQUIRE_THROWS_AS(matmul(a, b), std::runtime_error);
}

// ── Unary ─────────────────────────────────────────────────────────────────────

TEST_CASE("exp and log are inverses", "[ops]") {
    Tensor a = arange(4);
    a = add(a, 1.0f);              // [1,2,3,4] — avoid log(0)
    Tensor b = log(exp(a));
    require_near(a, b, 1e-5f);
}

TEST_CASE("neg", "[ops]") {
    Tensor a = ones({3});
    Tensor b = neg(a);
    for (float v : b.data_as<float>()) REQUIRE_THAT(v, WithinAbs(-1.0f, kEps));
}

TEST_CASE("norm of ones vector", "[ops]") {
    Tensor a = ones({4});
    REQUIRE_THAT(norm(a), WithinAbs(2.0f, kEps));  // sqrt(4)
}
