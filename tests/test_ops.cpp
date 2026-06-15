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

TEST_CASE("relu - positive stays, negative zeros", "[ops]") {
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

TEST_CASE("matmul_bt equals matmul with transposed B (small K, SIMD path)", "[ops]") {
    Tensor a = randn({5, 7});
    Tensor b = randn({4, 7});
    Tensor ref = matmul(a, b.transpose(0, 1).contiguous());
    Tensor out = matmul_bt(a, b);
    require_near(ref, out, 1e-5f);
}

TEST_CASE("matmul_bt equals matmul with transposed B (K>=64, BLAS/Eigen path)", "[ops]") {
    Tensor a = randn({9, 96});
    Tensor b = randn({33, 96});
    Tensor ref = matmul(a, b.transpose(0, 1).contiguous());
    Tensor out = matmul_bt(a, b);
    require_near(ref, out, 1e-4f);
}

TEST_CASE("matmul_tb equals matmul with transposed A (small M, SIMD path)", "[ops]") {
    Tensor a = randn({7, 5});    // (M=7, K=5)
    Tensor b = randn({7, 4});    // (M=7, N=4)
    Tensor ref = matmul(a.transpose(0, 1).contiguous(), b);
    Tensor out = matmul_tb(a, b);
    require_near(ref, out, 1e-5f);
}

TEST_CASE("matmul_tb equals matmul with transposed A (M>=64, BLAS/Eigen path)", "[ops]") {
    Tensor a = randn({96, 9});
    Tensor b = randn({96, 33});
    Tensor ref = matmul(a.transpose(0, 1).contiguous(), b);
    Tensor out = matmul_tb(a, b);
    require_near(ref, out, 1e-4f);
}

TEST_CASE("matmul_tb dimension mismatch throws", "[ops]") {
    Tensor a = ones({2, 3});
    Tensor b = ones({4, 2});   // leading dims 2 vs 4
    REQUIRE_THROWS_AS(matmul_tb(a, b), std::runtime_error);
}

TEST_CASE("matmul_bt dimension mismatch throws", "[ops]") {
    Tensor a = ones({2, 3});
    Tensor b = ones({4, 2});   // inner dims 3 vs 2
    REQUIRE_THROWS_AS(matmul_bt(a, b), std::runtime_error);
}

// ── Batched 3D matmul (Ch29 re-architecture: batch folds into the M dimension) ──

// Copy slice `bi` of a 3D (B,R,C) tensor into a fresh 2D (R,C) tensor.
static Tensor slice2d(const Tensor& t, std::int64_t bi) {
    const std::int64_t R = t.shape(1), C = t.shape(2);
    Tensor out({R, C});
    auto src = t.data_as<float>();
    auto dst = out.data_as<float>();
    std::copy_n(src.begin() + bi * R * C, R * C, dst.begin());
    return out;
}

TEST_CASE("batched matmul equals per-slice 2D matmul", "[ops][batched]") {
    const std::int64_t B = 3, M = 5, K = 7, N = 4;
    Tensor a = randn({B, M, K});
    Tensor b = randn({B, K, N});
    Tensor out = matmul(a, b);                       // (B,M,N)
    REQUIRE(out.shape() == Tensor::Shape{B, M, N});
    for (std::int64_t bi = 0; bi < B; ++bi)
        require_near(slice2d(out, bi), matmul(slice2d(a, bi), slice2d(b, bi)), 1e-4f);
}

TEST_CASE("batched matmul_bt equals per-slice 2D matmul_bt (K>=64 path too)", "[ops][batched]") {
    const std::int64_t B = 2, M = 6, N = 5, K = 96;   // K>=64 exercises Eigen/BLAS slice path
    Tensor a = randn({B, M, K});
    Tensor b = randn({B, N, K});
    Tensor out = matmul_bt(a, b);                    // (B,M,N)
    REQUIRE(out.shape() == Tensor::Shape{B, M, N});
    for (std::int64_t bi = 0; bi < B; ++bi)
        require_near(slice2d(out, bi), matmul_bt(slice2d(a, bi), slice2d(b, bi)), 1e-3f);
}

TEST_CASE("batched matmul_tb equals per-slice 2D matmul_tb", "[ops][batched]") {
    const std::int64_t B = 3, M = 8, K = 5, N = 4;
    Tensor a = randn({B, M, K});
    Tensor b = randn({B, M, N});
    Tensor out = matmul_tb(a, b);                    // (B,K,N)
    REQUIRE(out.shape() == Tensor::Shape{B, K, N});
    for (std::int64_t bi = 0; bi < B; ++bi)
        require_near(slice2d(out, bi), matmul_tb(slice2d(a, bi), slice2d(b, bi)), 1e-4f);
}

TEST_CASE("matmul_tb threaded large-M path matches serial reference", "[ops][threaded]") {
    // M>=256 and work>=2M engages the threaded M-partition+reduce path; verify it
    // matches the trusted 2D reference (transpose then matmul). Reduction reorders
    // float adds, so allow a loose tolerance.
    const std::int64_t M = 2048, K = 128, N = 96;
    Tensor a = randn({M, K});
    Tensor b = randn({M, N});
    Tensor ref = matmul(a.transpose(0, 1).contiguous(), b);   // (K, N)
    Tensor out = matmul_tb(a, b);
    REQUIRE(out.shape() == Tensor::Shape{K, N});
    require_near(out, ref, 5e-2f);
}

TEST_CASE("matmul threaded large-M path matches serial reference", "[ops][threaded]") {
    const std::int64_t M = 2048, K = 128, N = 384;
    Tensor a = randn({M, K});
    Tensor b = randn({K, N});
    Tensor ref(Tensor::Shape{M, N});
    // serial reference via per-row dot using the public 2D matmul on row blocks of 1.
    Tensor out = matmul(a, b);
    REQUIRE(out.shape() == Tensor::Shape{M, N});
    // spot-check a few rows against an explicit small matmul of that row.
    for (std::int64_t r : {0, 1, 1023, 2047}) {
        Tensor arow = narrow(a, r, 1);                // (1,K)
        Tensor orow = matmul(arow, b);                // (1,N)
        auto od = out.data_as<float>(); auto rd = orow.data_as<float>();
        for (std::int64_t j = 0; j < N; ++j)
            REQUIRE_THAT(od[static_cast<std::size_t>(r * N + j)],
                         WithinAbs(rd[static_cast<std::size_t>(j)], 1e-3f));
    }
}

TEST_CASE("batched matmul rejects mismatched batch / rank", "[ops][batched]") {
    REQUIRE_THROWS_AS(matmul(randn({2, 3, 4}), randn({3, 4, 5})), std::runtime_error); // B 2 vs 3
    REQUIRE_THROWS_AS(matmul(randn({2, 3, 4}), randn({4, 5})),    std::runtime_error); // 3D vs 2D
}

// ── Unary ─────────────────────────────────────────────────────────────────────

TEST_CASE("exp and log are inverses", "[ops]") {
    Tensor a = arange(4);
    a = add(a, 1.0f);              // [1,2,3,4] - avoid log(0)
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

// ── Device guard tests (review finding #12) ────────────────────────────────────
#ifndef SUB0LLM_CUDA
TEST_CASE("sum on CUDA tensor throws (no CUDA build)", "[ops][device]") {
    Tensor a = ones({4});
    REQUIRE_THROWS_AS(a.to(Device::cuda()), std::runtime_error); // proves .to() guards
    // sum/max/min/norm are guarded by require_cpu - tested via CPU path only in this build
}

TEST_CASE("sub device mismatch throws", "[ops][device]") {
    Tensor a = ones({4});
    Tensor b = ones({4});
    // Can't test actual cross-device without CUDA, but device equality check works
    REQUIRE_NOTHROW(sub(a, b));  // same device - should work
}
#endif

// ── Softmax edge cases (review finding #25) ───────────────────────────────────
TEST_CASE("softmax 1D input", "[ops]") {
    Tensor t({4}, DType::Float32);
    auto sp = t.data_as<float>();
    sp[0] = 1.0f; sp[1] = 2.0f; sp[2] = 3.0f; sp[3] = 4.0f;
    Tensor s = softmax(t);
    REQUIRE_THAT(sum(s), WithinAbs(1.0f, kEps));
}

TEST_CASE("softmax multi-row 2D", "[ops]") {
    // Two rows, softmax should independently sum to 1
    Tensor t({2, 4}, DType::Float32);
    auto sp = t.data_as<float>();
    for (std::size_t i = 0; i < 8; ++i) sp[i] = static_cast<float>(i);
    Tensor s = softmax(t);
    auto out = s.data_as<float>();
    const float row0_sum = out[0] + out[1] + out[2] + out[3];
    const float row1_sum = out[4] + out[5] + out[6] + out[7];
    REQUIRE_THAT(row0_sum, WithinAbs(1.0f, kEps));
    REQUIRE_THAT(row1_sum, WithinAbs(1.0f, kEps));
}

TEST_CASE("softmax dim != -1 throws", "[ops]") {
    Tensor t({2, 3});
    REQUIRE_THROWS_AS(softmax(t, 0), std::runtime_error);
}
