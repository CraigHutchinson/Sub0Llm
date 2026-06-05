#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/autograd/embedding_ops.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/positional_encoding.hpp"

#include <cmath>
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

static Tensor int_tensor(std::vector<int32_t> vals) {
    Tensor t = zeros({static_cast<int64_t>(vals.size())}, DType::Int32);
    auto s = t.data_as<int32_t>();
    for (std::size_t i = 0; i < vals.size(); ++i) s[i] = vals[i];
    return t;
}

// ── embedding_lookup forward ──────────────────────────────────────────────────

TEST_CASE("embedding_lookup - forward gathers correct rows", "[embed]") {
    // Weight: [[1,2],[3,4],[5,6]]  indices: [2, 0, 1]
    auto W = leaf2d({1.f,2.f, 3.f,4.f, 5.f,6.f}, 3, 2);
    auto idx = int_tensor({2, 0, 1});
    auto out = embedding_lookup(W, idx);
    const auto od = out.data().data_as<float>();
    REQUIRE_THAT(od[0], WithinAbs(5.0f, 1e-5f));  // row 2
    REQUIRE_THAT(od[1], WithinAbs(6.0f, 1e-5f));
    REQUIRE_THAT(od[2], WithinAbs(1.0f, 1e-5f));  // row 0
    REQUIRE_THAT(od[4], WithinAbs(3.0f, 1e-5f));  // row 1
}

TEST_CASE("embedding_lookup - output shape 1D indices", "[embed]") {
    auto W   = leaf2d(std::vector<float>(10 * 4, 0.f), 10, 4);
    auto idx = int_tensor({0, 3, 7});
    auto out = embedding_lookup(W, idx);
    REQUIRE(out.data().ndim() == 2);
    REQUIRE(out.data().shape()[0] == 3);
    REQUIRE(out.data().shape()[1] == 4);
}

TEST_CASE("embedding_lookup - out-of-range index throws", "[embed]") {
    auto W   = leaf2d(std::vector<float>(4 * 2, 0.f), 4, 2);
    auto idx = int_tensor({5});   // vocab_size=4, 5 out of range
    REQUIRE_THROWS_AS(embedding_lookup(W, idx), std::runtime_error);
}

TEST_CASE("sinusoidal_encoding - non-positive seq_len throws", "[embed][pe]") {
    REQUIRE_THROWS_AS(sinusoidal_encoding(0,  8), std::runtime_error);
    REQUIRE_THROWS_AS(sinusoidal_encoding(-1, 8), std::runtime_error);
}

TEST_CASE("LearnedPE - non-positive seq_len throws", "[embed][pe]") {
    LearnedPositionalEncoding lpe(16, 4);
    REQUIRE_THROWS_AS(lpe.forward(0),  std::runtime_error);
    REQUIRE_THROWS_AS(lpe.forward(-1), std::runtime_error);
}

// ── embedding_lookup backward ─────────────────────────────────────────────────

TEST_CASE("embedding_lookup - backward gathers correct grad rows", "[embed]") {
    // Weight (3,2), indices [0,2]: upstream = all-ones → each row gets (1,1)
    auto W   = leaf2d({0.f,0.f, 0.f,0.f, 0.f,0.f}, 3, 2);
    auto idx = int_tensor({0, 2});
    auto out = embedding_lookup(W, idx);
    sum(out).backward();
    const auto gd = W.grad().data_as<float>();
    REQUIRE_THAT(gd[0], WithinAbs(1.0f, 1e-5f));  // row 0, dim 0
    REQUIRE_THAT(gd[1], WithinAbs(1.0f, 1e-5f));  // row 0, dim 1
    REQUIRE_THAT(gd[2], WithinAbs(0.0f, 1e-5f));  // row 1 unused
    REQUIRE_THAT(gd[4], WithinAbs(1.0f, 1e-5f));  // row 2, dim 0
}

TEST_CASE("embedding_lookup - repeated token accumulates gradient", "[embed]") {
    // indices [1, 1]: token 1 appears twice, so grad[1] should be 2× upstream.
    auto W   = leaf2d({0.f,0.f, 0.f,0.f, 0.f,0.f}, 3, 2);
    auto idx = int_tensor({1, 1});
    auto out = embedding_lookup(W, idx);
    sum(out).backward();
    const auto gd = W.grad().data_as<float>();
    REQUIRE_THAT(gd[2], WithinAbs(2.0f, 1e-5f));  // row 1, dim 0 (accumulated)
    REQUIRE_THAT(gd[3], WithinAbs(2.0f, 1e-5f));  // row 1, dim 1
    REQUIRE_THAT(gd[0], WithinAbs(0.0f, 1e-5f));  // row 0 unused
}

TEST_CASE("embedding_lookup - unused tokens have zero gradient", "[embed]") {
    auto W   = leaf2d(std::vector<float>(5 * 3, 1.f), 5, 3);
    auto idx = int_tensor({0, 2});
    auto out = embedding_lookup(W, idx);
    sum(out).backward();
    const auto gd = W.grad().data_as<float>();
    // Rows 1, 3, 4 are unused → all-zero grad
    for (std::size_t r : {1u, 3u, 4u})
        for (std::size_t j = 0; j < 3u; ++j)
            REQUIRE_THAT(gd[r * 3u + j], WithinAbs(0.0f, 1e-5f));
}

// ── Embedding class ───────────────────────────────────────────────────────────

TEST_CASE("Embedding - weight shape is (vocab, dim)", "[embed]") {
    Embedding e(20, 8);
    REQUIRE(e.vocab_size() == 20);
    REQUIRE(e.embed_dim()  == 8);
    REQUIRE(e.weight().data().shape()[0] == 20);
    REQUIRE(e.weight().data().shape()[1] == 8);
}

TEST_CASE("Embedding - requires_grad=true on weight", "[embed]") {
    Embedding e(10, 4);
    REQUIRE(e.weight().requires_grad());
}

TEST_CASE("Embedding - forward matches manual lookup", "[embed]") {
    Embedding e(5, 3, 0);
    auto idx = int_tensor({2, 4});
    auto out = e.forward(idx);
    // Manual lookup
    const auto wd = e.weight().data().data_as<float>();
    const auto od = out.data().data_as<float>();
    for (std::size_t j = 0; j < 3u; ++j)
        REQUIRE_THAT(od[j],      WithinAbs(wd[2u * 3u + j], 1e-5f));  // row 2
    for (std::size_t j = 0; j < 3u; ++j)
        REQUIRE_THAT(od[3u + j], WithinAbs(wd[4u * 3u + j], 1e-5f));  // row 4
}

TEST_CASE("Embedding - invalid vocab/dim throws", "[embed]") {
    REQUIRE_THROWS_AS(Embedding(0, 4),  std::runtime_error);
    REQUIRE_THROWS_AS(Embedding(10, 0), std::runtime_error);
}

// ── sinusoidal_encoding ───────────────────────────────────────────────────────

TEST_CASE("sinusoidal_encoding - shape is (T, D)", "[embed][pe]") {
    auto pe = sinusoidal_encoding(8, 16);
    REQUIRE(pe.shape()[0] == 8);
    REQUIRE(pe.shape()[1] == 16);
}

TEST_CASE("sinusoidal_encoding - pos 0, even dims are 0", "[embed][pe]") {
    auto pe = sinusoidal_encoding(4, 8);
    const auto pd = pe.data_as<float>();
    // sin(0) = 0 for all even dims
    for (std::size_t i = 0; i < 4u; ++i)
        REQUIRE_THAT(pd[i * 2u], WithinAbs(0.0f, 1e-5f));
}

TEST_CASE("sinusoidal_encoding - pos 0, odd dims are 1", "[embed][pe]") {
    auto pe = sinusoidal_encoding(4, 8);
    const auto pd = pe.data_as<float>();
    // cos(0) = 1 for all odd dims
    for (std::size_t i = 0; i < 4u; ++i)
        REQUIRE_THAT(pd[i * 2u + 1u], WithinAbs(1.0f, 1e-5f));
}

TEST_CASE("sinusoidal_encoding - odd embed_dim throws", "[embed][pe]") {
    REQUIRE_THROWS_AS(sinusoidal_encoding(4, 7), std::runtime_error);
}

TEST_CASE("sinusoidal_encoding - rows have equal L2 norm", "[embed][pe]") {
    // All row norms should equal sqrt(D/2) (each sin²+cos² pair = 1).
    const int64_t T = 6, D = 12;
    auto pe = sinusoidal_encoding(T, D);
    const auto pd = pe.data_as<float>();
    const float expected = std::sqrt(static_cast<float>(D) / 2.0f);
    for (std::size_t pos = 0; pos < static_cast<std::size_t>(T); ++pos) {
        float sq = 0.0f;
        for (std::size_t j = 0; j < static_cast<std::size_t>(D); ++j)
            sq += pd[pos * static_cast<std::size_t>(D) + j] * pd[pos * static_cast<std::size_t>(D) + j];
        REQUIRE_THAT(std::sqrt(sq), WithinAbs(expected, 1e-4f));
    }
}

// ── LearnedPositionalEncoding ─────────────────────────────────────────────────

TEST_CASE("LearnedPE - weight shape is (max_seq_len, dim)", "[embed][pe]") {
    LearnedPositionalEncoding lpe(64, 8);
    REQUIRE(lpe.max_seq_len() == 64);
    REQUIRE(lpe.embed_dim()   == 8);
    REQUIRE(lpe.weight().data().shape()[0] == 64);
}

TEST_CASE("LearnedPE - forward shape is (seq_len, dim)", "[embed][pe]") {
    LearnedPositionalEncoding lpe(64, 8);
    auto out = lpe.forward(10);
    REQUIRE(out.data().shape()[0] == 10);
    REQUIRE(out.data().shape()[1] == 8);
}

TEST_CASE("LearnedPE - seq_len > max throws", "[embed][pe]") {
    LearnedPositionalEncoding lpe(8, 4);
    REQUIRE_THROWS_AS(lpe.forward(9), std::runtime_error);
}

// ── apply_rope ────────────────────────────────────────────────────────────────

TEST_CASE("apply_rope - output shape matches input", "[embed][rope]") {
    Tensor x = randn({6, 8});
    auto out = apply_rope(x);
    REQUIRE(out.shape()[0] == 6);
    REQUIRE(out.shape()[1] == 8);
}

TEST_CASE("apply_rope - position 0 is unchanged (theta=0 -> rotation is identity)", "[embed][rope]") {
    Tensor x = randn({4, 8});
    auto out = apply_rope(x);
    const auto xs  = x.data_as<float>();
    const auto os  = out.data_as<float>();
    // At pos=0, all thetas are 0 → cos=1, sin=0 → identity rotation
    for (std::size_t j = 0; j < 8u; ++j)
        REQUIRE_THAT(os[j], WithinAbs(xs[j], 1e-5f));
}

TEST_CASE("apply_rope - rotation preserves vector norms", "[embed][rope]") {
    Tensor x = randn({4, 8});
    auto out = apply_rope(x);
    const auto xs  = x.data_as<float>();
    const auto os  = out.data_as<float>();
    for (std::size_t pos = 0; pos < 4u; ++pos) {
        float sq_in = 0.0f, sq_out = 0.0f;
        for (std::size_t j = 0; j < 8u; ++j) {
            sq_in  += xs[pos * 8u + j] * xs[pos * 8u + j];
            sq_out += os[pos * 8u + j] * os[pos * 8u + j];
        }
        REQUIRE_THAT(std::sqrt(sq_out), WithinAbs(std::sqrt(sq_in), 1e-4f));
    }
}

TEST_CASE("apply_rope - odd embed_dim throws", "[embed][rope]") {
    Tensor x = zeros({4, 7});
    REQUIRE_THROWS_AS(apply_rope(x), std::runtime_error);
}
