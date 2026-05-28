#include "sub0llm/nn/math_nodes.hpp"
#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/tokenizer/numeric_tokenizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/autograd/ops.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static NumericTokenizer make_small_ntok() {
    auto bpe = BPETokenizer::train({"hello world"}, /*vocab_size=*/20);
    return NumericTokenizer(std::move(bpe));
}

static Variable make_rand_var(int64_t T, int64_t D) {
    Tensor t({T, D}, DType::Float32);
    auto sp = t.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i)
        sp[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;
    return Variable(t, /*requires_grad=*/true);
}

// ── NumericTokenizer: vocab sizes ─────────────────────────────────────────────

TEST_CASE("NumericTokenizer: vocab sizes") {
    auto ntok = make_small_ntok();
    REQUIRE(ntok.total_vocab_size() == ntok.bpe_vocab_size() + 65538);
    REQUIRE(ntok.numeric_range_start() == static_cast<NumericTokenizer::TokenId>(ntok.bpe_vocab_size()));

    // Boundaries: just below numeric range is not numeric
    auto just_below = static_cast<NumericTokenizer::TokenId>(ntok.numeric_range_start() - 1);
    REQUIRE_FALSE(ntok.is_numeric(just_below));

    // First numeric token
    REQUIRE(ntok.is_numeric(ntok.numeric_range_start()));

    // Last valid numeric (before nan/overflow sentinels)
    auto last_int = static_cast<NumericTokenizer::TokenId>(
        ntok.numeric_range_start() + 65535);
    REQUIRE(ntok.is_numeric(last_int));

    // nan and overflow tokens are also in the numeric range
    REQUIRE(ntok.is_numeric(ntok.nan_token()));
    REQUIRE(ntok.is_numeric(ntok.overflow_token()));
}

// ── NumericTokenizer: encode_int round-trip ───────────────────────────────────

TEST_CASE("NumericTokenizer: encode_int round-trip") {
    auto ntok = make_small_ntok();

    auto id42 = ntok.encode_int(42);
    REQUIRE(ntok.is_numeric(id42));
    REQUIRE_FALSE(ntok.is_nan_token(id42));
    REQUIRE_FALSE(ntok.is_overflow_token(id42));
    REQUIRE(ntok.numeric_value(id42) == Catch::Approx(42.0f));

    auto id_neg = ntok.encode_int(-100);
    REQUIRE(ntok.numeric_value(id_neg) == Catch::Approx(-100.0f));

    auto id_max = ntok.encode_int(32767);
    REQUIRE(ntok.numeric_value(id_max) == Catch::Approx(32767.0f));

    // Out of range → overflow_token
    auto id_ovf = ntok.encode_int(32768);
    REQUIRE(id_ovf == ntok.overflow_token());
}

// ── NumericTokenizer: nan and overflow tokens ─────────────────────────────────

TEST_CASE("NumericTokenizer: nan and overflow tokens") {
    auto ntok = make_small_ntok();

    REQUIRE(ntok.is_nan_token(ntok.nan_token()));
    REQUIRE(ntok.is_overflow_token(ntok.overflow_token()));

    REQUIRE(std::isnan(ntok.numeric_value(ntok.nan_token())));
    REQUIRE(std::isinf(ntok.numeric_value(ntok.overflow_token())));
}

// ── NumericTokenizer: encode splits words ─────────────────────────────────────

TEST_CASE("NumericTokenizer: encode splits words") {
    auto ntok = make_small_ntok();
    auto ids = ntok.encode("42 plus 7");

    // Must have at least 3 tokens (42, "plus", 7)
    REQUIRE(ids.size() >= 3);

    // First token: 42 → numeric
    REQUIRE(ntok.is_numeric(ids.front()));
    REQUIRE(ntok.numeric_value(ids.front()) == Catch::Approx(42.0f));

    // Last token: 7 → numeric
    REQUIRE(ntok.is_numeric(ids.back()));
    REQUIRE(ntok.numeric_value(ids.back()) == Catch::Approx(7.0f));
}

// ── apply_math_op: addition ───────────────────────────────────────────────────

TEST_CASE("apply_math_op: addition") {
    auto r = apply_math_op(RouteType::Add, 42.f, 7.f);
    REQUIRE(r.value == Catch::Approx(49.f));
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

// ── apply_math_op: subtraction ────────────────────────────────────────────────
// Convention: a = LEFT operand, b = RIGHT operand  ("a op b" in expression order).
// MathLayer::forward must pass (lhs=earlier_token, rhs=most_recent_token).
// Regression: "5 - 3" must yield 2, not -2 (was broken before the call-site fix).

TEST_CASE("apply_math_op: subtraction") {
    auto r = apply_math_op(RouteType::Sub, 100.f, 65.f);
    REQUIRE(r.value == Catch::Approx(35.f));
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

TEST_CASE("apply_math_op: subtraction left-right order") {
    // 5 - 3 = 2  (a=5=LEFT, b=3=RIGHT)
    auto r = apply_math_op(RouteType::Sub, 5.f, 3.f);
    REQUIRE(r.value == Catch::Approx(2.f));
    // Regression guard: b - a = 3 - 5 = -2 is the wrong result
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

// ── apply_math_op: multiplication ────────────────────────────────────────────

TEST_CASE("apply_math_op: multiplication") {
    auto r = apply_math_op(RouteType::Mul, 12.f, 5.f);
    REQUIRE(r.value == Catch::Approx(60.f));
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

// ── apply_math_op: division exact ────────────────────────────────────────────
// Convention: a = dividend (LEFT), b = divisor (RIGHT). "a / b" in expression order.

TEST_CASE("apply_math_op: division exact") {
    auto r = apply_math_op(RouteType::Div, 100.f, 4.f);
    REQUIRE(r.value == Catch::Approx(25.0f));
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

TEST_CASE("apply_math_op: division left-right order") {
    // 6 / 3 = 2  (a=6=LEFT/dividend, b=3=RIGHT/divisor)
    auto r = apply_math_op(RouteType::Div, 6.f, 3.f);
    REQUIRE(r.value == Catch::Approx(2.f));
}

// ── apply_math_op: division by zero ──────────────────────────────────────────

TEST_CASE("apply_math_op: division by zero") {
    auto r = apply_math_op(RouteType::Div, 7.f, 0.f);
    REQUIRE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

// ── apply_math_op: IsLessThan ─────────────────────────────────────────────────
// Convention: a = LEFT operand, b = RIGHT operand. Returns 1 if a < b, else 0.

TEST_CASE("apply_math_op: IsLessThan true") {
    auto r = apply_math_op(RouteType::IsLessThan, 3.f, 7.f);
    REQUIRE(r.value == Catch::Approx(1.0f));
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

TEST_CASE("apply_math_op: IsLessThan false") {
    auto r = apply_math_op(RouteType::IsLessThan, 7.f, 3.f);
    REQUIRE(r.value == Catch::Approx(0.0f));
    REQUIRE_FALSE(r.is_nan);
    REQUIRE_FALSE(r.is_overflow);
}

// ── apply_math_op: overflow ───────────────────────────────────────────────────

TEST_CASE("apply_math_op: overflow") {
    auto r = apply_math_op(RouteType::Mul, 300.f, 300.f);
    REQUIRE(r.is_overflow);
    REQUIRE_FALSE(r.is_nan);
}

// ── MathLayer: forward shape ──────────────────────────────────────────────────

TEST_CASE("MathLayer: forward shape") {
    const int64_t D = 16, T = 4;
    auto ntok = make_small_ntok();
    const int64_t total_V = ntok.total_vocab_size();

    MathLayer layer(D, /*d_ff=*/0, /*seed=*/7);

    Variable x = make_rand_var(T, D);

    // Build all-NaN register (no numeric tokens)
    std::vector<float> reg(static_cast<std::size_t>(T),
                            std::numeric_limits<float>::quiet_NaN());

    // Embedding weight: (total_V, D)
    Tensor emb_t({total_V, D}, DType::Float32);
    auto sp = emb_t.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(total_V * D); ++i)
        sp[i] = static_cast<float>(i % 5) * 0.05f;
    Variable emb_weight(emb_t, /*requires_grad=*/true);

    auto out = layer.forward(x, reg, emb_weight, ntok);

    REQUIRE(out.data().shape(0) == T);
    REQUIRE(out.data().shape(1) == D);
}

// ── MathGPT: vocab and layer sizes ───────────────────────────────────────────

TEST_CASE("MathGPT: vocab and layer sizes") {
    MathGPT m(200, 32, 2, 1, 4);
    REQUIRE(m.vocab_size() == 200);
    REQUIRE(m.embed_dim()  == 32);
    REQUIRE(m.num_layers() == 4);
    // l_math = round(0.7 * 4) = round(2.8) = 3
    REQUIRE(m.l_math() == 3);
}

// ── MathGPT: forward and forward_math same shape ──────────────────────────────

TEST_CASE("MathGPT: forward and forward_math same shape") {
    auto ntok = make_small_ntok();
    const int64_t V = ntok.total_vocab_size();

    MathGPT m(V, 32, 2, 1, 2, -1, 0, /*seed=*/7);

    const int64_t T = 5;
    Tensor ids({T}, DType::Int32);
    auto sp = ids.data_as<int32_t>();
    for (int64_t i = 0; i < T; ++i) {
        // Mix numeric and non-numeric tokens
        sp[static_cast<std::size_t>(i)] = (i % 2 == 0)
            ? static_cast<int32_t>(ntok.encode_int(static_cast<int32_t>(i)))
            : static_cast<int32_t>(i % ntok.bpe_vocab_size());
    }

    auto out_base = m.forward(ids);
    auto out_math = m.forward_math(ids, ntok);

    REQUIRE(out_base.data().shape(0) == T);
    REQUIRE(out_base.data().shape(1) == V);
    REQUIRE(out_math.data().shape(0) == T);
    REQUIRE(out_math.data().shape(1) == V);
}

// ── MathGPT: default l_math placement ────────────────────────────────────────

TEST_CASE("MathGPT: default l_math placement") {
    // n_layers=6 → l_math = round(0.7*6) = round(4.2) = 4
    MathGPT m6(100, 16, 2, 1, 6);
    REQUIRE(m6.l_math() == 4);

    // n_layers=2 → l_math = round(0.7*2) = round(1.4) = 1
    MathGPT m2(100, 16, 2, 1, 2);
    REQUIRE(m2.l_math() == 1);
}

// ── MathGPT: math_parameters subset of parameters ─────────────────────────────

TEST_CASE("MathGPT: math_parameters subset of parameters") {
    MathGPT m(100, 16, 2, 1, 4);
    auto all_p  = m.parameters();
    auto math_p = m.math_parameters();
    REQUIRE(math_p.size() < all_p.size());
}
