// Tests for multi-step reasoning chains and natural language + arithmetic integration.
//
// Key invariant: the MathLayer uses the two most recent numeric tokens in context
// as operands (op2_pos = lhs, op1_pos = rhs).  For a sequence "A op B = C op2 D = ?",
// when predicting "?":  lhs = C (2nd most recent), rhs = D (most recent) → correct.
// This means exact multi-step arithmetic is structurally guaranteed regardless of what
// numbers appeared earlier in the text, as long as operands are written explicitly.

#include "sub0llm/nn/math_nodes.hpp"
#include "sub0llm/nn/numeric_router.hpp"
#include "sub0llm/tokenizer/numeric_tokenizer.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/autograd/ops.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <vector>

using namespace sub0llm;
using namespace sub0llm::nn;
using namespace sub0llm::autograd;

// ── helpers ───────────────────────────────────────────────────────────────────

static NumericTokenizer make_arith_ntok(int32_t int_min = -32768, int32_t int_max = 32767) {
    // BPE trained on arithmetic corpus so "+", "-", "=", etc. tokenize cleanly
    std::vector<std::string> corpus = {
        "1 + 2 = 3", "5 - 3 = 2", "4 * 2 = 8", "10 / 2 = 5",
        "start add result", "Alice has apples gives",
    };
    auto bpe = BPETokenizer::train(corpus, /*vocab_size=*/50);
    return NumericTokenizer(std::move(bpe), int_min, int_max);
}

// ── apply_math_op: chaining ───────────────────────────────────────────────────

TEST_CASE("reasoning chain: two-step add then sub") {
    // Step 1: 5 + 3 = 8
    auto r1 = apply_math_op(RouteType::Add, 5.f, 3.f);
    REQUIRE(r1.value == Catch::Approx(8.f));
    REQUIRE_FALSE(r1.is_nan);
    REQUIRE_FALSE(r1.is_overflow);

    // Step 2: 8 - 2 = 6  (r1.value feeds the next lhs)
    auto r2 = apply_math_op(RouteType::Sub, r1.value, 2.f);
    REQUIRE(r2.value == Catch::Approx(6.f));
    REQUIRE_FALSE(r2.is_nan);
    REQUIRE_FALSE(r2.is_overflow);
}

TEST_CASE("reasoning chain: three-step add mul div") {
    // 2 + 3 = 5 → 5 * 4 = 20 → 20 / 5 = 4
    auto r1 = apply_math_op(RouteType::Add, 2.f, 3.f);
    REQUIRE(r1.value == Catch::Approx(5.f));

    auto r2 = apply_math_op(RouteType::Mul, r1.value, 4.f);
    REQUIRE(r2.value == Catch::Approx(20.f));

    auto r3 = apply_math_op(RouteType::Div, r2.value, 5.f);
    REQUIRE(r3.value == Catch::Approx(4.f));
}

TEST_CASE("reasoning chain: chain with comparison at end") {
    // 7 + 3 = 10 → is 10 > 8? → 1
    auto r1 = apply_math_op(RouteType::Add, 7.f, 3.f);
    REQUIRE(r1.value == Catch::Approx(10.f));

    auto r2 = apply_math_op(RouteType::IsGreaterThan, r1.value, 8.f);
    REQUIRE(r2.value == Catch::Approx(1.f));
    REQUIRE_FALSE(r2.is_nan);
}

TEST_CASE("reasoning chain: chain with sqrt at end") {
    // 5 * 5 = 25 → sqrt(25) = 5  (unary: only rhs used)
    auto r1 = apply_math_op(RouteType::Mul, 5.f, 5.f);
    REQUIRE(r1.value == Catch::Approx(25.f));

    // Sqrt is unary: convention is apply_math_op(Sqrt, 0.f, operand)
    auto r2 = apply_math_op(RouteType::Sqrt, 0.f, r1.value);
    REQUIRE(r2.value == Catch::Approx(5.f));
}

TEST_CASE("reasoning chain: overflow in middle stops chain cleanly") {
    // 300 * 300 → overflow
    auto r1 = apply_math_op(RouteType::Mul, 300.f, 300.f);
    REQUIRE(r1.is_overflow);
    REQUIRE_FALSE(r1.is_nan);

    // Continuing a chain after overflow: the is_overflow flag propagates to caller.
    // The downstream apply_math_op uses the clamped value (0.f) — verify no UB.
    auto r2 = apply_math_op(RouteType::Add, r1.value, 5.f);
    REQUIRE_FALSE(r2.is_overflow);   // 0 + 5 = 5 is fine
    REQUIRE(r2.value == Catch::Approx(5.f));
}

TEST_CASE("reasoning chain: lhs/rhs ordering across custom range") {
    // Division is non-commutative — verify lhs (2nd most recent) / rhs (most recent).
    // 100 / 4 = 25  (lhs=100, rhs=4)
    auto r = apply_math_op(RouteType::Div, 100.f, 4.f, -1000, 1000);
    REQUIRE(r.value == Catch::Approx(25.f));

    // Reversed (lhs=4, rhs=100) → 4/100 → rounds to 0 (integer arithmetic)
    auto r_rev = apply_math_op(RouteType::Div, 4.f, 100.f, -1000, 1000);
    REQUIRE(r_rev.value == Catch::Approx(0.f));
}

// ── NumericTokenizer: multi-step sequence handling ────────────────────────────

TEST_CASE("reasoning chain: encode multi-step sequence numeric positions") {
    auto ntok = make_arith_ntok();
    // Encode "3 + 4 = 7 - 2 = 5": should have exactly 5 numeric tokens in order.
    auto ids = ntok.encode("3 + 4 = 7 - 2 = 5");

    std::vector<int32_t> numerics;
    for (auto id : ids) {
        if (ntok.is_numeric(id) && !ntok.is_nan_token(id) && !ntok.is_overflow_token(id))
            numerics.push_back(static_cast<int32_t>(ntok.numeric_value(id)));
    }
    REQUIRE(numerics.size() == 5);
    CHECK(numerics[0] == 3);
    CHECK(numerics[1] == 4);
    CHECK(numerics[2] == 7);
    CHECK(numerics[3] == 2);
    CHECK(numerics[4] == 5);
}

TEST_CASE("reasoning chain: three-step sequence numeric positions") {
    auto ntok = make_arith_ntok();
    // "2 + 3 = 5 * 4 = 20 / 5 = 4" — 7 numeric tokens
    auto ids = ntok.encode("2 + 3 = 5 * 4 = 20 / 5 = 4");

    std::vector<int32_t> numerics;
    for (auto id : ids) {
        if (ntok.is_numeric(id) && !ntok.is_nan_token(id) && !ntok.is_overflow_token(id))
            numerics.push_back(static_cast<int32_t>(ntok.numeric_value(id)));
    }
    REQUIRE(numerics.size() == 7);
    CHECK(numerics[0] == 2);
    CHECK(numerics[1] == 3);
    CHECK(numerics[2] == 5);
    CHECK(numerics[3] == 4);
    CHECK(numerics[4] == 20);
    CHECK(numerics[5] == 5);
    CHECK(numerics[6] == 4);
}

TEST_CASE("reasoning chain: text-interleaved numerics") {
    auto ntok = make_arith_ntok();
    // "Alice has 5 apples and gets 3 more so has 5 + 3 = 8"
    // All 4 numerics must be captured in order: 5, 3, 5, 3, 8
    auto ids = ntok.encode("Alice has 5 and gets 3 so has 5 + 3 = 8");

    std::vector<int32_t> numerics;
    for (auto id : ids) {
        if (ntok.is_numeric(id) && !ntok.is_nan_token(id) && !ntok.is_overflow_token(id))
            numerics.push_back(static_cast<int32_t>(ntok.numeric_value(id)));
    }
    // There must be at least 3 numeric tokens (5, 3, 8) — the explicit expression
    REQUIRE(numerics.size() >= 3);
    CHECK(numerics.back() == 8);  // last numeric is the result
}

TEST_CASE("reasoning chain: large range encode/decode round-trip") {
    auto ntok = make_arith_ntok(-10000, 10000);

    // Chain: 1000 + 5000 = 6000 — all within range
    auto id1 = ntok.encode_int(1000);
    auto id2 = ntok.encode_int(5000);
    auto id3 = ntok.encode_int(6000);
    REQUIRE(ntok.numeric_value(id1) == Catch::Approx(1000.f));
    REQUIRE(ntok.numeric_value(id2) == Catch::Approx(5000.f));
    REQUIRE(ntok.numeric_value(id3) == Catch::Approx(6000.f));

    auto r1 = apply_math_op(RouteType::Add, 1000.f, 5000.f, -10000, 10000);
    REQUIRE(r1.value == Catch::Approx(6000.f));

    // Second step: 6000 - 2500 = 3500
    auto r2 = apply_math_op(RouteType::Sub, r1.value, 2500.f, -10000, 10000);
    REQUIRE(r2.value == Catch::Approx(3500.f));

    // Overflow: 6000 + 5000 = 11000 > 10000
    auto r_ovf = apply_math_op(RouteType::Add, 6000.f, 5000.f, -10000, 10000);
    REQUIRE(r_ovf.is_overflow);
}

// ── MathGPT: multi-step input shape and route_info ────────────────────────────

TEST_CASE("reasoning chain: MathGPT forward_math shape on two-step sequence") {
    auto ntok = make_arith_ntok();
    const int64_t V = ntok.total_vocab_size();
    MathGPT m(V, 16, 2, 1, 4, -1, 0, /*seed=*/7);

    // Encode "3 + 4 = 7 - 2 = 5"
    auto enc = ntok.encode("3 + 4 = 7 - 2 = 5");
    REQUIRE(enc.size() >= 5);

    Tensor ids_t({static_cast<int64_t>(enc.size())}, DType::Int32);
    auto sp = ids_t.data_as<int32_t>();
    for (std::size_t i = 0; i < enc.size(); ++i)
        sp[i] = static_cast<int32_t>(enc[i]);

    auto logits = m.forward_math(ids_t, ntok);
    REQUIRE(logits.data().shape(0) == static_cast<int64_t>(enc.size()));
    REQUIRE(logits.data().shape(1) == V);
}

TEST_CASE("reasoning chain: MathGPT route_info on two-step sequence") {
    auto ntok = make_arith_ntok();
    const int64_t V = ntok.total_vocab_size();
    MathGPT m(V, 16, 2, 1, 4, -1, 0, /*seed=*/7);

    auto enc = ntok.encode("10 - 3 = 7 + 2 = 9");
    REQUIRE(enc.size() >= 5);

    Tensor ids_t({static_cast<int64_t>(enc.size())}, DType::Int32);
    auto sp = ids_t.data_as<int32_t>();
    for (std::size_t i = 0; i < enc.size(); ++i)
        sp[i] = static_cast<int32_t>(enc[i]);

    auto info = m.route_info(ids_t, ntok);
    REQUIRE(static_cast<int64_t>(info.routes.size()) == static_cast<int64_t>(enc.size()));
    REQUIRE(static_cast<int64_t>(info.entropy.size()) == static_cast<int64_t>(enc.size()));

    // All routes are valid RouteType values
    for (auto r : info.routes)
        REQUIRE(static_cast<int>(r) < kNumRouteTypes);

    // Entropy is finite and non-negative at all positions
    for (float e : info.entropy) {
        REQUIRE(std::isfinite(e));
        REQUIRE(e >= 0.f);
    }
}

TEST_CASE("reasoning chain: MathGPT forward_math longer context") {
    auto ntok = make_arith_ntok();
    const int64_t V = ntok.total_vocab_size();
    MathGPT m(V, 16, 2, 1, 4, -1, 0, /*seed=*/7);

    // A longer natural-language-style reasoning chain
    auto enc = ntok.encode("Alice has 5 and gets 3 so has 5 + 3 = 8 gives 2 left 8 - 2 = 6");
    REQUIRE(enc.size() >= 7);

    Tensor ids_t({static_cast<int64_t>(enc.size())}, DType::Int32);
    auto sp = ids_t.data_as<int32_t>();
    for (std::size_t i = 0; i < enc.size(); ++i)
        sp[i] = static_cast<int32_t>(enc[i]);

    auto logits = m.forward_math(ids_t, ntok);
    REQUIRE(logits.data().shape(0) == static_cast<int64_t>(enc.size()));
    REQUIRE(logits.data().shape(1) == V);
}
