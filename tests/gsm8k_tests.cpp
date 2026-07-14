// gsm8k_tests.cpp -- the GSM8K op-curriculum front end (engine-free): the exact big-int primitives, the
// consolidated `math`-only registry, and the calc-annotation converter (sub0/gsm8k.hpp) that turns
// `<<48/2=24>>` spans into `[op math 48/2]` op-frames with masked results.

#include <catch2/catch_test_macros.hpp>

#include "sub0/nodes.hpp"
#include "sub0/gsm8k.hpp"
#include "sub0/node_frame.hpp"   // the emitted op-frame must dispatch through the production callback
#include "sub0/casing.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace nd  = sub0::nodes;
namespace gs  = sub0::gsm8k;
namespace cas = sub0::casing;

namespace {
std::vector<int> byte_encode(std::string_view s) {   // a trivial stand-in for the real tokenizer
    std::vector<int> v; v.reserve(s.size());
    for (char c : s) v.push_back(static_cast<unsigned char>(c));
    return v;
}
}  // namespace

// --- The exact big-integer primitives (mul, exact div) the `math` node runs on --------------------------
TEST_CASE("nodes: exact big-integer mul", "[gsm8k][nodes]") {
    REQUIRE(nd::mul("48", "24") == "1152");
    REQUIRE(nd::mul("0", "999") == "0");
    REQUIRE(nd::mul("999999", "999999") == "999998000001");   // beyond 32-bit -- string math, no overflow
    REQUIRE(nd::mul("125", "8") == "1000");
}

TEST_CASE("nodes: exact div returns the quotient only when it divides evenly", "[gsm8k][nodes]") {
    REQUIRE(nd::div("48", "2") == "24");
    REQUIRE(nd::div("1000", "8") == "125");
    REQUIRE(nd::div("0", "5") == "0");
    REQUIRE(nd::div("7", "2").empty());     // not exact -> declines (the annotation would be dropped)
    REQUIRE(nd::div("5", "0").empty());     // division by zero -> declines
    REQUIRE(nd::div("999998000001", "999999") == "999999");
}

// The registry is consolidated: one general `math` op supersedes the per-op arithmetic frames.
TEST_CASE("nodes: builtin registers the general math op, not per-op arithmetic frames", "[gsm8k][nodes]") {
    const nd::Registry r = nd::builtin();
    REQUIRE(r.find("math") != nullptr);
    REQUIRE(r.find("add") == nullptr);      // superseded by math (still an internal primitive)
    REQUIRE(r.find("mul") == nullptr);
    REQUIRE(r.find("div") == nullptr);
    REQUIRE(r.run("math", {"6", "*", "7"}) == "42");
}

// --- The annotation parser ------------------------------------------------------------------------------
TEST_CASE("gsm8k: parse_annotation keeps the verbatim expression, verified by the math node", "[gsm8k]") {
    gs::Op op;
    REQUIRE(gs::parse_annotation("48/2=24", op));  REQUIRE(op.expr == "48/2");  REQUIRE(op.result == "24");
    REQUIRE(gs::parse_annotation("48+24=72", op)); REQUIRE(op.expr == "48+24");
    REQUIRE(gs::parse_annotation("3*10=30", op));  REQUIRE(op.expr == "3*10");
    REQUIRE(gs::parse_annotation("100-40=60", op));
    REQUIRE(gs::parse_annotation("2+3*4=14", op)); REQUIRE(op.expr == "2+3*4");   // compound expr, math handles precedence

    // Rejected (the FILTER is just the math node declining): decimals, non-exact division, wrong result, no '='.
    REQUIRE_FALSE(gs::parse_annotation("0.5*20=10", op));
    REQUIRE_FALSE(gs::parse_annotation("5/2=2", op));
    REQUIRE_FALSE(gs::parse_annotation("48/2=25", op));   // wrong stated result
    REQUIRE_FALSE(gs::parse_annotation("nogeq", op));
}

// --- The converter on a real GSM8K solution -------------------------------------------------------------
TEST_CASE("gsm8k: segment a real solution into text + verified math expressions", "[gsm8k]") {
    const std::string sol =
        "Natalia sold 48/2 = <<48/2=24>> 24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>> 72 clips altogether.\n"
        "#### 72";

    const std::vector<gs::Segment> segs = gs::segment(sol);
    int ops = 0, verified = 0;
    std::vector<std::string> exprs;
    for (const gs::Segment& s : segs)
        if (s.is_op) { ++ops; verified += gs::verify(s.op); exprs.push_back(s.op.expr); }

    REQUIRE(ops == 2);
    REQUIRE(verified == 2);
    REQUIRE(exprs == std::vector<std::string>{"48/2", "48+24"});
    REQUIRE(segs.front().is_op == false);
    REQUIRE(segs.front().text.starts_with("Natalia sold 48/2 = "));
}

TEST_CASE("gsm8k: bad annotations are filtered, not learned", "[gsm8k]") {
    const std::vector<gs::Segment> segs = gs::segment("half of it is <<0.5*8=4>> 4 apples");   // decimal -> not parsed
    int ops = 0; for (const gs::Segment& s : segs) if (s.is_op) ++ops;
    REQUIRE(ops == 0);

    gs::Op wrong{"48/2", "25"};              // label says 25...
    REQUIRE_FALSE(gs::verify(wrong));        // ...but 48/2 = 24, so it's dropped
}

// --- The training-stream builder: grade routing, mask results, de-dup the repeated answer ---------------
TEST_CASE("gsm8k: build_stream emits an exact graded-frame / masked-result token stream", "[gsm8k]") {
    // Minimal case, token-for-token: `2+3=<<2+3=5>> 5 b` -> `2+3=` [op math] (5 masked) ` b`. The frame is
    // BARE (`op math`) -- the expression `2+3` is already in the graded prose right before it.
    const gs::Example ex = gs::build_stream("2+3=<<2+3=5>> 5 b", byte_encode);

    std::vector<int> et; std::vector<std::uint8_t> em;
    auto g = [&](int t) { et.push_back(t); em.push_back(1); };
    auto m = [&](int t) { et.push_back(t); em.push_back(0); };
    g('2'); g('+'); g('3'); g('=');
    g(cas::TOK_TURN_START); for (char c : std::string("op math")) g(static_cast<unsigned char>(c)); g(cas::TOK_TURN_END);
    m('5');
    g(' '); g('b');

    REQUIRE(ex.ops == 1);
    REQUIRE(ex.dropped == 0);
    REQUIRE(ex.tokens == et);
    REQUIRE(ex.mask == em);
}

TEST_CASE("gsm8k: build_stream on the real solution masks exactly the two results", "[gsm8k]") {
    const std::string sol =
        "Natalia sold 48/2 = <<48/2=24>> 24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>> 72 clips altogether.\n#### 72";
    const gs::Example ex = gs::build_stream(sol, byte_encode);

    REQUIRE(ex.ops == 2);
    std::string masked;
    for (std::size_t i = 0; i < ex.tokens.size(); ++i) if (ex.mask[i] == 0) masked.push_back(static_cast<char>(ex.tokens[i]));
    REQUIRE(masked == "2472");
    int frames = 0; for (int t : ex.tokens) if (t == cas::TOK_TURN_START) ++frames;
    REQUIRE(frames == 2);
}

// Close the train<->gen loop: the EXACT op-frame the builder emits dispatches through the PRODUCTION compute
// callback (node_frame.hpp) to the right result -- a model trained on these frames, run through gen's wired
// callback, gets the node's exact answer.
TEST_CASE("gsm8k: the emitted op-frame dispatches through the production callback", "[gsm8k]") {
    // The prose poses the expression, then the BARE `[op math]` routes -- the node reads `6*7` from context.
    std::vector<int> ctx;
    for (char c : std::string("6*7=")) ctx.push_back(static_cast<unsigned char>(c));
    for (int t : gs::op_frame()) ctx.push_back(t);
    const auto compute = nd::make_compute_callback(nd::builtin());
    const std::vector<int> inj = compute(ctx);
    const std::vector<int> want = { cas::TOK_TURN_START, '4', '2', cas::TOK_TURN_END };
    REQUIRE(inj == want);                                           // 6*7 -> [42] injected
}
