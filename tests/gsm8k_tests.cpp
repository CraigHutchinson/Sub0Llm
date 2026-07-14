// gsm8k_tests.cpp -- the GSM8K op-curriculum front end (engine-free): the exact mul/div nodes GSM8K needs,
// and the calc-annotation converter (sub0/gsm8k.hpp) that turns `<<48/2=24>>` spans into verifiable op calls.

#include <catch2/catch_test_macros.hpp>

#include "sub0/nodes.hpp"
#include "sub0/gsm8k.hpp"

#include <string>
#include <vector>

namespace nd = sub0::nodes;
namespace gs = sub0::gsm8k;

// --- The exact big-integer nodes GSM8K adds (mul, exact div) --------------------------------------------
TEST_CASE("nodes: exact big-integer mul", "[gsm8k][nodes]") {
    REQUIRE(nd::mul("48", "24") == "1152");
    REQUIRE(nd::mul("0", "999") == "0");
    REQUIRE(nd::mul("7", "1") == "7");
    REQUIRE(nd::mul("999999", "999999") == "999998000001");   // beyond 32-bit -- string math, no overflow
    REQUIRE(nd::mul("125", "8") == "1000");
}

TEST_CASE("nodes: exact div returns the quotient only when it divides evenly", "[gsm8k][nodes]") {
    REQUIRE(nd::div("48", "2") == "24");
    REQUIRE(nd::div("1152", "48") == "24");
    REQUIRE(nd::div("1000", "8") == "125");
    REQUIRE(nd::div("100", "10") == "10");
    REQUIRE(nd::div("0", "5") == "0");
    REQUIRE(nd::div("7", "2").empty());     // not exact -> node declines (the annotation would be dropped)
    REQUIRE(nd::div("5", "0").empty());     // division by zero -> declines
    REQUIRE(nd::div("999998000001", "999999") == "999999");
}

TEST_CASE("nodes: mul/div are registered in builtin()", "[gsm8k][nodes]") {
    const nd::Registry r = nd::builtin();
    REQUIRE(r.run("mul", {"6", "7"}) == "42");
    REQUIRE(r.run("div", {"42", "7"}) == "6");
    REQUIRE(r.find("mul") != nullptr);
    REQUIRE(r.find("div") != nullptr);
}

// --- The annotation parser ------------------------------------------------------------------------------
TEST_CASE("gsm8k: parse_annotation maps a calc symbol to an integer op", "[gsm8k]") {
    gs::Op op;
    REQUIRE(gs::parse_annotation("48/2", "24", op));
    REQUIRE(op.name == "div"); REQUIRE(op.a == "48"); REQUIRE(op.b == "2"); REQUIRE(op.result == "24");

    REQUIRE(gs::parse_annotation("48+24", "72", op));   REQUIRE(op.name == "add");
    REQUIRE(gs::parse_annotation("3*10", "30", op));    REQUIRE(op.name == "mul");
    REQUIRE(gs::parse_annotation("100-40", "60", op));  REQUIRE(op.name == "sub");

    // Rejected (integer-MVP filter): decimals, spaces, multi-op, empty operand.
    REQUIRE_FALSE(gs::parse_annotation("0.5*20", "10", op));
    REQUIRE_FALSE(gs::parse_annotation("48 / 2", "24", op));
    REQUIRE_FALSE(gs::parse_annotation("2+3+4", "9", op));
    REQUIRE_FALSE(gs::parse_annotation("5/2", "2.5", op));
    REQUIRE_FALSE(gs::parse_annotation("*5", "0", op));
}

// --- The converter on a real GSM8K solution -------------------------------------------------------------
TEST_CASE("gsm8k: segment a real solution into text + verified op calls", "[gsm8k]") {
    // The canonical GSM8K first example (Natalia's clips), verbatim annotation format.
    const std::string sol =
        "Natalia sold 48/2 = <<48/2=24>> 24 clips in May.\n"
        "Natalia sold 48+24 = <<48+24=72>> 72 clips altogether.\n"
        "#### 72";

    const std::vector<gs::Segment> segs = gs::segment(sol);
    const nd::Registry reg = nd::builtin();

    int ops = 0, verified = 0;
    for (const gs::Segment& s : segs)
        if (s.is_op) { ++ops; verified += gs::verify(s.op, reg); }

    REQUIRE(ops == 2);
    REQUIRE(verified == 2);   // both annotations reproduced exactly by the nodes (div 48/2, add 48+24)

    // The op segments carry the parsed ops, in order.
    std::vector<std::string> names;
    for (const gs::Segment& s : segs) if (s.is_op) names.push_back(s.op.name);
    REQUIRE(names == std::vector<std::string>{"div", "add"});

    // The literal text around the ops is preserved (delegation replaces only the `<<...>>`, not the prose).
    REQUIRE(segs.front().is_op == false);
    REQUIRE(segs.front().text.starts_with("Natalia sold 48/2 = "));
}

// A malformed annotation stays as literal text (not turned into an op) and a wrong label fails verification.
TEST_CASE("gsm8k: bad annotations are filtered, not learned", "[gsm8k]") {
    const nd::Registry reg = nd::builtin();

    // A decimal annotation cannot parse -> it remains literal text, contributing zero ops.
    const std::vector<gs::Segment> segs = gs::segment("half of it is <<0.5*8=4>> 4 apples");
    int ops = 0; for (const gs::Segment& s : segs) if (s.is_op) ++ops;
    REQUIRE(ops == 0);

    // A wrong stated result parses but fails verification (the node computes the truth).
    gs::Op wrong;
    REQUIRE(gs::parse_annotation("48/2", "25", wrong));   // label says 25...
    REQUIRE_FALSE(gs::verify(wrong, reg));                // ...but 48/2 = 24, so it's dropped
}
