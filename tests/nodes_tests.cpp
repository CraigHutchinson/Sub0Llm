// nodes_tests.cpp -- the production ComputeNode registry (sub0/nodes.hpp): exact big-number primitives, the
// built-in node set, name dispatch, and extensibility. Engine-free (no model), so it lives in the fast
// frontend suite -- this is the deterministic substrate the delegation mechanism routes to.

#include <catch2/catch_test_macros.hpp>

#include "sub0/nodes.hpp"
#include "sub0/node_frame.hpp"

#include <string>
#include <vector>

namespace nd = sub0::nodes;

namespace {
// Build a token context from a string, mapping '[' -> TOK_TURN_START and ']' -> TOK_TURN_END (so tests read
// like `A + B = [op add]`), every other char as its byte token.
std::vector<int> ctx_of(const std::string& s) {
    std::vector<int> v;
    for (char c : s) v.push_back(c == '[' ? nd::FRAME_OPEN : c == ']' ? nd::FRAME_CLOSE : static_cast<unsigned char>(c));
    return v;
}
std::string str_of(const std::vector<int>& v) {
    std::string s;
    for (int t : v) s += (t == nd::FRAME_OPEN ? '[' : t == nd::FRAME_CLOSE ? ']' : static_cast<char>(t));
    return s;
}
}

TEST_CASE("nodes: exact big-number primitives (arbitrary length)", "[nodes]") {
    REQUIRE(nd::add("12345678901234567890", "98765432109876543210") == "111111111011111111100");
    REQUIRE(nd::add("999", "1") == "1000");
    REQUIRE(nd::add("0", "0") == "0");
    REQUIRE(nd::sub("1000", "1") == "999");
    REQUIRE(nd::sub("1", "1000") == "999");                 // |a-b|, non-negative
    REQUIRE(nd::sub("500", "500") == "0");
    REQUIRE(nd::cmp("00042", "42") == 0);                   // leading zeros ignored
    REQUIRE(nd::cmp("100", "99") == 1);
    REQUIRE(nd::cmp("99", "100") == -1);
}

TEST_CASE("nodes: built-in registry dispatches by op-name", "[nodes]") {
    const nd::Registry r = nd::builtin();
    REQUIRE(r.size() == 4);
    REQUIRE(r.run("add", {"7", "8"}) == "15");
    REQUIRE(r.run("sub", {"8", "3"}) == "5");
    REQUIRE(r.run("max", {"8", "3"}) == "8");
    REQUIRE(r.run("min", {"8", "3"}) == "3");
    // Comparison ops return the WHOLE winning operand (big numbers), not a truncation.
    REQUIRE(r.run("max", {"12345678901234567890", "98765432109876543210"}) == "98765432109876543210");

    // Unknown op / malformed operands -> empty (a graceful miss, never a wrong answer).
    REQUIRE(r.run("divide", {"6", "2"}).empty());
    REQUIRE(r.run("add", {"7"}).empty());
    REQUIRE(r.find("nope") == nullptr);
}

TEST_CASE("nodes: registry is extensible -- a new node is one register_node call", "[nodes]") {
    nd::Registry r = nd::builtin();
    r.register_node("sum3", [](const std::vector<std::string>& o) {
        return o.size() >= 3 ? nd::add(nd::add(o[0], o[1]), o[2]) : std::string{};
    });
    REQUIRE(r.size() == 5);
    REQUIRE(r.run("sum3", {"100", "20", "3"}) == "123");    // a 3-ary node, zero new tokenizer cost
    REQUIRE(r.run("add",  {"100", "20"}) == "120");         // built-ins still work
}

// The PRODUCTION region-frame callback (node_frame.hpp): the TOK_TURN-delimited `[op <name> ...]` region the
// decode interceptor dispatches. Engine-free -- exercises exactly the callback kv_decode_generate invokes.
TEST_CASE("nodes: region-frame compute callback dispatches TOK_TURN op-regions", "[nodes]") {
    const auto compute = nd::make_compute_callback(nd::builtin());

    // Operands inside the region (self-contained tool-call).
    REQUIRE(str_of(compute(ctx_of("[op add 12 34]"))) == "[46]");
    REQUIRE(str_of(compute(ctx_of("[op max 5 900]"))) == "[900]");
    // Operands read from the preceding inline expression (region names only the op).
    REQUIRE(str_of(compute(ctx_of("100+20=[op add]"))) == "[120]");
    // Big numbers stay exact end to end.
    REQUIRE(str_of(compute(ctx_of("[op add 999999999999 1]"))) == "[1000000000000]");

    // Inert cases (no injection): a chat turn, an unknown op, and the injected RESULT region (no `op` word) --
    // so a result region never re-triggers the node.
    REQUIRE(compute(ctx_of("[user hello]")).empty());
    REQUIRE(compute(ctx_of("[op divide 6 2]")).empty());
    REQUIRE(compute(ctx_of("[46]")).empty());
}
