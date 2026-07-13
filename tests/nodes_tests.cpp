// nodes_tests.cpp -- the production ComputeNode registry (sub0/nodes.hpp): exact big-number primitives, the
// built-in node set, name dispatch, and extensibility. Engine-free (no model), so it lives in the fast
// frontend suite -- this is the deterministic substrate the delegation mechanism routes to.

#include <catch2/catch_test_macros.hpp>

#include "sub0/nodes.hpp"

#include <string>
#include <vector>

namespace nd = sub0::nodes;

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
