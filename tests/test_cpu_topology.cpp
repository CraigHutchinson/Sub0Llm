#include <catch2/catch_test_macros.hpp>

#include "sub0llm/core/cpu_topology.hpp"

using sub0llm::CpuTopology;
using sub0llm::parse_cpu_list;
using sub0llm::resolve_pin_set;

namespace {
// A synthetic hybrid topology (no real detection needed): 4 P-cores {0,1,2,3},
// 4 E-cores {4,5,6,7}, one logical per physical (no SMT).
CpuTopology fake_hybrid() {
    CpuTopology t;
    t.n_logical    = 8;
    t.perf         = {0, 1, 2, 3};
    t.efficiency   = {4, 5, 6, 7};
    t.perf_primary = {0, 1, 2, 3};
    t.eff_primary  = {4, 5, 6, 7};
    return t;
}
} // namespace

TEST_CASE("parse_cpu_list - ranges and comma lists", "[topology]") {
    REQUIRE(parse_cpu_list("0-3") == std::vector<int>{0, 1, 2, 3});
    REQUIRE(parse_cpu_list("12,13,22,23") == std::vector<int>{12, 13, 22, 23});
    REQUIRE(parse_cpu_list("5") == std::vector<int>{5});
    REQUIRE(parse_cpu_list("10-10") == std::vector<int>{10});
}

TEST_CASE("resolve_pin_set - policies map to the right cores", "[topology]") {
    const auto topo = fake_hybrid();

    SECTION("auto = compute_pin_set (P primaries first)") {
        REQUIRE(resolve_pin_set("auto", topo, 2) == std::vector<int>{0, 1});
    }
    SECTION("all = unpinned") {
        REQUIRE(resolve_pin_set("all", topo, 3) == std::vector<int>{-1, -1, -1});
    }
    SECTION("P / E select their core class") {
        REQUIRE(resolve_pin_set("P", topo, 3) == std::vector<int>{0, 1, 2});
        REQUIRE(resolve_pin_set("E", topo, 2) == std::vector<int>{4, 5});
    }
    SECTION("explicit list, exact and cycled") {
        REQUIRE(resolve_pin_set("5,6", topo, 2) == std::vector<int>{5, 6});
        // n_workers > list size → cycle (oversubscription is the caller's choice).
        REQUIRE(resolve_pin_set("5,6", topo, 3) == std::vector<int>{5, 6, 5});
    }
    SECTION("two jobs can take DISJOINT cores") {
        const auto a = resolve_pin_set("0,1", topo, 2);   // job A
        const auto b = resolve_pin_set("2,3", topo, 2);   // job B
        for (int x : a) for (int y : b) REQUIRE(x != y);  // no shared core
    }
}

TEST_CASE("resolve_pin_set - homogeneous CPU falls back gracefully", "[topology]") {
    CpuTopology homo;                       // no E-cores
    homo.n_logical = 4;
    homo.perf = homo.perf_primary = {0, 1, 2, 3};
    // "E" on a non-hybrid box has no E-cores → unpinned rather than crashing.
    REQUIRE(resolve_pin_set("E", homo, 2) == std::vector<int>{-1, -1});
    REQUIRE(resolve_pin_set("P", homo, 2) == std::vector<int>{0, 1});
}
