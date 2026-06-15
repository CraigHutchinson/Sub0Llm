#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "sub0llm/core/workspace.hpp"

using namespace sub0llm;
using Catch::Matchers::WithinAbs;

TEST_CASE("Workspace: take returns aligned, sized spans", "[workspace]") {
    Workspace ws(64);          // reserve up front so take() never reallocates mid-step
    auto a = ws.take(10);
    auto b = ws.take(7);
    REQUIRE(a.size() == 10);
    REQUIRE(b.size() == 7);
    // 16-float alignment: b starts at the next multiple of 16 after a (offset 0..10 → 16).
    REQUIRE(b.data() == a.data() + 16);
}

TEST_CASE("Workspace: reset rewinds, reuses the same backing storage", "[workspace]") {
    Workspace ws;
    auto a = ws.take(32);
    float* base = a.data();
    ws.reset();
    auto b = ws.take(32);
    REQUIRE(b.data() == base);          // same storage reused after reset
    REQUIRE(ws.used() == 32);
}

TEST_CASE("Workspace: stops growing after warmup (zero steady-state allocation)", "[workspace]") {
    Workspace ws;
    // Simulate a fixed-shape training step touching several buffers.
    auto step = [&] {
        ws.reset();
        (void)ws.take(256);
        (void)ws.take_zeroed(128);
        (void)ws.take(64);
    };
    step();                              // warmup grows the arena to the high-water mark
    const std::size_t hw = ws.high_water();
    REQUIRE(hw > 0);
    for (int i = 0; i < 1000; ++i) {
        step();
        REQUIRE(ws.high_water() == hw);  // never grows again → no further heap allocation
    }
}

TEST_CASE("Workspace: take_zeroed zeros the span", "[workspace]") {
    Workspace ws;
    auto a = ws.take(16);
    std::fill(a.begin(), a.end(), 9.0f);
    ws.reset();
    auto z = ws.take_zeroed(16);
    for (float v : z) REQUIRE_THAT(v, WithinAbs(0.0f, 0.0f));
}

TEST_CASE("MatView: row-major 2-D access over a workspace span", "[workspace]") {
    Workspace ws;
    auto s = ws.take(2 * 3);
    MatView m = mat(s, 2, 3);
    REQUIRE(m.rows == 2);
    REQUIRE(m.cols == 3);
    m.at(0, 0) = 1.0f; m.at(0, 2) = 3.0f; m.at(1, 1) = 5.0f;
    REQUIRE_THAT(s[0], WithinAbs(1.0f, 0.0f));
    REQUIRE_THAT(s[2], WithinAbs(3.0f, 0.0f));
    REQUIRE_THAT(s[4], WithinAbs(5.0f, 0.0f));
    REQUIRE(m.row(1).size() == 3);
    REQUIRE(m.row(1).data() == s.data() + 3);
}
