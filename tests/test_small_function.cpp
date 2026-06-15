#include <catch2/catch_test_macros.hpp>

#include "sub0llm/core/small_function.hpp"

#include <array>
#include <cstdint>

using sub0llm::SmallFunction;

TEST_CASE("SmallFunction - captureless + small capture (inline)", "[small_function]") {
    SmallFunction<int(int)> inc = [](int x) { return x + 1; };
    REQUIRE(static_cast<bool>(inc));
    REQUIRE(inc(41) == 42);

    int a = 10;
    SmallFunction<int(int)> add_a = [a](int x) { return x + a; };
    REQUIRE(add_a(5) == 15);

    SmallFunction<int(int)> empty;
    REQUIRE_FALSE(static_cast<bool>(empty));
}

TEST_CASE("SmallFunction - large capture spills to heap but still works", "[small_function]") {
    // A 64-int capture (256 B) exceeds Buf=32 → heap fallback path.
    std::array<int, 64> big{};
    for (int i = 0; i < 64; ++i) big[static_cast<std::size_t>(i)] = i;
    SmallFunction<int(int), 32> f = [big](int x) { return big[static_cast<std::size_t>(x)]; };
    REQUIRE(f(7) == 7);
    REQUIRE(f(63) == 63);
}

TEST_CASE("SmallFunction - copy and move preserve the callable", "[small_function]") {
    int a = 3;
    SmallFunction<int(int)> f = [a](int x) { return x * a; };

    SmallFunction<int(int)> g = f;                 // copy
    REQUIRE(g(4) == 12);
    REQUIRE(f(4) == 12);                           // source still valid

    SmallFunction<int(int)> h = std::move(f);      // move
    REQUIRE(h(5) == 15);

    SmallFunction<int(int)> k;
    k = h;                                         // copy-assign
    REQUIRE(k(6) == 18);
    k = std::move(g);                              // move-assign
    REQUIRE(k(7) == 21);
}

TEST_CASE("SmallFunction - heap-allocated callable copies correctly", "[small_function]") {
    std::array<std::int64_t, 40> big{};
    for (int i = 0; i < 40; ++i) big[static_cast<std::size_t>(i)] = i * 2;
    SmallFunction<std::int64_t(int), 32> f =
        [big](int x) { return big[static_cast<std::size_t>(x)]; };
    SmallFunction<std::int64_t(int), 32> g = f;    // copy of a heap-stored closure
    REQUIRE(g(10) == 20);
    REQUIRE(f(20) == 40);
    SmallFunction<std::int64_t(int), 32> h = std::move(g);
    REQUIRE(h(15) == 30);
}
