#include <catch2/catch_test_macros.hpp>

#include "sub0llm/core/inline_vector.hpp"

#include <memory>

using sub0llm::InlineVector;

namespace {
// Tracks live instances to catch leaks / double-frees in the placement-new/destroy paths.
struct Tracked {
    static int live;
    int v = 0;
    Tracked() { ++live; }
    explicit Tracked(int x) : v(x) { ++live; }
    Tracked(const Tracked& o) : v(o.v) { ++live; }
    Tracked(Tracked&& o) noexcept : v(o.v) { o.v = -1; ++live; }
    Tracked& operator=(const Tracked&) = default;
    Tracked& operator=(Tracked&&) noexcept = default;
    ~Tracked() { --live; }
};
int Tracked::live = 0;
} // namespace

TEST_CASE("InlineVector - inline, spill, destroy all balance live count", "[inline_vector]") {
    REQUIRE(Tracked::live == 0);
    {
        InlineVector<Tracked, 2> v;
        REQUIRE(v.empty());
        v.push_back(Tracked{1});
        v.push_back(Tracked{2});         // still inline (N=2)
        REQUIRE(v.size() == 2);
        v.push_back(Tracked{3});         // spills to heap
        v.push_back(Tracked{4});
        REQUIRE(v.size() == 4);
        REQUIRE(v[0].v == 1);
        REQUIRE(v[3].v == 4);
        int sum = 0;
        for (const auto& t : v) sum += t.v;
        REQUIRE(sum == 10);
        REQUIRE(Tracked::live == 4);     // exactly the 4 elements (moved-from temps gone)
    }
    REQUIRE(Tracked::live == 0);          // dtor freed everything, no leak/double-free
}

TEST_CASE("InlineVector - move and copy", "[inline_vector]") {
    REQUIRE(Tracked::live == 0);
    {
        InlineVector<Tracked, 2> a;
        for (int i = 0; i < 4; ++i) a.push_back(Tracked{i});   // on heap
        InlineVector<Tracked, 2> b = std::move(a);             // steal heap
        REQUIRE(b.size() == 4);
        REQUIRE(b[2].v == 2);
        InlineVector<Tracked, 2> c = b;                        // copy
        REQUIRE(c.size() == 4);
        REQUIRE(c[3].v == 3);
        REQUIRE(Tracked::live == 8);                           // b (4) + c (4)
    }
    REQUIRE(Tracked::live == 0);
}

TEST_CASE("InlineVector - works with move-only payloads", "[inline_vector]") {
    InlineVector<std::unique_ptr<int>, 2> v;
    v.push_back(std::make_unique<int>(7));
    v.push_back(std::make_unique<int>(8));
    v.push_back(std::make_unique<int>(9));   // spill
    REQUIRE(*v[0] == 7);
    REQUIRE(*v[2] == 9);
}
