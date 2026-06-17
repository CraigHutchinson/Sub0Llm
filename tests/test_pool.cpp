// test_pool.cpp — TensorPool + BlockPool: aligned allocation, free-list reuse, and the
// SELF-SCALING high-water-mark cache cap (the regression guard for the fixed-threshold fix
// in commit 8987949 — a class must cache a whole step's live buffers, not a fixed count).

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "sub0llm/core/block_pool.hpp"
#include "../src/core/pool.hpp"   // TensorPool lives in the private core dir (header-only)

using sub0llm::TensorPool;
using sub0llm::BlockPool;

namespace {
bool aligned64(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 63u) == 0; }
}

TEST_CASE("TensorPool hands out non-null 64-byte-aligned buffers", "[pool]") {
    TensorPool p;
    for (std::size_t bytes : {std::size_t{128}, std::size_t{1024}, std::size_t{1u << 20}}) {
        auto b = p.allocate_raw(bytes);
        REQUIRE(b.ptr != nullptr);
        CHECK(aligned64(b.ptr));
        p.reclaim_raw(b.ptr, b.idx);
    }
    auto sp = p.allocate(4096);
    REQUIRE(sp != nullptr);
    CHECK(aligned64(sp.get()));
}

TEST_CASE("TensorPool reuses a reclaimed buffer (free-list)", "[pool]") {
    TensorPool p;
    auto a = p.allocate_raw(1u << 16);
    std::byte* first = a.ptr;
    p.reclaim_raw(a.ptr, a.idx);
    auto b = p.allocate_raw(1u << 16);
    CHECK(b.ptr == first);   // same size class, pool not empty → exact reuse
    p.reclaim_raw(b.ptr, b.idx);
}

TEST_CASE("TensorPool cap self-scales to the working set (no fixed-threshold regression)", "[pool]") {
    // 10 concurrently-live 4 MB buffers — under the OLD fixed 16 MB/class cap a 4 MB class
    // held only 4, so 6 of every 10 would be evicted-then-realloc'd each cycle. The
    // high-water-mark cap learns peak=10 and caches all 10, so a free→re-take cycle reuses
    // every buffer (zero new allocation). This is the thrash fix, asserted behaviourally.
    constexpr std::size_t kSize = 4u << 20;   // 4 MB
    constexpr int         K     = 10;
    TensorPool p;

    std::vector<TensorPool::RawBuf> live;
    std::unordered_set<std::byte*>  first_round;
    for (int i = 0; i < K; ++i) {
        auto b = p.allocate_raw(kSize);
        live.push_back(b);
        first_round.insert(b.ptr);
    }
    for (auto& b : live) p.reclaim_raw(b.ptr, b.idx);   // free all 10 → peak=10, cap≥10

    int reused = 0;
    std::vector<TensorPool::RawBuf> live2;
    for (int i = 0; i < K; ++i) {
        auto b = p.allocate_raw(kSize);
        live2.push_back(b);
        if (first_round.count(b.ptr)) ++reused;
    }
    CHECK(reused == K);   // every buffer reused — the full working set stayed cached
    for (auto& b : live2) p.reclaim_raw(b.ptr, b.idx);
}

TEST_CASE("TensorPool bypass path (out-of-range sizes) is valid and freeable", "[pool]") {
    TensorPool p;
    auto small = p.allocate_raw(64);          // < 128 B floor → bypass (idx == -1)
    CHECK(small.idx == -1);
    REQUIRE(small.ptr != nullptr);
    CHECK(aligned64(small.ptr));
    p.reclaim_raw(small.ptr, small.idx);      // must not crash

    auto big = p.allocate_raw(std::size_t{128} << 20);   // 128 MB > 64 MB top → bypass
    CHECK(big.idx == -1);
    REQUIRE(big.ptr != nullptr);
    p.reclaim_raw(big.ptr, big.idx);
}

TEST_CASE("BlockPool reuses blocks and self-scales to the live count", "[pool]") {
    BlockPool bp;
    constexpr std::size_t kBytes = 96;   // a small size class (Node/Storage-sized)
    constexpr int         K      = 200;

    void* first = bp.take(kBytes);
    bp.give(first, kBytes);
    CHECK(bp.take(kBytes) == first);     // single-block reuse
    bp.give(first, kBytes);

    std::vector<void*>            live;
    std::unordered_set<void*>     seen;
    for (int i = 0; i < K; ++i) { void* p = bp.take(kBytes); live.push_back(p); seen.insert(p); }
    for (void* p : live) bp.give(p, kBytes);   // peak=200 → cap≥200

    int reused = 0;
    std::vector<void*> live2;
    for (int i = 0; i < K; ++i) { void* p = bp.take(kBytes); live2.push_back(p); if (seen.count(p)) ++reused; }
    CHECK(reused == K);
    for (void* p : live2) bp.give(p, kBytes);
}
