#pragma once

// pool_stats.hpp — opt-in allocator instrumentation for TensorPool + BlockPool.
//
// Enabled only with -DSUB0LLM_POOL_STATS (zero overhead otherwise: every count site
// compiles to a no-op). When on, a process-global set of relaxed atomics tallies, for
// each pool, cache HITS (reuse), MISSES (a real ::operator new / aligned_alloc), and on
// the return path CACHE (kept) vs EVICT (freed because the bucket was full). The churn
// the long run is suspected of is a per-step MISS+EVICT pair: a step's live working set
// exceeds a bucket's cache cap, so every step frees on return and re-allocates on the
// next — exactly the alloc/dealloc thrash to hunt. Diff two snapshots across N steps and
// divide by N to read per-step misses/evictions per pool. The global atomics make the
// measuring build's numbers correct across all worker threads (it is not a perf build —
// only the COUNTS matter, and they are identical to a release build's).

#ifdef SUB0LLM_POOL_STATS

#include <atomic>
#include <cstdint>

namespace sub0llm::poolstats {

struct Counters {
    std::atomic<std::uint64_t> tp_hit{0}, tp_miss{0}, tp_cache{0}, tp_evict{0};
    std::atomic<std::uint64_t> bp_hit{0}, bp_miss{0}, bp_cache{0}, bp_evict{0};
};

inline Counters& counters() noexcept {
    static Counters c;
    return c;
}

struct Snapshot {
    std::uint64_t tp_hit, tp_miss, tp_cache, tp_evict;
    std::uint64_t bp_hit, bp_miss, bp_cache, bp_evict;
};

inline Snapshot snapshot() noexcept {
    const Counters& c = counters();
    return {c.tp_hit.load(std::memory_order_relaxed), c.tp_miss.load(std::memory_order_relaxed),
            c.tp_cache.load(std::memory_order_relaxed), c.tp_evict.load(std::memory_order_relaxed),
            c.bp_hit.load(std::memory_order_relaxed), c.bp_miss.load(std::memory_order_relaxed),
            c.bp_cache.load(std::memory_order_relaxed), c.bp_evict.load(std::memory_order_relaxed)};
}

}  // namespace sub0llm::poolstats

#define SUB0LLM_POOL_COUNT(field) \
    (::sub0llm::poolstats::counters().field.fetch_add(1, std::memory_order_relaxed))

#else
#define SUB0LLM_POOL_COUNT(field) ((void)0)
#endif
