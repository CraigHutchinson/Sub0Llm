#pragma once

// block_pool.hpp — a thread-local free-list for small fixed-size object blocks, exposed as a
// std::allocator usable with std::allocate_shared. The autograd training graph creates one
// shared_ptr-managed Storage per tensor and one Node per op every step; make_shared heap-
// allocates the (control-block + object) block each time. Recycling those blocks (the graph
// shape is identical every step) drops them to zero allocations after warm-up — the buffers
// themselves are already pooled by TensorPool.
//
// Thread-safety: thread_local, no locks. A block freed on a different thread than it was
// taken lands in the freeing thread's pool (safe; reusable) — same model as TensorPool and
// fine for sub0llm's per-worker training threads.

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <vector>

#include "sub0llm/core/pool_stats.hpp"

namespace sub0llm {

class BlockPool {
public:
    static BlockPool& get() noexcept {
        thread_local BlockPool inst;
        return inst;
    }
    ~BlockPool() {
        for (auto& bkt : buckets_)
            for (void* p : bkt) ::operator delete(p);
    }

    [[nodiscard]] void* take(std::size_t bytes) {
        const int idx = bucket_for(bytes);
        if (idx < 0) { SUB0LLM_POOL_COUNT(bp_miss); return ::operator new(bytes); }  // outsized → bypass
        note_take(idx);
        auto& bkt = buckets_[static_cast<std::size_t>(idx)];
        if (!bkt.empty()) { SUB0LLM_POOL_COUNT(bp_hit); void* p = bkt.back(); bkt.pop_back(); return p; }
        SUB0LLM_POOL_COUNT(bp_miss);
        return ::operator new(bucket_size(idx));            // cold: size-class block
    }
    void give(void* p, std::size_t bytes) noexcept {
        if (!p) return;
        const int idx = bucket_for(bytes);
        if (idx < 0) { SUB0LLM_POOL_COUNT(bp_evict); ::operator delete(p); return; }
        const std::size_t i = static_cast<std::size_t>(idx);
        if (live_[i]) --live_[i];
        auto& bkt = buckets_[i];
        if (bkt.size() < cap(idx)) { SUB0LLM_POOL_COUNT(bp_cache); bkt.push_back(p); }
        else                       { SUB0LLM_POOL_COUNT(bp_evict); ::operator delete(p); }
    }

private:
    static constexpr std::size_t kMin       = 16;     // smallest size class
    static constexpr int         kNumBuckets = 12;    // 16 B … 32 KB
    // SELF-SCALING cap, same rationale as TensorPool (src/core/pool.hpp): a step's live Node/
    // Storage count is set by the graph size (∝ L·B), so learn it via a per-class high-water mark
    // rather than a fixed byte budget that would regress as layers grow. Memory-free (cached =
    // min(peak-live, cap)); kSafetyMax guards a leak only.
    static constexpr std::size_t kSafetyMax = 1u << 20;

    std::array<std::vector<void*>, kNumBuckets> buckets_;
    std::array<std::uint32_t, kNumBuckets>      live_{};
    std::array<std::uint32_t, kNumBuckets>      peak_{};

    void note_take(int idx) noexcept {
        const std::size_t i = static_cast<std::size_t>(idx);
        if (++live_[i] > peak_[i]) peak_[i] = live_[i];
    }
    [[nodiscard]] std::size_t cap(int idx) const noexcept {
        const std::size_t want = static_cast<std::size_t>(peak_[static_cast<std::size_t>(idx)]) + 2;
        return want < kSafetyMax ? want : kSafetyMax;
    }

    static int bucket_for(std::size_t bytes) noexcept {
        std::size_t s = kMin;
        int i = 0;
        while (s < bytes && i < kNumBuckets - 1) { s <<= 1; ++i; }
        return s >= bytes ? i : -1;
    }
    static std::size_t bucket_size(int idx) noexcept { return kMin << static_cast<std::size_t>(idx); }
};

// Stateless allocator over BlockPool. Used with std::allocate_shared<T>(PoolAllocator<T>{})
// to recycle the control-block+object allocation behind a shared_ptr.
template<class T>
struct PoolAllocator {
    using value_type = T;
    PoolAllocator() noexcept = default;
    template<class U> PoolAllocator(const PoolAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        return static_cast<T*>(BlockPool::get().take(n * sizeof(T)));
    }
    void deallocate(T* p, std::size_t n) noexcept {
        BlockPool::get().give(p, n * sizeof(T));
    }
    template<class U> bool operator==(const PoolAllocator<U>&) const noexcept { return true; }
    template<class U> bool operator!=(const PoolAllocator<U>&) const noexcept { return false; }
};

} // namespace sub0llm
