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
#include <new>
#include <vector>

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
        if (idx < 0) return ::operator new(bytes);          // outsized → bypass
        auto& bkt = buckets_[static_cast<std::size_t>(idx)];
        if (!bkt.empty()) { void* p = bkt.back(); bkt.pop_back(); return p; }
        return ::operator new(bucket_size(idx));            // cold: size-class block
    }
    void give(void* p, std::size_t bytes) noexcept {
        if (!p) return;
        const int idx = bucket_for(bytes);
        if (idx < 0) { ::operator delete(p); return; }
        auto& bkt = buckets_[static_cast<std::size_t>(idx)];
        if (bkt.size() < max_cached(idx)) bkt.push_back(p);
        else                              ::operator delete(p);
    }

private:
    static constexpr std::size_t kMin       = 16;     // smallest size class
    static constexpr int         kNumBuckets = 12;    // 16 B … 32 KB
    // Size-aware cap (~8 MB cached bytes/class) so the small classes hold a whole step's live
    // graph (hundreds–thousands of Nodes/Storages) while bounding total cached memory.
    static constexpr std::size_t kCacheBytes = 8u << 20;
    static std::size_t max_cached(int idx) noexcept {
        const std::size_t cap = kCacheBytes / bucket_size(idx);
        return cap < 8 ? 8 : (cap > 16384 ? 16384 : cap);
    }

    std::array<std::vector<void*>, kNumBuckets> buckets_;

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
