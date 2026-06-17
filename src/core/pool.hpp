#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

#include "sub0llm/core/pool_stats.hpp"

// Windows (MSVC CRT + clang-cl) does not provide std::aligned_alloc.
// Use _aligned_malloc/_aligned_free instead, noting the reversed argument order.
#ifdef _WIN32
namespace sub0llm { namespace detail {
    inline void* aligned_alloc(std::size_t alignment, std::size_t size) noexcept {
        return ::_aligned_malloc(size, alignment);
    }
    inline void aligned_free(void* ptr) noexcept { ::_aligned_free(ptr); }
} }
#else
namespace sub0llm { namespace detail {
    inline void* aligned_alloc(std::size_t alignment, std::size_t size) noexcept {
        return std::aligned_alloc(alignment, size);
    }
    inline void aligned_free(void* ptr) noexcept { std::free(ptr); }
} }
#endif

namespace sub0llm {

// Thread-local tensor buffer pool.
//
// Problem: the autograd training loop creates O(n_layers * ops_per_layer) temporary
// Tensor buffers per backward pass, each going through glibc's ptmalloc.  For a
// D=64 model with 6 layers this is ~150 alloc/free pairs per step.  At 1000 steps
// the ptmalloc overhead (brk/mmap syscalls, free-list lock) dominates sys time.
//
// Solution: power-of-2 size classes, 64-byte aligned, max kMaxCached per class.
// Freed buffers go back to the pool instead of hitting the allocator; the next
// allocation of the same size-class picks them up instantly with zero syscalls.
//
// Size range: 128 B … 64 MB (20 buckets, powers of 2).
// Sizes < 128 B or > 64 MB bypass the pool and use operator new/delete as before.
//
// Thread-safety: the pool is thread_local — no locks needed.
// Cross-thread freeing: the deleter always calls `TensorPool::get()` on the
// freeing thread, so cross-thread frees land in the freeing thread's pool. This is
// safe and the buffer is still reusable. For sub0llm's single-threaded training
// loop there is never a cross-thread free.

class TensorPool {
public:
    static TensorPool& get() noexcept {
        thread_local TensorPool inst;
        return inst;
    }

    ~TensorPool() {
        for (auto& bkt : buckets_)
            for (auto* p : bkt) detail::aligned_free(p);
    }

    // A pooled buffer + the bucket index needed to return it. idx >= 0 ⇒ a power-of-2
    // pool bucket; idx == -1 ⇒ the oversized/undersized bypass path (plain aligned_free).
    struct RawBuf { std::byte* ptr = nullptr; int idx = -1; };

    // Allocate a 64-byte-aligned buffer WITHOUT a shared_ptr control block (which the
    // shared_ptr `allocate()` below heap-allocates on every call even for a pooled buffer).
    // The caller (Storage) owns the buffer and MUST return it via reclaim_raw(ptr, idx).
    [[nodiscard]] RawBuf allocate_raw(std::size_t bytes) {
        const int idx = bucket_for(bytes);
        if (idx >= 0) {
            note_take(idx);
            auto& bkt = buckets_[static_cast<std::size_t>(idx)];
            if (!bkt.empty()) { SUB0LLM_POOL_COUNT(tp_hit); std::byte* p = bkt.back(); bkt.pop_back(); return {p, idx}; }
            SUB0LLM_POOL_COUNT(tp_miss);
            auto* p = static_cast<std::byte*>(detail::aligned_alloc(kAlign, bucket_size(idx)));
            if (!p) throw std::bad_alloc{};
            return {p, idx};
        }
        SUB0LLM_POOL_COUNT(tp_miss);
        const std::size_t aligned = (bytes + kAlign - 1u) & ~(kAlign - 1u);
        auto* p = static_cast<std::byte*>(detail::aligned_alloc(kAlign, aligned));
        if (!p) throw std::bad_alloc{};
        return {p, -1};
    }

    // Return a buffer from allocate_raw to the pool (or free it on the bypass path).
    void reclaim_raw(std::byte* ptr, int idx) noexcept {
        if (!ptr) return;
        if (idx >= 0) reclaim(ptr, idx);
        else        { SUB0LLM_POOL_COUNT(tp_evict); detail::aligned_free(ptr); }
    }

    // Allocate a buffer of at least `bytes` bytes, 64-byte aligned.
    // Returns a shared_ptr whose custom deleter returns the buffer to the pool.
    [[nodiscard]] std::shared_ptr<std::byte[]> allocate(std::size_t bytes) {
        const int idx = bucket_for(bytes);
        if (idx >= 0) {
            note_take(idx);
            auto&             bkt       = buckets_[static_cast<std::size_t>(idx)];
            const std::size_t alloc_sz  = bucket_size(idx);
            std::byte* ptr = nullptr;
            if (!bkt.empty()) {
                SUB0LLM_POOL_COUNT(tp_hit);
                ptr = bkt.back();
                bkt.pop_back();
            } else {
                SUB0LLM_POOL_COUNT(tp_miss);
                ptr = static_cast<std::byte*>(detail::aligned_alloc(kAlign, alloc_sz));
                if (!ptr) throw std::bad_alloc{};
            }
            try {
                return std::shared_ptr<std::byte[]>(ptr,
                    [idx](std::byte* p) noexcept {
                        TensorPool::get().reclaim(p, idx);
                    });
            } catch (...) {
                detail::aligned_free(ptr);
                throw;
            }
        }
        // Bypass: size outside pooled range — still 64-byte aligned for SIMD correctness.
        SUB0LLM_POOL_COUNT(tp_miss);
        const std::size_t aligned = (bytes + kAlign - 1u) & ~(kAlign - 1u);
        std::byte* ptr = static_cast<std::byte*>(detail::aligned_alloc(kAlign, aligned));
        if (!ptr) throw std::bad_alloc{};
        return std::shared_ptr<std::byte[]>(ptr, [](std::byte* p) noexcept { detail::aligned_free(p); });
    }

private:
    static constexpr std::size_t kAlign      = 64;    // cache line = SIMD granule
    static constexpr std::size_t kMinBytes   = 128;   // below this → bypass
    static constexpr int         kNumBuckets = 20;    // 128 B … 64 MB

    // SELF-SCALING per-class cache cap. The right cap is "hold a whole step's live buffers of
    // this size" — a quantity SET BY THE WORKLOAD (D·L·B·T), not a constant. A fixed cap (the
    // old 16 MB/class) is correct only at the scale it was tuned for and silently regresses when
    // the model grows: at the FOUNDED default it left 242 free+realloc thrash per per-worker step.
    // So instead of any magic number we LEARN it: track the high-water mark of simultaneously-live
    // buffers per class (live_ in/decremented on take/reclaim; peak_ its running max) and cache up
    // to that peak. After a few warm-up steps peak_ equals the step's working set for each size and
    // every reclaim is a hit — and it auto-scales to ANY D/B/T with no threshold to revisit. This
    // is memory-free: a bucket can only fill with buffers that were actually freed, so cached =
    // min(peak-live, cap); the cap merely stops eviction, it reserves nothing. kSafetyMax is a
    // generous backstop so a leak (ever-growing live_) can't inflate a cache unboundedly.
    static constexpr std::size_t kSafetyMax = 1u << 20;   // 1M buffers/class — leak guard only

    std::array<std::vector<std::byte*>, kNumBuckets> buckets_;
    std::array<std::uint32_t, kNumBuckets>           live_{};   // currently outstanding per class
    std::array<std::uint32_t, kNumBuckets>           peak_{};   // high-water mark of live_

    void note_take(int idx) noexcept {
        const std::size_t i = static_cast<std::size_t>(idx);
        if (++live_[i] > peak_[i]) peak_[i] = live_[i];
    }
    [[nodiscard]] std::size_t cap(int idx) const noexcept {
        // +2 absorbs minor per-step graph-shape variation (e.g. a differing masked count); the
        // operating value is peak_, derived entirely from the observed working set.
        const std::size_t want = static_cast<std::size_t>(peak_[static_cast<std::size_t>(idx)]) + 2;
        return want < kSafetyMax ? want : kSafetyMax;
    }

    void reclaim(std::byte* ptr, int idx) noexcept {
        const std::size_t i = static_cast<std::size_t>(idx);
        if (live_[i]) --live_[i];
        auto& bkt = buckets_[i];
        if (bkt.size() < cap(idx)) { SUB0LLM_POOL_COUNT(tp_cache); bkt.push_back(ptr); }
        else                       { SUB0LLM_POOL_COUNT(tp_evict); detail::aligned_free(ptr); }
    }

    // Bucket index whose size covers `bytes` bytes, or -1 if out of range.
    static int bucket_for(std::size_t bytes) noexcept {
        if (bytes < kMinBytes) return -1;
        std::size_t sz  = kMinBytes;
        int         idx = 0;
        while (sz < bytes && idx < kNumBuckets - 1) { sz <<= 1; ++idx; }
        return (sz >= bytes) ? idx : -1;
    }

    static std::size_t bucket_size(int idx) noexcept {
        return kMinBytes << static_cast<std::size_t>(idx);
    }
};

} // namespace sub0llm
