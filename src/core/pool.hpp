#pragma once
#include <array>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <vector>

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
            auto& bkt = buckets_[static_cast<std::size_t>(idx)];
            if (!bkt.empty()) { std::byte* p = bkt.back(); bkt.pop_back(); return {p, idx}; }
            auto* p = static_cast<std::byte*>(detail::aligned_alloc(kAlign, bucket_size(idx)));
            if (!p) throw std::bad_alloc{};
            return {p, idx};
        }
        const std::size_t aligned = (bytes + kAlign - 1u) & ~(kAlign - 1u);
        auto* p = static_cast<std::byte*>(detail::aligned_alloc(kAlign, aligned));
        if (!p) throw std::bad_alloc{};
        return {p, -1};
    }

    // Return a buffer from allocate_raw to the pool (or free it on the bypass path).
    void reclaim_raw(std::byte* ptr, int idx) noexcept {
        if (!ptr) return;
        if (idx >= 0) reclaim(ptr, idx);
        else          detail::aligned_free(ptr);
    }

    // Allocate a buffer of at least `bytes` bytes, 64-byte aligned.
    // Returns a shared_ptr whose custom deleter returns the buffer to the pool.
    [[nodiscard]] std::shared_ptr<std::byte[]> allocate(std::size_t bytes) {
        const int idx = bucket_for(bytes);
        if (idx >= 0) {
            auto&             bkt       = buckets_[static_cast<std::size_t>(idx)];
            const std::size_t alloc_sz  = bucket_size(idx);
            std::byte* ptr = nullptr;
            if (!bkt.empty()) {
                ptr = bkt.back();
                bkt.pop_back();
            } else {
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
        const std::size_t aligned = (bytes + kAlign - 1u) & ~(kAlign - 1u);
        std::byte* ptr = static_cast<std::byte*>(detail::aligned_alloc(kAlign, aligned));
        if (!ptr) throw std::bad_alloc{};
        return std::shared_ptr<std::byte[]>(ptr, [](std::byte* p) noexcept { detail::aligned_free(p); });
    }

private:
    static constexpr std::size_t kAlign      = 64;    // cache line = SIMD granule
    static constexpr std::size_t kMinBytes   = 128;   // below this → bypass
    static constexpr int         kNumBuckets = 20;    // 128 B … 64 MB
    static constexpr std::size_t kMaxCached  = 16;    // per bucket

    std::array<std::vector<std::byte*>, kNumBuckets> buckets_;

    void reclaim(std::byte* ptr, int idx) noexcept {
        auto& bkt = buckets_[static_cast<std::size_t>(idx)];
        if (bkt.size() < kMaxCached) bkt.push_back(ptr);
        else                          detail::aligned_free(ptr);
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
