#pragma once

#include "sub0/moe_quant.hpp"
#include "sub0/residency.hpp"

#include <sub0mempage/local_file_backend.hpp>
#include <sub0tieredcache/row_cache.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace sub0::moeio {

/** An owned cache of routed experts: one TieredCache row per (layer, expert), holding the contiguous
 *  encoded bytes of that expert's three planes in caller-owned RAM.
 *
 *  This replaces OS paging of the S0Q1 sidecar for the regime where it does not fit in RAM
 *  (docs/STORAGE_STACK_PLAN.md "Core use case"). A hit costs no copy, no syscall and no page fault:
 *  plane() points straight into the resident row. A miss is one read of the whole expert, started at
 *  router time by prefetch(). Rows are bounded-extent (sub0tieredcache::RowExtent::bounded) because
 *  expert sizes differ between layers.
 *
 *  Per layer: prefetch(layer, experts), then acquire(k) for each selected expert before reading
 *  plane(k, ...). The pins taken by acquire are released by the next prefetch, which the caller makes
 *  only after every reader of the previous layer has joined. acquire may run concurrently for
 *  different k; open, close and prefetch require exclusive access. Nothing here allocates after open.
 */
class ExpertRowCache {
public:
    /// Page counts over the cache's own RAM (sub0/residency.hpp).
    using Residency = residency::Report;

    /// Constructs a closed cache; no storage, worker or file handle exists until open.
    ExpertRowCache() = default;
    ExpertRowCache(const ExpertRowCache&) = delete;
    ExpertRowCache& operator=(const ExpertRowCache&) = delete;
    /// Releases pins, drains fills, then frees storage; callers must first join all readers.
    ~ExpertRowCache();

    /** Registers `sidecar` (the file `store` was opened from) and sizes the cache.
     *  @param store         Must outlive this cache; its descriptors locate and size every row.
     *  @param budget_bytes  RAM for resident rows, rounded down to whole rows and capped at the table.
     *  @param max_selected  Experts selected per layer: the per-call batch bound.
     *  @param readers       Backend read threads (the I/O queue depth).
     *  @param concurrent_pins Extra rows pin() may hold at once (one per batched-forward worker).
     *  @return invalid_argument if an expert's planes are not contiguous in the file, if the budget
     *          holds fewer than two layers' selections plus `concurrent_pins`, or if the OS cannot
     *          commit the budget's storage; pool_exhausted if the OS refuses to lock that storage in
     *          RAM. An unpinned cache is refused rather than used: the OS would page it out under
     *          memory pressure, turning hits into pagefile reads.
     *  @note Administrative: allocates, faults in and locks the whole budget, so it takes time
     *        proportional to the budget. Any previous session is closed first.
     */
    [[nodiscard]] sub0tieredcache::Status open(const std::filesystem::path& sidecar, const moeq::Store& store,
                                               std::uint64_t budget_bytes, std::uint32_t max_selected,
                                               std::uint32_t readers, std::uint32_t concurrent_pins);
    /// Administrative; idempotent. Releases pins and drains in-flight fills before freeing storage.
    void close() noexcept;

    /** Releases the previous layer's pins, then starts fills for `experts` of `layer` (never blocks).
     *  @pre Every reader of the previously acquired experts has joined.
     */
    [[nodiscard]] sub0tieredcache::Status prefetch(int layer, std::span<const int> experts) noexcept;
    /// Blocks until selected expert `k` of the last prefetch is resident, then pins it.
    [[nodiscard]] sub0tieredcache::Status acquire(int k) noexcept;
    /// Pins selected expert `k` only if it is already resident; never blocks and never starts I/O.
    /// Lets decode compute resident experts while the others are still filling.
    [[nodiscard]] bool try_acquire(int k) noexcept;

    /** Blocks until `expert` of `layer` is resident, then pins it for the returned lease's lifetime.
     *  Thread-safe and independent of prefetch/acquire: for the batched forward() path, which decodes
     *  an expert from its bytes and releases it at once. Fails with pool_exhausted if more than
     *  `concurrent_pins` such leases are held at once.
     */
    [[nodiscard]] std::expected<sub0tieredcache::RowLease, sub0tieredcache::Status> pin(int layer, int expert) noexcept;
    /// Encoded bytes of plane `which` within a row returned by pin(layer, expert).
    [[nodiscard]] std::span<const std::uint8_t> plane(const sub0tieredcache::RowLease& row, int layer, int expert,
                                                      int which) const noexcept;
    /// Encoded bytes of plane `which` (moeq::Gate/Up/Down) of expert `k`.
    /// @pre acquire(k) succeeded since the last prefetch; the span is valid until the next prefetch.
    [[nodiscard]] std::span<const std::uint8_t> plane(int k, int which) const noexcept;

    [[nodiscard]] sub0tieredcache::TableStats stats() const noexcept;
    [[nodiscard]] std::uint32_t resident_rows() const noexcept { return budget_rows_; }
    /// Whether the rows are locked resident. Always true for an open cache: open() refuses otherwise.
    [[nodiscard]] bool pinned() const noexcept { return storage_.pinned; }
    /// Asks the OS which of the cache's pages are resident and locked right now. Administrative: walks
    /// every page. A pool that is not complete() means memory the cache owns was paged out.
    [[nodiscard]] Residency verify_resident() const noexcept { return storage_.verify(); }

private:
    /// Row r is expert r % num_experts of layer r / num_experts: the span from its gate plane's start
    /// to its down plane's end, which open() verified is contiguous.
    struct Resolver {
        const moeq::Store* store = nullptr; // non-owning; outlives the table
        [[nodiscard]] std::expected<sub0tieredcache::RowLocation, sub0tieredcache::Status>
        resolve_extent(std::uint64_t row) const noexcept;
    };
    /// The cache's RAM, pinned when the OS allows. An unpinned pool is private, dirty memory, so under
    /// pressure the OS writes it to the pagefile and a "hit" becomes a pagefile read: measured
    /// 2026-10-01, pagefile use rose 20% -> 47% and pages read in ran at 100-236k/s during a 10 GiB cache
    /// run under a RAM ballast, while reactive mmap's clean file pages are simply dropped instead.
    struct Storage {
        std::byte* data = nullptr; // owning
        std::size_t bytes = 0;
        bool pinned = false;       // locked resident (VirtualLock/mlock) rather than pageable
        Storage() = default;
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        ~Storage() { release(); }
        /// Allocates `size` bytes and tries to lock them; `pinned` reports whether the lock held.
        [[nodiscard]] bool reserve(std::size_t size) noexcept;
        void release() noexcept;
        [[nodiscard]] Residency verify() const noexcept { return residency::query(data, bytes, pinned); }
    };

    // Destruction order matters (reverse of declaration): leases, then table, then backend, then storage.
    Storage storage_;
    Resolver resolver_;
    std::unique_ptr<sub0mempage::LocalFileBackend> backend_;
    std::unique_ptr<sub0tieredcache::Table> table_;
    std::vector<std::uint64_t> rows_;                 // the last prefetch's rows, by selection index k
    std::vector<sub0tieredcache::RowLease> leases_;   // pins taken by acquire, by k
    std::uint32_t selected_ = 0;
    std::uint32_t budget_rows_ = 0;
    int layer_ = -1;
};

/// Diagnostic name for a TieredCache status.
[[nodiscard]] std::string_view status_name(sub0tieredcache::Status status) noexcept;

} // namespace sub0::moeio
