#pragma once

#include "sub0/moe_quant.hpp"

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
     *  @return invalid_argument if an expert's planes are not contiguous in the file, if the budget
     *          holds fewer than two layers' selections (one pinned while the next is prefetched), or if
     *          the OS cannot commit the budget's storage.
     *  @note Administrative: allocates and may block. Any previous session is closed first.
     */
    [[nodiscard]] sub0tieredcache::Status open(const std::filesystem::path& sidecar, const moeq::Store& store,
                                               std::uint64_t budget_bytes, std::uint32_t max_selected,
                                               std::uint32_t readers);
    /// Administrative; idempotent. Releases pins and drains in-flight fills before freeing storage.
    void close() noexcept;

    /** Releases the previous layer's pins, then starts fills for `experts` of `layer` (never blocks).
     *  @pre Every reader of the previously acquired experts has joined.
     */
    [[nodiscard]] sub0tieredcache::Status prefetch(int layer, std::span<const int> experts) noexcept;
    /// Blocks until selected expert `k` of the last prefetch is resident, then pins it.
    [[nodiscard]] sub0tieredcache::Status acquire(int k) noexcept;
    /// Encoded bytes of plane `which` (moeq::Gate/Up/Down) of expert `k`.
    /// @pre acquire(k) succeeded since the last prefetch; the span is valid until the next prefetch.
    [[nodiscard]] std::span<const std::uint8_t> plane(int k, int which) const noexcept;

    [[nodiscard]] sub0tieredcache::TableStats stats() const noexcept;
    [[nodiscard]] std::uint32_t resident_rows() const noexcept { return budget_rows_; }

private:
    /// Row r is expert r % num_experts of layer r / num_experts: the span from its gate plane's start
    /// to its down plane's end, which open() verified is contiguous.
    struct Resolver {
        const moeq::Store* store = nullptr; // non-owning; outlives the table
        [[nodiscard]] std::expected<sub0tieredcache::RowLocation, sub0tieredcache::Status>
        resolve_extent(std::uint64_t row) const noexcept;
    };
    /// Page-granular RAM committed on first touch, so a multi-GB budget costs nothing until filled.
    struct Storage {
        std::byte* data = nullptr; // owning
        std::size_t bytes = 0;
        Storage() = default;
        Storage(const Storage&) = delete;
        Storage& operator=(const Storage&) = delete;
        ~Storage() { release(); }
        [[nodiscard]] bool reserve(std::size_t size) noexcept;
        void release() noexcept;
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
