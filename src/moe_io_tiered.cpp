#include "sub0/moe_io_tiered.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace sub0::moeio {
using sub0tieredcache::Status;

namespace {
constexpr auto kSidecarSource = static_cast<sub0mempage::SourceId>(1);
} // namespace

bool ExpertRowCache::Storage::reserve(std::size_t size) noexcept {
    release();
#if defined(_WIN32)
    void* p = ::VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
    void* p = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) p = nullptr;
#endif
    if (p == nullptr) return false;
    data = static_cast<std::byte*>(p);
    bytes = size;
    pinned = residency::pin(p, size);
    return true;
}

void ExpertRowCache::Storage::release() noexcept {
    if (data == nullptr) return;
#if defined(_WIN32)
    ::VirtualFree(data, 0, MEM_RELEASE); // also releases the lock
#else
    ::munmap(data, bytes);               // also releases the lock
#endif
    if (pinned) residency::release_quota(bytes);
    data = nullptr;
    bytes = 0;
    pinned = false;
}

std::expected<sub0tieredcache::RowLocation, Status>
ExpertRowCache::Resolver::resolve_extent(std::uint64_t row) const noexcept {
    const auto& h = store->header();
    const auto experts = static_cast<std::uint64_t>(h.num_experts);
    if (row >= experts * static_cast<std::uint64_t>(h.n_layers)) return std::unexpected(Status::out_of_range);
    const int layer = static_cast<int>(row / experts);
    const int expert = static_cast<int>(row % experts);
    const moeq::Desc& gate = store->desc(layer, expert, moeq::Gate);
    const moeq::Desc& down = store->desc(layer, expert, moeq::Down);
    return sub0tieredcache::RowLocation{
        0, sub0mempage::ByteRange{h.data_off + gate.off, down.off + down.bytes - gate.off}};
}

ExpertRowCache::~ExpertRowCache() { close(); }

Status ExpertRowCache::open(const std::filesystem::path& sidecar, const moeq::Store& store,
                            std::uint64_t budget_bytes, std::uint32_t max_selected, std::uint32_t readers,
                            std::uint32_t concurrent_pins, std::uint64_t fill_chunk_bytes,
                            sub0mempage::FileAccess access) {
    close();
    // Uncached fills read the block-aligned window around each expert, so every slot holds that window.
    const std::uint64_t alignment = access == sub0mempage::FileAccess::uncached ? sub0mempage::kUncachedAlignment : 0;
    const auto slot_bytes = [&](std::uint64_t expert_bytes) {
        return alignment != 0 ? sub0tieredcache::aligned_slot_bytes(expert_bytes, alignment) : expert_bytes;
    };
    const auto& h = store.header();
    if (max_selected == 0 || readers == 0 || h.num_experts <= 0 || h.n_layers <= 0) return Status::invalid_argument;
    const auto row_count = static_cast<std::uint64_t>(h.num_experts) * static_cast<std::uint64_t>(h.n_layers);

    // One row must be one read: refuse a sidecar whose expert planes are not stored back to back.
    std::uint64_t row_bytes = 0, table_bytes = 0;
    for (int layer = 0; layer < h.n_layers; ++layer) {
        for (int expert = 0; expert < h.num_experts; ++expert) {
            const moeq::Desc& gate = store.desc(layer, expert, moeq::Gate);
            const moeq::Desc& up = store.desc(layer, expert, moeq::Up);
            const moeq::Desc& down = store.desc(layer, expert, moeq::Down);
            if (up.off != gate.off + gate.bytes || down.off != up.off + up.bytes) return Status::invalid_argument;
            row_bytes = std::max(row_bytes, slot_bytes(gate.bytes + up.bytes + down.bytes));
            table_bytes += slot_bytes(gate.bytes + up.bytes + down.bytes);
        }
    }
    const std::uint64_t storage_bytes = std::min(budget_bytes, table_bytes);

    std::error_code error;
    const auto file_bytes = std::filesystem::file_size(sidecar, error);
    if (error) return Status::io_error;

    if (!storage_.reserve(static_cast<std::size_t>(storage_bytes))) return Status::invalid_argument;
    if (!storage_.pinned) { // see open()'s contract: an unpinned owned cache is worse than none
        close();
        return Status::pool_exhausted;
    }
    // Every chunk of every in-flight fill must fit in the queue: two layers' selections plus the pins.
    const auto chunks = static_cast<std::uint32_t>(
        fill_chunk_bytes != 0 && fill_chunk_bytes < row_bytes ? (row_bytes + fill_chunk_bytes - 1) / fill_chunk_bytes : 1);
    auto backend = sub0mempage::LocalFileBackend::create(
        {.workers = readers, .queue_capacity = (4 * max_selected + concurrent_pins) * chunks, .max_sources = 1});
    if (!backend) {
        close();
        return Status::io_error;
    }
    backend_ = std::move(*backend);
    if (backend_->register_file(kSidecarSource, sidecar, access) != sub0mempage::Status::ok) {
        close();
        return Status::io_error;
    }

    resolver_.store = &store;
    const auto sources = sub0tieredcache::single_source(kSidecarSource, file_bytes);
    sub0tieredcache::SizeClassedTableConfig config{};
    config.row_count = row_count;
    config.sources = sources;
    config.generation = 1;
    config.resolve_extent = sub0tieredcache::RowExtentResolverRef(resolver_);
    config.output_storage = std::span(storage_.data, storage_.bytes);
    config.max_tickets = 4;
    config.max_batch_rows = max_selected;
    config.fill_chunk_bytes = fill_chunk_bytes;
    config.fill_alignment = alignment;
    auto table = sub0tieredcache::SizeClassedTable::create(config, sub0mempage::FillBackendRef(*backend_));
    if (!table) {
        close();
        return table.error();
    }
    table_ = std::move(*table);
    // Every expert size must fit two layers' selections (the current layer stays pinned while the next
    // is prefetched), plus whatever pin() may hold concurrently: a layer's selection can be all one size.
    std::uint64_t resident = 0;
    for (const auto& cls : table_->classes()) {
        if (cls.budget_rows < std::min<std::uint64_t>(2ull * max_selected + concurrent_pins, cls.rows)) {
            close();
            return Status::invalid_argument;
        }
        resident += cls.budget_rows;
    }
    rows_.assign(max_selected, 0);
    leases_.resize(max_selected);
    budget_rows_ = static_cast<std::uint32_t>(resident);
    return Status::ok;
}

void ExpertRowCache::close() noexcept {
    for (auto& lease : leases_) lease.reset();
    if (table_) (void)table_->drain(); // fills still need the backend's workers to finish
    table_.reset();
    if (backend_) backend_->shutdown();
    backend_.reset();
    storage_.release();
    rows_.clear();
    leases_.clear();
    resolver_.store = nullptr;
    wait_count_.store(0, std::memory_order_relaxed);
    wait_ns_.store(0, std::memory_order_relaxed);
    selected_ = 0;
    budget_rows_ = 0;
    layer_ = -1;
}

Status ExpertRowCache::prefetch(int layer, std::span<const int> experts) noexcept {
    if (!table_) return Status::invalid_argument;
    if (experts.empty()) return Status::empty_range;
    if (experts.size() > rows_.size()) return Status::batch_too_large;
    for (std::uint32_t k = 0; k < selected_; ++k) leases_[k].reset();
    const auto per_layer = static_cast<std::uint64_t>(resolver_.store->header().num_experts);
    for (std::size_t k = 0; k < experts.size(); ++k)
        rows_[k] = static_cast<std::uint64_t>(layer) * per_layer + static_cast<std::uint64_t>(experts[k]);
    layer_ = layer;
    selected_ = static_cast<std::uint32_t>(experts.size());
    // Fills continue in the background; acquire() waits on the rows themselves.
    return table_->prefetch(std::span(rows_).first(selected_));
}

Status ExpertRowCache::acquire(int k) noexcept {
    if (k < 0 || static_cast<std::uint32_t>(k) >= selected_) return Status::out_of_range;
    const auto at = static_cast<std::size_t>(k);
    const auto start = std::chrono::steady_clock::now();
    const auto resolved = table_->resolve_into(std::span(rows_).subspan(at, 1), std::span(leases_).subspan(at, 1));
    const auto waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start);
    wait_count_.fetch_add(1, std::memory_order_relaxed);
    wait_ns_.fetch_add(static_cast<std::uint64_t>(waited.count()), std::memory_order_relaxed);
    return resolved ? Status::ok : resolved.error();
}

bool ExpertRowCache::try_acquire(int k) noexcept {
    if (k < 0 || static_cast<std::uint32_t>(k) >= selected_) return false;
    const auto at = static_cast<std::size_t>(k);
    auto lease = table_->try_get(rows_[at]);
    if (!lease) return false;
    leases_[at] = std::move(*lease);
    return true;
}

std::expected<sub0tieredcache::RowLease, Status> ExpertRowCache::pin(int layer, int expert) noexcept {
    if (!table_) return std::unexpected(Status::invalid_argument);
    const auto& h = resolver_.store->header();
    if (layer < 0 || layer >= h.n_layers || expert < 0 || expert >= h.num_experts)
        return std::unexpected(Status::out_of_range);
    const std::array<std::uint64_t, 1> row{static_cast<std::uint64_t>(layer) * static_cast<std::uint64_t>(h.num_experts) +
                                           static_cast<std::uint64_t>(expert)};
    std::array<sub0tieredcache::RowLease, 1> lease;
    if (const auto resolved = table_->resolve_into(row, lease); !resolved) return std::unexpected(resolved.error());
    return std::move(lease[0]);
}

std::span<const std::uint8_t> ExpertRowCache::plane(const sub0tieredcache::RowLease& row, int layer, int expert,
                                                    int which) const noexcept {
    const moeq::Desc& gate = resolver_.store->desc(layer, expert, moeq::Gate);
    const moeq::Desc& d = resolver_.store->desc(layer, expert, which);
    return {reinterpret_cast<const std::uint8_t*>(row.bytes().data()) + (d.off - gate.off), static_cast<std::size_t>(d.bytes)};
}

std::span<const std::uint8_t> ExpertRowCache::plane(int k, int which) const noexcept {
    const auto at = static_cast<std::size_t>(k);
    const auto expert = static_cast<int>(rows_[at] % static_cast<std::uint64_t>(resolver_.store->header().num_experts));
    const moeq::Desc& gate = resolver_.store->desc(layer_, expert, moeq::Gate);
    const moeq::Desc& d = resolver_.store->desc(layer_, expert, which);
    const auto* row = reinterpret_cast<const std::uint8_t*>(leases_[at].bytes().data());
    return {row + (d.off - gate.off), static_cast<std::size_t>(d.bytes)};
}

sub0tieredcache::TableStats ExpertRowCache::stats() const noexcept {
    return table_ ? table_->stats() : sub0tieredcache::TableStats{};
}

std::string_view status_name(Status status) noexcept {
    using enum Status;
    switch (status) {
    case ok: return "ok";
    case pending: return "pending";
    case not_resident: return "not_resident";
    case pool_exhausted: return "pool_exhausted";
    case batch_too_large: return "batch_too_large";
    case ticket_exhausted: return "ticket_exhausted";
    case out_of_range: return "out_of_range";
    case empty_range: return "empty_range";
    case invalid_argument: return "invalid_argument";
    case busy: return "busy";
    case short_read: return "short_read";
    case io_error: return "io_error";
    case cancelled: return "cancelled";
    case timeout: return "timeout";
    case codec_failed: return "codec_failed";
    case unsupported_conversion: return "unsupported_conversion";
    case declined: return "declined";
    }
    return "unknown";
}

} // namespace sub0::moeio
