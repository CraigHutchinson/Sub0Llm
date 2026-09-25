#pragma once

/** @file ngram_tiered_storage.hpp
 *  @brief S1 (storage-stack plan, docs/STORAGE_STACK_PLAN.md "Joined-up development sequence"): a thin
 *  adapter over sub0tieredcache::Table for a flat, row-major, bf16-on-disk / f32-resolved external row
 *  table -- the shape a frozen external n-gram/PLE table needs (docs/NGRAM_TABLE_TIERED_STORAGE.md sec
 *  0's case 1, the real Qwen3.8-Flash-Next table). Single shard only (FlatFileResolver's flat-file
 *  case) -- sec 2d's sharded-shard-set generalization is not attempted here.
 *
 *  ISOLATED, per this pass's own scope: nothing in src/ or tools/ includes this header yet. Stage 3 of
 *  docs/NGRAM_TABLE_TIERED_STORAGE.md sec 5 ("thin-client op + resolve-pass wiring") is the future
 *  engine-wiring step; this pass's whole correctness gate is tests/ngram_tiered_storage_tests.cpp,
 *  built only under the default-OFF SUB0_STORAGE_TIEREDCACHE CMake option -- the same
 *  isolated-header-gated-by-its-own-test-target shape backbone_quant_dot.hpp / backbone_quant_dot_tests.cpp
 *  already established (see tests/CMakeLists.txt's comment on that target).
 *
 *  AGENTS.md sec 1 (no heap allocation on a per-call hot path): every buffer this adapter's
 *  resolve_rows() touches is caller-owned and sized exactly ONCE, at create() time -- the Table's own
 *  output_storage/scratch_storage spans, and this adapter's own lease scratch array (member, sized to
 *  max_batch_rows once, reused every call, never resized/reallocated per call). resolve_rows() itself
 *  performs no heap allocation; the only allocation on this whole path is Table::create()'s and
 *  LocalFileBackend::create()'s own administrative, once-only setup.
 */

#include <sub0mempage/local_file_backend.hpp>
#include <sub0tieredcache/local_file_source.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sub0::storage {

/// A flat row-major bf16-on-disk table, widened to f32 on resolve
/// (sub0tieredcache::Representation::bf16_to_f32). One registered file (single shard).
class NgramTieredAdapter {
    struct Passkey {};

public:
    struct Config {
        std::uint64_t row_count = 0;     ///< Addressable rows, [0, row_count).
        std::uint64_t row_width = 0;     ///< Elements per row (source dtype: bf16, 2 bytes/element).
        std::uint32_t budget_rows = 0;   ///< Resident (widened, f32) output-row capacity.
        std::uint32_t scratch_rows = 0;  ///< Bounded bf16 staging rows the widen codec reads through.
        std::uint32_t max_tickets = 1;
        std::uint32_t max_batch_rows = 0; ///< Bound on one resolve_rows() call.
        std::uint32_t backend_workers = 1;
        std::uint32_t backend_queue_capacity = 0;
    };

    [[nodiscard]] static std::expected<std::unique_ptr<NgramTieredAdapter>, sub0tieredcache::Status>
    create(const Config& config, const std::filesystem::path& source_path);

    NgramTieredAdapter(Passkey, const Config& config) noexcept
        : row_width_(config.row_width), max_batch_rows_(config.max_batch_rows) {}
    NgramTieredAdapter(const NgramTieredAdapter&) = delete;
    NgramTieredAdapter& operator=(const NgramTieredAdapter&) = delete;

    /** @brief Resolves `rows` (in request order; duplicates and non-monotonic order are both honoured,
     *  exactly as sub0tieredcache::Table::resolve_into itself honours them) and copies each row's
     *  widened f32 bytes into `out`, which must hold at least `rows.size() * row_width` floats --
     *  caller-owned, sized once outside this call (see the file comment). All-or-nothing: on any
     *  failure (including one out-of-range id anywhere in `rows`) `out` is left entirely unwritten,
     *  matching Table::resolve_into's own R14 contract -- this adapter never writes a partial result.
     */
    [[nodiscard]] std::expected<std::size_t, sub0tieredcache::Status>
    resolve_rows(std::span<const std::uint64_t> rows, std::span<float> out) noexcept;

    [[nodiscard]] sub0tieredcache::TableStats stats() const noexcept { return table_->stats(); }

private:
    std::uint64_t row_width_ = 0;
    std::uint32_t max_batch_rows_ = 0;
    std::unique_ptr<sub0mempage::LocalFileBackend> backend_;
    std::unique_ptr<sub0tieredcache::FlatFileResolver> resolver_;
    std::vector<std::byte> output_storage_;
    std::vector<std::byte> scratch_storage_;
    std::unique_ptr<sub0tieredcache::Table> table_;
    std::vector<sub0tieredcache::RowLease> lease_scratch_; // sized once to max_batch_rows (create())
};

inline std::expected<std::unique_ptr<NgramTieredAdapter>, sub0tieredcache::Status>
NgramTieredAdapter::create(const Config& config, const std::filesystem::path& source_path) {
    using sub0tieredcache::Status;

    if (config.row_count == 0 || config.row_width == 0 || config.budget_rows == 0 ||
        config.scratch_rows == 0 || config.max_batch_rows == 0) {
        return std::unexpected(Status::invalid_argument);
    }

    auto adapter = std::make_unique<NgramTieredAdapter>(Passkey{}, config);

    auto backend = sub0mempage::LocalFileBackend::create({.workers = config.backend_workers,
                                                            .queue_capacity = config.backend_queue_capacity,
                                                            .max_sources = 1});
    if (!backend) {
        return std::unexpected(Status::io_error);
    }
    adapter->backend_ = std::move(*backend);

    const std::uint64_t source_row_bytes = config.row_width * 2; // bf16 elements
    const std::uint64_t output_row_bytes = config.row_width * 4; // f32 elements

    auto shard = sub0tieredcache::register_local_file_shard(
        *adapter->backend_, static_cast<sub0mempage::SourceId>(1), source_path);
    if (!shard) {
        return std::unexpected(shard.error());
    }

    // Single shard, flat file, no padding: row_stride == row_width (source_row_bytes).
    adapter->resolver_ = std::make_unique<sub0tieredcache::FlatFileResolver>(
        config.row_count, /*rows_per_shard=*/config.row_count, /*row_stride=*/source_row_bytes,
        /*row_width=*/source_row_bytes, /*base_offset=*/0);

    adapter->output_storage_.resize(static_cast<std::size_t>(config.budget_rows) * output_row_bytes);
    adapter->scratch_storage_.resize(static_cast<std::size_t>(config.scratch_rows) * source_row_bytes);
    adapter->lease_scratch_.resize(config.max_batch_rows);

    const std::array<sub0tieredcache::ShardSource, 1> sources{*shard};
    const sub0tieredcache::TableConfig table_config{
        .row_count = config.row_count,
        .source_row_bytes = source_row_bytes,
        .output_row_bytes = output_row_bytes,
        .representation = sub0tieredcache::Representation::bf16_to_f32,
        .codec = nullptr,
        .sources = sources,
        .generation = 1,
        .resolve_extent = sub0tieredcache::RowExtentResolverRef(*adapter->resolver_),
        .output_storage = adapter->output_storage_,
        .budget_rows = config.budget_rows,
        .scratch_storage = adapter->scratch_storage_,
        .scratch_rows = config.scratch_rows,
        .max_tickets = config.max_tickets,
        .max_batch_rows = config.max_batch_rows,
    };

    auto table = sub0tieredcache::Table::create(table_config, sub0mempage::FillBackendRef(*adapter->backend_));
    if (!table) {
        return std::unexpected(table.error());
    }
    adapter->table_ = std::move(*table);
    return adapter;
}

inline std::expected<std::size_t, sub0tieredcache::Status>
NgramTieredAdapter::resolve_rows(std::span<const std::uint64_t> rows, std::span<float> out) noexcept {
    using sub0tieredcache::Status;

    if (rows.size() > max_batch_rows_ || out.size() < rows.size() * row_width_) {
        return std::unexpected(Status::invalid_argument);
    }

    const std::span<sub0tieredcache::RowLease> leases(lease_scratch_.data(), rows.size());
    const auto resolved = table_->resolve_into(rows, leases);
    if (!resolved) {
        return std::unexpected(resolved.error()); // R14: Table already left nothing pinned -- `out`
                                                    // is never touched below, so this call is all-or-nothing too.
    }

    for (std::size_t i = 0; i < rows.size(); ++i) {
        sub0tieredcache::RowLease& lease = leases[i];
        const std::span<const std::byte> bytes = lease.bytes();
        std::memcpy(out.data() + i * row_width_, bytes.data(), bytes.size());
        lease.reset(); // release the pin now that the caller's own flat buffer holds the bytes
    }
    return rows.size();
}

} // namespace sub0::storage
