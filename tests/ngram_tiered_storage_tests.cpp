// ngram_tiered_storage_tests.cpp -- Storage-stack S1 (docs/STORAGE_STACK_PLAN.md, the "Joined-up
// development sequence" S1 row applied to workload contract E1): the Sub0Llm-side correctness gate for
// include/sub0/ngram_tiered_storage.hpp, a thin adapter over sub0tieredcache::Table on real
// sub0mempage::LocalFileBackend local-file transport. Only built when SUB0_STORAGE_TIEREDCACHE is ON
// (default OFF, tests/CMakeLists.txt), which FetchContent's the two pinned sibling libraries at their
// pinned revisions (docs/STORAGE_STACK_PLAN.md "Current checkpoint" -> "Pinned lower revisions").
//
// E1's fixture shape (docs/STORAGE_STACK_PLAN.md step 2, mirroring the real Qwen4 n-gram table in
// miniature per docs/NGRAM_TABLE_TIERED_STORAGE.md sec 0 and
// tests/fixtures/qwen4_preview/ngram_embedding_manifest.json -- 160 elements/row, source dtype bf16,
// output dtype f32): a small, deterministic, generated external table written to a temp file by this
// test, an ordered request with duplicates/non-monotonic order/row-0/the-last-row, and a separate
// out-of-range request that must fail as a whole. The expected-bytes oracle is computed from the
// fixture's OWN generator formula below -- never from the adapter, sub0tieredcache::codec.hpp's
// bf16->f32 widening, or sub0::bf16_widen (include/sub0/bf16.hpp) -- and cross-checked a second,
// independent way against an engine-free re-expression of op_embed's own plain-gather convention
// (src/backends/cpu/backend.cpp's "o[t,j] = tab[ids[t]*C+j]" branch), the same choice
// tests/ngram_qwen4_fixture_tests.cpp already makes for the "concat" convention rather than touching
// backend.cpp itself (a shared file another agent's O8 track owns this pass).

#include <catch2/catch_test_macros.hpp>

#include <sub0/ngram_tiered_storage.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

// Deletes the fixture file on scope exit. Must outlive the adapter that reads it only in the sense of
// being declared BEFORE it in a test's local scope, so C++'s reverse-destruction-order rule closes the
// adapter's LocalFileBackend (and its open file handle) before this tries to remove the file --
// unlinking a still-open handle fails outright on Windows (no FILE_SHARE_DELETE was requested).
struct TempFile {
    std::filesystem::path path;
    explicit TempFile(std::filesystem::path p) : path(std::move(p)) {}
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec); // best-effort; a leaked temp file is not a test failure
    }
};

std::filesystem::path make_temp_path(std::string_view tag) {
    static std::atomic<std::uint64_t> counter{0};
    const auto now = static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    return std::filesystem::temp_directory_path() /
           ("sub0_s1_ngram_" + std::string(tag) + "_" + std::to_string(now) + "_" +
            std::to_string(counter.fetch_add(1)) + ".bin");
}

// The fixture's OWN generator formula (deliberately unrelated to any real hashing scheme, real n-gram
// formula, or either library's codec): a reproducible per-(row,col) 16-bit pattern, recomputable here
// without touching sub0::bf16_widen or sub0tieredcache::detail::bf16_bits_to_f32_bits.
constexpr std::uint16_t gen_bits(std::uint64_t row, std::uint64_t col) noexcept {
    const std::uint64_t mixed = row * 2654435761ull + col * 40503ull + 0x9E3779B9ull;
    return static_cast<std::uint16_t>((mixed ^ (mixed >> 17)) & 0xFFFFull);
}

// bf16 -> f32 widening is a 16-bit left shift (a bf16 bit pattern occupies an f32's high 16 bits) --
// re-derived independently here as the fixture's own oracle (docs/STORAGE_STACK_PLAN.md step 2: "an
// expected-bytes oracle derived from the generator formula, never from the adapter"), not borrowed from
// include/sub0/bf16.hpp's bf16_widen or codec.hpp's Bf16ToF32Codec, even though all three happen to
// compute the same bit-layout fact.
constexpr std::uint32_t gen_widen_bits(std::uint16_t bits) noexcept { return static_cast<std::uint32_t>(bits) << 16; }

float gen_expected_float(std::uint64_t row, std::uint64_t col) noexcept {
    const std::uint32_t f32_bits = gen_widen_bits(gen_bits(row, col));
    float value;
    std::memcpy(&value, &f32_bits, sizeof(value));
    return value;
}

bool write_fixture_file(const std::filesystem::path& path, std::uint64_t row_count, std::uint64_t row_width) {
    std::vector<std::uint16_t> bits(row_count * row_width);
    for (std::uint64_t r = 0; r < row_count; ++r) {
        for (std::uint64_t c = 0; c < row_width; ++c) {
            bits[r * row_width + c] = gen_bits(r, c);
        }
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(bits.data()), static_cast<std::streamsize>(bits.size() * sizeof(std::uint16_t)));
    return static_cast<bool>(f);
}

std::vector<float> build_resident_table(std::uint64_t row_count, std::uint64_t row_width) {
    std::vector<float> table(row_count * row_width);
    for (std::uint64_t r = 0; r < row_count; ++r) {
        for (std::uint64_t c = 0; c < row_width; ++c) {
            table[r * row_width + c] = gen_expected_float(r, c);
        }
    }
    return table;
}

// Re-expresses op_embed's plain-lookup branch (src/backends/cpu/backend.cpp: "const auto trow = tab +
// ids[t]*C; for j: o[t,j] = trow[j]", where `tab` is the resident table already widened through
// ParamCPtr's sub0::bf16_widen) engine-free -- NOT a call into backend.cpp (a shared file this pass
// must not touch; op_embed is also `static`, not exported). `resident_table` must already be the
// widened (f32) resident table, exactly as ParamCPtr presents it to the real op_embed.
void resident_gather(const std::vector<float>& resident_table, std::uint64_t row_width,
                      std::span<const std::uint64_t> ids, std::vector<float>& out) {
    out.resize(ids.size() * row_width);
    for (std::size_t t = 0; t < ids.size(); ++t) {
        const float* row = resident_table.data() + ids[t] * row_width;
        std::copy(row, row + row_width, out.data() + t * row_width);
    }
}

constexpr std::uint64_t kRowCount = 24;
constexpr std::uint64_t kRowWidth = 160; // mirrors the real Qwen4 head_dim_per_ngram, kept row-count small
constexpr std::uint32_t kMaxBatchRows = 16; // mirrors "16 rows per token position" (manifest's ngram_heads)

sub0::storage::NgramTieredAdapter::Config test_config() {
    return sub0::storage::NgramTieredAdapter::Config{
        .row_count = kRowCount,
        .row_width = kRowWidth,
        .budget_rows = static_cast<std::uint32_t>(kRowCount),
        .scratch_rows = static_cast<std::uint32_t>(kRowCount),
        .max_tickets = 4,
        .max_batch_rows = kMaxBatchRows,
        .backend_workers = 2,
        .backend_queue_capacity = 32,
    };
}

} // namespace

TEST_CASE("S1 storage-stack: adapter matches the generator oracle and the resident reference, one fetch per duplicate row",
          "[storage][tiered_cache]") {
    TempFile fixture(make_temp_path("adapter_oracle"));
    REQUIRE(write_fixture_file(fixture.path, kRowCount, kRowWidth));

    auto created = sub0::storage::NgramTieredAdapter::create(test_config(), fixture.path);
    REQUIRE(created.has_value());
    auto adapter = std::move(*created);

    // Ordered request: duplicates (5 and 23 each appear twice), non-monotonic order (12 then 5), row 0
    // and the last row (23 == kRowCount - 1).
    const std::vector<std::uint64_t> rows{0, 5, 12, 5, 23, 1, 23, 9};
    REQUIRE(rows.front() == 0);
    REQUIRE(rows[4] == kRowCount - 1);

    std::vector<float> out(rows.size() * kRowWidth, -1.0f);
    const auto resolved = adapter->resolve_rows(rows, out);
    REQUIRE(resolved.has_value());
    REQUIRE(*resolved == rows.size());

    // (1) Generator-oracle comparison: bit-exact against bytes derived straight from the fixture's own
    // formula -- never from the adapter or either library's codec.
    std::vector<float> expected(rows.size() * kRowWidth);
    for (std::size_t t = 0; t < rows.size(); ++t) {
        for (std::uint64_t c = 0; c < kRowWidth; ++c) {
            expected[t * kRowWidth + c] = gen_expected_float(rows[t], c);
        }
    }
    REQUIRE(std::memcmp(out.data(), expected.data(), expected.size() * sizeof(float)) == 0);

    // (2) Resident-reference comparison: op_embed's own plain-gather convention, re-expressed engine-free.
    const std::vector<float> resident_table = build_resident_table(kRowCount, kRowWidth);
    std::vector<float> resident_out;
    resident_gather(resident_table, kRowWidth, rows, resident_out);
    REQUIRE(std::memcmp(out.data(), resident_out.data(), resident_out.size() * sizeof(float)) == 0);

    // (3) One fetch per DISTINCT row: 0, 5, 12, 23, 1, 9 (6 distinct); the second "5" and second "23"
    // each coalesce onto the fetch already in flight for that row rather than starting a new one.
    const sub0tieredcache::TableStats stats = adapter->stats();
    REQUIRE(stats.fetches == 6);
    REQUIRE(stats.coalesced == 2);
}

TEST_CASE("S1 storage-stack: an out-of-range row fails resolve_rows as a whole and leaves the output untouched",
          "[storage][tiered_cache]") {
    TempFile fixture(make_temp_path("adapter_oob"));
    REQUIRE(write_fixture_file(fixture.path, kRowCount, kRowWidth));

    auto created = sub0::storage::NgramTieredAdapter::create(test_config(), fixture.path);
    REQUIRE(created.has_value());
    auto adapter = std::move(*created);

    // kRowCount itself is one past the valid range [0, kRowCount); the other two ids are ordinary valid
    // rows -- the whole request must still fail, not just the one bad id (R14, all-or-nothing).
    const std::vector<std::uint64_t> rows{2, kRowCount, 7};
    std::vector<float> out(rows.size() * kRowWidth);
    constexpr float kSentinel = -12345.0f;
    std::fill(out.begin(), out.end(), kSentinel);

    const auto resolved = adapter->resolve_rows(rows, out);
    REQUIRE_FALSE(resolved.has_value());
    REQUIRE(resolved.error() == sub0tieredcache::Status::out_of_range);

    // All-or-nothing: `out` was never written to, not even for the two otherwise-valid ids.
    REQUIRE(std::all_of(out.begin(), out.end(), [](float v) { return v == kSentinel; }));

    // Table checks every id up front (row_cache.hpp's resolve_into): nothing was ever admitted, so no
    // fetch was attempted for the two valid ids either.
    const sub0tieredcache::TableStats stats = adapter->stats();
    REQUIRE(stats.fetches == 0);
    REQUIRE(stats.coalesced == 0);
}
