// moe_io_tiered_tests.cpp -- the owned routed-expert cache (include/sub0/moe_io_tiered.hpp) against
// independent positional reads of the sidecar file, never against the cache itself. Built with the
// storage tests and in any engine build configured --moe-io-mode cache (tests/CMakeLists.txt).
//
// The synthetic sidecars give layers two different expert sizes, as the real Qwen4 sidecar has
// (1,561,600 and 1,766,400 bytes), so every case exercises bounded-extent rows. The budget holds fewer
// rows than the table, so walking all layers forces eviction and slot reuse.

#include <catch2/catch_test_macros.hpp>

#include <sub0/moe_io_tiered.hpp>
#include <sub0/moe_quant.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace moeq = sub0::moeq;
using sub0::moeio::ExpertRowCache;
using sub0tieredcache::Status;

constexpr int kLayers = 3;
constexpr int kExperts = 4;
constexpr std::uint32_t kSelected = 2;

constexpr std::uint8_t payload_byte(std::uint64_t i) { return static_cast<std::uint8_t>((i * 167u) ^ (i >> 9)); }

// Plane sizes differ by layer parity and by plane, and are deliberately not multiples of anything.
constexpr std::uint64_t plane_bytes(int layer, int which) {
    return (layer % 2 == 0 ? 1000u : 1300u) + 211u * static_cast<std::uint64_t>(which);
}

// Removes the file on scope exit; declared before the cache so the cache closes its handle first.
struct Sidecar {
    std::filesystem::path path;
    Sidecar(const char* name, std::uint64_t gap_after_gate)
        : path(std::filesystem::temp_directory_path() / name) {
        const std::size_t n = static_cast<std::size_t>(kLayers) * kExperts * moeq::PerExpert;
        std::vector<moeq::Desc> descs(n);
        std::uint64_t cursor = 0;
        for (int layer = 0; layer < kLayers; ++layer)
            for (int expert = 0; expert < kExperts; ++expert)
                for (int which = 0; which < moeq::PerExpert; ++which) {
                    const auto bytes = plane_bytes(layer, which);
                    descs[moeq::desc_index(kExperts, layer, expert, which)] = moeq::Desc{0, 1, 1, 0, cursor, bytes};
                    cursor += bytes + (which == moeq::Gate ? gap_after_gate : 0);
                }
        moeq::Header h;
        h.n_layers = kLayers;
        h.num_experts = kExperts;
        h.n_tensors = n;
        h.data_off = sizeof(moeq::Header) + n * sizeof(moeq::Desc);
        h.data_bytes = cursor;
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        os.write(reinterpret_cast<const char*>(&h), sizeof h);
        os.write(reinterpret_cast<const char*>(descs.data()), static_cast<std::streamsize>(n * sizeof(moeq::Desc)));
        for (std::uint64_t i = 0; i < cursor; ++i) os.put(static_cast<char>(payload_byte(i)));
    }
    ~Sidecar() { std::error_code ec; std::filesystem::remove(path, ec); }
    Sidecar(const Sidecar&) = delete;
    Sidecar& operator=(const Sidecar&) = delete;
};

std::vector<std::uint8_t> positional_read(const std::filesystem::path& path, std::uint64_t off, std::uint64_t bytes) {
    std::vector<std::uint8_t> out(bytes);
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(off));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    REQUIRE(in.gcount() == static_cast<std::streamsize>(bytes));
    return out;
}

bool plane_matches_file(const ExpertRowCache& cache, const moeq::Store& store, const std::filesystem::path& path,
                        int k, int layer, int expert) {
    for (int which = 0; which < moeq::PerExpert; ++which) {
        const moeq::Desc& d = store.desc(layer, expert, which);
        const auto got = cache.plane(k, which);
        const auto want = positional_read(path, store.header().data_off + d.off, d.bytes);
        if (got.size() != want.size() || !std::equal(want.begin(), want.end(), got.begin())) return false;
    }
    return true;
}

std::uint64_t largest_expert_bytes() {
    std::uint64_t largest = 0;
    for (int layer = 0; layer < kLayers; ++layer)
        largest = std::max(largest, plane_bytes(layer, 0) + plane_bytes(layer, 1) + plane_bytes(layer, 2));
    return largest;
}

} // namespace

TEST_CASE("ExpertRowCache: every plane matches the file across eviction and reuse", "[moeio][tiered]") {
    const Sidecar file("sub0_moe_cache_bytes.moeq", 0);
    moeq::Store store;
    std::string error;
    REQUIRE(store.open(file.path.string(), error));
    ExpertRowCache cache;
    // Four rows of twelve: two layers' selections, so walking every layer evicts and reuses slots.
    REQUIRE(cache.open(file.path, store, 4 * largest_expert_bytes(), kSelected, 2) == Status::ok);
    REQUIRE(cache.resident_rows() == 4);

    for (int pass = 0; pass < 2; ++pass) {
        for (int layer = 0; layer < kLayers; ++layer) {
            const std::array<int, kSelected> experts{(layer + pass) % kExperts, (layer + 3) % kExperts};
            REQUIRE(cache.prefetch(layer, experts) == Status::ok);
            for (int k = static_cast<int>(kSelected); k-- > 0;) { // out of selection order
                REQUIRE(cache.acquire(k) == Status::ok);
                CHECK(plane_matches_file(cache, store, file.path, k, layer, experts[static_cast<std::size_t>(k)]));
            }
        }
    }
}

TEST_CASE("ExpertRowCache: a resident expert is served without a second read", "[moeio][tiered]") {
    const Sidecar file("sub0_moe_cache_hit.moeq", 0);
    moeq::Store store;
    std::string error;
    REQUIRE(store.open(file.path.string(), error));
    ExpertRowCache cache;
    REQUIRE(cache.open(file.path, store, 12 * largest_expert_bytes(), kSelected, 2) == Status::ok);

    const std::array<int, kSelected> experts{2, 1};
    REQUIRE(cache.prefetch(1, experts) == Status::ok);
    for (int k = 0; k < static_cast<int>(kSelected); ++k) REQUIRE(cache.acquire(k) == Status::ok);
    const auto fetched = cache.stats().fetches;
    CHECK(fetched == kSelected);

    REQUIRE(cache.prefetch(1, experts) == Status::ok); // same layer and experts again
    for (int k = 0; k < static_cast<int>(kSelected); ++k) {
        REQUIRE(cache.acquire(k) == Status::ok);
        CHECK(plane_matches_file(cache, store, file.path, k, 1, experts[static_cast<std::size_t>(k)]));
    }
    CHECK(cache.stats().fetches == fetched);
}

TEST_CASE("ExpertRowCache: open refuses layouts and budgets it cannot serve", "[moeio][tiered]") {
    const Sidecar gapped("sub0_moe_cache_gapped.moeq", 64);
    moeq::Store gapped_store;
    std::string error;
    REQUIRE(gapped_store.open(gapped.path.string(), error));
    ExpertRowCache cache;
    CHECK(cache.open(gapped.path, gapped_store, 12 * largest_expert_bytes(), kSelected, 2) ==
          Status::invalid_argument); // planes not contiguous: one row would not be one read

    const Sidecar file("sub0_moe_cache_budget.moeq", 0);
    moeq::Store store;
    REQUIRE(store.open(file.path.string(), error));
    CHECK(cache.open(file.path, store, 3 * largest_expert_bytes(), kSelected, 2) == Status::invalid_argument);
    CHECK(cache.open(file.path, store, 12 * largest_expert_bytes(), 0, 2) == Status::invalid_argument);
    CHECK(cache.prefetch(0, std::array<int, 1>{0}) == Status::invalid_argument); // closed
    REQUIRE(cache.open(file.path, store, 12 * largest_expert_bytes(), kSelected, 2) == Status::ok);
    CHECK(cache.prefetch(0, std::array<int, 3>{0, 1, 2}) == Status::batch_too_large);
    CHECK(cache.acquire(0) == Status::out_of_range); // nothing prefetched yet
}

TEST_CASE("ExpertRowCache: real S0Q1 experts match the mapped sidecar", "[moeio][tiered][real]") {
    // Local deprecation silencing, as tests/backbone_quant_dot_tests.cpp's real_gguf_dir() does.
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    const char* path = std::getenv("SUB0_QWEN4_MOEQ_PATH");
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    if (!path || !*path) SKIP("Set SUB0_QWEN4_MOEQ_PATH to qualify a real S0Q1 sidecar");
    moeq::Store store;
    std::string error;
    REQUIRE(store.open(path, error));
    const auto& h = store.header();
    ExpertRowCache cache;
    REQUIRE(cache.open(path, store, std::uint64_t{1} << 30, 3, 3) == Status::ok); // 1 GiB of rows
    const std::array<int, 3> experts{h.num_experts - 1, 0, 7};
    for (const int layer : {0, h.n_layers - 1}) {
        REQUIRE(cache.prefetch(layer, experts) == Status::ok);
        for (int k = 0; k < 3; ++k) {
            REQUIRE(cache.acquire(k) == Status::ok);
            for (int which = 0; which < moeq::PerExpert; ++which) {
                const auto mapped = store.raw(store.desc(layer, experts[static_cast<std::size_t>(k)], which));
                const auto got = cache.plane(k, which);
                REQUIRE(std::equal(mapped.begin(), mapped.end(), got.begin(), got.end()));
            }
        }
    }
}
