// moe_io_mempage_tests.cpp -- Storage-stack S1b step 4 (docs/STORAGE_STACK_PLAN.md "Next: S1b routed-MoE
// byte adapter"): acceptance for include/sub0/moe_io_mempage.hpp BEFORE it is wired into decode. Only
// built when SUB0_STORAGE_TIEREDCACHE is ON (default OFF, tests/CMakeLists.txt).
//
// Two independent oracles, never the adapter itself: a plain std::ifstream positional read, and the
// existing comparison implementation `moeio::PlaneIo` (include/sub0/moe_io.hpp) that the adapter is
// meant to be interchangeable with. The fixture is a deterministic three-planes-per-expert file with a
// deliberately odd tail so the last expert's planes end exactly at EOF.

#include <catch2/catch_test_macros.hpp>

#include <sub0/moe_io.hpp>
#include <sub0/moe_io_mempage.hpp>
#include <sub0/moe_quant.hpp>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace {

using sub0mempage::Status;
using sub0::moeio::MemPagePlaneIo;
using sub0::moeio::Request;

constexpr std::uint32_t kExperts = 7;
constexpr std::array<std::uint32_t, 3> kPlaneBytes{4096 + 17, 8192 - 3, 1031}; // gate, up, down; unaligned
constexpr std::uint64_t kExpertBytes = kPlaneBytes[0] + kPlaneBytes[1] + kPlaneBytes[2];
constexpr std::uint64_t kHeaderBytes = 61; // payload does not start at 0, like a real sidecar

constexpr std::uint8_t fixture_byte(std::uint64_t i) {
    return static_cast<std::uint8_t>((i * 131u) ^ (i >> 8) ^ (i >> 16));
}

std::uint64_t plane_offset(std::uint32_t expert, std::size_t plane) {
    std::uint64_t off = kHeaderBytes + expert * kExpertBytes;
    for (std::size_t p = 0; p < plane; ++p) off += kPlaneBytes[p];
    return off;
}

// Removes the fixture on scope exit; declared before any reader so the reader closes its handle first.
struct TempFile {
    std::filesystem::path path;
    explicit TempFile(const std::string& name, std::uint64_t bytes)
        : path(std::filesystem::temp_directory_path() / name) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        for (std::uint64_t i = 0; i < bytes; ++i) out.put(static_cast<char>(fixture_byte(i)));
    }
    ~TempFile() { std::error_code ec; std::filesystem::remove(path, ec); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
};

std::vector<std::uint8_t> positional_read(const std::filesystem::path& path, std::uint64_t off,
                                          std::uint32_t bytes) {
    std::vector<std::uint8_t> out(bytes);
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(off));
    in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    REQUIRE(in.gcount() == static_cast<std::streamsize>(bytes));
    return out;
}

// One destination per tag, sized to the largest plane, the shape decode's staging buffer has.
struct Staging {
    std::vector<std::vector<std::byte>> buffers;
    std::vector<std::span<std::byte>> spans;
    explicit Staging(std::size_t tags) : buffers(tags, std::vector<std::byte>(8192)) {
        for (auto& b : buffers) spans.emplace_back(b);
    }
};

std::vector<Request> plane_requests(std::span<const std::uint32_t> experts, Staging& staging) {
    std::vector<Request> requests;
    for (const std::uint32_t e : experts)
        for (std::size_t p = 0; p < kPlaneBytes.size(); ++p) {
            const std::size_t tag = requests.size();
            requests.push_back({plane_offset(e, p), kPlaneBytes[p],
                                reinterpret_cast<std::uint8_t*>(staging.spans[tag].data())});
        }
    return requests;
}

bool equal_bytes(std::span<const std::byte> got, const std::vector<std::uint8_t>& want) {
    return got.size() >= want.size() &&
           std::equal(want.begin(), want.end(), reinterpret_cast<const std::uint8_t*>(got.data()));
}

} // namespace

TEST_CASE("MemPagePlaneIo: selected planes match positional reads and PlaneIo", "[moeio][mempage]") {
    const TempFile file("sub0_moeio_mempage_exact.bin", kHeaderBytes + kExperts * kExpertBytes);
    // Duplicates, non-monotonic order, expert 0 and the last expert (whose down plane ends at EOF).
    const std::array<std::uint32_t, 5> selected{5, 2, 5, 0, kExperts - 1};
    constexpr std::size_t kTags = selected.size() * kPlaneBytes.size();

    Staging staging(kTags);
    MemPagePlaneIo io;
    REQUIRE(io.open(file.path, staging.spans) == Status::ok);
    const auto requests = plane_requests(selected, staging);
    REQUIRE(io.submit(requests) == Status::ok);
    for (std::size_t i = kTags; i-- > 0;) REQUIRE(io.wait(static_cast<int>(i)) == Status::ok); // reverse order

    Staging oracle_staging(kTags);
    sub0::moeio::PlaneIo<static_cast<int>(kTags)> oracle;
    std::string err;
    REQUIRE(oracle.open(file.path.string(), err));
    const auto oracle_requests = plane_requests(selected, oracle_staging);
    REQUIRE(oracle.submit(oracle_requests, err));
    for (std::size_t i = 0; i < kTags; ++i) REQUIRE(oracle.wait(static_cast<int>(i), err));

    for (std::size_t i = 0; i < kTags; ++i) {
        const auto want = positional_read(file.path, requests[i].abs_off, requests[i].bytes);
        CHECK(equal_bytes(staging.spans[i], want));
        CHECK(equal_bytes(oracle_staging.spans[i], want));
    }
    CHECK(io.retire_batch() == Status::ok);
}

TEST_CASE("MemPagePlaneIo: batches are reused only after retirement", "[moeio][mempage]") {
    const TempFile file("sub0_moeio_mempage_reuse.bin", kHeaderBytes + kExperts * kExpertBytes);
    Staging staging(6);
    MemPagePlaneIo io;
    REQUIRE(io.open(file.path, staging.spans) == Status::ok);

    const std::array<std::uint32_t, 2> first{3, 1};
    const auto first_requests = plane_requests(first, staging);
    REQUIRE(io.submit(first_requests) == Status::ok);
    // Unretired: the compute readers may still hold these buffers, so nothing may be rewritten.
    CHECK(io.submit(first_requests) == Status::busy);
    for (int i = 0; i < 6; ++i) REQUIRE(io.wait(i) == Status::ok);
    REQUIRE(io.retire_batch() == Status::ok);

    for (std::uint32_t round = 0; round < 20; ++round) {
        const std::array<std::uint32_t, 2> next{round % kExperts, (round * 3 + 1) % kExperts};
        const auto requests = plane_requests(next, staging);
        REQUIRE(io.submit(requests) == Status::ok);
        for (int i = 0; i < 6; ++i) REQUIRE(io.wait(i) == Status::ok);
        for (std::size_t i = 0; i < requests.size(); ++i)
            CHECK(equal_bytes(staging.spans[i],
                              positional_read(file.path, requests[i].abs_off, requests[i].bytes)));
        REQUIRE(io.retire_batch() == Status::ok);
    }
    // A shorter batch uses a prefix of the registered tags; unsubmitted tags are out of range.
    const std::array<std::uint32_t, 1> one{6};
    REQUIRE(io.submit(plane_requests(one, staging)) == Status::ok);
    CHECK(io.wait(3) == Status::out_of_range);
    for (int i = 0; i < 3; ++i) REQUIRE(io.wait(i) == Status::ok);
    REQUIRE(io.retire_batch() == Status::ok);
}

TEST_CASE("MemPagePlaneIo: invalid batches are refused before any read is issued", "[moeio][mempage]") {
    const std::uint64_t file_bytes = kHeaderBytes + kExperts * kExpertBytes;
    const TempFile file("sub0_moeio_mempage_invalid.bin", file_bytes);
    Staging staging(3);
    MemPagePlaneIo io;
    CHECK(io.submit({}) == Status::invalid_argument); // closed
    REQUIRE(io.open(file.path, staging.spans) == Status::ok);

    auto dst = [&](std::size_t tag) { return reinterpret_cast<std::uint8_t*>(staging.spans[tag].data()); };
    const std::array<Request, 1> past_eof{{{file_bytes - 10, 11, dst(0)}}};
    CHECK(io.submit(past_eof) == Status::out_of_range);
    const std::array<Request, 1> overflow{{{UINT64_MAX - 4, 8, dst(0)}}};
    CHECK(io.submit(overflow) == Status::out_of_range);
    const std::array<Request, 1> too_big_for_slot{{{0, 8193, dst(0)}}};
    CHECK(io.submit(too_big_for_slot) == Status::out_of_range);
    const std::array<Request, 1> zero{{{0, 0, dst(0)}}};
    CHECK(io.submit(zero) == Status::empty_range);
    const std::array<Request, 1> wrong_tag_buffer{{{0, 16, dst(1)}}};
    CHECK(io.submit(wrong_tag_buffer) == Status::invalid_argument);
    CHECK(io.submit({}) == Status::empty_range);
    const std::array<Request, 4> too_many{{{0, 1, dst(0)}, {0, 1, dst(1)}, {0, 1, dst(2)}, {0, 1, dst(2)}}};
    CHECK(io.submit(too_many) == Status::batch_too_large);

    // None of the refusals left a batch behind: the last byte of the file is still readable.
    const std::array<Request, 1> last{{{file_bytes - 1, 1, dst(0)}}};
    REQUIRE(io.submit(last) == Status::ok);
    REQUIRE(io.wait(0) == Status::ok);
    CHECK(static_cast<std::uint8_t>(staging.spans[0][0]) == fixture_byte(file_bytes - 1));
    REQUIRE(io.retire_batch() == Status::ok);
}

TEST_CASE("MemPagePlaneIo: open validates registrations and sessions repeat", "[moeio][mempage]") {
    const TempFile a("sub0_moeio_mempage_session_a.bin", kHeaderBytes + kExperts * kExpertBytes);
    const TempFile b("sub0_moeio_mempage_session_b.bin", 4096);
    Staging staging(3);
    MemPagePlaneIo io;

    CHECK(io.open(a.path.string() + ".missing", staging.spans) == Status::io_error);
    CHECK(io.open(a.path, {}) == Status::invalid_argument);
    const std::array<std::span<std::byte>, 2> overlapping{staging.spans[0], staging.spans[0].subspan(100)};
    CHECK(io.open(a.path, overlapping) == Status::invalid_argument);
    const std::array<std::span<std::byte>, 1> empty_slot{std::span<std::byte>{}};
    CHECK(io.open(a.path, empty_slot) == Status::invalid_argument);

    for (int session = 0; session < 5; ++session) {
        const auto& file = (session % 2 == 0) ? a : b;
        REQUIRE(io.open(file.path, staging.spans) == Status::ok);
        const std::array<Request, 3> requests{{
            {0, 4096, reinterpret_cast<std::uint8_t*>(staging.spans[0].data())},
            {100, 7, reinterpret_cast<std::uint8_t*>(staging.spans[1].data())},
            {4095, 1, reinterpret_cast<std::uint8_t*>(staging.spans[2].data())}}};
        REQUIRE(io.submit(requests) == Status::ok);
        if (session == 3) {
            // Close with the batch outstanding: close() must drain writers before returning.
            io.close();
            continue;
        }
        for (int i = 0; i < 3; ++i) REQUIRE(io.wait(i) == Status::ok);
        for (std::size_t i = 0; i < requests.size(); ++i)
            CHECK(equal_bytes(staging.spans[i],
                              positional_read(file.path, requests[i].abs_off, requests[i].bytes)));
        REQUIRE(io.retire_batch() == Status::ok);
    }
    io.close();
    io.close(); // idempotent
    CHECK(io.retire_batch() == Status::ok);
}

TEST_CASE("MemPagePlaneIo: real S0Q1 planes preserve encoded and decoded values", "[moeio][mempage][real]") {
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
    sub0::moeq::Store store;
    std::string error;
    REQUIRE(store.open(path, error));
    const auto& header = store.header();
    REQUIRE(header.num_experts >= 2);
    const std::array<int, 3> experts{header.num_experts - 1, 0, header.num_experts - 1};
    constexpr std::size_t tags = experts.size() * sub0::moeq::PerExpert;
    std::array<std::vector<std::byte>, tags> buffers;
    std::array<std::span<std::byte>, tags> destinations;
    std::array<Request, tags> requests;
    for (std::size_t k = 0; k < experts.size(); ++k) {
        for (int plane = 0; plane < sub0::moeq::PerExpert; ++plane) {
            const auto& desc = store.desc(0, experts[k], plane);
            REQUIRE(desc.bytes <= UINT32_MAX);
            const auto tag = k * sub0::moeq::PerExpert + plane;
            buffers[tag].resize(static_cast<std::size_t>(desc.bytes));
            destinations[tag] = buffers[tag];
            requests[tag] = {header.data_off + desc.off, static_cast<std::uint32_t>(desc.bytes),
                             reinterpret_cast<std::uint8_t*>(buffers[tag].data())};
        }
    }
    MemPagePlaneIo io;
    REQUIRE(io.open(path, destinations) == Status::ok);
    REQUIRE(io.submit(requests) == Status::ok);
    for (std::size_t i = tags; i-- > 0;) REQUIRE(io.wait(static_cast<int>(i)) == Status::ok);
    for (std::size_t k = 0; k < experts.size(); ++k) {
        for (int plane = 0; plane < sub0::moeq::PerExpert; ++plane) {
            const auto tag = k * sub0::moeq::PerExpert + plane;
            const auto& desc = store.desc(0, experts[k], plane);
            const auto reference = positional_read(path, requests[tag].abs_off, requests[tag].bytes);
            REQUIRE(equal_bytes(destinations[tag], reference));
            const auto mapped = store.raw(desc);
            REQUIRE(std::equal(reference.begin(), reference.end(), mapped.begin(), mapped.end()));
            const std::span<const std::uint8_t> staged(requests[tag].dst, requests[tag].bytes);
            std::vector<float> want, got;
            REQUIRE(sub0::moeq::dequantize_expert_source(desc, mapped, want));
            REQUIRE(sub0::moeq::dequantize_expert_source(desc, staged, got));
            REQUIRE(got == want);
        }
    }
    REQUIRE(io.retire_batch() == Status::ok);
}
