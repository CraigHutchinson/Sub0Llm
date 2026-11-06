// quant_registry_tests.cpp -- generic conformance for the quantisation-format registry in sub0/gguf.hpp
// (kFormats geometry + kDequant dequantisers). These tests ITERATE the tables, so a newly registered
// format is covered by every check below with no new test code; per-format MATH (a format's values
// against an independent reference) still belongs in that format's own tests. See docs/QUANT_FORMATS.md.

#include "sub0/gguf.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace sub0::gguf;

namespace {
bool is_block_quantised(const FormatInfo& f) { return f.block.elems > 1; }

std::vector<std::uint8_t> filled(std::size_t n, std::uint8_t v) { return std::vector<std::uint8_t>(n, v); }

// Deterministic non-trivial bytes (a fixed LCG), so a decoder is exercised beyond the all-zero block.
std::vector<std::uint8_t> noise(std::size_t n) {
    std::vector<std::uint8_t> v(n);
    std::uint32_t s = 0x9e3779b9u;
    for (auto& b : v) { s = s * 1664525u + 1013904223u; b = static_cast<std::uint8_t>(s >> 24); }
    return v;
}
}  // namespace

TEST_CASE("quant registry: ids and names are unique, geometry is sane", "[quant][registry]") {
    std::set<std::uint32_t> ids;
    std::set<std::string> names;
    for (const FormatInfo& f : kFormats) {
        INFO(f.name);
        CHECK(ids.insert(static_cast<std::uint32_t>(f.type)).second);
        CHECK(names.insert(f.name).second);
        CHECK(f.block.elems >= 1);
        CHECK(f.block.bytes >= 1);
        if (f.block.elems == 1) CHECK((f.block.bytes == 2 || f.block.bytes == 4));   // plain F16/BF16/F32 only
        else CHECK((f.block.elems == 32 || f.block.elems == 256));                    // ggml block widths
        // Lookup agrees with the table, and block_spec is the table's geometry.
        REQUIRE(find_format(static_cast<std::uint32_t>(f.type)) == &f);
        CHECK(block_spec(static_cast<std::uint32_t>(f.type)).bytes == f.block.bytes);
    }
}

TEST_CASE("quant registry: an unknown type id is unknown everywhere, never guessed", "[quant][registry]") {
    for (const std::uint32_t id : {9999u, 10u /*Q2_K: real ggml id, not implemented*/, 0xffffffffu}) {
        CHECK(find_format(id) == nullptr);
        CHECK(block_spec(id).elems == 0);
        CHECK(dequant_fn(id) == nullptr);
    }
}

TEST_CASE("quant registry: every block-quantised format has a dequantiser, plain ones do not", "[quant][registry]") {
    for (const FormatInfo& f : kFormats) {
        INFO(f.name);
        const bool has = dequant_fn(static_cast<std::uint32_t>(f.type)) != nullptr;
        CHECK(has == is_block_quantised(f));
    }
    for (const DequantEntry& e : kDequant) {
        INFO(static_cast<unsigned>(e.type));
        REQUIRE(find_format(static_cast<std::uint32_t>(e.type)) != nullptr);   // no orphan dequantiser rows
        CHECK(e.fn != nullptr);
    }
}

TEST_CASE("quant registry: dequantisers honour the shared input contract", "[quant][registry]") {
    for (const FormatInfo& f : kFormats) {
        if (!is_block_quantised(f)) continue;
        INFO(f.name);
        const DequantFn fn = dequant_fn(static_cast<std::uint32_t>(f.type));
        REQUIRE(fn != nullptr);
        const std::uint64_t be = f.block.elems, bb = f.block.bytes;

        // Zero-filled blocks (scale 0) decode to exactly n zeros, including a PARTIAL final block.
        for (const std::uint64_t n : {be, 4 * be, 2 * be + be / 2}) {
            const std::uint64_t blocks = (n + be - 1) / be;
            const auto raw = filled(static_cast<std::size_t>(blocks * bb), 0);
            std::vector<float> out;
            REQUIRE(fn(raw, n, out));
            REQUIRE(out.size() == n);
            for (const float v : out) CHECK(v == 0.f);
        }
        // One byte short of the buffer the size rule demands is refused, not read past.
        {
            const std::uint64_t n = 3 * be;
            const auto raw = filled(static_cast<std::size_t>(3 * bb - 1), 0);
            std::vector<float> out;
            CHECK_FALSE(fn(raw, n, out));
        }
        // Zero elements is a valid empty tensor.
        {
            std::vector<float> out;
            CHECK(fn({}, 0, out));
            CHECK(out.empty());
        }
        // Arbitrary bytes decode deterministically (bit-identical on a repeat) and to the requested length.
        {
            const auto raw = noise(static_cast<std::size_t>(5 * bb));
            std::vector<float> a, b;
            REQUIRE(fn(raw, 5 * be, a));
            REQUIRE(fn(raw, 5 * be, b));
            REQUIRE(a.size() == 5 * be);
            CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);   // memcmp: NaN-safe
        }
    }
}

TEST_CASE("quant registry: to_f32 dispatches through the registry on the tensor's own type", "[quant][registry]") {
    for (const FormatInfo& f : kFormats) {
        if (!is_block_quantised(f)) continue;
        INFO(f.name);
        TensorInfo t;
        t.type_raw = static_cast<std::uint32_t>(f.type);
        t.dims = {static_cast<std::uint64_t>(f.block.elems) * 2};
        const auto raw = filled(static_cast<std::size_t>(2 * f.block.bytes), 0);
        std::vector<float> out;
        CHECK(to_f32(t, raw, out));
        CHECK(out.size() == 2 * f.block.elems);
    }
    TensorInfo unknown;
    unknown.type_raw = 9999;
    unknown.dims = {32};
    std::vector<float> out;
    CHECK_FALSE(to_f32(unknown, filled(1024, 0), out));
}
