// WP4f: the S0HD hidden-state container (include/sub0/hidden_dump.hpp).
//
// The format exists so TWO programs -- this engine and something on the llama.cpp side that is not this
// repo -- can exchange per-layer hidden states for a divergence comparison. Its whole value is that a
// second, independently-written producer can emit a file this reader accepts, so the properties worth
// pinning are the ones a second implementation could get wrong while still "working":
//
//   * the exact BYTE LAYOUT, asserted offset by offset against a hand-written expectation rather than
//     by writing and reading back with the same code (a self-consistent writer/reader pair proves
//     nothing about what a foreign producer must emit -- this project's own
//     [[independent-reimplementation-catches-identity-swap-bugs]] lesson at file-format scale);
//   * that the tensor count really is BACK-PATCHED, i.e. a dump whose producer died mid-pass is
//     REJECTED rather than read as a valid short one. That is the failure mode a long real forward
//     pass actually has;
//   * that the input token array survives the round trip, since sub0llm-hidden-diff's premise check
//     ("same tokens in") is built on it.

#include "sub0/hidden_dump.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace sub0::hidden;

namespace {

std::string temp_path(const char* stem) {
    return (std::filesystem::temp_directory_path() / stem).string();
}

std::vector<unsigned char> read_all(const std::string& p) {
    std::ifstream is(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>()};
}

std::uint32_t le_u32(const std::vector<unsigned char>& b, std::size_t off) {
    return static_cast<std::uint32_t>(b[off]) | (static_cast<std::uint32_t>(b[off + 1]) << 8) |
           (static_cast<std::uint32_t>(b[off + 2]) << 16) | (static_cast<std::uint32_t>(b[off + 3]) << 24);
}

}  // namespace

TEST_CASE("S0HD byte layout is exactly what a foreign producer must emit", "[hidden][wp4f]") {
    const std::string p = temp_path("sub0_s0hd_layout.bin");
    const int tokens[] = {1543, 88123, 7};
    const float a[] = {1.f, 2.f, 3.f, 4.f};       // [2 x 2]

    Writer w;
    REQUIRE(w.open(p, tokens, 3));
    w.add("blk.0.out", 2, 2, a);
    REQUIRE(w.close());
    REQUIRE(w.count() == 1);

    const auto b = read_all(p);
    // header 16 + tokens 12 + (4 + 9 + 4 + 4 + 16) = 65 bytes. No padding anywhere: a producer that
    // aligns records would write a longer file and this is the assertion that would catch it.
    REQUIRE(b.size() == 65);
    REQUIRE(std::memcmp(b.data(), "S0HD", 4) == 0);
    REQUIRE(le_u32(b, 4) == 1);          // version
    REQUIRE(le_u32(b, 8) == 1);          // n_tensors -- back-patched by close()
    REQUIRE(le_u32(b, 12) == 3);         // n_tokens
    REQUIRE(le_u32(b, 16) == 1543);
    REQUIRE(le_u32(b, 20) == 88123);
    REQUIRE(le_u32(b, 24) == 7);
    REQUIRE(le_u32(b, 28) == 9);         // name_len
    REQUIRE(std::string(reinterpret_cast<const char*>(b.data() + 32), 9) == "blk.0.out");
    REQUIRE(le_u32(b, 41) == 2);         // rows
    REQUIRE(le_u32(b, 45) == 2);         // cols
    float v = 0.f;
    std::memcpy(&v, b.data() + 49, 4);
    REQUIRE(v == 1.f);
    std::memcpy(&v, b.data() + 61, 4);   // the last of the four floats
    REQUIRE(v == 4.f);

    std::filesystem::remove(p);
}

TEST_CASE("S0HD round trip preserves tokens, names, shapes and values", "[hidden][wp4f]") {
    const std::string p = temp_path("sub0_s0hd_roundtrip.bin");
    const int tokens[] = {1543, 88123, 245000, 7, 99999, 156789};   // WP4f's canonical array
    std::vector<float> big(6 * 5);
    for (std::size_t i = 0; i < big.size(); ++i) big[i] = static_cast<float>(i) * 0.25f - 3.f;

    Writer w;
    REQUIRE(w.open(p, tokens, 6));
    w.add("tok_embd", 6, 5, big.data());
    w.add("blk.0.attn_out", 6, 5, big.data());
    w.add("final_hidden", 6, 5, big.data());
    REQUIRE(w.close());

    Dump d;
    std::string err;
    REQUIRE(read(p, d, err));
    REQUIRE(err.empty());
    REQUIRE(d.tokens == std::vector<int>{1543, 88123, 245000, 7, 99999, 156789});
    REQUIRE(d.tensors.size() == 3);
    const Tensor* t = d.find("blk.0.attn_out");
    REQUIRE(t != nullptr);
    REQUIRE(t->rows == 6);
    REQUIRE(t->cols == 5);
    REQUIRE(t->n() == big.size());
    for (std::size_t i = 0; i < big.size(); ++i) REQUIRE(t->data[i] == big[i]);
    REQUIRE(d.find("no_such_tensor") == nullptr);

    std::filesystem::remove(p);
}

TEST_CASE("S0HD reports duplicate names rather than resolving them silently", "[hidden][wp4f]") {
    // A LoopSplit build re-executes the same layer index, so two `blk.3.out` tensors is a legitimate
    // dump, not a corrupt one -- but a differ must not pick one at random.
    const std::string p = temp_path("sub0_s0hd_dup.bin");
    const int tokens[] = {1, 2};
    const float first[] = {1.f, 1.f}, second[] = {9.f, 9.f};

    Writer w;
    REQUIRE(w.open(p, tokens, 2));
    w.add("blk.3.out", 1, 2, first);
    w.add("blk.3.out", 1, 2, second);
    REQUIRE(w.close());

    Dump d;
    std::string err;
    REQUIRE(read(p, d, err));
    int dup = -1;
    const Tensor* t = d.find("blk.3.out", &dup);
    REQUIRE(t != nullptr);
    REQUIRE(dup == 1);                 // one FURTHER occurrence beyond the one returned
    REQUIRE(t->data[0] == 1.f);        // and it is the first, not the last

    std::filesystem::remove(p);
}

TEST_CASE("S0HD rejects a truncated / aborted dump instead of reading it short", "[hidden][wp4f]") {
    Dump d;
    std::string err;

    SECTION("a writer that never closed leaves n_tensors == 0, which is rejected") {
        const std::string p = temp_path("sub0_s0hd_aborted.bin");
        const int tokens[] = {1, 2};
        {
            Writer w;
            REQUIRE(w.open(p, tokens, 2));
            const float x[] = {1.f, 2.f};
            w.add("blk.0.out", 1, 2, x);
            // deliberately NO close() -- this is what a killed forward pass leaves behind
        }
        REQUIRE_FALSE(read(p, d, err));
        REQUIRE(err.find("0 tensors") != std::string::npos);
        std::filesystem::remove(p);
    }

    SECTION("a file cut short mid-payload is rejected, not zero-filled") {
        const std::string p = temp_path("sub0_s0hd_cut.bin");
        const int tokens[] = {1, 2};
        {
            Writer w;
            REQUIRE(w.open(p, tokens, 2));
            const float x[] = {1.f, 2.f, 3.f, 4.f};
            w.add("blk.0.out", 2, 2, x);
            REQUIRE(w.close());
        }
        auto bytes = read_all(p);
        bytes.resize(bytes.size() - 5);          // lop off part of the last float
        { std::ofstream os(p, std::ios::binary | std::ios::trunc);
          os.write(reinterpret_cast<const char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size())); }
        REQUIRE_FALSE(read(p, d, err));
        REQUIRE(err.find("truncated data") != std::string::npos);
        std::filesystem::remove(p);
    }

    SECTION("a file that is not S0HD at all is rejected by magic, not by shape") {
        const std::string p = temp_path("sub0_s0hd_notours.bin");
        { std::ofstream os(p, std::ios::binary | std::ios::trunc); os << "GGUF....garbage"; }
        REQUIRE_FALSE(read(p, d, err));
        REQUIRE(err.find("bad magic") != std::string::npos);
        std::filesystem::remove(p);
    }
}
