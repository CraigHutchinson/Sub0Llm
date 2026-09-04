// moe_quant_tests.cpp -- unit tests for sub0::moeq, the S0Q1 quantized-resident routed-expert sidecar
// (docs/WP4_SCOPE.md WP4e).
//
// Engine-free, like transplant_tests.cpp/gguf_tests.cpp: moe_quant.hpp deliberately does not include
// sub0_config.hpp, so none of this needs a compiled model -- which is the point, since a real
// MOE_QUANT_EXPERTS build is a 4-layer, multi-GB, real-axes affair that no unit test can host.
//
// WHAT EACH TEST EXISTS TO RULE OUT, rather than merely to exercise:
//   * expert_byte_range: a MISALIGNED per-expert slice. Every format here is block-structured, so a
//     slice that does not land on a block boundary decodes a neighbouring expert's values with no
//     other symptom -- there is no shape or magnitude signal at all. It must refuse, not round.
//   * the descriptor ORDER: a real bug found by the offline tool's own bit-for-bit check during this
//     stage -- the table was built walking (layer, plane, expert) while the payload was written in
//     descriptor-index order (layer, expert, plane), so every offset pointed at the wrong place.
//     Pinned here as an ordering property rather than left to the tool's own check (AGENTS.md's
//     "regression test on a reproducible bug").
//   * dequantize_expert reproducing the f32 transplant path EXACTLY. The whole two-file design rests
//     on this: it must be the same gguf::to_f32 + transpose_out_in on the same bytes, bit for bit, or
//     WP4e's end-to-end gate is measuring a coincidence.
//   * the cache changing the answer. A resolve pool is a cache, and a cache that can alter output is
//     not a cache -- 1 slot and 4 slots must agree bit-for-bit over a selection pattern that forces
//     both eviction and reuse.

#include "sub0/moe_quant.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

using namespace sub0;

namespace {

// A tiny stand-in geometry. Q8_0 is the format used here because it is the one whose block layout this
// repo has had validated against a real file the longest (gguf_tests.cpp), and because 32-element
// blocks make both an aligned and a MISALIGNED per-expert extent easy to construct -- the real axes
// happen to be aligned for every format, so a test that only used them could not exercise the refusal.
constexpr int kExperts = 4, kIn = 8, kOut = 8;        // 64 elements per expert == 2 whole Q8_0 blocks
constexpr std::uint64_t kPerExpert = kIn * kOut;

// Deterministic pseudo-random Q8_0 blocks: an f16 scale followed by 32 int8 quants, repeated. Values
// are index-distinguishable so a wrong slice or a missed transpose cannot hide behind repetition.
std::vector<std::uint8_t> make_q8_0(std::uint64_t n_elements, std::uint32_t seed) {
    std::mt19937 rng(seed);
    const std::uint64_t blocks = (n_elements + 31) / 32;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(blocks) * 34);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        std::uint8_t* blk = raw.data() + b * 34;
        const std::uint16_t d_bits = static_cast<std::uint16_t>(0x3000u + (rng() & 0x0FFFu));  // a small positive f16
        std::memcpy(blk, &d_bits, 2);
        for (int i = 0; i < 32; ++i) blk[2 + i] = static_cast<std::uint8_t>(rng() & 0xFFu);
    }
    return raw;
}

// The f32 side of the comparison, spelled OUT rather than reusing dequantize_expert: this is what
// tools/sub0llm-transplant.cpp's own f32 path does (read the slice, gguf::to_f32, transpose_out_in), so
// writing it independently here is what makes "the two produce identical floats" a real check and not a
// function compared against itself.
std::vector<float> f32_reference(std::span<const std::uint8_t> slice, int in_f, int out_f) {
    gguf::TensorInfo t;
    t.type_raw = static_cast<std::uint32_t>(gguf::TensorType::Q8_0);
    t.dims = {static_cast<std::uint64_t>(in_f) * out_f};
    std::vector<float> src;
    REQUIRE(gguf::to_f32(t, slice, src));
    std::vector<float> dst(static_cast<std::size_t>(in_f) * out_f, 0.f);
    transplant::transpose_out_in(src.data(), out_f, in_f, dst.data());
    return dst;
}

// Builds a complete, valid S0Q1 file for `n_layers` layers of kExperts experts and returns its path
// alongside the raw source bytes each plane came from, so a test can compare against them directly.
struct Built {
    std::string path;
    std::vector<std::vector<std::uint8_t>> plane_bytes;   // indexed by moeq::desc_index
};

Built build_sidecar(int n_layers, const std::string& path) {
    Built out;
    out.path = path;
    const std::size_t n = static_cast<std::size_t>(n_layers) * kExperts * moeq::PerExpert;
    out.plane_bytes.resize(n);
    std::vector<moeq::Desc> descs(n);
    std::uint64_t cursor = 0;
    // Built in DESCRIPTOR-INDEX order deliberately -- see this file's header comment on the ordering
    // bug. The writer in the tool builds in a different order and then assigns offsets in this one;
    // both must agree, which is what the ordering test below checks.
    for (std::size_t i = 0; i < n; ++i) {
        out.plane_bytes[i] = make_q8_0(kPerExpert, static_cast<std::uint32_t>(1000 + i));
        descs[i] = moeq::Desc{static_cast<std::uint32_t>(gguf::TensorType::Q8_0), kIn, kOut, 0, cursor,
                              out.plane_bytes[i].size()};
        cursor += out.plane_bytes[i].size();
    }
    moeq::Header h;
    h.n_layers = n_layers;
    h.num_experts = kExperts;
    h.d_model = kIn;
    h.d_ff = kOut;
    h.n_tensors = n;
    h.data_off = sizeof(moeq::Header) + n * sizeof(moeq::Desc);
    h.data_bytes = cursor;
    h.model_param_floats = 12345;
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    os.write(reinterpret_cast<const char*>(&h), sizeof h);
    os.write(reinterpret_cast<const char*>(descs.data()),
             static_cast<std::streamsize>(n * sizeof(moeq::Desc)));
    for (const auto& b : out.plane_bytes)
        os.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
    os.close();
    return out;
}

std::string temp_path(const char* stem) {
    return (std::filesystem::temp_directory_path() / stem).string();
}

}  // namespace

TEST_CASE("moeq: the on-disk structs are the size the format says", "[moequant]") {
    // Belt and braces beside moe_quant.hpp's own static_asserts: these are the numbers a reader of the
    // format needs, so a change that moves them should fail here too, with a name.
    REQUIRE(sizeof(moeq::Header) == 56);
    REQUIRE(sizeof(moeq::Desc) == 32);
    REQUIRE(moeq::PerExpert == 3);
    REQUIRE(moeq::Gate == 0);
    REQUIRE(moeq::Up == 1);
    REQUIRE(moeq::Down == 2);
}

TEST_CASE("moeq: expert_byte_range refuses a misaligned slice rather than rounding", "[moequant]") {
    using gguf::TensorType;
    const auto q8 = static_cast<std::uint32_t>(TensorType::Q8_0);          // 32 elems / 34 bytes
    const auto iq1 = static_cast<std::uint32_t>(TensorType::IQ1_S);        // 256 / 50
    const auto iq4 = static_cast<std::uint32_t>(TensorType::IQ4_NL);       // 32 / 18

    // Aligned: expert e starts exactly e whole blocks in.
    REQUIRE(moeq::expert_byte_range(q8, 64, 0).off == 0);
    REQUIRE(moeq::expert_byte_range(q8, 64, 0).bytes == 68);
    REQUIRE(moeq::expert_byte_range(q8, 64, 3).off == 3 * 68);

    // The real axes, both formats a routed expert actually uses (2560*640 = 1,638,400 elements).
    REQUIRE(moeq::expert_byte_range(iq1, 1'638'400, 1).bytes == 1'638'400 / 256 * 50);
    REQUIRE(moeq::expert_byte_range(iq4, 1'638'400, 1).bytes == 1'638'400 / 32 * 18);

    // MISALIGNED: 100 elements is not a whole number of 32-element blocks. Expert 1 would begin
    // mid-block, so its decode would start from a neighbour's scale -- refused, not rounded.
    REQUIRE(moeq::expert_byte_range(q8, 100, 1).bytes == 0);
    // Likewise a 256-block format at a non-multiple-of-256 extent.
    REQUIRE(moeq::expert_byte_range(iq1, 1'638'400 - 1, 1).bytes == 0);
    // An unknown type has no size rule at all.
    REQUIRE(moeq::expert_byte_range(999u, 64, 0).bytes == 0);
}

TEST_CASE("moeq: desc_index is (layer, expert, plane)-major, in that order", "[moequant]") {
    // The ordering property the real bug violated: consecutive indices walk PLANES first, then
    // EXPERTS, then LAYERS -- the same order make_param_layout() emits an expert's own triple in, and
    // the order the payload must be written in.
    constexpr int NE = 4;
    REQUIRE(moeq::desc_index(NE, 0, 0, moeq::Gate) == 0);
    REQUIRE(moeq::desc_index(NE, 0, 0, moeq::Up) == 1);
    REQUIRE(moeq::desc_index(NE, 0, 0, moeq::Down) == 2);
    REQUIRE(moeq::desc_index(NE, 0, 1, moeq::Gate) == 3);
    REQUIRE(moeq::desc_index(NE, 1, 0, moeq::Gate) == NE * 3);
    // And it is a bijection over the whole table -- an index collision would silently alias two
    // experts onto one payload range, which is exactly what a transposed index expression does.
    std::vector<bool> seen(static_cast<std::size_t>(2 * NE * 3), false);
    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < NE; ++e)
            for (int w = 0; w < moeq::PerExpert; ++w) {
                const auto i = static_cast<std::size_t>(moeq::desc_index(NE, l, e, w));
                REQUIRE(i < seen.size());
                REQUIRE_FALSE(seen[i]);
                seen[i] = true;
            }
    for (bool b : seen) REQUIRE(b);
}

TEST_CASE("moeq: a written sidecar reads back, and its planes match the f32 path bit for bit",
          "[moequant]") {
    const std::string path = temp_path("sub0_moeq_roundtrip.bin");
    const Built built = build_sidecar(2, path);

    moeq::Store store;
    std::string err;
    REQUIRE(store.open(path, err));
    REQUIRE(err.empty());
    REQUIRE(store.header().n_layers == 2);
    REQUIRE(store.header().num_experts == kExperts);
    REQUIRE(store.header().n_tensors == 2 * kExperts * moeq::PerExpert);
    REQUIRE(store.resident_bytes() == 2 * kExperts * moeq::PerExpert * 68);

    std::vector<float> scratch, dst(kPerExpert, 0.f);
    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < kExperts; ++e)
            for (int w = 0; w < moeq::PerExpert; ++w) {
                const moeq::Desc& d = store.desc(l, e, w);
                // The bytes the store hands back must be the bytes that were written for THIS plane --
                // the offset check the ordering bug failed.
                const auto& want_bytes = built.plane_bytes[
                    static_cast<std::size_t>(moeq::desc_index(kExperts, l, e, w))];
                const auto got_bytes = store.raw(d);
                REQUIRE(got_bytes.size() == want_bytes.size());
                REQUIRE(std::memcmp(got_bytes.data(), want_bytes.data(), want_bytes.size()) == 0);

                REQUIRE(moeq::dequantize_expert(d, got_bytes, dst.data(), scratch));
                const std::vector<float> ref = f32_reference(want_bytes, kIn, kOut);
                // EXACT, not approximate: both sides run the same decode and the same permutation on
                // the same bytes, so any difference at all is a real defect, not rounding.
                REQUIRE(std::memcmp(dst.data(), ref.data(), ref.size() * sizeof(float)) == 0);
            }
    std::filesystem::remove(path);
}

TEST_CASE("moeq: Store refuses a truncated or foreign file rather than reading garbage", "[moequant]") {
    const std::string path = temp_path("sub0_moeq_bad.bin");
    moeq::Store store;
    std::string err;

    {   // not an S0Q1 file at all
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        const char junk[64] = {'N', 'O', 'P', 'E'};
        os.write(junk, sizeof junk);
    }
    REQUIRE_FALSE(store.open(path, err));
    REQUIRE_FALSE(err.empty());

    {   // right magic, header truncated
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        os.write("S0Q1", 4);
    }
    REQUIRE_FALSE(store.open(path, err));

    {   // valid header, payload missing
        const Built b = build_sidecar(1, path);
        std::filesystem::resize_file(path, sizeof(moeq::Header) +
                                                b.plane_bytes.size() * sizeof(moeq::Desc) + 4);
    }
    REQUIRE_FALSE(store.open(path, err));

    REQUIRE_FALSE(store.open(temp_path("sub0_moeq_does_not_exist.bin"), err));
    std::filesystem::remove(path);
}

TEST_CASE("moeq: the resolve pool is a cache -- its capacity cannot change the answer", "[moequant]") {
    // The claim the whole WP4e design rests on at the engine seam. A one-slot pool re-dequantizes on
    // every single resolve; a four-slot pool serves repeats from memory. Over a selection pattern
    // chosen to force BOTH eviction (more distinct experts than slots) and reuse (the same expert
    // revisited after an eviction and after a hit), the floats must be identical -- and identical to
    // the f32 reference, so this is not two caches agreeing on the same wrong value.
    const std::string path = temp_path("sub0_moeq_cache.bin");
    const Built built = build_sidecar(2, path);
    moeq::Store store;
    std::string err;
    REQUIRE(store.open(path, err));

    moeq::ExpertCache<1, kPerExpert> one;
    moeq::ExpertCache<4, kPerExpert> four;
    one.allocate();
    four.allocate();

    struct Pick { int layer, expert; };
    const std::vector<Pick> pattern = {
        {0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 0},   // revisit after filling four slots
        {1, 0}, {1, 1}, {0, 0},                    // revisit across a layer boundary
        {0, 1}, {0, 1},                            // an immediate repeat (a guaranteed hit)
        {1, 3}, {0, 2}, {1, 0},
    };
    std::vector<float> scratch, ref(kPerExpert, 0.f);
    for (const Pick& p : pattern) {
        const auto a = one.resolve(store, p.layer, p.expert);
        const auto b = four.resolve(store, p.layer, p.expert);
        REQUIRE(a.gate != nullptr);
        REQUIRE(b.gate != nullptr);
        const float* aps[3] = {a.gate, a.up, a.down};
        const float* bps[3] = {b.gate, b.up, b.down};
        for (int w = 0; w < moeq::PerExpert; ++w) {
            REQUIRE(std::memcmp(aps[w], bps[w], kPerExpert * sizeof(float)) == 0);
            const auto& bytes = built.plane_bytes[
                static_cast<std::size_t>(moeq::desc_index(kExperts, p.layer, p.expert, w))];
            ref = f32_reference(bytes, kIn, kOut);
            REQUIRE(std::memcmp(aps[w], ref.data(), kPerExpert * sizeof(float)) == 0);
        }
    }
    // And the cache is really caching: the one-slot pool cannot hit at all on this pattern except on
    // the immediate repeat, while the four-slot pool must hit strictly more often. (If these were
    // equal, the "cache" would be doing nothing and the slots would be dead weight.)
    REQUIRE(one.hits() < four.hits());
    REQUIRE(one.misses() + one.hits() == pattern.size());
    REQUIRE(four.misses() + four.hits() == pattern.size());
    std::filesystem::remove(path);
}
