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
//   * WP5b: the MAPPING changing the answer. Store::open now memory-maps the payload instead of
//     reading it into an owned buffer, so that the ~38 GiB sidecar the full 48-layer model needs is
//     never eagerly resident. The claim that this is invisible above `raw()` is checked by
//     reproducing the OLD eager read INDEPENDENTLY (EagerStore below -- an ifstream into an owned
//     buffer, written from the format spec, not by calling the new code) and requiring the two to
//     agree bit-for-bit on the raw bytes, on the dequantized planes, AND on moe::expert_ffn_row's own
//     output. The last of those is the one that matters: it is the only consumer the engine actually
//     has, and it is what makes this a structural claim rather than a spot check
//     ([[independent-reimplementation-catches-identity-swap-bugs]], the same discipline WP4e's own
//     forward_row_via refactor was gated on).

#include "sub0/moe_math.hpp"
#include "sub0/moe_quant.hpp"
#include "sub0/moe_quant_dot.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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

// WP5b: the residency form moeq::Store used BEFORE it mapped its payload -- an ifstream read into an
// owned buffer. Written here from the S0Q1 format spec (moe_quant.hpp's Header/Desc + data_off), NOT by
// calling the current Store, because a reader compared against itself proves nothing about a change to
// how it acquires its bytes. This is the "b" side of WP5b's own gate; it exists only in this file and
// nothing in src/ or tools/ can reach it.
class EagerStore {
public:
    bool open(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        f.read(reinterpret_cast<char*>(&h_), sizeof h_);
        if (f.gcount() != static_cast<std::streamsize>(sizeof h_)) return false;
        if (std::memcmp(h_.magic, "S0Q1", 4) != 0) return false;
        descs_.resize(static_cast<std::size_t>(h_.n_tensors));
        const auto table_bytes = static_cast<std::streamsize>(descs_.size() * sizeof(moeq::Desc));
        f.read(reinterpret_cast<char*>(descs_.data()), table_bytes);
        if (f.gcount() != table_bytes) return false;
        data_ = std::make_unique<std::uint8_t[]>(static_cast<std::size_t>(h_.data_bytes));
        f.clear();
        f.seekg(static_cast<std::streamoff>(h_.data_off));
        const auto payload_bytes = static_cast<std::streamsize>(h_.data_bytes);
        f.read(reinterpret_cast<char*>(data_.get()), payload_bytes);
        return f.gcount() == payload_bytes;
    }
    const moeq::Header& header() const { return h_; }
    const moeq::Desc& desc(int layer, int expert, int which) const {
        return descs_[static_cast<std::size_t>(moeq::desc_index(h_.num_experts, layer, expert, which))];
    }
    std::span<const std::uint8_t> raw(const moeq::Desc& d) const {
        return {data_.get() + d.off, static_cast<std::size_t>(d.bytes)};
    }

private:
    moeq::Header                    h_{};
    std::vector<moeq::Desc>         descs_;
    std::unique_ptr<std::uint8_t[]> data_;
};

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

    // WP5b: the Store is SCOPED so its mapping is released before the remove() at the end. That is a
    // real behavioural difference the mmap upgrade introduced, and it is the correct one: on Windows a
    // mapped file cannot be deleted or truncated while the view is open, which for a live model is
    // exactly the property you want (the weights cannot change underneath a running forward pass).
    {
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
    }
    REQUIRE(std::filesystem::remove(path));
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
    {   // scoped: the mapping must be released before the remove() below (see the roundtrip case)
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
    }
    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("moeq (WP5b): mapping the payload instead of reading it changes nothing above raw()",
          "[moequant]") {
    // THE GATE for the mmap upgrade. The eager read is reproduced independently (EagerStore above) and
    // the two are compared at all three levels a consumer can observe:
    //   1. the raw encoded bytes handed back by raw()      -- the accessor's own contract
    //   2. the dequantized [in, out] plane                  -- what dequantize_expert produces from them
    //   3. moe::expert_ffn_row's output                     -- the ONLY thing the engine actually does
    //                                                          with a resolved expert
    // Level 3 is the one that makes this structural: an identical byte span could still be consumed
    // differently, and the whole point of the change is that it cannot be.
    const std::string path = temp_path("sub0_moeq_mmap.bin");
    const Built built = build_sidecar(2, path);
    {   // scoped: the mapping must be released before the remove() below (see the roundtrip case)
    moeq::Store mapped;
    std::string err;
    REQUIRE(mapped.open(path, err));
    REQUIRE(mapped.loaded());
    EagerStore eager;
    REQUIRE(eager.open(path));

    // The two agree on the table itself before anything is decoded, so a later disagreement cannot be
    // blamed on a mis-parsed header.
    REQUIRE(mapped.header().n_tensors == eager.header().n_tensors);
    REQUIRE(mapped.header().data_off == eager.header().data_off);
    REQUIRE(mapped.header().data_bytes == eager.header().data_bytes);

    // A row of input with no zeros anywhere: expert_ffn_row skips zero inputs, and a row of zeros would
    // make every expert produce the same output and hide a wrong plane entirely.
    const moe::Dims dims{kIn, kOut, kExperts, 2};
    std::vector<float> x(kIn);
    for (int i = 0; i < kIn; ++i) x[static_cast<std::size_t>(i)] = 0.25f + 0.125f * static_cast<float>(i);

    std::vector<float> plane_m(kPerExpert, 0.f), plane_e(kPerExpert, 0.f);
    std::vector<float> scratch_m, scratch_e;
    std::vector<float> gate_m(kPerExpert), up_m(kPerExpert), down_m(kPerExpert);
    std::vector<float> gate_e(kPerExpert), up_e(kPerExpert), down_e(kPerExpert);
    std::vector<float> out_m(kIn), out_e(kIn), pre(kOut), g(kOut);

    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < kExperts; ++e) {
            std::vector<float>* planes_m[3] = {&gate_m, &up_m, &down_m};
            std::vector<float>* planes_e[3] = {&gate_e, &up_e, &down_e};
            for (int w = 0; w < moeq::PerExpert; ++w) {
                const moeq::Desc& dm = mapped.desc(l, e, w);
                const moeq::Desc& de = eager.desc(l, e, w);
                // 1. the same descriptor and the same bytes.
                REQUIRE(dm.off == de.off);
                REQUIRE(dm.bytes == de.bytes);
                REQUIRE(dm.type_raw == de.type_raw);
                const auto bm = mapped.raw(dm);
                const auto be = eager.raw(de);
                REQUIRE(bm.size() == be.size());
                REQUIRE(std::memcmp(bm.data(), be.data(), bm.size()) == 0);
                // 2. the same dequantized plane, bit for bit.
                REQUIRE(moeq::dequantize_expert(dm, bm, plane_m.data(), scratch_m));
                REQUIRE(moeq::dequantize_expert(de, be, plane_e.data(), scratch_e));
                REQUIRE(std::memcmp(plane_m.data(), plane_e.data(), kPerExpert * sizeof(float)) == 0);
                *planes_m[w] = plane_m;
                *planes_e[w] = plane_e;
            }
            // 3. the same expert_ffn_row output, bit for bit -- the engine's only consumer.
            moe::expert_ffn_row(dims, x.data(), gate_m.data(), up_m.data(), down_m.data(),
                                out_m.data(), pre.data(), g.data());
            moe::expert_ffn_row(dims, x.data(), gate_e.data(), up_e.data(), down_e.data(),
                                out_e.data(), pre.data(), g.data());
            REQUIRE(std::memcmp(out_m.data(), out_e.data(), out_m.size() * sizeof(float)) == 0);
            // And it is not trivially zero on both sides, which would make the comparison vacuous.
            bool any_nonzero = false;
            for (float v : out_m) any_nonzero = any_nonzero || (v != 0.f);
            REQUIRE(any_nonzero);
        }
    }
    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("moeq (WP5b): a Store outlives the scope its FileMap was opened in", "[moequant]") {
    // The one lifetime property a mapping has that an owned buffer did not: `raw()` hands back a span
    // over pages the Store must keep mapped. A Store moved/held past the open() call's own scope, with
    // the source file deleted underneath it on POSIX and held open on Windows, must still read.
    const std::string path = temp_path("sub0_moeq_lifetime.bin");
    const Built built = build_sidecar(1, path);
    std::vector<float> scratch, dst(kPerExpert, 0.f);
    {
        moeq::Store store;
        std::string err;
        REQUIRE(store.open(path, err));
        // Re-opening the SAME Store must release the previous mapping and succeed, not leak a handle
        // or refuse -- load_model calls open() exactly once, but nothing in the type says so.
        REQUIRE(store.open(path, err));
        const moeq::Desc& d = store.desc(0, kExperts - 1, moeq::Down);
        REQUIRE(moeq::dequantize_expert(d, store.raw(d), dst.data(), scratch));
        const std::vector<float> ref = f32_reference(
            built.plane_bytes[static_cast<std::size_t>(
                moeq::desc_index(kExperts, 0, kExperts - 1, moeq::Down))], kIn, kOut);
        REQUIRE(std::memcmp(dst.data(), ref.data(), ref.size() * sizeof(float)) == 0);
    }
    // The mapping is released with the Store, so the file is removable straight afterwards on Windows
    // too -- which is also the check that nothing leaked the HANDLE.
    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("moeq (B31/B34/B38): the fused no-transpose resolve produces the SAME output as dequant+transpose",
          "[moequant]") {
    // THE GATE for the whole B31 design. dequantize_expert_source + moe::expert_ffn_row_source is a
    // from-scratch REORDERING of the same math dequantize_expert + moe::expert_ffn_row already compute
    // (see moe_math.hpp's own comment on expert_ffn_row_source for the per-output summation-order
    // argument this test exists to check, not merely restate): both paths must produce the SAME FFN
    // output over the SAME raw encoded bytes, for every one of the three plane roles, or the "one DRAM
    // touch instead of three" design is measuring a coincidence rather than an identity.
    //
    // B34/B38 (docs/INDEPENDENT_REVIEW_BACKLOG.md B38): expert_ffn_row_source's gate/up/down reductions
    // go through simd_reduce.hpp's `dot_seq` (strict left-to-right), NEVER `dot()`/`dot_choice<UseSimd>`
    // -- and NOT gated on USE_SIMD_REDUCE at all, regardless of the build's own flag setting: an earlier
    // pass of this work used the reordered multi-accumulator `dot()` here, which broke this test's own
    // bit-exactness (expert_ffn_row's own scatter-accumulated sum is NOT reorderable without losing its
    // deliberate input-major cache locality, so it stays a strict scalar sum, and expert_ffn_row_source
    // has to match that order term-for-term to stay bit-identical). `dot_seq` still drops the old
    // `if(xi==0.f) continue` branch (see moe_math.hpp's own comment) -- that part is a genuine, proven
    // no-op (0.f*finite=0.f exactly), not a reordering. So this stays a bit-exact memcmp, unaffected by
    // whichever way USE_SIMD_REDUCE is set for the rest of the build: the two paths compute the same
    // addends in the same order, only the branch is gone. See simd_reduce.hpp's own `dot_seq` comment.
    const std::string path = temp_path("sub0_moeq_fused.bin");
    const Built built = build_sidecar(2, path);
    {
    moeq::Store store;
    std::string err;
    REQUIRE(store.open(path, err));

    moeq::ExpertCache<1, kPerExpert>       transposed;
    moeq::ExpertCacheSource<1, kPerExpert> fused;
    transposed.allocate();
    fused.allocate();

    // A row of input with no zeros anywhere, same reasoning as the WP5b case above: expert_ffn_row(_source)
    // skips zero inputs, and an all-zero row would make a wrong plane invisible.
    const moe::Dims dims{kIn, kOut, kExperts, 2};
    std::vector<float> x(kIn);
    for (int i = 0; i < kIn; ++i) x[static_cast<std::size_t>(i)] = 0.25f + 0.125f * static_cast<float>(i);
    std::vector<float> out_t(kIn), out_f(kIn), pre(kOut), g(kOut);

    for (int l = 0; l < 2; ++l)
        for (int e = 0; e < kExperts; ++e) {
            const auto rt = transposed.resolve(store, l, e);
            const auto rf = fused.resolve(store, l, e);
            REQUIRE(rt.gate != nullptr);
            REQUIRE(rf.gate != nullptr);

            moe::expert_ffn_row(dims, x.data(), rt.gate, rt.up, rt.down, out_t.data(), pre.data(), g.data());
            moe::expert_ffn_row_source(dims, x.data(), rf.gate, rf.up, rf.down, out_f.data(), pre.data(), g.data());
            REQUIRE(std::memcmp(out_t.data(), out_f.data(), out_t.size() * sizeof(float)) == 0);
            bool any_nonzero = false;
            for (float v : out_t) any_nonzero = any_nonzero || (v != 0.f);
            REQUIRE(any_nonzero);
        }
    }
    REQUIRE(std::filesystem::remove(path));
}

TEST_CASE("moeq (B38): forward_row_via_run_ex<UseSimd=true> builds, runs, and stays close to the "
          "default scalar arm, while expert_ffn_row_source stays untouched by the flag",
          "[moequant]") {
    // B38's own correctness gate for the ONE call site forward_row_via_run_ex's `UseSimd` template
    // parameter actually controls (the shared-expert gate logit, moe_math.hpp): instantiate both arms
    // over the SAME real sidecar bytes and SAME input row, and require (a) both compile/run/produce
    // finite output -- the templating itself is exercised, not merely assumed to compile because
    // `sub0_core` happened to link somewhere else, (b) the two arms' outputs are close (tolerance, NOT
    // bit-exact -- reassociating a sum changes the last bits by construction, same as every other
    // reordering this project has already accepted, e.g. B24/B31's own precedent), and (c) the routed
    // experts' own output (expert_ffn_row_source, untouched by `UseSimd`) is IDENTICAL between the two
    // runs -- proving the flag really does stay scoped to only the shared-expert gate logit, not leak
    // into the routed-expert path it must never touch (docs/INDEPENDENT_REVIEW_BACKLOG.md B38).
    const std::string path = temp_path("sub0_moeq_b38_simd.bin");
    const Built built = build_sidecar(1, path);
    {
    moeq::Store store;
    std::string err;
    REQUIRE(store.open(path, err));

    const moe::Dims dims{kIn, kOut, kExperts, 2};
    std::vector<float> x(kIn);
    for (int i = 0; i < kIn; ++i) x[static_cast<std::size_t>(i)] = 0.3f - 0.017f * static_cast<float>(i);

    // Shared-expert weights: reuse routed expert 0's own planes as stand-ins (this test only needs SOME
    // real, non-degenerate weight bytes; the shared expert's identity is irrelevant to what's gated).
    moeq::ExpertCacheSource<1, kPerExpert> fused;
    fused.allocate();
    const auto shared = fused.resolve(store, 0, 0);
    REQUIRE(shared.gate != nullptr);

    // A trivial router: [hidden_size, num_experts], f32, zero-initialized (uniform routing -- this test
    // does not care WHICH experts are picked, only that the shared-expert gate logit reduction differs).
    std::vector<float> router_w_mut(static_cast<std::size_t>(kIn) * kExperts, 0.01f);
    std::vector<float> shared_gate_proj_w_mut(kIn);
    for (int i = 0; i < kIn; ++i) shared_gate_proj_w_mut[static_cast<std::size_t>(i)] = 0.02f * static_cast<float>(i % 7 - 3);
    const std::vector<float>& router_w = router_w_mut;
    const std::vector<float>& shared_gate_proj_w = shared_gate_proj_w_mut;

    std::vector<float> scratch(moe::scratch_floats(dims));
    std::vector<float> out_false(kIn), out_true(kIn);
    std::vector<float> routed_false(kIn), routed_true(kIn);   // expert 0's own routed output, captured

    auto run = [&](bool use_simd, std::vector<float>& out, std::vector<float>& routed_out) {
        auto compute_expert = [&](int /*k*/, int e, float* out_ptr, float* ffn, float* g) {
            const auto r = fused.resolve(store, 0, e % kExperts);
            moe::expert_ffn_row_source(dims, x.data(), r.gate, r.up, r.down, out_ptr, ffn, g);
            std::memcpy(routed_out.data(), out_ptr, routed_out.size() * sizeof(float));
        };
        if (use_simd) {
            moe::forward_row_via_run_ex<true>(dims, x.data(), router_w.data(), compute_expert,
                                               moe::SerialExperts{}, shared.gate, shared.up, shared.down,
                                               shared_gate_proj_w.data(), out.data(), scratch.data());
        } else {
            moe::forward_row_via_run_ex<false>(dims, x.data(), router_w.data(), compute_expert,
                                                moe::SerialExperts{}, shared.gate, shared.up, shared.down,
                                                shared_gate_proj_w.data(), out.data(), scratch.data());
        }
    };
    run(false, out_false, routed_false);
    run(true,  out_true,  routed_true);

    // (c) the routed expert's own output (expert_ffn_row_source, never touched by UseSimd) is identical.
    REQUIRE(std::memcmp(routed_false.data(), routed_true.data(), routed_false.size() * sizeof(float)) == 0);

    // (a)/(b): both finite; close but not required to be bit-identical (the shared-expert gate logit's
    // reduction order differs, so sigmoid(gate_logit) differs at ULP scale, which the weighted-sum
    // combine can then amplify slightly -- bounded by a loose tolerance, not asserted to be zero).
    double max_abs_diff = 0.0, max_abs = 0.0;
    for (std::size_t i = 0; i < out_false.size(); ++i) {
        REQUIRE(std::isfinite(out_false[i]));
        REQUIRE(std::isfinite(out_true[i]));
        max_abs_diff = std::max(max_abs_diff, static_cast<double>(std::fabs(out_false[i] - out_true[i])));
        max_abs = std::max(max_abs, static_cast<double>(std::fabs(out_false[i])));
    }
    REQUIRE(max_abs > 0.0);                        // not a degenerate all-zero run
    REQUIRE(max_abs_diff < 1e-3 * std::max(1.0, max_abs));   // same loose tolerance engine_tests.cpp's
                                                              // own forward_one-vs-forward parity check uses
    }
    REQUIRE(std::filesystem::remove(path));
}

// --- B35: the fused quantized dot path (docs/MOE_QUANT_DOT.md) ------------------------------------
//
// The cases below split the one question that matters about B35 into its two independent halves,
// because a single end-to-end "is it close enough" number cannot tell them apart and a surprise in
// either has a completely different cause:
//
//   (a) is the WEIGHT decode exact? moeqd's three unpackers re-derive each format's block layout for
//       the integer path. If one of them has the layout even slightly wrong -- a shifted qh field, the
//       IQ4_NL nibble split read as adjacent pairs, IQ2_XXS's sign index off by a group -- the result
//       is still finite, still roughly the right magnitude, and still "close-ish" on random data. The
//       only way to catch that is to remove the OTHER error source entirely: feed an activation row
//       that int8-quantizes EXACTLY (integers in [-127,127] with a +/-127 in every group, so the scale
//       is exactly 1.0f and lrintf is the identity), leaving float-rounding as the only difference from
//       gguf::to_f32's own dequantized dot. Any real layout error blows straight through that.
//
//   (b) how big is the ACTIVATION quantization error on its own? That is the one genuinely new error
//       source B35 introduces into this codebase (docs/MOE_QUANT_DOT.md S5), and the end-to-end logit
//       diff on the real artifact cannot be called explained-or-surprising without a number for it
//       measured in isolation first.
//
// Both run over all THREE real sidecar formats, because the sidecar carries all three (IQ1_S 34,816
// planes, IQ2_XXS 14,336, IQ4_NL 24,576) and a layout error in any one of them would otherwise hide
// behind the two that are right.
namespace {

// Deterministic pseudo-random bytes in one of the three fused formats. Every byte is random EXCEPT each
// block's leading f16 scale, which is patched to a small positive value the same way make_q8_0 does --
// a random f16 would be inf/NaN often enough to make the comparison meaningless. No other field needs
// constraining: every grid index these formats can encode is in range by construction (IQ1_S's 8+3 bits
// address 2048 of 2048 grid entries, IQ2_XXS's byte addresses 256 of 256 and its 7-bit sign index 128
// of 128, IQ4_NL's nibble 16 of 16), which is itself worth knowing -- there is no such thing as a
// malformed block here, only a misread one.
std::vector<std::uint8_t> make_iq_blocks(gguf::TensorType type, std::uint64_t n_elements,
                                          std::uint32_t seed) {
    const gguf::BlockSpec spec = gguf::block_spec(static_cast<std::uint32_t>(type));
    REQUIRE(spec.elems != 0);
    REQUIRE(n_elements % spec.elems == 0);
    std::mt19937 rng(seed);
    const std::uint64_t blocks = n_elements / spec.elems;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(blocks * spec.bytes));
    for (auto& b : raw) b = static_cast<std::uint8_t>(rng() & 0xFFu);
    for (std::uint64_t b = 0; b < blocks; ++b) {
        const auto d_bits = static_cast<std::uint16_t>(0x3000u + (rng() & 0x0FFFu));
        std::memcpy(raw.data() + b * spec.bytes, &d_bits, sizeof d_bits);
    }
    return raw;
}

// The f32 side of every B35 comparison: gguf::to_f32 on the same bytes, then a plain sequential dot --
// i.e. exactly what dequantize_expert_source + moe::expert_ffn_row_source's simd::dot_seq compute
// today, spelled out here rather than called, so the fused path is compared against the CONTRACT and
// not against a second caller of itself.
double reference_dot(gguf::TensorType type, std::span<const std::uint8_t> raw, int n,
                      const std::vector<float>& x) {
    gguf::TensorInfo t;
    t.type_raw = static_cast<std::uint32_t>(type);
    t.dims = {static_cast<std::uint64_t>(n)};
    std::vector<float> w;
    REQUIRE(gguf::to_f32(t, raw, w));
    double s = 0.0;
    for (int i = 0; i < n; ++i)
        s += static_cast<double>(x[static_cast<std::size_t>(i)]) * w[static_cast<std::size_t>(i)];
    return s;
}

constexpr gguf::TensorType kFusedFormats[3] = {gguf::TensorType::IQ1_S, gguf::TensorType::IQ2_XXS,
                                                gguf::TensorType::IQ4_NL};
const char* format_name(gguf::TensorType t) {
    switch (t) {
        case gguf::TensorType::IQ1_S:   return "IQ1_S";
        case gguf::TensorType::IQ2_XXS: return "IQ2_XXS";
        default:                        return "IQ4_NL";
    }
}

}  // namespace

TEST_CASE("moeq (B35): the fused unpackers decode the SAME weights gguf::to_f32 does", "[moequant]") {
    // Half (a) above. With the activation quantization made lossless by construction, any remaining
    // disagreement is a weight-decode disagreement -- i.e. a block-layout bug -- and nothing else.
    constexpr int kN = 256;                     // one IQ1_S/IQ2_XXS super-block; 8 IQ4_NL blocks
    constexpr int kRows = 4;                    // several rows, so a non-zero row base is exercised

    std::vector<float> x(kN);
    std::mt19937 rng(4242);
    for (int g = 0; g < kN / moeqd::GROUP; ++g)
        for (int j = 0; j < moeqd::GROUP; ++j) {
            // Integers in [-126, 126], with an exact +/-127 planted in each group so amax is 127 and the
            // quantizer's scale is exactly 1.0f -- see this case's own comment.
            const int v = (j == 0) ? ((g % 2) ? -127 : 127)
                                    : static_cast<int>(rng() % 253u) - 126;
            x[static_cast<std::size_t>(g * moeqd::GROUP + j)] = static_cast<float>(v);
        }
    moeqd::ActBlocks xq;
    xq.quantize(x.data(), kN);
    for (int g = 0; g < kN / moeqd::GROUP; ++g)
        REQUIRE(xq.scale[static_cast<std::size_t>(g)] == 1.0f);      // the lossless premise, checked
    for (int i = 0; i < kN; ++i)
        REQUIRE(static_cast<float>(xq.qs[static_cast<std::size_t>(i)]) == x[static_cast<std::size_t>(i)]);

    for (const gguf::TensorType type : kFusedFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const gguf::BlockSpec spec = gguf::block_spec(raw_t);
        const std::vector<std::uint8_t> raw = make_iq_blocks(type, static_cast<std::uint64_t>(kRows) * kN,
                                                              77u + raw_t);
        std::vector<float> fused(kRows, 0.f);
        REQUIRE(moeqd::gemv_plane(raw_t, std::span<const std::uint8_t>(raw), kRows, kN, xq, fused.data()));

        double worst_rel = 0.0;
        for (int row = 0; row < kRows; ++row) {
            // The reference decodes THIS row's own bytes only -- row r begins at element r*kN, which is
            // a whole number of blocks for every format here.
            const auto byte_off = static_cast<std::size_t>(
                (static_cast<std::uint64_t>(row) * kN / spec.elems) * spec.bytes);
            const double ref = reference_dot(
                type, std::span<const std::uint8_t>(raw).subspan(byte_off,
                                                                  static_cast<std::size_t>(kN / spec.elems * spec.bytes)),
                kN, x);
            REQUIRE(std::isfinite(fused[static_cast<std::size_t>(row)]));
            REQUIRE(std::fabs(ref) > 0.0);
            worst_rel = std::max(worst_rel,
                                  std::fabs(fused[static_cast<std::size_t>(row)] - ref) / std::fabs(ref));
        }
        INFO("worst relative disagreement vs gguf::to_f32 = " << worst_rel);
        // Float-rounding scale only. A real layout error is orders of magnitude above this: the fused
        // sum and the reference sum are the same addends in the same order, differing only in where the
        // per-group scale is applied.
        REQUIRE(worst_rel < 1e-5);
    }
}

TEST_CASE("moeq (B35): the int8 activation is the only new error source, and this is its size",
          "[moequant]") {
    // Half (b) above -- the number docs/INDEPENDENT_REVIEW_BACKLOG.md B35 cites, measured here rather
    // than inferred from the end-to-end logit diff. A Gaussian activation row is the realistic case
    // (real MoE inputs are post-norm and roughly Gaussian); the bound below is what this actually
    // measures with room for a different seed, not a target picked first and hoped for.
    constexpr int kN = 256;
    std::vector<float> x(kN);
    std::mt19937 rng(20260921);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (float& v : x) v = normal(rng);

    moeqd::ActBlocks xq;
    xq.quantize(x.data(), kN);

    for (const gguf::TensorType type : kFusedFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const std::vector<std::uint8_t> raw = make_iq_blocks(type, kN, 909u + raw_t);
        float fused = 0.f;
        REQUIRE(moeqd::gemv_plane(raw_t, std::span<const std::uint8_t>(raw), 1, kN, xq, &fused));
        const double ref = reference_dot(type, std::span<const std::uint8_t>(raw), kN, x);
        REQUIRE(std::isfinite(fused));
        REQUIRE(std::fabs(ref) > 0.0);
        const double rel = std::fabs(fused - ref) / std::fabs(ref);
        INFO("relative error from int8 activation quantization = " << rel);
        REQUIRE(rel < 0.05);
    }
}

TEST_CASE("moeq (O1): the AVX2 kernel computes what the portable kernel does, at the real plane shapes",
          "[moequant]") {
    // gemv_plane runs detail::gemv_avx2 wherever the target has AVX2 (moeqd::kAvx2Kernels); the portable
    // detail::gemv is kept as its reference. The two share each format's bit-field parse, so what this
    // pins is everything they do NOT share: the vector materialisation of each format (IQ1_S's grid
    // qwords, IQ2_XXS's magnitude/sign split, IQ4_NL's vpshufb nibble decode), the vpsignb/vpmaddubsw
    // MAC, F16C's scale conversion and the per-lane accumulation.
    //
    // Real shapes, not a toy block: gate/up are 640 rows of 2560, down is 2560 rows of 640 -- the odd
    // rows of the latter begin half-way into an IQ1_S/IQ2_XXS super-block, which a 256-wide case never
    // reaches. 1.6M random elements per plane also cover every grid index and all 128 IQ2_XXS sign
    // patterns many times over.
    //
    // Tolerance: the kernels differ only in float summation order, so the right yardstick is the sum of
    // the MAGNITUDES of the per-group terms (the quantity reassociation error scales with), not the
    // possibly-cancelling row result. Any decode or sign error is O(1) against that, not O(1e-6).
    INFO("AVX2 kernels compiled in: " << moeqd::kAvx2Kernels);
    struct Shape { int rows, cols; };
    for (const Shape sh : {Shape{640, 2560}, Shape{2560, 640}}) {
        std::vector<float> x(static_cast<std::size_t>(sh.cols));
        std::mt19937 rng(31337u + static_cast<std::uint32_t>(sh.cols));
        std::normal_distribution<float> normal(0.f, 1.f);
        for (float& v : x) v = normal(rng);
        moeqd::ActBlocks xq;
        xq.quantize(x.data(), sh.cols);

        for (const gguf::TensorType type : kFusedFormats) {
            INFO("format " << format_name(type) << ", " << sh.rows << " x " << sh.cols);
            const auto raw_t = static_cast<std::uint32_t>(type);
            const std::vector<std::uint8_t> raw = make_iq_blocks(
                type, static_cast<std::uint64_t>(sh.rows) * static_cast<std::uint64_t>(sh.cols), 4040u + raw_t);

            std::vector<float> best(static_cast<std::size_t>(sh.rows));
            std::vector<float> portable(static_cast<std::size_t>(sh.rows));
            std::vector<double> magnitude(static_cast<std::size_t>(sh.rows), 0.0);
            REQUIRE(moeqd::gemv_plane(raw_t, std::span<const std::uint8_t>(raw), sh.rows, sh.cols, xq, best.data()));

            const auto reference = [&](const auto& plane) {
                moeqd::detail::gemv(plane, sh.rows, sh.cols, xq, portable.data());
                for (int r = 0; r < sh.rows; ++r)
                    for (int g = 0; g < sh.cols / moeqd::GROUP; ++g) {
                        const auto gi = static_cast<std::size_t>(g);
                        const moeqd::WeightGroup wg = plane.group(
                            static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(sh.cols) + gi * moeqd::GROUP);
                        double term = moeqd::detail::dot_group(wg.q.data(), xq.qs.data() + gi * moeqd::GROUP);
                        term += static_cast<double>(wg.delta) * xq.gsum[gi];
                        magnitude[static_cast<std::size_t>(r)] += std::fabs(
                            static_cast<double>(xq.scale[gi]) * wg.scale * term);
                    }
            };
            switch (type) {
                case gguf::TensorType::IQ1_S:   reference(moeqd::Iq1SPlane{raw.data()}); break;
                case gguf::TensorType::IQ2_XXS: reference(moeqd::Iq2XxsPlane{raw.data()}); break;
                default:                        reference(moeqd::Iq4NlPlane{raw.data()}); break;
            }

            double worst = 0.0;
            for (std::size_t r = 0; r < best.size(); ++r) {
                REQUIRE(std::isfinite(best[r]));
                REQUIRE(magnitude[r] > 0.0);
                worst = std::max(worst, std::fabs(static_cast<double>(best[r]) - portable[r]) / magnitude[r]);
            }
            INFO("worst |avx2 - portable| / sum|group terms| = " << worst);
            REQUIRE(worst < 2e-5);
        }
    }
}

TEST_CASE("moeq (B35): expert_ffn_row_quant computes the same expert the dequantize path does",
          "[moequant]") {
    // The whole fused FFN against the whole existing one, on the SAME encoded bytes: gate, up, SiLU,
    // the per-expert re-quantization of the down projection's input, and the down projection itself.
    // Close, NOT bit-exact, and deliberately not asserted to be (docs/MOE_QUANT_DOT.md S5) -- what this
    // pins is that the fused path is the same COMPUTATION at reduced precision, not a different one:
    // a swapped plane role or a transposed traversal would miss this tolerance by a wide margin even
    // though both paths would still produce finite, plausibly-scaled output.
    constexpr int kHidden = 256, kFf = 64;      // both multiples of 32; kFf deliberately NOT a multiple
                                                 // of 256, so down-projection rows start at every
                                                 // sub-block offset a real d_ff=640 plane would hit
    const moe::Dims dims{kHidden, kFf, 4, 2};

    std::vector<float> x(kHidden);
    std::mt19937 rng(31337);
    std::normal_distribution<float> normal(0.f, 1.f);
    for (float& v : x) v = normal(rng);
    moeqd::ActBlocks xq;
    xq.quantize(x.data(), kHidden);

    for (const gguf::TensorType type : kFusedFormats) {
        INFO("format " << format_name(type));
        const auto raw_t = static_cast<std::uint32_t>(type);
        const auto plane_elems = static_cast<std::uint64_t>(kHidden) * kFf;
        const std::vector<std::uint8_t> rg = make_iq_blocks(type, plane_elems, 11u + raw_t);
        const std::vector<std::uint8_t> ru = make_iq_blocks(type, plane_elems, 22u + raw_t);
        const std::vector<std::uint8_t> rd = make_iq_blocks(type, plane_elems, 33u + raw_t);
        const moeq::Desc dg{raw_t, kHidden, kFf, 0, 0, rg.size()};
        const moeq::Desc du{raw_t, kHidden, kFf, 0, 0, ru.size()};
        const moeq::Desc dd{raw_t, kFf, kHidden, 0, 0, rd.size()};

        // The existing path, in full: dequantize each plane to SOURCE-order f32, then run the f32 FFN.
        std::vector<float> wg, wu, wd;
        REQUIRE(moeq::dequantize_expert_source(dg, std::span<const std::uint8_t>(rg), wg));
        REQUIRE(moeq::dequantize_expert_source(du, std::span<const std::uint8_t>(ru), wu));
        REQUIRE(moeq::dequantize_expert_source(dd, std::span<const std::uint8_t>(rd), wd));
        std::vector<float> out_ref(kHidden), pre(kFf), g(kFf);
        moe::expert_ffn_row_source(dims, x.data(), wg.data(), wu.data(), wd.data(), out_ref.data(),
                                    pre.data(), g.data());

        moeqd::ActBlocks pq;
        std::vector<float> out_fused(kHidden);
        const moeqd::ExpertPlanes planes{{dg, std::span<const std::uint8_t>(rg)},
                                          {du, std::span<const std::uint8_t>(ru)},
                                          {dd, std::span<const std::uint8_t>(rd)}};
        REQUIRE(moeqd::expert_ffn_row_quant(dims, xq, planes, out_fused.data(), pre.data(), g.data(),
                                             pq));

        // The geometry guard is not decoration: a swapped gate/down pair, or a sidecar built at other
        // axes, must be refused rather than decoded into plausible garbage.
        const moeqd::ExpertPlanes swapped{planes.down, planes.up, planes.gate};
        std::vector<float> out_reject(kHidden, 7.f);
        REQUIRE_FALSE(moeqd::expert_ffn_row_quant(dims, xq, swapped, out_reject.data(), pre.data(),
                                                   g.data(), pq));
        REQUIRE(out_reject[0] == 7.f);

        double num = 0.0, den = 0.0;
        for (int j = 0; j < kHidden; ++j) {
            REQUIRE(std::isfinite(out_fused[static_cast<std::size_t>(j)]));
            const double d = static_cast<double>(out_fused[static_cast<std::size_t>(j)])
                             - out_ref[static_cast<std::size_t>(j)];
            num += d * d;
            den += static_cast<double>(out_ref[static_cast<std::size_t>(j)])
                   * out_ref[static_cast<std::size_t>(j)];
        }
        REQUIRE(den > 0.0);                       // not a degenerate all-zero expert
        const double l2_rel = std::sqrt(num / den);
        INFO("L2-relative difference, fused vs dequantize path = " << l2_rel);
        REQUIRE(l2_rel < 0.10);
    }
}

TEST_CASE("moeq (B35): a format the fused path cannot handle is refused, not silently mis-decoded",
          "[moequant]") {
    // The sidecar's formats are per-tensor, not per-role (moe_quant.hpp's own header comment), so a
    // model quantized with a mix this path does not cover is a real possibility rather than a
    // hypothetical -- and the failure mode of guessing would be a plausible-looking wrong answer.
    // Q8_0 stands in for "some other real format"; the width checks use rows that are not a multiple
    // of GROUP, which no real Qwen4 axis produces but a future one could.
    REQUIRE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S), 2560));
    REQUIRE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ2_XXS), 640));
    REQUIRE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ4_NL), 32));
    REQUIRE_FALSE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::Q8_0), 2560));
    REQUIRE_FALSE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::F32), 2560));
    REQUIRE_FALSE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S), 48));
    REQUIRE_FALSE(moeqd::fusable(static_cast<std::uint32_t>(gguf::TensorType::IQ1_S), 0));

    const std::vector<std::uint8_t> q8 = make_q8_0(256, 5u);
    std::vector<float> x(256, 0.5f);
    moeqd::ActBlocks xq;
    xq.quantize(x.data(), 256);
    float out = 1234.f;
    REQUIRE_FALSE(moeqd::gemv_plane(static_cast<std::uint32_t>(gguf::TensorType::Q8_0),
                                     std::span<const std::uint8_t>(q8), 1, 256, xq, &out));
    REQUIRE(out == 1234.f);                       // refused means untouched, not partially written
}
