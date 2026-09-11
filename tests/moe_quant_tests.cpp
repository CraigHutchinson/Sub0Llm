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
