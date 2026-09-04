// sub0llm-transplant -- offline weight transplant: a real Qwen3.8-Flash-Next GGUF file in, this
// project's own `S0L5` model file out (docs/WP4_SCOPE.md S4 / WP4c).
//
// WHERE THIS LIVES, AND WHY (the question WP4c's brief asks explicitly). `tools/`, one file, no
// subdirectory of its own -- because what is big about a transplant is the DATA, not the code, and the
// code that IS interesting has already been moved somewhere it can be tested:
//
//   * the mapping (names, axis order, the granularity reconstructions) is include/sub0/transplant.hpp,
//     engine-free, exercised by tests/transplant_tests.cpp and replayed against real fixtures by
//     tests/transplant_fixture_tests.cpp;
//   * the decoders are include/sub0/gguf.hpp, likewise;
//   * the destination layout is include/sub0/layout.hpp, which this file merely WALKS.
//
// What is left here is file plumbing and a reconciliation report -- the same shape as
// tools/configurator.cpp, which also reads an input and writes a generated artifact and also has no
// business in the hot path. A tools/transplant/ subdirectory would have been the right call only if
// the logic had stayed here, and the reason it did not is that levels 3 and 4 of the correctness gate
// are impossible to write against code buried in a tool binary.
//
// HOW IT IS COMPILED, which is the one genuinely unusual thing about this target. layout.hpp is closed
// over a sub0_config.hpp, so this tool is compiled against tests/qwen4_real_axes/sub0_config.hpp --
// the REAL model's axes -- with -DSUB0_QWEN4_LAYERS=4 for the 4-layer sub-stack (docs/WP4_SCOPE.md S7
// Q3: 3 GDN layers + 1 QSA layer, the real stack's own repeating unit). It therefore links NO engine
// library: sub0_core is compiled against the BUILD's generated config, and two definitions of
// sub0::PARAM_LAYOUT in one binary is an ODR violation. Same reasoning as sub0_qwen4_shape_tests.
//
// SCOPE, deliberately (docs/WP4_SCOPE.md S5): the n-gram/PLE table and its six per-layer tensors at
// layer 1, and the MTP and vision blocks, are NOT transplanted. They are reported by name in the
// unmatched-source list rather than silently skipped.

#include "sub0/gguf.hpp"
#include "sub0/layout.hpp"
#include "sub0/model_file.hpp"
#include "sub0/transplant.hpp"
#include "sub4_prefix.hpp"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <print>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sub0;
using namespace sub0::transplant;

namespace {

// --- the compile-time claim that this really is the real model's first four layers ----------------
// Asserted here as well as in tests/qwen4_real_shape_tests.cpp, against the SAME two literals, because
// the two layouts cannot be compared directly in one translation unit (sub4_prefix.hpp's own comment).
// The shape test owns the 48-layer half; this is the 4-layer half.
static_assert(N_LAYERS == 4, "this tool targets the 4-layer real sub-stack -- see the header comment");
static_assert(NUM_PARAMS == qwen4_sub4::NUM_PARAMS);
static_assert(PARAM_FLOATS == qwen4_sub4::PARAM_FLOATS);
static_assert(PARAM_LAYOUT[qwen4_sub4::PREFIX_TENSORS].off == qwen4_sub4::PREFIX_FLOATS,
              "layers 0-3 must occupy exactly the float span the 48-layer real-shape test also "
              "asserts -- otherwise this artifact is not a sub-stack of the real model");
static_assert(MIXER_SCHEDULE[0] == LayerMixer::Gdn && MIXER_SCHEDULE[3] == LayerMixer::Qsa,
              "the 3-GDN-then-1-QSA repeating unit");
static_assert(sizeof(ModelHeader) == 48);

// --- one GGUF shard, header parsed and kept ------------------------------------------------------
// The tensor DATA is not held: a shard is 50 GB and every tensor is read on demand from its own byte
// range. Only the header (a few MB) is ever buffered, and only long enough to copy the tensor table
// out -- gguf::Reader holds spans into its input, so the TensorInfos are copied, not referenced.
struct Shard {
    fs::path path;
    std::uint64_t data_offset = 0;
    std::vector<gguf::TensorInfo> tensors;
};

struct Source {
    const Shard* shard = nullptr;
    gguf::TensorInfo info;
    bool consumed = false;
};

// --- destination plan ------------------------------------------------------------------------------
// One entry per PARAM_LAYOUT tensor, in PARAM_LAYOUT order. Built by MIRRORING make_param_layout()'s
// own append order, then verified against PARAM_LAYOUT entry by entry (see verify_plan). The mirror is
// a second derivation of the same order, which is the only honest way to recover the (layer, expert)
// context PARAM_LAYOUT deliberately does not carry -- and the verification is what makes the
// duplication safe rather than a second source of truth that can drift silently. Same discipline as
// memplan::param_floats() being pinned against PARAM_FLOATS.
struct Slot {
    Dest dest;
    int  layer = -1;      // -1 for model-level tensors
    int  expert = -1;     // MoE routed experts only
    PKind expect_kind{};
    int  rows = 0, cols = 0;
};

std::vector<Slot> build_plan() {
    std::vector<Slot> plan;
    plan.reserve(NUM_PARAMS);
    auto add = [&](Dest d, PKind k, int layer, int rows, int cols, int expert = -1) {
        plan.push_back(Slot{d, layer, expert, k, rows, cols});
    };
    constexpr int WIDE = HC_COUNT * D_MODEL;

    add(Dest::TokEmb, PKind::TokEmb, -1, VOCAB, D_MODEL);
    for (int l = 0; l < N_LAYERS; ++l) {
        add(Dest::GrAttnNorm,   PKind::GrHcNorm,      l, 1, WIDE);
        add(Dest::GrAttnDown,   PKind::GrMixDown,     l, WIDE, HC_LOWRANK);
        add(Dest::GrAttnUp,     PKind::GrMixUp,       l, HC_LOWRANK, WIDE);
        add(Dest::GrAttnInject, PKind::GrBlockInject, l, WIDE, HC_COUNT);
        if (MIXER_SCHEDULE[static_cast<std::size_t>(l)] == LayerMixer::Qsa) {
            add(Dest::QsaQProj,     PKind::QsaQProj,     l, D_MODEL, D_Q);
            add(Dest::QsaGateProj,  PKind::QsaGateProj,  l, D_MODEL, D_Q);
            add(Dest::QsaKProj,     PKind::QsaKProj,     l, D_MODEL, D_KV);
            add(Dest::QsaVProj,     PKind::QsaVProj,     l, D_MODEL, D_KV);
            add(Dest::QsaOProj,     PKind::QsaOProj,     l, D_Q, D_MODEL);
            add(Dest::QsaQNorm,     PKind::QsaQNorm,     l, 1, D_HEAD);
            add(Dest::QsaKNorm,     PKind::QsaKNorm,     l, 1, D_HEAD);
            add(Dest::QsaIdxQkProj, PKind::QsaIdxQkProj, l, D_MODEL, QSA_IDX_QK_OUT);
            add(Dest::QsaIdxQNorm,  PKind::QsaIdxQNorm,  l, 1, QSA_INDEXER_HEAD_DIM);
            add(Dest::QsaIdxKNorm,  PKind::QsaIdxKNorm,  l, 1, QSA_INDEXER_HEAD_DIM);
        } else {
            add(Dest::GdnInProjQkv, PKind::GdnInProjQkv, l, D_MODEL, GDN_CONV_DIM);
            add(Dest::GdnInProjZ,   PKind::GdnInProjZ,   l, D_MODEL, GDN_VALUE_DIM);
            add(Dest::GdnInProjB,   PKind::GdnInProjB,   l, D_MODEL, GDN_V_HEADS);
            add(Dest::GdnInProjA,   PKind::GdnInProjA,   l, D_MODEL, GDN_V_HEADS);
            add(Dest::GdnConv,      PKind::GdnConv,      l, GDN_CONV_DIM, GDN_CONV_KERNEL);
            add(Dest::GdnALog,      PKind::GdnALog,      l, 1, GDN_V_HEADS);
            add(Dest::GdnDtBias,    PKind::GdnDtBias,    l, 1, GDN_V_HEADS);
            add(Dest::GdnNorm,      PKind::GdnNorm,      l, 1, GDN_V_HEAD_DIM);
            add(Dest::GdnOutProj,   PKind::GdnOutProj,   l, GDN_VALUE_DIM, D_MODEL);
        }
        add(Dest::GrFfnNorm,   PKind::GrHcNorm,      l, 1, WIDE);
        add(Dest::GrFfnDown,   PKind::GrMixDown,     l, WIDE, HC_LOWRANK);
        add(Dest::GrFfnUp,     PKind::GrMixUp,       l, HC_LOWRANK, WIDE);
        add(Dest::GrFfnInject, PKind::GrBlockInject, l, WIDE, HC_COUNT);
        add(Dest::MoeRouter, PKind::MoeRouter, l, D_MODEL, NUM_EXPERTS);
        for (int e = 0; e < NUM_EXPERTS; ++e) {
            add(Dest::MoeGate, PKind::MoeGate, l, D_MODEL, D_FF, e);
            add(Dest::MoeUp,   PKind::MoeUp,   l, D_MODEL, D_FF, e);
            add(Dest::MoeDown, PKind::MoeDown, l, D_FF, D_MODEL, e);
        }
        add(Dest::MoeSharedGate,     PKind::MoeSharedGate,     l, D_MODEL, D_FF);
        add(Dest::MoeSharedUp,       PKind::MoeSharedUp,       l, D_MODEL, D_FF);
        add(Dest::MoeSharedDown,     PKind::MoeSharedDown,     l, D_FF, D_MODEL);
        add(Dest::MoeSharedGateProj, PKind::MoeSharedGateProj, l, D_MODEL, 1);
    }
    add(Dest::GrExitNorm, PKind::GrHcNorm,  -1, 1, WIDE);
    add(Dest::GrExitDown, PKind::GrMixDown, -1, WIDE, HC_LOWRANK);
    add(Dest::GrExitUp,   PKind::GrMixUp,   -1, HC_LOWRANK, WIDE);
    add(Dest::LnF,    PKind::LnF,    -1, 1, D_MODEL);
    add(Dest::LmHead, PKind::LmHead, -1, D_MODEL, VOCAB);
    add(Dest::LmBias, PKind::LmBias, -1, 1, VOCAB);
    return plan;
}

// The mirror above is only safe because of this: every slot must line up with PARAM_LAYOUT's own entry
// at the same index, in kind AND in both extents. A reordering, an omission or a transposed shape in
// either derivation is a hard failure here rather than a 43 GB file with its tensors in the wrong
// places.
bool verify_plan(const std::vector<Slot>& plan) {
    if (plan.size() != static_cast<std::size_t>(NUM_PARAMS)) {
        std::println(stderr, "plan has {} slots, PARAM_LAYOUT has {}", plan.size(), NUM_PARAMS);
        return false;
    }
    for (std::size_t i = 0; i < plan.size(); ++i) {
        const ParamDesc& p = PARAM_LAYOUT[i];
        const Slot& s = plan[i];
        if (p.kind != s.expect_kind || p.rows != s.rows || p.cols != s.cols) {
            std::println(stderr,
                         "plan/PARAM_LAYOUT disagree at index {}: layout kind {} [{}x{}], plan kind {} "
                         "[{}x{}]", i, static_cast<int>(p.kind), p.rows, p.cols,
                         static_cast<int>(s.expect_kind), s.rows, s.cols);
            return false;
        }
    }
    return true;
}

// --- source reads ---------------------------------------------------------------------------------

// Reads and dequantizes a contiguous ELEMENT range of one source tensor. `first`/`count` must land on
// block boundaries for a quantized format -- which every slice this tool takes does, because an
// expert's own [out, in] plane is a whole multiple of 256 at these axes (2560*640 = 1,638,400). The
// check is explicit rather than assumed: a misaligned slice would decode neighbouring experts' values
// with no other symptom.
bool read_range(std::ifstream& f, const Shard& sh, const gguf::TensorInfo& t, std::uint64_t first,
                std::uint64_t count, std::vector<std::uint8_t>& raw, std::vector<float>& out) {
    const gguf::BlockSpec b = gguf::block_spec(t.type_raw);
    if (b.elems == 0) {
        std::println(stderr, "  {}: unsupported GGML type {}", t.name, t.type_raw);
        return false;
    }
    if (first % b.elems != 0 || (count % b.elems != 0 && first + count != t.element_count())) {
        std::println(stderr, "  {}: slice [{}, {}) is not block-aligned for type {} (block {} elems)",
                     t.name, first, first + count, t.type_raw, b.elems);
        return false;
    }
    const std::uint64_t byte_off = sh.data_offset + t.offset + (first / b.elems) * b.bytes;
    const std::uint64_t byte_len = ((count + b.elems - 1) / b.elems) * b.bytes;
    raw.assign(static_cast<std::size_t>(byte_len), 0);
    f.clear();
    f.seekg(static_cast<std::streamoff>(byte_off));
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byte_len));
    if (static_cast<std::uint64_t>(f.gcount()) != byte_len) {
        std::println(stderr, "  {}: short read ({} of {} bytes at offset {})", t.name,
                     static_cast<std::uint64_t>(f.gcount()), byte_len, byte_off);
        return false;
    }
    // A shape-carrying copy so to_f32 dispatches on this tensor's OWN type_raw with the sliced count.
    gguf::TensorInfo slice = t;
    slice.dims = {count};
    return gguf::to_f32(slice, raw, out);
}

// The GGUF `ne` array is fastest-varying-first, so a 2-D nn.Linear weight is declared {in, out} and a
// 3-D expert stack {in, out, n_experts}. Named here rather than indexed inline at four call sites,
// because "dims[0] is the input width" is the single most inversion-prone line in this file.
std::uint64_t ne_in(const gguf::TensorInfo& t)  { return t.dims.empty() ? 0 : t.dims[0]; }
std::uint64_t ne_out(const gguf::TensorInfo& t) { return t.dims.size() < 2 ? 1 : t.dims[1]; }

struct Totals {
    std::uint64_t dest_tensors = 0, dest_floats = 0, src_bytes_read = 0;
    int synthesized = 0, stats_checked = 0, stats_failed = 0;
};

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"sub0llm-transplant: a real Qwen4-preview GGUF -> this project's own S0L5 model file"};
    std::string gguf_dir, out_path, verify_path;
    bool dry_run = false;
    app.add_option("--gguf", gguf_dir, "directory holding the model's .gguf shards")->required();
    app.add_option("--out", out_path, "destination .bin (omit with --dry-run or --verify)");
    app.add_flag("--dry-run", dry_run,
                 "run the full reconciliation and per-tensor statistics without writing the artifact");
    // Re-runs the ENTIRE pipeline against an existing artifact and compares every destination tensor
    // BIT-FOR-BIT with what the transplant computes. Not redundant with the write: it is the only
    // thing that checks the bytes that actually landed on disk -- header size, tensor placement,
    // stream state, all of it -- rather than the bytes the tool believed it wrote.
    app.add_option("--verify", verify_path, "compare an existing artifact against a fresh transplant");
    CLI11_PARSE(app, argc, argv);
    if (!dry_run && out_path.empty() && verify_path.empty()) {
        std::println(stderr, "error: one of --out, --dry-run or --verify is required");
        return 2;
    }
    if (!verify_path.empty()) dry_run = true;   // verifying never writes

    // --- index every shard's header ---------------------------------------------------------------
    std::vector<Shard> shards;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(gguf_dir))
        if (e.is_regular_file() && e.path().extension() == ".gguf") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::println(stderr, "error: no .gguf files under {}", gguf_dir); return 2; }

    for (const auto& p : files) {
        std::ifstream f(p, std::ios::binary);
        if (!f) { std::println(stderr, "error: cannot open {}", p.string()); return 2; }
        // Enough for any shard's header + tensor table; the vocab string array in shard 1 is the
        // largest of these and fits comfortably.
        std::vector<std::uint8_t> head(64ull * 1024 * 1024);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        head.resize(static_cast<std::size_t>(f.gcount()));
        gguf::Reader r(head);
        if (!r.ok()) {
            std::println(stderr, "error: {} is not a readable GGUF (err {})", p.filename().string(),
                         static_cast<int>(r.error()));
            return 2;
        }
        Shard sh;
        sh.path = p;
        sh.data_offset = r.data_offset();
        sh.tensors.assign(r.tensors().begin(), r.tensors().end());
        std::println("shard {:<48} tensors {:>5}  data_offset {}", p.filename().string(),
                     sh.tensors.size(), sh.data_offset);
        shards.push_back(std::move(sh));
    }

    std::map<std::string, Source> by_name;
    for (const Shard& sh : shards)
        for (const gguf::TensorInfo& t : sh.tensors) by_name[t.name] = Source{&sh, t, false};
    std::println("indexed {} source tensors across {} shards", by_name.size(), shards.size());

    // --- destination plan, cross-checked against PARAM_LAYOUT --------------------------------------
    const std::vector<Slot> plan = build_plan();
    if (!verify_plan(plan)) return 3;
    std::println("destination: {} tensors, {} floats ({:.2f} GiB as f32), {} layers at the real axes",
                 NUM_PARAMS, PARAM_FLOATS,
                 static_cast<double>(PARAM_FLOATS) * 4.0 / (1024.0 * 1024.0 * 1024.0), N_LAYERS);

    std::ifstream vs;
    std::vector<float> vbuf;
    std::uint64_t verify_mismatches = 0;
    if (!verify_path.empty()) {
        vs.open(verify_path, std::ios::binary);
        if (!vs) { std::println(stderr, "error: cannot open {}", verify_path); return 2; }
        ModelHeader want, got;
        vs.read(reinterpret_cast<char*>(&got), sizeof got);
        if (std::memcmp(&want, &got, sizeof want) != 0) {
            std::println(stderr, "error: {}'s header does not match this build's config", verify_path);
            return 11;
        }
    }

    std::ofstream os;
    if (!dry_run) {
        os.open(out_path, std::ios::binary | std::ios::trunc);
        if (!os) { std::println(stderr, "error: cannot create {}", out_path); return 2; }
        const ModelHeader h;
        os.write(reinterpret_cast<const char*>(&h), sizeof h);
        if (!os) { std::println(stderr, "error: header write failed"); return 4; }
    }

    // --- execute -----------------------------------------------------------------------------------
    // One open file handle per shard, kept for the whole run: every tensor is read from its own byte
    // range and the destinations are walked in PARAM_LAYOUT order, so the writes are sequential even
    // though the reads are not.
    std::map<const Shard*, std::ifstream> handles;
    for (const Shard& sh : shards) handles[&sh].open(sh.path, std::ios::binary);

    Totals tot;
    std::vector<std::uint8_t> raw;
    std::vector<float> src, src_b, dst;
    const auto t0 = std::chrono::steady_clock::now();
    int last_pct = -1;

    for (std::size_t i = 0; i < plan.size(); ++i) {
        const Slot& s = plan[i];
        const ParamDesc& p = PARAM_LAYOUT[i];
        const Recipe rec = recipe_for(s.dest);
        const std::size_t n = static_cast<std::size_t>(s.rows) * s.cols;
        dst.assign(n, 0.f);

        if (rec.op == Op::Synthetic) {
            std::fill(dst.begin(), dst.end(), rec.fill);
            ++tot.synthesized;
        } else {
            const std::string name = gguf_name(rec.src, s.layer);
            auto it = by_name.find(name);
            if (it == by_name.end()) {
                std::println(stderr, "error: destination {} wants missing source tensor '{}'",
                             static_cast<int>(s.dest), name);
                return 5;
            }
            Source& srcT = it->second;
            srcT.consumed = true;
            const gguf::TensorInfo& t = srcT.info;

            const std::uint64_t in_f = ne_in(t), out_f = ne_out(t);
            std::uint64_t first = 0, count = t.element_count();
            if (rec.op == Op::ExpertSlice) {
                count = in_f * out_f;
                first = static_cast<std::uint64_t>(s.expert) * count;
            }
            if (!read_range(handles[srcT.shard], *srcT.shard, t, first, count, raw, src)) return 6;
            tot.src_bytes_read += raw.size();

            Stats before = stats_of(src);
            switch (rec.op) {
                case Op::Copy:
                    if (src.size() != n) { std::println(stderr, "error: {} size {} != {}", name, src.size(), n); return 7; }
                    dst = src;
                    break;
                case Op::Transpose:
                case Op::ExpertSlice:
                    if (src.size() != n) { std::println(stderr, "error: {} size {} != {}", name, src.size(), n); return 7; }
                    transpose_out_in(src.data(), static_cast<int>(out_f), static_cast<int>(in_f), dst.data());
                    break;
                case Op::PerHeadHalf:
                    if (src.size() != n * 2) { std::println(stderr, "error: {} size {} != 2*{}", name, src.size(), n); return 7; }
                    per_head_half_transpose(src.data(), N_HEADS, D_HEAD, static_cast<int>(in_f),
                                            rec.half, dst.data());
                    break;
                case Op::ConcatOut: {
                    const std::string name_b = gguf_name(rec.src2, s.layer);
                    auto it_b = by_name.find(name_b);
                    if (it_b == by_name.end()) { std::println(stderr, "error: missing '{}'", name_b); return 5; }
                    it_b->second.consumed = true;
                    const gguf::TensorInfo& tb = it_b->second.info;
                    std::vector<std::uint8_t> raw_b;
                    if (!read_range(handles[it_b->second.shard], *it_b->second.shard, tb, 0,
                                    tb.element_count(), raw_b, src_b)) return 6;
                    tot.src_bytes_read += raw_b.size();
                    concat_out_transpose(src.data(), static_cast<int>(out_f), src_b.data(),
                                         static_cast<int>(ne_out(tb)), static_cast<int>(in_f), dst.data());
                    // Level 2 compares the destination against BOTH halves, not just the q half --
                    // the concatenation is what makes them one tensor, so the statistics of the pair
                    // are the ones that must survive it.
                    src.insert(src.end(), src_b.begin(), src_b.end());
                    before = stats_of(src);
                    break;
                }
                case Op::Synthetic: break;
            }

            // LEVEL 2 (docs/WP4_SCOPE.md S4c): every op above is a permutation of its source, so the
            // statistics must survive it -- exactly, for the count and the extrema. Checked on EVERY
            // tensor, not a sample: the cost is one pass over data already in cache.
            const Stats after = stats_of(dst);
            ++tot.stats_checked;
            // PerHeadHalf is the one op whose source is larger than its destination (it takes half of
            // the fused projection), so its "before" is not comparable and is checked by extent only.
            const bool ok = (rec.op == Op::PerHeadHalf)
                                ? (after.n == n && after.nonfinite == 0)
                                : stats_consistent(before, after);
            if (!ok) {
                ++tot.stats_failed;
                std::println(stderr,
                             "LEVEL-2 FAIL {} <- {}: src n={} mean={:.6g} std={:.6g} min={:.6g} max={:.6g} | "
                             "dst n={} mean={:.6g} std={:.6g} min={:.6g} max={:.6g}",
                             static_cast<int>(s.dest), name, before.n, before.mean, before.stddev,
                             before.min, before.max, after.n, after.mean, after.stddev, after.min, after.max);
            }
        }

        if (!dry_run) {
            os.write(reinterpret_cast<const char*>(dst.data()),
                     static_cast<std::streamsize>(dst.size() * sizeof(float)));
            if (!os) { std::println(stderr, "error: write failed at destination {}", i); return 4; }
        } else if (vs.is_open()) {
            vbuf.assign(n, 0.f);
            vs.seekg(static_cast<std::streamoff>(sizeof(ModelHeader) + p.off * sizeof(float)));
            vs.read(reinterpret_cast<char*>(vbuf.data()), static_cast<std::streamsize>(n * sizeof(float)));
            if (static_cast<std::size_t>(vs.gcount()) != n * sizeof(float) || vbuf != dst) {
                ++verify_mismatches;
                if (verify_mismatches <= 10)
                    std::println(stderr, "VERIFY FAIL at destination {} (kind {}, layer {}, expert {})",
                                 i, static_cast<int>(p.kind), s.layer, s.expert);
            }
        }
        tot.dest_tensors += 1;
        tot.dest_floats += n;
        if (p.off + n != (i + 1 < plan.size() ? PARAM_LAYOUT[i + 1].off : PARAM_FLOATS)) {
            std::println(stderr, "error: destination {} does not abut the next -- layout walk desynced", i);
            return 8;
        }
        if (const int pct = static_cast<int>(100.0 * static_cast<double>(tot.dest_floats) /
                                              static_cast<double>(PARAM_FLOATS));
            pct != last_pct && pct % 5 == 0) {
            last_pct = pct;
            std::println("  {:3}%  {} / {} tensors, {} floats", pct, tot.dest_tensors, NUM_PARAMS,
                         tot.dest_floats);
        }
    }

    if (!dry_run) {
        // The three trailing records, in the order load_model reads them (see model_file.hpp). The
        // tokenizer fingerprint is 0 -- "unknown", which load_model treats as "no guard" -- and that
        // is the honest value: these weights were trained against QWEN'S vocabulary, which is
        // emphatically not this project's tokenizer, so stamping any fingerprint this build could
        // produce would assert something false.
        const std::uint64_t tok_fp = 0;
        const std::uint64_t arch = ARCH_FINGERPRINT, arch2 = ARCH_FINGERPRINT2;
        os.write(reinterpret_cast<const char*>(&tok_fp), sizeof tok_fp);
        os.write(reinterpret_cast<const char*>(&arch), sizeof arch);
        os.write(reinterpret_cast<const char*>(&arch2), sizeof arch2);
        os.flush();
        if (!os) { std::println(stderr, "error: trailer write failed"); return 4; }
    }

    // --- LEVEL 1: reconciliation --------------------------------------------------------------------
    // Every destination consumed exactly once (guaranteed by the single pass over PARAM_LAYOUT above),
    // and every source that is IN SCOPE consumed. The unmatched list is printed in full rather than
    // counted: "a whole mechanism silently skipped" is the failure it exists to catch, and a count
    // would not say which.
    std::vector<std::string> unmatched;
    for (const auto& [name, srcT] : by_name) {
        if (srcT.consumed) continue;
        // Layers >= N_LAYERS are simply not part of this sub-stack.
        bool other_layer = false;
        if (name.rfind("blk.", 0) == 0) {
            const std::size_t dot = name.find('.', 4);
            const int l = std::stoi(name.substr(4, dot - 4));
            other_layer = (l >= N_LAYERS);
        }
        if (!other_layer) unmatched.push_back(name);
    }
    const auto secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::println("");
    std::println("--- level 1: reconciliation -----------------------------------------------");
    std::println("destinations filled      : {} of {}", tot.dest_tensors, NUM_PARAMS);
    std::println("destination floats       : {} of {} ({} bytes)", tot.dest_floats, PARAM_FLOATS,
                 tot.dest_floats * 4);
    std::println("synthesized (no source)  : {}", tot.synthesized);
    std::println("encoded source bytes read: {}", tot.src_bytes_read);
    std::println("unmatched in-scope source tensors: {}", unmatched.size());
    for (const std::string& n : unmatched) std::println("    {}", n);
    std::println("--- level 2: per-tensor statistics ---------------------------------------");
    std::println("tensors checked          : {}", tot.stats_checked);
    std::println("mismatches               : {}", tot.stats_failed);
    std::println("elapsed                  : {:.1f}s", secs);
    if (vs.is_open())
        std::println("--- artifact verification ------------------------------------------------\n"
                     "bit-for-bit mismatches   : {} of {} destination tensors ({})",
                     verify_mismatches, NUM_PARAMS, verify_path);
    if (!dry_run) std::println("wrote {} ({} bytes incl. header + trailers)", out_path,
                               sizeof(ModelHeader) + PARAM_FLOATS * 4 + 24);

    if (tot.dest_tensors != static_cast<std::uint64_t>(NUM_PARAMS)) return 9;
    if (tot.dest_floats != PARAM_FLOATS) return 9;
    if (tot.stats_failed != 0) return 10;
    if (verify_mismatches != 0) return 12;
    return 0;
}
