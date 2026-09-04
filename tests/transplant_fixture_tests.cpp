// transplant_fixture_tests.cpp -- WP4c's PRIMARY correctness gate: docs/WP4_SCOPE.md S4c levels 3
// and 4, the two that a shape check and a statistical check cannot substitute for.
//
// WHAT IS ACTUALLY UNDER TEST, and why this construction rather than "load the real 50GB file":
//
// The thing most likely to be wrong in a weight transplant is the MAPPING -- which GGUF name feeds
// which destination slot, which way each 2-D tensor turns, and how the three granularity mismatches
// are reconstructed. None of that depends on the model's SIZE. So each test here:
//
//   1. takes an EXISTING real-weight fixture (tests/fixtures/qwen4_preview/, extracted from the real
//      Qwen/Qwen3.8-Flash-Next checkpoint and already trusted -- it is what gdn_qwen4_fixture_tests.cpp
//      and qsa_qwen4_fixture_tests.cpp match against);
//   2. RE-ENCODES it as a GGUF file, under the REAL model's own GGUF tensor names, in GGUF's own byte
//      order -- which for every one of these is the fixture .bin's existing [out_features, in_features]
//      row-major layout, since that is exactly what a PyTorch nn.Linear weight is;
//   3. reads it back through sub0::gguf::Reader + to_f32 and runs the real sub0::transplant recipes;
//   4. feeds the result straight into the mechanism's own math core and compares against the
//      fixture's real reference output.
//
// Step 2 is the load-bearing part. The fixture files are stored SPLIT the way this engine wants
// (q_proj and gate_proj separately; the indexer's qk_proj fused) while GGUF stores them the other way
// round (attn_q fused per head; the indexer's q_proj and k_proj separate) -- so re-encoding forces the
// test to perform the INVERSE of each granularity change, and the transplant must undo it exactly. A
// per-head interleave that the transplant then split down the middle would fail here and pass every
// shape assertion. That is the whole point.
//
// The tolerance is the same one the existing fixture tests use, and for the same reason: this path
// computes in float32 while the reference ran in PyTorch. Any mapping error is orders of magnitude
// larger than that, so the tolerance is not what makes this pass.

#include "sub0/gdn_math.hpp"
#include "sub0/gguf.hpp"
#include "sub0/qsa_math.hpp"
#include "sub0/transplant.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#ifdef SUB0_SOURCE_DIR

namespace {
namespace fs = std::filesystem;
using namespace sub0::transplant;

std::vector<float> read_f32(const fs::path& p, std::size_t expect_n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> out(expect_n);
    if (!f) return {};
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expect_n * sizeof(float)));
    if (static_cast<std::size_t>(f.gcount()) != expect_n * sizeof(float)) return {};
    return out;
}

// A minimal GGUF v3 writer. Deliberately NOT shared with gguf_tests.cpp's own GgufBuilder (which lives
// in that file's anonymous namespace for the same reason): a builder that shares code with the reader
// would hide a bug common to both, and one that shares code with the other test file would couple two
// independent checks. This one only needs F32 tensors, which keeps it short.
class GgufWriter {
public:
    GgufWriter() {
        raw("GGUF", 4);
        pod<std::uint32_t>(3);
        tensor_count_pos_ = buf_.size(); pod<std::uint64_t>(0);
        kv_count_pos_     = buf_.size(); pod<std::uint64_t>(0);
    }

    // `dims` in GGML ne order (fastest-varying first). For a [out, in] row-major PyTorch weight that
    // is {in, out} -- the very inversion this file exists to exercise, so it is spelled at every call
    // site rather than hidden in a helper.
    void add(const std::string& name, std::vector<std::uint64_t> dims, const std::vector<float>& data) {
        std::size_t n = 1;
        for (auto d : dims) n *= static_cast<std::size_t>(d);
        REQUIRE(n == data.size());
        pending_.push_back({name, std::move(dims), data});
    }

    std::vector<std::uint8_t> finish() {
        // Two passes: the tensor-info table must carry each tensor's offset into the data section, and
        // the data section only starts once the whole table is written and padded.
        std::uint64_t off = 0;
        for (auto& t : pending_) {
            t.offset = off;
            off += t.data.size() * 4;
            off = (off + 31) / 32 * 32;   // ggml keeps every tensor's data aligned
        }
        for (const auto& t : pending_) {
            str(t.name);
            pod<std::uint32_t>(static_cast<std::uint32_t>(t.dims.size()));
            for (auto d : t.dims) pod(d);
            pod<std::uint32_t>(0);        // F32
            pod(t.offset);
            ++tensor_count_;
        }
        std::memcpy(buf_.data() + tensor_count_pos_, &tensor_count_, sizeof tensor_count_);
        std::memcpy(buf_.data() + kv_count_pos_, &kv_count_, sizeof kv_count_);
        if (const std::size_t rem = buf_.size() % 32; rem != 0) buf_.insert(buf_.end(), 32 - rem, std::uint8_t{0});
        const std::size_t data_start = buf_.size();
        buf_.resize(data_start + static_cast<std::size_t>(off), std::uint8_t{0});
        for (const auto& t : pending_)
            std::memcpy(buf_.data() + data_start + static_cast<std::size_t>(t.offset), t.data.data(),
                        t.data.size() * 4);
        return buf_;
    }

private:
    struct Pending {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::vector<float> data;
        std::uint64_t offset = 0;
    };
    template <class T> void pod(T v) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        buf_.insert(buf_.end(), p, p + sizeof(T));
    }
    void raw(const char* s, std::size_t n) { buf_.insert(buf_.end(), s, s + n); }
    void str(const std::string& s) { pod<std::uint64_t>(s.size()); buf_.insert(buf_.end(), s.begin(), s.end()); }

    std::vector<std::uint8_t> buf_;
    std::vector<Pending> pending_;
    std::size_t tensor_count_pos_ = 0, kv_count_pos_ = 0;
    std::uint64_t tensor_count_ = 0, kv_count_ = 0;
};

// Runs one destination's recipe against a parsed GGUF, exactly as the offline tool does -- same
// recipe_for(), same ops, same name substitution. The tool adds file/shard plumbing and PARAM_LAYOUT
// placement around this; the mapping decisions themselves are all here.
struct Applied { std::vector<float> data; Stats src_stats, dst_stats; };

Applied apply(const sub0::gguf::Reader& r, Dest d, int layer, int rows, int cols,
              int n_heads = 0, int head_dim = 0) {
    const Recipe rec = recipe_for(d);
    Applied a;
    a.data.assign(static_cast<std::size_t>(rows) * cols, 0.f);

    auto decode = [&r](const std::string& name, std::vector<float>& out) {
        const sub0::gguf::TensorInfo* t = r.find_tensor(name);
        REQUIRE(t != nullptr);
        REQUIRE(sub0::gguf::to_f32(*t, r.tensor_bytes(*t), out));
    };

    if (rec.op == Op::Synthetic) {
        std::fill(a.data.begin(), a.data.end(), rec.fill);
        a.dst_stats = a.src_stats = stats_of(a.data);
        return a;
    }

    std::vector<float> src;
    decode(gguf_name(rec.src, layer), src);
    switch (rec.op) {
        case Op::Copy:
            REQUIRE(src.size() == a.data.size());
            a.data = src;
            a.src_stats = stats_of(src);
            break;
        case Op::Transpose:
            REQUIRE(src.size() == a.data.size());
            transpose_out_in(src.data(), /*out_f=*/cols, /*in_f=*/rows, a.data.data());
            a.src_stats = stats_of(src);
            break;
        case Op::PerHeadHalf:
            REQUIRE(src.size() == a.data.size() * 2);
            per_head_half_transpose(src.data(), n_heads, head_dim, /*in_f=*/rows, rec.half, a.data.data());
            // A half of the source, so the statistics compare against that half, not the whole tensor.
            a.src_stats = stats_of(a.data);
            break;
        case Op::ConcatOut: {
            std::vector<float> src_b;
            decode(gguf_name(rec.src2, layer), src_b);
            const int out_a = static_cast<int>(src.size() / static_cast<std::size_t>(rows));
            const int out_b = static_cast<int>(src_b.size() / static_cast<std::size_t>(rows));
            REQUIRE(out_a + out_b == cols);
            concat_out_transpose(src.data(), out_a, src_b.data(), out_b, /*in_f=*/rows, a.data.data());
            std::vector<float> joined = src;
            joined.insert(joined.end(), src_b.begin(), src_b.end());
            a.src_stats = stats_of(joined);
            break;
        }
        default:
            FAIL("op not exercised by this fixture");
    }
    a.dst_stats = stats_of(a.data);
    return a;
}

}  // namespace

// --- LEVEL 3: layer 0, a GDN layer -----------------------------------------------------------------

TEST_CASE("WP4c level 3: the GDN mapping replays layer 0's real fixture through a GGUF round trip",
          "[transplant][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    if (!fs::exists(dir / "gdn_layer0_small_input.bin")) {
        WARN("Qwen4 GDN fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }
    // The fixture's own sliced config (gdn_layer0_small_manifest.json). head_k_dim/head_v_dim are the
    // REAL 128; only hidden_size and the head COUNTS were sliced, and the real 3:1 value:key head
    // ratio is preserved -- so the repeat_interleave fold this mapping feeds is genuinely exercised.
    constexpr int T = 6, hidden = 32, k_heads = 1, v_heads = 3, hk = 128, hv = 128, kernel = 4;
    const sub0::gdn::Dims dims{hidden, k_heads, v_heads, hk, hv, kernel};
    const int key_dim = dims.key_dim(), value_dim = dims.value_dim(), conv_dim = dims.conv_dim();
    REQUIRE(conv_dim == 640);
    REQUIRE(value_dim == 384);
    (void)key_dim;

    const auto input    = read_f32(dir / "gdn_layer0_small_input.bin", static_cast<std::size_t>(T) * hidden);
    const auto expected = read_f32(dir / "gdn_layer0_small_output.bin", static_cast<std::size_t>(T) * hidden);
    REQUIRE(input.size() == static_cast<std::size_t>(T) * hidden);

    // Re-encode the fixture as GGUF. Every .bin is already [out, in] row-major (PyTorch's own
    // convention), which is exactly GGUF's byte order -- so the ne arrays below are {in, out}, the
    // inversion described in transplant.hpp's header. The names are the REAL model's, read out of the
    // real file's tensor table, not invented for this test.
    GgufWriter w;
    w.add("blk.0.attn_qkv.weight",   {hidden, static_cast<std::uint64_t>(conv_dim)},
          read_f32(dir / "gdn_layer0_small_weight_in_proj_qkv.bin", static_cast<std::size_t>(conv_dim) * hidden));
    w.add("blk.0.attn_gate.weight",  {hidden, static_cast<std::uint64_t>(value_dim)},
          read_f32(dir / "gdn_layer0_small_weight_in_proj_z.bin", static_cast<std::size_t>(value_dim) * hidden));
    w.add("blk.0.ssm_beta.weight",   {hidden, v_heads},
          read_f32(dir / "gdn_layer0_small_weight_in_proj_b.bin", static_cast<std::size_t>(v_heads) * hidden));
    w.add("blk.0.ssm_alpha.weight",  {hidden, v_heads},
          read_f32(dir / "gdn_layer0_small_weight_in_proj_a.bin", static_cast<std::size_t>(v_heads) * hidden));
    // Depthwise conv: PyTorch [C, 1, K] row-major == ne {K, C}. NOT a transpose, unlike every
    // projection above -- the one 2-D-shaped GDN tensor that stays put.
    w.add("blk.0.ssm_conv1d.weight", {kernel, static_cast<std::uint64_t>(conv_dim)},
          read_f32(dir / "gdn_layer0_small_weight_conv1d.bin", static_cast<std::size_t>(conv_dim) * kernel));
    w.add("blk.0.ssm_dt.bias",       {v_heads}, read_f32(dir / "gdn_layer0_small_weight_dt_bias.bin", v_heads));
    w.add("blk.0.ssm_a",             {v_heads}, read_f32(dir / "gdn_layer0_small_weight_A_log.bin", v_heads));
    w.add("blk.0.ssm_norm.weight",   {hv},      read_f32(dir / "gdn_layer0_small_weight_norm.bin", hv));
    w.add("blk.0.ssm_out.weight",    {static_cast<std::uint64_t>(value_dim), hidden},
          read_f32(dir / "gdn_layer0_small_weight_out_proj.bin", static_cast<std::size_t>(hidden) * value_dim));
    const std::vector<std::uint8_t> gguf_buf = w.finish();

    sub0::gguf::Reader r(gguf_buf);
    REQUIRE(r.ok());

    // Now the transplant proper: every buffer below comes out of recipe_for(), by name.
    const Applied w_qkv = apply(r, Dest::GdnInProjQkv, 0, hidden, conv_dim);
    const Applied w_z   = apply(r, Dest::GdnInProjZ,   0, hidden, value_dim);
    const Applied w_b   = apply(r, Dest::GdnInProjB,   0, hidden, v_heads);
    const Applied w_a   = apply(r, Dest::GdnInProjA,   0, hidden, v_heads);
    const Applied conv  = apply(r, Dest::GdnConv,      0, conv_dim, kernel);
    const Applied dt    = apply(r, Dest::GdnDtBias,    0, 1, v_heads);
    const Applied alog  = apply(r, Dest::GdnALog,      0, 1, v_heads);
    const Applied nrm   = apply(r, Dest::GdnNorm,      0, 1, hv);
    const Applied w_out = apply(r, Dest::GdnOutProj,   0, value_dim, hidden);

    // Level 2 in situ: every op above is a permutation, so the statistics must survive it exactly.
    for (const auto* a : {&w_qkv, &w_z, &w_b, &w_a, &conv, &dt, &alog, &nrm, &w_out})
        CHECK(stats_consistent(a->src_stats, a->dst_stats));
    // ...and the fixture is not degenerate, so a mapping that produced zeros could not pass.
    CHECK(w_qkv.dst_stats.stddev > 0.0);
    CHECK(w_out.dst_stats.stddev > 0.0);

    std::vector<float> state(sub0::gdn::state_floats(dims), 0.f);
    std::vector<float> conv_hist(sub0::gdn::conv_hist_floats(dims), 0.f);
    std::vector<float> scratch(sub0::gdn::scratch_floats(dims, T), 0.f);
    std::vector<float> out(static_cast<std::size_t>(T) * hidden, 0.f);
    sub0::gdn::forward(dims, T, input.data(), w_qkv.data.data(), w_z.data.data(), w_b.data.data(),
                       w_a.data.data(), conv.data.data(), dt.data.data(), alog.data.data(),
                       nrm.data.data(), w_out.data.data(), state.data(), conv_hist.data(), out.data(),
                       scratch.data());

    double max_abs = 0.0, sum_expected = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(out[i]) - expected[i]));
        sum_expected += std::fabs(static_cast<double>(expected[i]));
    }
    INFO("max |transplanted - real reference| = " << max_abs);
    REQUIRE(sum_expected > 0.0);
    REQUIRE(max_abs < 5e-5);   // the same float32 tolerance gdn_qwen4_fixture_tests.cpp records

    // MUTATION CHECK (AGENTS.md S6): ssm_alpha and ssm_beta are both [hidden, num_v_heads] -- the exact
    // same-shaped-pair identity swap that WP-GDN Stage 3 found in already-"verified" code, and one that
    // no shape check and no statistical check can see. Swapping them must MOVE the output, or the match
    // above proves nothing about which name feeds which gate.
    std::vector<float> swapped(out.size(), 0.f);
    std::fill(state.begin(), state.end(), 0.f);
    std::fill(conv_hist.begin(), conv_hist.end(), 0.f);
    sub0::gdn::forward(dims, T, input.data(), w_qkv.data.data(), w_z.data.data(),
                       /*b <- a*/ w_a.data.data(), /*a <- b*/ w_b.data.data(),
                       conv.data.data(), dt.data.data(), alog.data.data(), nrm.data.data(),
                       w_out.data.data(), state.data(), conv_hist.data(), swapped.data(), scratch.data());
    double swap_diff = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
        swap_diff = std::max(swap_diff, std::fabs(static_cast<double>(swapped[i]) - out[i]));
    INFO("alpha/beta swap moves the output by " << swap_diff);
    // The bar is stated RELATIVE to the reference agreement above, and that is not a cosmetic choice:
    // measured, the swap moves this fixture's output by only ~4.1e-8, while the correct mapping agrees
    // with the real reference to ~4.4e-11. So the mutation IS detectable -- by nearly three orders of
    // magnitude -- but an absolute 1e-6 bar would have rejected it, and an absolute bar loose enough to
    // pass would have been satisfied by float32 noise too. Recording the actual weakness of this
    // particular signal matters: A_log/dt_bias/alpha/beta all feed gates through softplus and sigmoid,
    // which compress differences hard, which is exactly why the original dt_bias/A_log swap survived a
    // fixture test and a gradient check both (docs/GATED_DELTANET.md S6).
    CHECK(swap_diff > 100.0 * max_abs);
    CHECK(swap_diff > 1e-9);
}

// --- LEVEL 4: layer 3, the first QSA layer ---------------------------------------------------------

TEST_CASE("WP4c level 4: the QSA mapping replays layer 3's real fixture through a GGUF round trip",
          "[transplant][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    if (!fs::exists(dir / "qsa_layer3_small_input.bin")) {
        WARN("Qwen4 QSA fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }
    // The fixture's own sliced config (qsa_layer3_small_manifest.json). n_heads == 2 is what makes the
    // per-head q|gate split distinguishable from a down-the-middle one at all.
    constexpr int H = 16, NH = 2, HD = 8, NKV = 1;
    constexpr int IN_H = 2, IKV = 1, IHD = 8, BUDGET = 8, RATIO = 4, ROT = 2, T = 24;
    const sub0::qsa::Dims dims{H, NH, HD, NKV, IN_H, IKV, IHD, BUDGET, RATIO, ROT};

    const auto input    = read_f32(dir / "qsa_layer3_small_input.bin", static_cast<std::size_t>(T) * H);
    const auto cos      = read_f32(dir / "qsa_layer3_small_cos.bin", static_cast<std::size_t>(T) * ROT);
    const auto sin      = read_f32(dir / "qsa_layer3_small_sin.bin", static_cast<std::size_t>(T) * ROT);
    const auto expected = read_f32(dir / "qsa_layer3_small_output.bin", static_cast<std::size_t>(T) * H);
    REQUIRE(input.size() == static_cast<std::size_t>(T) * H);

    const auto q_w    = read_f32(dir / "qsa_layer3_small_weight_q_proj.bin", static_cast<std::size_t>(NH * HD) * H);
    const auto gate_w = read_f32(dir / "qsa_layer3_small_weight_gate_proj.bin", static_cast<std::size_t>(NH * HD) * H);
    const auto idx_qk = read_f32(dir / "qsa_layer3_small_weight_idx_qk_proj.bin",
                                  static_cast<std::size_t>((IN_H + IKV) * IHD) * H);
    REQUIRE(q_w.size() == static_cast<std::size_t>(NH * HD) * H);
    REQUIRE(idx_qk.size() == static_cast<std::size_t>((IN_H + IKV) * IHD) * H);

    // GRANULARITY INVERSION 1 -- FUSE q_proj and gate_proj back into the real `attn_q.weight`. The
    // fixture stores them already split (the engine's own shape); GGUF stores one tensor whose output
    // rows are, per head, [query_h | gate_h] ADJACENT. Building it that way here is what makes the
    // transplant's per-head split a real test: a down-the-middle split would read head 1's query rows
    // out of head 0's gate.
    std::vector<float> fused_q(static_cast<std::size_t>(NH) * 2 * HD * H);
    for (int h = 0; h < NH; ++h)
        for (int d = 0; d < HD; ++d)
            for (int i = 0; i < H; ++i) {
                const std::size_t src_row = static_cast<std::size_t>(h) * HD + d;
                fused_q[(static_cast<std::size_t>(h) * 2 * HD + d) * H + i] = q_w[src_row * H + i];
                fused_q[(static_cast<std::size_t>(h) * 2 * HD + HD + d) * H + i] = gate_w[src_row * H + i];
            }
    // GRANULARITY INVERSION 2 -- SPLIT the fused indexer projection into the real file's two tensors.
    // The reference's own torch.split is [n_heads*head_dim, kv_heads*head_dim] on the OUTPUT axis, so
    // the q half is the first IN_H*IHD output rows and the k half the remaining IKV*IHD.
    const std::size_t q_rows = static_cast<std::size_t>(IN_H) * IHD, k_rows = static_cast<std::size_t>(IKV) * IHD;
    const std::vector<float> idx_q(idx_qk.begin(), idx_qk.begin() + static_cast<std::ptrdiff_t>(q_rows * H));
    const std::vector<float> idx_k(idx_qk.begin() + static_cast<std::ptrdiff_t>(q_rows * H), idx_qk.end());
    REQUIRE(idx_k.size() == k_rows * H);

    GgufWriter w;
    w.add("blk.3.attn_q.weight",      {H, static_cast<std::uint64_t>(NH * 2 * HD)}, fused_q);
    w.add("blk.3.attn_k.weight",      {H, static_cast<std::uint64_t>(NKV * HD)},
          read_f32(dir / "qsa_layer3_small_weight_k_proj.bin", static_cast<std::size_t>(NKV * HD) * H));
    w.add("blk.3.attn_v.weight",      {H, static_cast<std::uint64_t>(NKV * HD)},
          read_f32(dir / "qsa_layer3_small_weight_v_proj.bin", static_cast<std::size_t>(NKV * HD) * H));
    w.add("blk.3.attn_output.weight", {static_cast<std::uint64_t>(NH * HD), H},
          read_f32(dir / "qsa_layer3_small_weight_o_proj.bin", static_cast<std::size_t>(H) * NH * HD));
    w.add("blk.3.attn_q_norm.weight", {HD}, read_f32(dir / "qsa_layer3_small_weight_q_norm.bin", HD));
    w.add("blk.3.attn_k_norm.weight", {HD}, read_f32(dir / "qsa_layer3_small_weight_k_norm.bin", HD));
    w.add("blk.3.indexer.q_proj.weight", {H, q_rows}, idx_q);
    w.add("blk.3.indexer.k_proj.weight", {H, k_rows}, idx_k);
    w.add("blk.3.indexer.q_norm.weight", {IHD}, read_f32(dir / "qsa_layer3_small_weight_idx_q_norm.bin", IHD));
    w.add("blk.3.indexer.k_norm.weight", {IHD}, read_f32(dir / "qsa_layer3_small_weight_idx_k_norm.bin", IHD));
    const std::vector<std::uint8_t> gguf_buf = w.finish();

    sub0::gguf::Reader r(gguf_buf);
    REQUIRE(r.ok());

    const Applied q     = apply(r, Dest::QsaQProj,     3, H, NH * HD, NH, HD);
    const Applied gate  = apply(r, Dest::QsaGateProj,  3, H, NH * HD, NH, HD);
    const Applied k     = apply(r, Dest::QsaKProj,     3, H, NKV * HD);
    const Applied v     = apply(r, Dest::QsaVProj,     3, H, NKV * HD);
    const Applied o     = apply(r, Dest::QsaOProj,     3, NH * HD, H);
    const Applied qn    = apply(r, Dest::QsaQNorm,     3, 1, HD);
    const Applied kn    = apply(r, Dest::QsaKNorm,     3, 1, HD);
    const Applied idx   = apply(r, Dest::QsaIdxQkProj, 3, H, (IN_H + IKV) * IHD);
    const Applied idx_qn = apply(r, Dest::QsaIdxQNorm, 3, 1, IHD);
    const Applied idx_kn = apply(r, Dest::QsaIdxKNorm, 3, 1, IHD);

    for (const auto* a : {&q, &gate, &k, &v, &o, &qn, &kn, &idx, &idx_qn, &idx_kn})
        CHECK(stats_consistent(a->src_stats, a->dst_stats));

    auto run = [&](const float* q_ptr, const float* gate_ptr) {
        std::vector<float> out(static_cast<std::size_t>(T) * H, 0.f);
        std::vector<float> scratch(sub0::qsa::scratch_floats(dims, T), 0.f);
        sub0::qsa::forward(dims, T, input.data(), idx.data.data(), idx_qn.data.data(), idx_kn.data.data(),
                           q_ptr, gate_ptr, k.data.data(), v.data.data(), qn.data.data(), kn.data.data(),
                           o.data.data(), cos.data(), sin.data(), sub0::qsa::RMS_EPS, out.data(),
                           scratch.data());
        return out;
    };

    const std::vector<float> out = run(q.data.data(), gate.data.data());
    double max_abs = 0.0, sum_expected = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(out[i]) - expected[i]));
        sum_expected += std::fabs(static_cast<double>(expected[i]));
    }
    INFO("max |transplanted - real reference| = " << max_abs);
    REQUIRE(sum_expected > 0.0);
    REQUIRE(max_abs < 1e-4);

    // MUTATION CHECK: the WRONG per-head split -- take the first NH*HD fused rows as the query and the
    // rest as the gate, the naive reading of "chunk the doubled projection in two". It is correct for
    // head 0 and wrong for head 1, so it must produce a genuinely different output. Without this, the
    // match above would also hold for a single-head fixture where the two splits coincide.
    std::vector<float> naive_q(q.data.size(), 0.f), naive_gate(gate.data.size(), 0.f);
    {
        std::vector<float> fused;
        const sub0::gguf::TensorInfo* t = r.find_tensor("blk.3.attn_q.weight");
        REQUIRE(t != nullptr);
        REQUIRE(sub0::gguf::to_f32(*t, r.tensor_bytes(*t), fused));
        transpose_out_in(fused.data(), NH * HD, H, naive_q.data());
        transpose_out_in(fused.data() + static_cast<std::size_t>(NH) * HD * H, NH * HD, H, naive_gate.data());
    }
    const std::vector<float> naive_out = run(naive_q.data(), naive_gate.data());
    double naive_diff = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i)
        naive_diff = std::max(naive_diff, std::fabs(static_cast<double>(naive_out[i]) - out[i]));
    INFO("a down-the-middle q|gate split moves the output by " << naive_diff);
    CHECK(naive_diff > 1e-6);

    // MUTATION CHECK: the indexer's concat ORDER. Joining k-then-q instead of q-then-k keeps the shape
    // and every statistic identical (it is still a permutation of the same values) and computes a
    // different attention -- level 2 provably cannot see this, which is why level 4 exists.
    {
        std::vector<float> swapped(idx.data.size(), 0.f);
        concat_out_transpose(idx_k.data(), static_cast<int>(k_rows), idx_q.data(), static_cast<int>(q_rows),
                             H, swapped.data());
        CHECK(stats_consistent(stats_of(idx.data), stats_of(swapped)));   // ...statistically identical
        std::vector<float> out2(static_cast<std::size_t>(T) * H, 0.f);
        std::vector<float> scratch(sub0::qsa::scratch_floats(dims, T), 0.f);
        sub0::qsa::forward(dims, T, input.data(), swapped.data(), idx_qn.data.data(), idx_kn.data.data(),
                           q.data.data(), gate.data.data(), k.data.data(), v.data.data(), qn.data.data(),
                           kn.data.data(), o.data.data(), cos.data(), sin.data(), sub0::qsa::RMS_EPS,
                           out2.data(), scratch.data());
        double d = 0.0;
        for (std::size_t i = 0; i < out.size(); ++i)
            d = std::max(d, std::fabs(static_cast<double>(out2[i]) - out[i]));
        INFO("a swapped indexer concat order moves the output by " << d);
        CHECK(d > 1e-6);
    }
}

#endif  // SUB0_SOURCE_DIR
