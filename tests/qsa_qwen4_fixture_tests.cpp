// qsa_qwen4_fixture_tests.cpp -- Qwen Sparse Attention Stage 1's PRIMARY correctness gate
// (docs/QSA.md S7): sub0::qsa::forward and the row helpers it is built from (include/sub0/qsa_math.hpp)
// run against the REAL extracted Qwen/Qwen3.8-Flash-Next weights + activations at
// tests/fixtures/qwen4_preview/qsa_layer3_small_*.
//
// Engine-free, mirroring gdn_qwen4_fixture_tests.cpp's / gated_residual_qwen4_fixture_tests.cpp's /
// moe_qwen4_fixture_tests.cpp's own pattern: calls sub0::qsa:: directly with the fixture's own raw dims
// rather than routing through whatever QSA_INDEXER_* this binary happens to be built for.
//
// Layer 3, not layer 0, because the real config.json's layer_types array is `full_attention` iff
// `l % 4 == 3` -- layer 3 is the FIRST layer in the real 48-layer stack that has an indexer at all.

#include <catch2/catch_test_macros.hpp>

#include "sub0/qsa_math.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef SUB0_SOURCE_DIR

namespace {
namespace fs = std::filesystem;

std::vector<float> read_f32(const fs::path& p, std::size_t expect_n) {
    std::ifstream f(p, std::ios::binary);
    std::vector<float> out(expect_n);
    if (!f) return {};
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(expect_n * sizeof(float)));
    if (static_cast<std::size_t>(f.gcount()) != expect_n * sizeof(float)) return {};
    return out;
}

// The fixture's raw weight files are PyTorch nn.Linear convention, [out_features, in_features] row-major
// -- the TRANSPOSE of this project's own [rows=in, cols=out] convention qsa_math.hpp expects (see that
// file's header comment). Re-derived explicitly per AGENTS.md S5, not assumed.
std::vector<float> transpose(const std::vector<float>& w, int out_f, int in_f) {
    std::vector<float> t(static_cast<std::size_t>(out_f) * in_f);
    for (int o = 0; o < out_f; ++o)
        for (int i = 0; i < in_f; ++i)
            t[static_cast<std::size_t>(i) * out_f + o] = w[static_cast<std::size_t>(o) * in_f + i];
    return t;
}

// The fixture's own config_small (qsa_layer3_small_manifest.json). num_attention_heads is 2 (not 1) so
// the real PER-HEAD query|gate chunk order is distinguishable from a flat-row split; indexer_n_heads is 2
// so the relu-then-sum-over-heads score is distinguishable from a single head; num_key_value_heads is 1
// so GQA repeat_kv is exercised; budget/compress_ratio are set so block_topk (2) is genuinely smaller
// than the block count at most positions (docs/QSA.md S7).
constexpr int kH = 16, kNH = 2, kHD = 8, kNKV = 1;
constexpr int kIN = 2, kIKV = 1, kIHD = 8;
constexpr int kBudget = 8, kRatio = 4, kRot = 2, kT = 24;
const sub0::qsa::Dims kDims{kH, kNH, kHD, kNKV, kIN, kIKV, kIHD, kBudget, kRatio, kRot};

struct Fixture {
    bool ok = false;
    std::vector<float> input, cos, sin, expected, expected_mutant, mask, mask_mutant;
    std::vector<float> q_w, gate_w, k_w, v_w, o_w, q_norm, k_norm;
    std::vector<float> idx_qk, idx_qk_mutant, idx_q_norm, idx_k_norm;
};

Fixture load(const fs::path& dir) {
    Fixture f;
    const auto sz = [](std::size_t a, std::size_t b) { return a * b; };
    f.input           = read_f32(dir / "qsa_layer3_small_input.bin", sz(kT, kH));
    if (f.input.empty()) return f;
    f.cos             = read_f32(dir / "qsa_layer3_small_cos.bin", sz(kT, kRot));
    f.sin             = read_f32(dir / "qsa_layer3_small_sin.bin", sz(kT, kRot));
    f.expected        = read_f32(dir / "qsa_layer3_small_output.bin", sz(kT, kH));
    f.expected_mutant = read_f32(dir / "qsa_layer3_small_output_mutant.bin", sz(kT, kH));
    f.mask            = read_f32(dir / "qsa_layer3_small_mask.bin", sz(kT, kT));
    f.mask_mutant     = read_f32(dir / "qsa_layer3_small_mask_mutant.bin", sz(kT, kT));
    // [out, in] -> [in, out] for every projection.
    f.q_w    = transpose(read_f32(dir / "qsa_layer3_small_weight_q_proj.bin", sz(kNH * kHD, kH)), kNH * kHD, kH);
    f.gate_w = transpose(read_f32(dir / "qsa_layer3_small_weight_gate_proj.bin", sz(kNH * kHD, kH)), kNH * kHD, kH);
    f.k_w    = transpose(read_f32(dir / "qsa_layer3_small_weight_k_proj.bin", sz(kNKV * kHD, kH)), kNKV * kHD, kH);
    f.v_w    = transpose(read_f32(dir / "qsa_layer3_small_weight_v_proj.bin", sz(kNKV * kHD, kH)), kNKV * kHD, kH);
    f.o_w    = transpose(read_f32(dir / "qsa_layer3_small_weight_o_proj.bin", sz(kH, kNH * kHD)), kH, kNH * kHD);
    f.q_norm = read_f32(dir / "qsa_layer3_small_weight_q_norm.bin", kHD);
    f.k_norm = read_f32(dir / "qsa_layer3_small_weight_k_norm.bin", kHD);
    f.idx_qk = transpose(read_f32(dir / "qsa_layer3_small_weight_idx_qk_proj.bin", sz((kIN + kIKV) * kIHD, kH)),
                          (kIN + kIKV) * kIHD, kH);
    f.idx_qk_mutant = transpose(
        read_f32(dir / "qsa_layer3_small_weight_idx_qk_proj_mutant.bin", sz((kIN + kIKV) * kIHD, kH)),
        (kIN + kIKV) * kIHD, kH);
    f.idx_q_norm = read_f32(dir / "qsa_layer3_small_weight_idx_q_norm.bin", kIHD);
    f.idx_k_norm = read_f32(dir / "qsa_layer3_small_weight_idx_k_norm.bin", kIHD);
    f.ok = !f.cos.empty() && !f.expected.empty() && !f.q_w.empty() && !f.idx_qk.empty() &&
           !f.o_w.empty() && !f.mask.empty() && !f.expected_mutant.empty();
    return f;
}

std::vector<float> run(const Fixture& f, const std::vector<float>& idx_qk) {
    std::vector<float> out(static_cast<std::size_t>(kT) * kH);
    std::vector<float> scratch(sub0::qsa::scratch_floats(kDims, kT));
    sub0::qsa::forward(kDims, kT, f.input.data(), idx_qk.data(), f.idx_q_norm.data(), f.idx_k_norm.data(),
                       f.q_w.data(), f.gate_w.data(), f.k_w.data(), f.v_w.data(),
                       f.q_norm.data(), f.k_norm.data(), f.o_w.data(),
                       f.cos.data(), f.sin.data(), sub0::qsa::RMS_EPS, out.data(), scratch.data());
    return out;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

}  // namespace

TEST_CASE("QSA CPU forward matches the real Qwen4-preview reference at layer 3's small attention fixture",
          "[qsa][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    if (!fs::exists(dir / "qsa_layer3_small_input.bin")) {
        WARN("Qwen4 QSA fixtures not present at " << dir.string() << " -- skipping (optional)");
        return;
    }
    const Fixture f = load(dir);
    REQUIRE(f.ok);

    // GUARD THE GUARD: the fixture is only a real test of QSA if the indexer actually DROPS causally
    // visible tokens. At the real budget/ratio on a 24-token sequence every block would be selected and
    // this whole file would pass identically for an implementation that never ran the indexer at all
    // (docs/QSA.md S7). Assert the reference's own mask is genuinely sparse BEFORE trusting any match.
    int causal = 0, dropped = 0;
    for (int q = 0; q < kT; ++q)
        for (int j = 0; j <= q; ++j) {
            ++causal;
            if (f.mask[static_cast<std::size_t>(q) * kT + j] == 0.f) ++dropped;
        }
    REQUIRE(causal == kT * (kT + 1) / 2);
    REQUIRE(dropped > 0);

    const std::vector<float> out = run(f, f.idx_qk);
    double sum_expected = 0.0;
    for (float v : f.expected) sum_expected += std::fabs(static_cast<double>(v));
    const double worst = max_abs_diff(out, f.expected);
    WARN("QSA fixture: max|out - expected| = " << worst << " over " << out.size()
         << " values, sum|expected| = " << sum_expected << "; indexer dropped " << dropped
         << " of " << causal << " causally-visible entries");
    REQUIRE(worst < 1e-5);
}

// PRESENCE / MUTATION check (docs/QSA.md S7): a second REAL reference, produced by re-running the real,
// unmodified module with ONLY the indexer's key-half projection replaced, so it selects a genuinely
// different block set. Two claims, both needed: our port must MATCH that second real reference under the
// same perturbation (so it really is reading the indexer's weights), and the two real references must
// genuinely DIFFER (so the fixture can distinguish "consulted the indexer" from "attended densely").
TEST_CASE("QSA output depends on WHICH blocks the indexer selected", "[qsa][qwen4_fixture]") {
    const fs::path dir = fs::path(SUB0_SOURCE_DIR) / "tests" / "fixtures" / "qwen4_preview";
    if (!fs::exists(dir / "qsa_layer3_small_input.bin")) {
        WARN("Qwen4 QSA fixtures not present -- skipping (optional)");
        return;
    }
    const Fixture f = load(dir);
    REQUIRE(f.ok);

    int mask_changed = 0;
    for (std::size_t i = 0; i < f.mask.size(); ++i) if (f.mask[i] != f.mask_mutant[i]) ++mask_changed;
    REQUIRE(mask_changed > 0);            // the mutant really does select different blocks

    const std::vector<float> out_real   = run(f, f.idx_qk);
    const std::vector<float> out_mutant = run(f, f.idx_qk_mutant);
    const double mutant_match = max_abs_diff(out_mutant, f.expected_mutant);
    const double real_vs_mutant = max_abs_diff(out_real, out_mutant);
    WARN("QSA mutation check: max|out_mutant - expected_mutant| = " << mutant_match
         << "; max|out(real) - out(mutant)| = " << real_vs_mutant
         << "; mask entries changed = " << mask_changed);
    REQUIRE(mutant_match < 1e-5);
    REQUIRE(real_vs_mutant > 1e-4);       // a dense-attention mutant would score 0 here
}

#endif  // SUB0_SOURCE_DIR

// --- fixture-free property checks ---------------------------------------------------------------
// These need no fixture and run everywhere, pinning the three rules a plausible-looking but wrong
// implementation would most easily get backwards (docs/QSA.md S7's checks 2 and 3).

TEST_CASE("QSA block selection keeps the highest-scoring blocks AND always keeps the tail", "[qsa]") {
    // 2 indexer heads, head_dim 4, ratio 3, budget 3 => block_topk 1: exactly one block survives the
    // top-k, and the incomplete tail must survive REGARDLESS of any score.
    const sub0::qsa::Dims d{8, 1, 4, 1, 2, 1, 4, 3, 3, 4};
    REQUIRE(d.block_topk() == 1);
    constexpr int kv = 8;                       // 2 complete blocks (0..2, 3..5) + a 2-token tail (6,7)
    std::vector<float> raw_keys(static_cast<std::size_t>(kv) * 4, 0.f);
    // Block 1's keys point along +e0; block 0's along -e0. Any query along +e0 must pick block 1.
    for (int t = 0; t < 3; ++t) raw_keys[static_cast<std::size_t>(t) * 4 + 0] = -1.f;
    for (int t = 3; t < 6; ++t) raw_keys[static_cast<std::size_t>(t) * 4 + 0] = +1.f;
    for (int t = 6; t < 8; ++t) raw_keys[static_cast<std::size_t>(t) * 4 + 1] = +1.f;
    std::vector<float> q(static_cast<std::size_t>(d.idx_q_width()), 0.f);
    for (int h = 0; h < d.idx_n_heads; ++h) q[static_cast<std::size_t>(h) * 4 + 0] = 1.f;
    const std::vector<float> k_ln(4, 0.f);      // (1 + 0) == identity gain
    std::vector<float> cos(static_cast<std::size_t>(kv) * 4, 1.f), sin(static_cast<std::size_t>(kv) * 4, 0.f);
    std::vector<float> mask(kv), scr(sub0::qsa::select_scratch_floats(d, kv));

    sub0::qsa::indexer_select_row(d, q.data(), raw_keys.data(), kv, k_ln.data(), cos.data(), sin.data(),
                                   sub0::qsa::RMS_EPS, mask.data(), scr.data());
    for (int t = 0; t < 3; ++t) REQUIRE(mask[static_cast<std::size_t>(t)] == 0.f);   // block 0 dropped
    for (int t = 3; t < 6; ++t) REQUIRE(mask[static_cast<std::size_t>(t)] == 1.f);   // block 1 selected
    for (int t = 6; t < 8; ++t) REQUIRE(mask[static_cast<std::size_t>(t)] == 1.f);   // tail ALWAYS visible

    // Flip the query and the selected block must flip with it -- proving the selection tracks the score
    // rather than a fixed position (e.g. "always keep the most recent block").
    for (int h = 0; h < d.idx_n_heads; ++h) q[static_cast<std::size_t>(h) * 4 + 0] = -1.f;
    sub0::qsa::indexer_select_row(d, q.data(), raw_keys.data(), kv, k_ln.data(), cos.data(), sin.data(),
                                   sub0::qsa::RMS_EPS, mask.data(), scr.data());
    for (int t = 0; t < 3; ++t) REQUIRE(mask[static_cast<std::size_t>(t)] == 1.f);
    for (int t = 3; t < 6; ++t) REQUIRE(mask[static_cast<std::size_t>(t)] == 0.f);
    for (int t = 6; t < 8; ++t) REQUIRE(mask[static_cast<std::size_t>(t)] == 1.f);

    // Fewer complete blocks than block_topk => every block visible (min(block_topk, nb), docs/QSA.md S1a).
    std::vector<float> mask2(3);
    sub0::qsa::indexer_select_row(d, q.data(), raw_keys.data(), 3, k_ln.data(), cos.data(), sin.data(),
                                   sub0::qsa::RMS_EPS, mask2.data(), scr.data());
    for (int t = 0; t < 3; ++t) REQUIRE(mask2[static_cast<std::size_t>(t)] == 1.f);
}

TEST_CASE("QSA's batched prefill and its incremental row composition agree bitwise", "[qsa]") {
    // The engine runs qsa::forward() from Model::forward and the SAME row helpers, one position at a
    // time, from Model::forward_one -- so a difference between the two composition orders is a real
    // engine bug, not a tolerance question. This is the math-core form of that check, and it is how this
    // stage's own rope_apply_row out-of-bounds write was found (docs/QSA.md S10).
    const sub0::qsa::Dims d{16, 2, 8, 2, 2, 1, 8, 8, 4, 8};
    constexpr int T = 20;
    auto fill = [](std::size_t n, unsigned seed) {
        std::vector<float> v(n);
        unsigned s = seed;
        for (auto& x : v) { s = s * 1664525u + 1013904223u; x = static_cast<float>((s >> 8) % 2003) / 1000.f - 1.f; }
        return v;
    };
    const auto hidden = fill(static_cast<std::size_t>(T) * d.hidden_size, 11);
    const auto qk = fill(static_cast<std::size_t>(d.hidden_size) * d.idx_qk_out(), 12);
    const auto qln = fill(static_cast<std::size_t>(d.idx_head_dim), 13);
    const auto kln = fill(static_cast<std::size_t>(d.idx_head_dim), 14);
    const auto qw = fill(static_cast<std::size_t>(d.hidden_size) * d.q_width(), 15);
    const auto gw = fill(static_cast<std::size_t>(d.hidden_size) * d.q_width(), 16);
    const auto kw = fill(static_cast<std::size_t>(d.hidden_size) * d.kv_width(), 17);
    const auto vw = fill(static_cast<std::size_t>(d.hidden_size) * d.kv_width(), 18);
    const auto qn = fill(static_cast<std::size_t>(d.head_dim), 19);
    const auto kn = fill(static_cast<std::size_t>(d.head_dim), 20);
    const auto ow = fill(static_cast<std::size_t>(d.q_width()) * d.hidden_size, 21);
    std::vector<float> cos(static_cast<std::size_t>(T) * d.rotary_dim), sin(cos.size());
    for (int p = 0; p < T; ++p)
        for (int m = 0; m < d.rotary_dim / 2; ++m) {
            const float ang = static_cast<float>(p) * std::pow(10000.f, -2.f * m / d.rotary_dim);
            const std::size_t b = static_cast<std::size_t>(p) * d.rotary_dim;
            cos[b + m] = cos[b + m + d.rotary_dim / 2] = std::cos(ang);
            sin[b + m] = sin[b + m + d.rotary_dim / 2] = std::sin(ang);
        }

    std::vector<float> out_batch(static_cast<std::size_t>(T) * d.hidden_size);
    std::vector<float> scr(sub0::qsa::scratch_floats(d, T));
    sub0::qsa::forward(d, T, hidden.data(), qk.data(), qln.data(), kln.data(), qw.data(), gw.data(),
                       kw.data(), vw.data(), qn.data(), kn.data(), ow.data(), cos.data(), sin.data(),
                       sub0::qsa::RMS_EPS, out_batch.data(), scr.data());

    std::vector<float> kc(static_cast<std::size_t>(T) * d.kv_width()), vc(kc.size());
    std::vector<float> rk(static_cast<std::size_t>(T) * d.idx_head_dim);
    std::vector<float> iq(static_cast<std::size_t>(d.idx_q_width())), q(static_cast<std::size_t>(d.q_width()));
    std::vector<float> g(q.size()), mask(T);
    std::vector<float> ss(sub0::qsa::select_scratch_floats(d, T)), as(sub0::qsa::attn_scratch_floats(d, T));
    std::vector<float> out_dec(out_batch.size());
    int total_dropped = 0;
    for (int p = 0; p < T; ++p) {
        const float* x = hidden.data() + static_cast<std::size_t>(p) * d.hidden_size;
        const float* cp = cos.data() + static_cast<std::size_t>(p) * d.rotary_dim;
        const float* sp = sin.data() + static_cast<std::size_t>(p) * d.rotary_dim;
        sub0::qsa::indexer_project_row(d, x, qk.data(), qln.data(), cp, sp, sub0::qsa::RMS_EPS,
                                        iq.data(), rk.data() + static_cast<std::size_t>(p) * d.idx_head_dim);
        sub0::qsa::attn_project_row(d, x, qw.data(), gw.data(), kw.data(), vw.data(), qn.data(), kn.data(),
                                     cp, sp, sub0::qsa::RMS_EPS, q.data(), g.data(),
                                     kc.data() + static_cast<std::size_t>(p) * d.kv_width(),
                                     vc.data() + static_cast<std::size_t>(p) * d.kv_width());
        const int visible = sub0::qsa::indexer_select_row(d, iq.data(), rk.data(), p + 1, kln.data(),
                                                           cos.data(), sin.data(), sub0::qsa::RMS_EPS,
                                                           mask.data(), ss.data());
        total_dropped += (p + 1) - visible;
        sub0::qsa::attn_row(d, q.data(), g.data(), kc.data(), vc.data(), p + 1, mask.data(), ow.data(),
                             out_dec.data() + static_cast<std::size_t>(p) * d.hidden_size, as.data());
    }
    // Guard the guard again: if nothing was dropped, this check would also pass for a dense mutant.
    REQUIRE(total_dropped > 0);
    for (std::size_t i = 0; i < out_dec.size(); ++i) REQUIRE(out_dec[i] == out_batch[i]);
}
