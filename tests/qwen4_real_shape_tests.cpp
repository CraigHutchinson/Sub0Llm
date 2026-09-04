// qwen4_real_shape_tests.cpp -- WP4b's headline gate: this engine's OWN make_param_layout(), evaluated
// at Qwen3.8-Flash-Next's REAL RunConfig axes, must produce the REAL model's tensor count and float
// total. Entirely compile-time: `consteval` evaluation, zero runtime memory, no weights, no download.
//
// WHY THIS EXISTS (docs/WP4_SCOPE.md S0/S2/S6): before the four shape blockers, evaluating this repo's
// own layout at the real axes yielded 74,899 tensors / 124,027,786,776 floats against the real model's
// 125,711,064,960 -- a 1,683,278,184-float (1.34%) gap that made a real weight transplant IMPOSSIBLE,
// because the destination tensors were literally the wrong shape (GdnInProjQkv was [2560, 2984] here
// against [2560, 10240] in the checkpoint). Closing that gap IS this test.
//
// WHY IT IS ITS OWN EXECUTABLE, not a case in sub0_tests: layout.hpp is closed over the BUILD's
// generated sub0_config.hpp, so checking a different config means compiling it a second time against a
// different one. Two definitions of sub0::PARAM_LAYOUT in one binary is an ODR violation, so this TU
// gets its own target whose include path carries tests/qwen4_real_axes/sub0_config.hpp INSTEAD of the
// generated dir -- the same "separate binary because the seam is compiled differently" reasoning
// sub0_eval_seam_tests already uses for SUB0_BUILD_MOCK_DEVICE.
//
// It links no engine library: nothing here runs, so there is nothing to link. If it COMPILES, it passed.

#include <catch2/catch_test_macros.hpp>

#include "sub0/layout.hpp"
#include "sub4_prefix.hpp"

#include <cstddef>

namespace {

// The real model's own parameter total, from docs/QWEN4_MEMORY_ORCHESTRATION.md S2f/S2a (which derives
// it from the real checkpoint's shard headers, not from the published "125B" round number). NOT the
// 124,027,786,776 the engine produced BEFORE the blockers -- closing that gap is the point.
constexpr std::size_t kRealParamFloats = 125'711'064'960ull;

// The tensor count make_param_layout() emits at these axes. Derived here by hand from the real
// architecture rather than pasted from the compiler, so a future change that moves it has to be
// understood rather than merely re-recorded:
//   tok_emb                                                             1
//   no pos_emb (RoPE)                                                   0
//   36 GDN layers   x (4 GR + 9 GDN + 4 GR + MoE)
//   12 QSA layers   x (4 GR + 10 QSA + 4 GR + MoE)
//   MoE per layer = router + 512*(gate,up,down) + shared triple + shared gate = 1 + 1536 + 3 + 1 = 1541
//   NO Ln1/Ln2 anywhere (WP4b blocker D -- the real decoder layer has neither)
//   model-level GR exit collapse (no block_inject)                      3
//   ln_f + lm_head + lm_bias (untied)                                   3
constexpr int kMoePerLayer  = 1 + 3 * NUM_EXPERTS + 3 + 1;                    // 1541
constexpr int kGdnLayer     = 4 + 9  + 4 + kMoePerLayer;                      // 1558
constexpr int kQsaLayer     = 4 + 10 + 4 + kMoePerLayer;                      // 1559
constexpr int kGdnLayers    = 36, kQsaLayers = 12;
constexpr int kExpectedNumParams = 1 + kGdnLayers * kGdnLayer + kQsaLayers * kQsaLayer + 3 + 3;

}  // namespace

// --- The four blockers, each re-asserted at the real axes ----------------------------------------
// Stated as individual claims rather than only through the two totals, so a failure names WHICH axis
// regressed instead of only "the total moved" -- the same reason ARCH_FINGERPRINT is packed rather
// than hashed (layout.hpp's own comment: a fingerprint you cannot decode says only "different").

// Blocker A -- head_dim is independent, and n_heads * head_dim != hidden_size.
static_assert(D_HEAD == 256, "the real model's head_dim");
static_assert(N_HEADS * D_HEAD == 6144, "the real query width");
static_assert(sub0::D_Q == 6144);
static_assert(sub0::D_Q != D_MODEL, "the whole point of blocker A: 6144 != 2560");
static_assert(sub0::D_KV == 2 * 256, "num_key_value_heads * head_dim");

// Blocker B -- GDN's four head axes are its own, not aliases of the attention ones.
static_assert(sub0::GDN_K_HEADS    == 16  && sub0::GDN_K_HEADS    != N_KV_HEADS);
static_assert(sub0::GDN_V_HEADS    == 48  && sub0::GDN_V_HEADS    != N_HEADS);
static_assert(sub0::GDN_K_HEAD_DIM == 128 && sub0::GDN_K_HEAD_DIM != D_HEAD);
static_assert(sub0::GDN_V_HEAD_DIM == 128 && sub0::GDN_V_HEAD_DIM != D_HEAD);
static_assert(sub0::GDN_KEY_DIM   == 2048, "linear_num_key_heads * linear_key_head_dim");
static_assert(sub0::GDN_VALUE_DIM == 6144, "linear_num_value_heads * linear_value_head_dim");
static_assert(sub0::GDN_DIMS.rep() == 3, "the reference's repeat_interleave factor, 48 / 16");
// The single number docs/QWEN4_MEMORY_ORCHESTRATION.md S2h named as the proof the shape was wrong:
// GdnInProjQkv was [2560, 2984] and the real checkpoint's is [2560, 10240].
static_assert(sub0::GDN_CONV_DIM == 10240, "in_proj_qkv's real output width: 2*2048 + 6144");

// Blocker C -- partial rotary, and the indexer bound that the old `>= D_HEAD` form would have rejected.
static_assert(ROTARY_DIM == 64, "partial_rotary_factor 0.25 * head_dim 256");
static_assert(ROTARY_DIM < D_HEAD, "the real model rotates only a prefix");
static_assert(QSA_INDEXER_HEAD_DIM >= ROTARY_DIM && QSA_INDEXER_HEAD_DIM < D_HEAD,
              "128 is >= the rotary prefix 64 but < head_dim 256 -- the old assert demanded >= 256 and "
              "would have refused the real configuration outright");
static_assert(sub0::QSA_DIMS.rotary_dim == 64 && sub0::QSA_DIMS.head_dim == 256);

// Blocker D -- no Ln1/Ln2 anywhere, because the real decoder layer has neither.
consteval int count_kind(sub0::PKind k) {
    int n = 0;
    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) if (p.kind == k) ++n;
    return n;
}
static_assert(count_kind(sub0::PKind::Ln1) == 0);
static_assert(count_kind(sub0::PKind::Ln2) == 0);
static_assert(sub0::USE_GATED_RESIDUAL);

// --- Per-mechanism tensor census, so a total mismatch localizes ----------------------------------
// 36 GDN layers x 9 tensors; 12 QSA layers x 10 (7 attention + 3 indexer); 512 routed experts x 3 per
// layer; 2 GR instances per layer + 1 model-level exit.
static_assert(count_kind(sub0::PKind::GdnInProjQkv) == kGdnLayers);
static_assert(count_kind(sub0::PKind::GdnOutProj)   == kGdnLayers);
static_assert(count_kind(sub0::PKind::QsaQProj)     == kQsaLayers);
static_assert(count_kind(sub0::PKind::QsaIdxQkProj) == kQsaLayers);
static_assert(count_kind(sub0::PKind::MoeGate)      == N_LAYERS * NUM_EXPERTS);
static_assert(count_kind(sub0::PKind::MoeRouter)    == N_LAYERS);
static_assert(count_kind(sub0::PKind::GrHcNorm)     == 2 * N_LAYERS + 1);
static_assert(count_kind(sub0::PKind::GrBlockInject) == 2 * N_LAYERS,
              "the model-level exit instance has use_combine=False -- 96, not 97");
// Wq/Wo do not exist at all here: every full-attention layer is a QSA layer in the real model.
static_assert(count_kind(sub0::PKind::Wq) == 0 && count_kind(sub0::PKind::Wo) == 0);

// --- An INDEPENDENT float census -------------------------------------------------------------------
// Re-derived from the real architecture by hand, tensor by tensor, WITHOUT calling make_param_layout()
// -- so a bug that happened to total correctly by coincidence, or a pair of same-shaped tensors swapped
// for each other, still has to survive a second, differently-written derivation. This project's own
// [[independent-reimplementation-catches-identity-swap-bugs]] lesson, applied to the shape rather than
// to the math: WP-GDN Stage 3 found a real dt_bias/A_log swap in already-"verified" code exactly this way.
//
// The four per-mechanism subtotals below are ALSO each individually checkable against
// docs/QWEN4_MEMORY_ORCHESTRATION.md S2h's own gap table, which lists the real model's GR = 640,624,640,
// MoE = 121,094,922,240, QSA = 617,358,336 and GDN = 2,086,510,464.
namespace census {
constexpr long long WIDE = 4LL * 2560;                     // hc_count * hidden_size = 10240
constexpr long long gr_with_inject = WIDE + 2 * WIDE * 320 + WIDE * 4;      //  6,604,800
constexpr long long gr_top         = WIDE + 2 * WIDE * 320;                 //  6,563,840
constexpr long long gdn_layer =
      2560LL * 10240        // in_proj_qkv [hidden, 2*key_dim + value_dim]
    + 2560LL * 6144         // in_proj_z   [hidden, value_dim]
    + 2560LL * 48 * 2       // in_proj_b + in_proj_a [hidden, num_v_heads]
    + 10240LL * 4           // conv1d      [conv_dim, kernel]
    + 48 + 48               // A_log + dt_bias [num_v_heads]
    + 128                   // RMSNormGated gamma [head_v_dim]
    + 6144LL * 2560;        // out_proj    [value_dim, hidden]
constexpr long long qsa_layer =
      2560LL * 6144 * 2     // q_proj + gate_proj (the real double-width q_proj, stored as two)
    + 2560LL * 512 * 2      // k_proj + v_proj  [hidden, kv_width]
    + 6144LL * 2560         // o_proj  [q_width, hidden]  -- NOT square
    + 256 + 256             // q_norm + k_norm [head_dim]
    + 2560LL * 640          // index_qk_proj [hidden, (idx_n + idx_kv) * idx_head_dim]
    + 128 + 128;            // indexer q/k layernorms [idx_head_dim]
constexpr long long moe_layer =
      2560LL * 512                                   // router [hidden, num_experts]
    + 512LL * (2560LL * 640 * 2 + 640LL * 2560)      // gate + up + down, per routed expert
    + (2560LL * 640 * 2 + 640LL * 2560)              // the shared expert's own triple
    + 2560;                                          // shared_expert_gate [hidden, 1]
constexpr long long total =
      248320LL * 2560                       // tok_emb (no pos_emb: RoPE)
    + 48 * (2 * gr_with_inject + moe_layer) // two GR instances + the MoE block, every layer
    + 36 * gdn_layer + 12 * qsa_layer       // the 3-GDN-then-1-QSA repeating unit, 48 layers
    + gr_top                                // the model-level exit collapse (use_combine=False)
    + 2560 + 2560LL * 248320 + 248320;      // ln_f + lm_head + lm_bias (untied)

// The per-mechanism subtotals, checked against S2h's own table before the grand total is trusted.
static_assert(48 * 2 * gr_with_inject + gr_top == 640'624'640LL, "GR, per S2h's table");
static_assert(48LL * moe_layer          == 121'094'922'240LL, "MoE, per S2h's table");
static_assert(36LL * gdn_layer          ==   2'086'510'464LL, "GDN's REAL total, per S2h's table -- "
              "the engine produced 751,723,560 before blocker B");
static_assert(12LL * qsa_layer          ==     617'358'336LL, "QSA's REAL total, per S2h's table -- "
              "the engine produced 268,621,296 before blockers A and C");
}  // namespace census
static_assert(static_cast<long long>(sub0::PARAM_FLOATS) == census::total,
              "make_param_layout()'s total disagrees with the hand-written per-tensor census -- one of "
              "the two is wrong, and a coincidental total cannot hide it");

// --- memplan's own lock-step, at the real axes ----------------------------------------------------
// memplan.hpp re-derives D_KV / QKV_STRIDE / QK_PRE_STRIDE independently of layout.hpp (it must: the
// configurator calls it and cannot include layout.hpp), and its own comment requires the two to stay in
// lock-step. Blocker A moved all three off D_MODEL onto D_Q, so this is exactly where a drift would hide.
// Checked HERE, at the real axes, because that is the configuration where d_q != d_model and a stale
// `d_model + 2*d_kv` would still look right at every neutral shape.
namespace {
constexpr sub0::memplan::Dims kRealDims = sub0::current_build_dims();
}
static_assert(sub0::memplan::head_dim(kRealDims)      == D_HEAD);
static_assert(sub0::memplan::d_q(kRealDims)           == sub0::D_Q);
static_assert(sub0::memplan::d_kv(kRealDims)          == sub0::D_KV);
static_assert(sub0::memplan::qkv_stride(kRealDims)    == sub0::QKV_STRIDE);
static_assert(sub0::memplan::qk_pre_stride(kRealDims) == sub0::QK_PRE_STRIDE);
// ...and the full param_floats() identity holds for the DENSE architecture memplan actually models --
// it has no term for GDN/GR/MoE/QSA (see layout.hpp's own guarded static_assert on the same identity),
// so at these axes it is checked against the dense-equivalent build, not against PARAM_FLOATS. What
// that pins is the piece blocker A touched: Wq/Wo are d_q-sided on BOTH sides of the derivation.
static_assert(sub0::memplan::param_floats(kRealDims) ==
                  248320ULL * 2560                                   // tok_emb
                + 48ULL * (2ULL * 2560                               // ln1 + ln2
                           + 2ULL * 2560 * 6144                      // Wq [C,D_Q] + Wo [D_Q,C]
                           + 2ULL * 2560 * 512                       // Wk + Wv [C,D_KV]
                           + 2ULL * 256                              // q_norm + k_norm [D_HEAD]
                           + 3ULL * 2560 * 640)                      // gated FFN
                + 2560ULL + 2560ULL * 248320 + 248320,               // ln_f + lm_head + lm_bias
              "memplan::param_floats() has drifted from layout.hpp's own arithmetic at the real axes");

// --- The two headline numbers ---------------------------------------------------------------------
static_assert(sub0::NUM_PARAMS == kExpectedNumParams,
              "make_param_layout()'s tensor count at the real axes disagrees with the hand-derived "
              "per-layer census above");
static_assert(sub0::PARAM_FLOATS == kRealParamFloats,
              "make_param_layout() at Qwen3.8-Flash-Next's real axes must total the REAL model's own "
              "parameter count -- docs/WP4_SCOPE.md S0's 1,683,278,184-float shape gap, closed");

// --- WP4c: the 4-layer sub-stack really is layers 0-3 of this model ------------------------------
// The offline transplant tool (tools/sub0llm-transplant.cpp) compiles the SAME axes header with
// N_LAYERS = 4 and fills a 4-layer PARAM_LAYOUT. That is only a faithful sub-stack of the real model
// if its tensors are, entry for entry, the real model's first four layers -- and the two layouts
// cannot be compared in one TU (see sub4_prefix.hpp's own header comment for why). This half of the
// claim lives here; the tool asserts the matching half against the same two literals.
//
// PARAM_LAYOUT[PREFIX_TENSORS] is the FIRST tensor after layers 0-3: layer 4's leading GR tensor here,
// the model-level GR exit's leading tensor in the tool's build. Its float offset is the total the four
// layers occupy, so equality means both builds lay those four layers out identically.
static_assert(sub0::PARAM_LAYOUT[sub0::qwen4_sub4::PREFIX_TENSORS].off == sub0::qwen4_sub4::PREFIX_FLOATS,
              "the real 48-layer layout's first four layers must occupy exactly the float span the "
              "4-layer transplant target claims -- if these disagree, the transplanted artifact is not "
              "a sub-stack of this model");
// ...and the boundary really is a layer boundary, not an accident of arithmetic: entry PREFIX_TENSORS
// is a GR hc_norm (the first tensor of layer 4's attn_hyper_connection) and the entry before it is the
// last of layer 3's MoE block.
static_assert(sub0::PARAM_LAYOUT[sub0::qwen4_sub4::PREFIX_TENSORS].kind == sub0::PKind::GrHcNorm);
static_assert(sub0::PARAM_LAYOUT[sub0::qwen4_sub4::PREFIX_TENSORS - 1].kind == sub0::PKind::MoeSharedGateProj);
// Layer 3 is the QSA layer -- the reason a 4-layer sub-stack exercises every mechanism at all.
static_assert(sub0::MIXER_SCHEDULE[0] == sub0::LayerMixer::Gdn);
static_assert(sub0::MIXER_SCHEDULE[1] == sub0::LayerMixer::Gdn);
static_assert(sub0::MIXER_SCHEDULE[2] == sub0::LayerMixer::Gdn);
static_assert(sub0::MIXER_SCHEDULE[3] == sub0::LayerMixer::Qsa);

// A single Catch2 case so a run REPORTS the numbers rather than only the build succeeding. Everything
// above already failed the compile if it was wrong; this exists to print the census.
TEST_CASE("make_param_layout() at Qwen3.8-Flash-Next's real axes matches the real model's shape",
          "[qwen4][layout][realshape]") {
    WARN("real-axes shape: NUM_PARAMS = " << sub0::NUM_PARAMS
         << ", PARAM_FLOATS = " << sub0::PARAM_FLOATS
         << " (real model " << kRealParamFloats << ")");
    REQUIRE(sub0::NUM_PARAMS == kExpectedNumParams);
    REQUIRE(sub0::PARAM_FLOATS == kRealParamFloats);
}
