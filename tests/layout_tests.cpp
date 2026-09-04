// layout_tests.cpp — the constexpr parameter layout table and the baked compute-backend
// facts. These guard the shared "shape of the weights" truth (include/sub0/layout.hpp)
// that both the CPU and the future GPU backend address, and the configure-time GPU facts
// the engine compiles against.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"     // pulls sub0_config.hpp: HAS_CUDA, COMPUTE_MODE, CUDA_ARCH, ...
#include "sub0/layout.hpp"   // PARAM_LAYOUT, NUM_PARAMS, PARAM_FLOATS

#include <cstddef>

TEST_CASE("param layout is contiguous and totals PARAM_FLOATS", "[layout]") {
    // Gated FFN drops the per-layer count from 10 (..., W1, b1, W2, b2) to 9
    // (..., Wg, W1, W2 -- no FFN bias, see layout.hpp). QK-norm adds 2 more per-layer slots
    // (q_norm, k_norm). Tied embeddings drops the tail from 3 (ln_f, lm_head, lm_bias) to 1
    // (ln_f alone -- the head reuses tok_emb, no separate slot).
    // tok_emb is always present; pos_emb only under absolute positions (RoPE omits the table
    // entirely -- see layout.hpp's HAS_POS_EMB, which is what decouples SEQ_LEN from the checkpoint).
    // N-gram embeddings add NGRAM_NUM_EMBEDDERS tables + one concat_proj tensor, 0 at the neutral
    // (NGRAM_MAX_N == 0) setting -- see layout.hpp's own NGRAM_EMBED section.
    //
    // Per-layer mixer slot count now depends on GDN_SCHEDULE (Stage 1): an attention layer keeps the
    // Wq/Wk/Wv/Wo (4) [+QNorm/KNorm (2)] shape; a GDN layer is a FIXED 9 slots regardless of QK-norm
    // (GDN never uses it -- layout.hpp's PKind comment). At GDN_FULL_ATTN_STRIDE == 0, gdn_layers == 0
    // and this collapses to exactly the pre-Stage-1 formula below, independently re-derived (not by
    // calling count_num_params() itself, which would make this circular).
    // QSA (Stage 1, docs/QSA.md S3b): a QSA layer is a FULL-ATTENTION layer whose mixer is replaced by a
    // FIXED 10 slots (QsaQProj/GateProj/KProj/VProj/OProj/QNorm/KNorm + the indexer's QkProj/QNorm/KNorm),
    // regardless of USE_QK_NORM (the real Qwen4ExpTextAttention always has its own q_norm/k_norm). At
    // USE_QSA == false there are no QSA layers and this collapses to the pre-QSA formula exactly.
    constexpr int kQsaLayers  = sub0::USE_QSA ? (N_LAYERS - sub0::GDN_SCHEDULE.gdn_layers) : 0;
    constexpr int kAttnLayers = N_LAYERS - sub0::GDN_SCHEDULE.gdn_layers - kQsaLayers;
    constexpr int kAttnMixer  = 4 + (USE_QK_NORM ? 2 : 0);
    constexpr int kGdnMixer   = 9;
    constexpr int kQsaMixer   = 10;
    // Mixture of Experts (Stage 1, docs/MOE.md S3b): REPLACES the FFN's own kFfnSlots with MoeRouter (1)
    // + NUM_EXPERTS*(MoeGate,MoeUp,MoeDown) + the shared expert's own SwiGLU triple (3) + its gate
    // projection (1), for EVERY layer (no per-layer schedule -- see USE_MOE's own comment). 0 at the
    // neutral (NUM_EXPERTS < 2) setting, where kFfnSlots is used unchanged.
    constexpr int kFfnSlots   = sub0::USE_MOE ? (1 + 3 * NUM_EXPERTS + 3 + 1) : (USE_GATED_FFN ? 3 : 4);
    // Gated Residual (Stage 1, docs/GATED_RESIDUAL.md S3b): 4 slots per instance (GrHcNorm/MixDown/
    // MixUp/BlockInject), TWO full instances per layer (attn-wrapping, mlp-wrapping) plus one top-level
    // instance WITHOUT BlockInject (3 slots), appended once regardless of N_LAYERS. Zero at HC_COUNT==0.
    constexpr int kGrPerLayer = sub0::USE_GATED_RESIDUAL ? 2 * 4 : 0;
    constexpr int kGrTop      = sub0::USE_GATED_RESIDUAL ? 3 : 0;
    STATIC_REQUIRE(sub0::NUM_PARAMS == 1 + (sub0::HAS_POS_EMB ? 1 : 0)
                                        + N_LAYERS * (2 + kFfnSlots + kGrPerLayer)
                                        + kAttnLayers * kAttnMixer
                                        + sub0::GDN_SCHEDULE.gdn_layers * kGdnMixer
                                        + kQsaLayers * kQsaMixer
                                        + (USE_TIED_EMBEDDINGS ? 1 : 3)
                                        + (sub0::NGRAM_EMBED ? sub0::NGRAM_NUM_EMBEDDERS + 1 : 0)
                                        + kGrTop);
    REQUIRE(sub0::PARAM_LAYOUT.size() == static_cast<std::size_t>(sub0::NUM_PARAMS));

    std::size_t off = 0;
    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
        REQUIRE(p.off == off);          // tensors are back-to-back: no gaps, no overlaps
        REQUIRE(p.rows >= 1);
        REQUIRE(p.cols >= 1);
        off += p.n();
    }
    REQUIRE(off == sub0::PARAM_FLOATS);                       // table totals the blob size
    REQUIRE(sub0::PARAM_FLOATS == sub0::trainable_floats());  // and the engine agrees
}

TEST_CASE("param layout roles, decay and ternary flags are consistent", "[layout]") {
    using sub0::PKind;
    REQUIRE(sub0::PARAM_LAYOUT.front().kind == PKind::TokEmb);   // first tensor
    // Last tensor: n-gram's concat_proj when the feature is on (appended at the very end -- see
    // layout.hpp's make_param_layout()); else ln_f when tied (no separate head slot -- see
    // op_tied_head), else lm_bias.
    const PKind expected_last = sub0::NGRAM_EMBED ? PKind::NgramProj
                              : (USE_TIED_EMBEDDINGS ? PKind::LnF : PKind::LmBias);
    REQUIRE(sub0::PARAM_LAYOUT.back().kind == expected_last);

    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
        // NgramProj is a GEMM weight like LmHead: full precision (not ternary-eligible), AdamW-only
        // (not Muon-eligible -- see is_muon_kind). NgramEmb is an embedding-row lookup like TokEmb:
        // no decay, not ternary -- it already satisfies the `!is_matrix` branch below with no special
        // case needed.
        //
        // Gated DeltaNet (Stage 1): GdnInProjQkv/Z/B/A and GdnOutProj are real 2D GEMM projections
        // (decay=true, same as Wq/Wk/Wv/Wo), but -- like LmHead/NgramProj -- deliberately kept FULL
        // PRECISION (ternary=false): ternary quantization's interaction with GDN's recurrence is
        // genuinely unexplored (AGENTS.md S8, don't add unvalidated surface), not merely an oversight.
        // GdnConv/GdnALog/GdnDtBias/GdnNorm are bias/gain-shaped (decay=false, not GEMM weights),
        // already satisfying `!is_matrix` with no special case, same as NgramEmb.
        //
        // Gated Residual (Stage 1): GrMixDown/GrMixUp/GrBlockInject are real 2D GEMM projections
        // (decay=true), also kept FULL PRECISION (ternary=false) for the same "genuinely unexplored,
        // don't add unvalidated surface" reason as GDN's own projections. GrHcNorm is gain-shaped
        // (decay=false), already satisfying `!is_matrix`.
        // Mixture of Experts (Stage 1, docs/MOE.md S3b): MoeGate/MoeUp/MoeDown/MoeSharedGate/
        // MoeSharedUp/MoeSharedDown are ordinary GEMM weights -- ternary-eligible, same treatment as
        // Wg/W1/W2 (they route through Muon too, see is_muon_kind). MoeRouter/MoeSharedGateProj are
        // real 2D GEMM projections (decay=true) kept FULL PRECISION (ternary=false), for the same
        // "routing precision matters" reasoning LmHead stays full precision for its own different reason.
        const bool is_moe_expert_matrix =
            p.kind == PKind::MoeGate || p.kind == PKind::MoeUp || p.kind == PKind::MoeDown ||
            p.kind == PKind::MoeSharedGate || p.kind == PKind::MoeSharedUp || p.kind == PKind::MoeSharedDown;
        const bool is_moe_router_matrix =
            p.kind == PKind::MoeRouter || p.kind == PKind::MoeSharedGateProj;
        // QSA (Stage 1, docs/QSA.md S3b): the five gated-attention projections are ordinary GEMM weights
        // -- ternary-eligible, Muon-eligible, exactly Wq/Wk/Wv/Wo's own treatment. QsaIdxQkProj is a real
        // 2D GEMM projection (decay=true) kept FULL PRECISION (ternary=false): it makes a SELECTION
        // decision, the same "routing precision matters" reasoning MoeRouter stays full precision for.
        // QsaQNorm/QsaKNorm/QsaIdxQNorm/QsaIdxKNorm are gain-shaped (decay=false), already satisfying
        // `!is_matrix` with no special case, same as NgramEmb/GrHcNorm.
        const bool is_qsa_attn_matrix =
            p.kind == PKind::QsaQProj || p.kind == PKind::QsaGateProj || p.kind == PKind::QsaKProj ||
            p.kind == PKind::QsaVProj || p.kind == PKind::QsaOProj;
        const bool is_qsa_index_matrix = p.kind == PKind::QsaIdxQkProj;
        const bool is_matrix =
            p.kind == PKind::Wq || p.kind == PKind::Wk || p.kind == PKind::Wv ||
            p.kind == PKind::Wo || p.kind == PKind::W1 || p.kind == PKind::W2 ||
            p.kind == PKind::Wg || p.kind == PKind::LmHead || p.kind == PKind::NgramProj ||
            p.kind == PKind::GdnInProjQkv || p.kind == PKind::GdnInProjZ ||
            p.kind == PKind::GdnInProjB || p.kind == PKind::GdnInProjA || p.kind == PKind::GdnOutProj ||
            p.kind == PKind::GrMixDown || p.kind == PKind::GrMixUp || p.kind == PKind::GrBlockInject ||
            is_moe_expert_matrix || is_moe_router_matrix ||
            is_qsa_attn_matrix || is_qsa_index_matrix;
        // AdamW weight decay applies only to the GEMM weight matrices.
        REQUIRE(p.decay == is_matrix);
        // Ternary quantization covers the block matrices but NOT the full-precision head/ngram-proj/
        // any GDN/GR/MoE-router projection (see the comment above for why each stays full precision).
        const bool is_gdn_matrix =
            p.kind == PKind::GdnInProjQkv || p.kind == PKind::GdnInProjZ ||
            p.kind == PKind::GdnInProjB || p.kind == PKind::GdnInProjA || p.kind == PKind::GdnOutProj;
        const bool is_gr_matrix =
            p.kind == PKind::GrMixDown || p.kind == PKind::GrMixUp || p.kind == PKind::GrBlockInject;
        const bool ternary_eligible = is_matrix && p.kind != PKind::LmHead && p.kind != PKind::NgramProj
                                      && !is_gdn_matrix && !is_gr_matrix && !is_moe_router_matrix
                                      && !is_qsa_index_matrix;
        REQUIRE(p.ternary == ternary_eligible);
    }
}

TEST_CASE("DECAY_RANGES exactly reproduces PARAM_LAYOUT's per-tensor decay flag, offset by offset", "[layout]") {
    // Exhaustive: for every flat float offset in the blob, "is it covered by some DECAY_RANGES range"
    // must agree with whichever PARAM_LAYOUT tensor owns that offset. This is the direct correctness
    // proof for backend_cuda.cu's adam_step_kernel lookup replacing the old per-parameter mask buffer --
    // a wrong range here would silently mis-apply (or skip) weight decay on GPU-trained models.
    auto covered = [](std::size_t off) {
        for (const auto& r : sub0::DECAY_RANGES) if (off >= r.start && off < r.end) return true;
        return false;
    };
    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
        for (std::size_t i = 0; i < p.n(); ++i) {
            REQUIRE(covered(p.off + i) == p.decay);
        }
    }
    // Ranges themselves are sorted, non-overlapping and non-empty (sanity on the construction, not
    // just the coverage property above).
    for (std::size_t i = 0; i < sub0::DECAY_RANGES.size(); ++i) {
        REQUIRE(sub0::DECAY_RANGES[i].end > sub0::DECAY_RANGES[i].start);
        if (i > 0) REQUIRE(sub0::DECAY_RANGES[i].start >= sub0::DECAY_RANGES[i - 1].end);
    }
}

TEST_CASE("baked compute-backend facts are self-consistent", "[config]") {
    // BitNet/ternary is CPU-only: a ternary build must never select a GPU/HYBRID backend
    // (the same invariant the CMake guard and the backend_cuda.cu static_assert enforce).
    STATIC_REQUIRE(!USE_TERNARY || COMPUTE_MODE == ComputeBackend::Cpu);
    // Tied embeddings: GPU support landed (launch_tied_head/launch_tied_head_bwd in backend_cuda.cu,
    // `if constexpr`-gated), so there is no CPU-only invariant left to assert for this axis -- it is
    // valid with any COMPUTE_MODE now.
    // QK-norm: GPU support landed the same way (qknorm_act_kernel/qknorm_backward_act_kernel in
    // backend_cuda.cu, `if constexpr`-gated) -- no CPU-only invariant left for this axis either.
    // SwiGLU-gated FFN: GPU support landed the same way too (swiglu_kernel/swiglu_act_kernel/
    // swiglu_backward_act_kernel in backend_cuda.cu, `if constexpr`-gated) -- no CPU-only invariant
    // left for this axis either.
    // A detected CUDA host has its arch + VRAM facts populated; an absent one zeroes them.
    STATIC_REQUIRE(!HAS_CUDA || (CUDA_ARCH > 0 && GPU_VRAM_MB > 0));
    STATIC_REQUIRE(HAS_CUDA || CUDA_ARCH == 0);
    // Shared/overflow memory is a non-negative MB figure (0 where there is no WDDM shared mem).
    STATIC_REQUIRE(GPU_SHARED_MEM_MB >= 0);
}

TEST_CASE("positional encoding facts are consistent", "[config]") {
    // RoPE (the default) injects RELATIVE position inside attention via a rotation of Q/K, and has
    // NO position table at all; Absolute uses a learned pos_emb table added to the token embedding.
    // The table used to keep its layout slot under RoPE "for format stability", allocated and zeroed
    // and never read -- that cost SEQ_LEN*D_MODEL floats in every checkpoint AND, more importantly,
    // made PARAM_FLOATS depend on SEQ_LEN, which pinned a model to the exact window it trained at.
    STATIC_REQUIRE((POS_ENCODING == PosEncoding::Rope || POS_ENCODING == PosEncoding::Absolute));
    STATIC_REQUIRE(ROPE_THETA > 0.0f);
    STATIC_REQUIRE(sub0::HAS_POS_EMB == (POS_ENCODING == PosEncoding::Absolute));
    // Written as an implication rather than an if constexpr branch: outside a template, if constexpr
    // does NOT discard the untaken branch from semantic analysis, so a STATIC_REQUIRE inside it still
    // fires. `||` short-circuits in a constant expression, so PARAM_LAYOUT[1] is only inspected when
    // the table really does carry pos_emb there.
    STATIC_REQUIRE(!sub0::HAS_POS_EMB || sub0::PARAM_LAYOUT[1].kind == sub0::PKind::PosEmb);
    // The decoupling this buys: with no pos_emb, NO parameter tensor's shape depends on SEQ_LEN, so a
    // checkpoint trained at one window is loadable by a binary built with a larger one. Absence of the
    // slot IS that property, so absence is what gets asserted.
    if constexpr (!sub0::HAS_POS_EMB)
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) REQUIRE(p.kind != sub0::PKind::PosEmb);
}

// --- ARCH_FINGERPRINT: adding an axis must not invalidate existing checkpoints ------------------
// depth_attn_stride was added to the packed fingerprint (2026-07-29) in the byte middle_layers can
// never reach. The claim that made that safe -- "with the stride at its default 0 every fingerprint
// this function has ever produced is reproduced BIT-IDENTICALLY" -- is exactly the kind of claim that
// is easy to assert in a comment and expensive to be wrong about: every .ckpt and .bin on disk is
// rejected the moment it stops holding. So it is pinned here against a literal transcription of the
// PREVIOUS packing rather than against the function itself.
namespace {
constexpr std::uint64_t arch_fingerprint_v1(int middle_layers, int repeats, float rope_theta) {
    return (static_cast<std::uint64_t>(static_cast<unsigned>(middle_layers) & 0xffffu) << 48)
         | (static_cast<std::uint64_t>(static_cast<unsigned>(repeats)       & 0xffffu) << 32)
         |  static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(rope_theta));
}
}  // namespace

TEST_CASE("arch fingerprint stays bit-identical when depth attention is off", "[config]") {
    // The three real shapes on disk right now: the LoopSplit arms (deep16 / loop10x6 / shallow10).
    for (auto [mid, rep] : {std::pair{0, 1}, std::pair{6, 2}, std::pair{0, 1}}) {
        for (float theta : {10000.0f, 500000.0f}) {
            REQUIRE(sub0::arch_fingerprint(mid, rep, theta, /*depth_attn_stride=*/0)
                    == arch_fingerprint_v1(mid, rep, theta));
        }
    }
    // ...and a non-zero stride MUST differ, or a depth-attention model would load into a plain build
    // and silently compute something else -- there is no shape difference to catch it (depth
    // attention adds no parameters at all, so PARAM_FLOATS is identical).
    REQUIRE(sub0::arch_fingerprint(0, 1, 10000.0f, 4) != sub0::arch_fingerprint(0, 1, 10000.0f, 0));
    REQUIRE(sub0::arch_fingerprint(0, 1, 10000.0f, 4) != sub0::arch_fingerprint(0, 1, 10000.0f, 2));

    // Round-trips through the decoder, so the diagnostic can name the mismatched value.
    const sub0::ArchAxes got = sub0::arch_axes_of(sub0::arch_fingerprint(6, 2, 10000.0f, 4));
    REQUIRE(got.middle_layers == 6);
    REQUIRE(got.repeats == 2);
    REQUIRE(got.depth_attn_stride == 4);
    REQUIRE(got.rope_theta == 10000.0f);

    // This build's own fingerprint agrees with its own axes (the invariant the loaders compare).
    REQUIRE(sub0::ARCH_FINGERPRINT
            == sub0::arch_fingerprint(LOOP_MIDDLE_LAYERS, LOOP_REPEATS, ROPE_THETA, DEPTH_ATTN_STRIDE));
}

// sub0::DEPTH_SCHEDULE is the single source of truth for depth-cache slot bookkeeping across four sites
// (CPU forward, CPU decode, CUDA forward, CUDA backward). An off-by-one in it is a silent device
// out-of-bounds write and a silently wrong cross-execution gradient -- neither of which a loss curve
// would reveal. depth_schedule_for<EXECS>(stride) is parameterised precisely so the shapes that MATTER
// can be asserted here rather than only in whichever build happens to be configured.
TEST_CASE("depth-attention schedule is correct at the shapes that matter", "[layout][depth]") {
    // ARM D's actual configuration: 16 executions (L10, middle 6 x2), stride 4. Verified here because
    // this is the shape a multi-hour training run uses, and no build in the test matrix is configured
    // for it. Appends land at 0/4/8/12; the middle block is executions 2-7 (pass 1) and 8-13 (pass 2).
    constexpr auto armd = sub0::depth_schedule_for<16>(4);
    static_assert(armd.slots == 4);
    static_assert(armd.own[0] == 0 && armd.own[4] == 1 && armd.own[8] == 2 && armd.own[12] == 3);
    static_assert(armd.own[1] == -1 && armd.own[7] == -1 && armd.own[15] == -1);
    static_assert(armd.live[0] == 0 && armd.live[4] == 1 && armd.live[8] == 2 && armd.live[12] == 3);
    static_assert(armd.live[13] == 4 && armd.live[15] == 4);
    // The mechanism arm D exists to test: pass 2 (executions 8-13) must read slots contributed EARLIER.
    // live[8] == 2 means execution 8 mixes over the head's slot 0 and pass 1's slot 1. A stride that
    // happened to append nothing before execution 8 would have made the whole experiment null.
    static_assert(armd.live[8] >= 2, "arm D's pass 2 must see cross-pass depth entries");
    REQUIRE(armd.live[8] == 2);

    // Stride 1 (every execution participates) and stride 2, the shapes the parity gates ran at.
    constexpr auto s1 = sub0::depth_schedule_for<16>(1);
    static_assert(s1.slots == 16);
    static_assert(s1.live[0] == 0 && s1.live[15] == 15);
    constexpr auto s2 = sub0::depth_schedule_for<10>(2);
    static_assert(s2.slots == 5);
    static_assert(s2.own[0] == 0 && s2.own[2] == 1 && s2.own[8] == 4 && s2.own[9] == -1);
    static_assert(s2.live[9] == 5);
    REQUIRE(s2.slots == 5);

    // Stride 0 is OFF: no slot is ever claimed, so nothing allocates and every kernel is bypassed.
    constexpr auto off = sub0::depth_schedule_for<16>(0);
    static_assert(off.slots == 0);
    static_assert(off.own[0] == -1 && off.live[15] == 0);

    // Two invariants the kernels depend on, checked across a sweep rather than at one point.
    //  1. own[e] == live[e] whenever e participates -- an execution appends AFTER mixing, so it never
    //     reads its own slot. The backward's `own_slot` sits exactly one PAST the readable range, which
    //     is why the cache views hand out every slot instead of the first live[e].
    //  2. live[e] <= slots -- what makes the kernels' DEPTH_CACHE_MAX+1 shared arrays exactly sufficient.
    auto check = [](auto sched) {
        for (std::size_t e = 0; e < sched.live.size(); ++e) {
            if (sched.own[e] >= 0) REQUIRE(sched.own[e] == sched.live[e]);
            REQUIRE(sched.live[e] <= sched.slots);
        }
    };
    check(sub0::depth_schedule_for<16>(1));
    check(sub0::depth_schedule_for<16>(3));   // does NOT divide 16 -- the ragged case
    check(sub0::depth_schedule_for<16>(4));
    check(sub0::depth_schedule_for<10>(2));
    check(sub0::depth_schedule_for<7>(5));    // stride larger than half the executions
    check(sub0::depth_schedule_for<1>(1));    // single execution: mixes over itself alone (identity)

    // ...and this build's own schedule obeys them too.
    check(sub0::DEPTH_SCHEDULE);
    REQUIRE(sub0::DEPTH_SCHEDULE.slots == sub0::DEPTH_CACHE_MAX);
}

// --- Gated DeltaNet -- DESIGN + Stage 0 skeleton only (docs/GATED_DELTANET.md) ------------------
// No forward/backward op exists yet (layout.hpp's static_assert refuses any nonzero
// GDN_FULL_ATTN_STRIDE at compile time), so there is nothing here to gradient-check or run at
// production dims. What Stage 0 DOES add -- the per-layer classification schedule and the second
// fingerprint word -- is pure compile-time bookkeeping, and per AGENTS.md S7 it is pinned at TWO
// shapes, not one: an 8-layer count divisible by the stride and an 11-layer (ODD) count that is not,
// mirroring the exact lesson that caught LoopSplit's head/tail static_assert only at odd N_LAYERS.
TEST_CASE("Gated DeltaNet layer schedule is correct at two shapes, one of them odd/ragged", "[layout][gdn]") {
    // Shape 1: 8 layers, stride 4 -- the Qwen3-Next/Qwen3.5 hybrid ratio (3x GDN -> 1x full attention),
    // divides evenly into two groups. Full attention lands at layers 3 and 7.
    constexpr auto s8 = sub0::gdn_schedule_for<8>(4);
    static_assert(s8.gdn_layers == 6);
    static_assert(!s8.full_attn[0] && !s8.full_attn[1] && !s8.full_attn[2] && s8.full_attn[3]);
    static_assert(!s8.full_attn[4] && !s8.full_attn[5] && !s8.full_attn[6] && s8.full_attn[7]);
    REQUIRE(s8.gdn_layers == 6);

    // Shape 2: 11 layers (ODD, does NOT divide by stride 3) -- the ragged case. Full attention lands at
    // layers 2, 5, 8; the trailing partial group (layers 9, 10) never reaches stride-1, so both stay GDN.
    constexpr auto s11 = sub0::gdn_schedule_for<11>(3);
    static_assert(s11.gdn_layers == 8);
    static_assert(s11.full_attn[2] && s11.full_attn[5] && s11.full_attn[8]);
    static_assert(!s11.full_attn[9] && !s11.full_attn[10]);
    REQUIRE(s11.gdn_layers == 8);

    // Stride 0 is OFF at any layer count: every layer stays full (ordinary softmax) attention -- today's
    // only architecture, and the only one either shape above can actually be built with right now.
    constexpr auto off8  = sub0::gdn_schedule_for<8>(0);
    constexpr auto off11 = sub0::gdn_schedule_for<11>(0);
    static_assert(off8.gdn_layers == 0 && off11.gdn_layers == 0);
    for (bool b : off8.full_attn)  REQUIRE(b);
    for (bool b : off11.full_attn) REQUIRE(b);

    // ...and this build's own schedule agrees with gdn_schedule_for<N_LAYERS>(GDN_FULL_ATTN_STRIDE) --
    // at the neutral setting (the only one every build before Stage 1 could take) that means every layer
    // is full attention, exactly as before; a Stage-1 GDN build (this test binary may now be one) has a
    // real, nonzero gdn_layers count instead, which this same check still verifies is self-consistent.
    static_assert(GDN_FULL_ATTN_STRIDE > 0 || sub0::GDN_SCHEDULE.gdn_layers == 0);
    constexpr auto this_build = sub0::gdn_schedule_for<N_LAYERS>(GDN_FULL_ATTN_STRIDE);
    static_assert(this_build.gdn_layers == sub0::GDN_SCHEDULE.gdn_layers);
    for (int l = 0; l < N_LAYERS; ++l)
        REQUIRE(sub0::GDN_SCHEDULE.full_attn[static_cast<std::size_t>(l)] == this_build.full_attn[static_cast<std::size_t>(l)]);
}

// GDN's own four head axes (WP4b blocker B, docs/WP4_SCOPE.md S2). These used to be ALIASED onto the
// ordinary attention axes, which is why a real weight transplant was impossible: the real model wants
// 16 / 48 / 128 / 128 against attention's 2 / 24 / 256. What must be pinned is (a) the alias default
// still reproduces the pre-blocker-B geometry EXACTLY -- the "zero effect when neutral" contract -- and
// (b) the derived widths track GDN's own axes rather than D_KV/D_MODEL once they diverge.
TEST_CASE("Gated DeltaNet head axes alias the attention axes at their neutral setting", "[layout][gdn]") {
    // (a) The alias identity, for THIS build. GDN_KEY_HEADS.. are emitted RESOLVED by the configurator,
    // so "neutral" is spelled as "the resolved value equals the attention axis it used to alias".
    if constexpr (GDN_KEY_HEADS == N_KV_HEADS && GDN_VALUE_HEADS == N_HEADS &&
                  GDN_KEY_HEAD_DIM == D_HEAD && GDN_VALUE_HEAD_DIM == D_HEAD) {
        STATIC_REQUIRE(sub0::GDN_KEY_DIM   == sub0::D_KV);
        STATIC_REQUIRE(sub0::GDN_VALUE_DIM == N_HEADS * D_HEAD);
        // The exact expression layout.hpp carried before this axis existed.
        STATIC_REQUIRE(sub0::GDN_CONV_DIM == 2 * sub0::D_KV + N_HEADS * D_HEAD);
    }
    // (b) GDN_DIMS is the single source of truth the math core consumes; the derived widths must agree
    // with gdn::Dims' own accessors, so a future edit cannot move one without the other.
    STATIC_REQUIRE(sub0::GDN_DIMS.key_dim()   == sub0::GDN_KEY_DIM);
    STATIC_REQUIRE(sub0::GDN_DIMS.value_dim() == sub0::GDN_VALUE_DIM);
    STATIC_REQUIRE(sub0::GDN_DIMS.conv_dim()  == sub0::GDN_CONV_DIM);
    STATIC_REQUIRE(sub0::GDN_DIMS.num_k_heads == sub0::GDN_K_HEADS);
    STATIC_REQUIRE(sub0::GDN_DIMS.num_v_heads == sub0::GDN_V_HEADS);
    STATIC_REQUIRE(sub0::GDN_DIMS.head_k_dim  == sub0::GDN_K_HEAD_DIM);
    STATIC_REQUIRE(sub0::GDN_DIMS.head_v_dim  == sub0::GDN_V_HEAD_DIM);
    STATIC_REQUIRE(sub0::GDN_DIMS.rep() == sub0::GDN_V_HEADS / sub0::GDN_K_HEADS);

    // (c) The nine GDN tensors in PARAM_LAYOUT must be shaped from GDN's OWN axes -- checked against the
    // real model's ratios via a standalone gdn::Dims rather than this build's (which is GDN-off by
    // default), so the claim is about the geometry, not about whatever shape this binary compiled.
    constexpr sub0::gdn::Dims real{2560, 16, 48, 128, 128, 4};
    STATIC_REQUIRE(real.key_dim()   == 2048);
    STATIC_REQUIRE(real.value_dim() == 6144);
    STATIC_REQUIRE(real.conv_dim()  == 2 * 2048 + 6144);
    STATIC_REQUIRE(real.rep() == 3);   // the reference's repeat_interleave factor, 48 / 16
}

// Partial rotary (WP4b blocker C, docs/WP4_SCOPE.md S2). The engine's rotary prefix used to BE D_HEAD;
// --rotary-dim makes it its own axis so the real model's partial_rotary_factor (0.25 x head_dim 256 =
// 64 of 256) is expressible. What must be pinned: the neutral setting still means "rotate the whole
// head", the prefix is a legal one, and it is the quantity QSA_DIMS actually carries.
TEST_CASE("rotary prefix defaults to the full head width and stays a legal prefix", "[layout][rope]") {
    STATIC_REQUIRE(ROTARY_DIM >= 2);
    STATIC_REQUIRE(ROTARY_DIM <= D_HEAD);
    // Even only under QSA (its half-split pairing needs it). An ODD prefix is legal for the interleaved
    // op_rope convention and is a shape this project actually builds -- d132 H4 gives D_HEAD 33.
    STATIC_REQUIRE(!sub0::USE_QSA || ROTARY_DIM % 2 == 0);
    STATIC_REQUIRE(sub0::ROTARY_HALF == ROTARY_DIM / 2);
    // QSA_DIMS.rotary_dim is the ONE value qsa_math.hpp's rope_apply_row indexes; it must be the axis,
    // not a second derivation of it. (Its own precondition, rotary_dim <= the width being rotated, is
    // what layout.hpp's QSA_INDEXER_HEAD_DIM >= ROTARY_DIM static_assert now enforces -- the old
    // `>= D_HEAD` form would have REJECTED the real model, whose indexer heads are 128 < head_dim 256.)
    STATIC_REQUIRE(sub0::QSA_DIMS.rotary_dim == ROTARY_DIM);
    STATIC_REQUIRE(sub0::QSA_DIMS_BUF.rotary_dim == ROTARY_DIM);
    // The real model's own relationship, checked on a standalone Dims rather than this build's (which is
    // full-width by default): the prefix is narrower than the attention head AND than the indexer head.
    constexpr sub0::qsa::Dims real{2560, 24, 256, 2, 4, 1, 128, 2048, 4, 64};
    STATIC_REQUIRE(real.rotary_dim < real.head_dim);
    STATIC_REQUIRE(real.rotary_dim <= real.idx_head_dim);
}

// ARCH_FINGERPRINT2 must reproduce 0 at the only value GDN_FULL_ATTN_STRIDE can currently take, and
// must differ once a nonzero stride is EVER passed to the free function -- the exact same "neutral is
// bit-identical, non-neutral is distinguishable" property ARCH_FINGERPRINT's own DEPTH_ATTN_STRIDE test
// pins above. Calling arch_fingerprint2() directly (rather than requiring an actual GDN build) is what
// makes this testable at all while the static_assert in layout.hpp forbids ever compiling one.
TEST_CASE("arch fingerprint2 stays bit-identical when Gated DeltaNet and MoE are both off", "[layout][gdn]") {
    REQUIRE(sub0::arch_fingerprint2(0) == 0);              // default experts_per_tok=0 too
    REQUIRE(sub0::arch_fingerprint2(0, 0) == 0);
    // This build's own word is 0 only when BOTH axes are off -- docs/MOE.md S3c added EXPERTS_PER_TOK
    // into byte 1 of this SAME word (this stage's own real build under test may have MoE genuinely ON,
    // e.g. the two-scale/parity verification config, so this is an implication, not an unconditional 0).
    // ... and, since docs/QSA.md S3a, only when QSA's own budget/compress-ratio (bytes 2-3, 4) are off too.
    if constexpr (EXPERTS_PER_TOK == 0 && QSA_INDEXER_BUDGET == 0 && QSA_INDEXER_COMPRESS_RATIO == 0) {
        REQUIRE(sub0::ARCH_FINGERPRINT2 == 0);
        REQUIRE(sub0::ARCH_FINGERPRINT2 == sub0::ARCH_FINGERPRINT2_LEGACY);
    } else {
        REQUIRE(sub0::ARCH_FINGERPRINT2 == sub0::arch_fingerprint2(GDN_FULL_ATTN_STRIDE, EXPERTS_PER_TOK,
                                                                    QSA_INDEXER_BUDGET,
                                                                    QSA_INDEXER_COMPRESS_RATIO,
                                                                    sub0::USE_GATED_DELTANET
                                                                        ? sub0::GDN_K_HEADS : 0,
                                                                    ROTARY_DIM == D_HEAD ? 0 : ROTARY_DIM));
        REQUIRE(sub0::ARCH_FINGERPRINT2 != sub0::ARCH_FINGERPRINT2_LEGACY);
    }

    // A nonzero stride MUST differ, or a (future) GDN checkpoint would load into a plain build and
    // silently compute something else -- there is no shape difference to catch it in general (a GDN
    // layer's parameter count is a different SHAPE from a softmax layer's, but nothing here proves that
    // holds at every possible dims/stride combination the way GQA's monotonic D_KV substitution does --
    // see docs/GATED_DELTANET.md's checkpoint-design section for the worked argument).
    REQUIRE(sub0::arch_fingerprint2(4) != sub0::arch_fingerprint2(0));
    REQUIRE(sub0::arch_fingerprint2(4) != sub0::arch_fingerprint2(3));

    // Round-trips through the decoder, so the diagnostic can name the mismatched value.
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4)).gdn_full_attn_stride == 4);

    // Mixture of Experts (docs/MOE.md S3c): EXPERTS_PER_TOK occupies byte 1, independently of GDN's own
    // byte 0 -- a nonzero experts-per-tok MUST differ from the all-zero word for the same "would silently
    // compute something else" reason, and the two bytes must not collide with each other.
    REQUIRE(sub0::arch_fingerprint2(0, 10) != sub0::arch_fingerprint2(0, 0));
    REQUIRE(sub0::arch_fingerprint2(0, 10) != sub0::arch_fingerprint2(4, 0));      // byte 0 vs byte 1 alone
    REQUIRE(sub0::arch_fingerprint2(4, 10) != sub0::arch_fingerprint2(4, 0));      // same GDN stride, differ by top-k
    REQUIRE(sub0::arch_fingerprint2(4, 10) != sub0::arch_fingerprint2(0, 10));     // same top-k, differ by GDN stride
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10)).gdn_full_attn_stride == 4);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10)).experts_per_tok == 10);

    // QSA (docs/QSA.md S3a): indexer_budget occupies [31:16] (SIXTEEN bits -- the real value 2048 does
    // not fit 8) and indexer_compress_ratio [39:32], both independent of GDN's byte 0 and MoE's byte 1.
    // The same "would silently attend to a different token set" reason applies: neither changes any
    // tensor shape, so nothing else can catch a cross-load.
    REQUIRE(sub0::arch_fingerprint2(0, 0, 2048, 4) != sub0::arch_fingerprint2(0, 0, 0, 0));
    REQUIRE(sub0::arch_fingerprint2(0, 0, 2048, 4) != sub0::arch_fingerprint2(0, 0, 2048, 8));  // ratio alone
    REQUIRE(sub0::arch_fingerprint2(0, 0, 2048, 4) != sub0::arch_fingerprint2(0, 0, 256, 4));   // budget alone
    REQUIRE(sub0::arch_fingerprint2(0, 0, 2048, 4) != sub0::arch_fingerprint2(4, 10, 0, 0));    // no collision
    REQUIRE(sub0::arch_fingerprint2(4, 10, 2048, 4) != sub0::arch_fingerprint2(4, 10, 0, 0));
    // A budget above 255 must survive intact -- the direct check that 16 bits, not 8, were allocated.
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4)).qsa_indexer_budget == 2048);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4)).qsa_indexer_compress_ratio == 4);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4)).gdn_full_attn_stride == 4);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4)).experts_per_tok == 10);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(0, 0, 0xffff, 0xff)).qsa_indexer_budget == 0xffff);

    // GDN's own KEY HEAD COUNT (WP4b blocker B) occupies byte 5, [47:40]. It is the one of GDN's four
    // new head axes that PARAM_FLOATS cannot see: every GDN tensor shape depends on it only through
    // key_dim = k_heads * k_head_dim, so 2x64 and 4x32 are byte-identical blobs computing a different
    // recurrence (rep() = v_heads / k_heads). Independent of every other field, and zero-by-default so
    // the neutral word is bit-identical to every value this function has ever produced.
    REQUIRE(sub0::arch_fingerprint2(0, 0, 0, 0, 16) != sub0::arch_fingerprint2(0, 0, 0, 0, 0));
    REQUIRE(sub0::arch_fingerprint2(0, 0, 0, 0, 16) != sub0::arch_fingerprint2(0, 0, 0, 0, 2));
    REQUIRE(sub0::arch_fingerprint2(0, 0, 0, 0, 16) != sub0::arch_fingerprint2(16, 0, 0, 0, 0));  // vs byte 0
    REQUIRE(sub0::arch_fingerprint2(4, 10, 2048, 4, 16) != sub0::arch_fingerprint2(4, 10, 2048, 4, 0));
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4, 16)).gdn_key_heads == 16);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4, 16)).qsa_indexer_budget == 2048);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(0, 0, 0, 0, 0xff)).gdn_key_heads == 0xff);

    // The PARTIAL ROTARY prefix (WP4b blocker C) occupies the top 16 bits, [63:48]. Sixteen and not
    // eight because it is bounded by D_HEAD, which is 256 in the real model. It changes NO tensor shape
    // while rotating fewer channels of every head, so nothing but this word can catch a cross-load --
    // the same reason ROPE_THETA sits in ARCH_FINGERPRINT. Canonicalized to 0 at full-width rotary.
    REQUIRE(sub0::arch_fingerprint2(0, 0, 0, 0, 0, 64) != sub0::arch_fingerprint2(0, 0, 0, 0, 0, 0));
    REQUIRE(sub0::arch_fingerprint2(0, 0, 0, 0, 0, 64) != sub0::arch_fingerprint2(0, 0, 0, 0, 0, 32));
    REQUIRE(sub0::arch_fingerprint2(0, 0, 0, 0, 0, 64) != sub0::arch_fingerprint2(0, 0, 64, 0, 0, 0));
    REQUIRE(sub0::arch_fingerprint2(4, 10, 2048, 4, 16, 64) != sub0::arch_fingerprint2(4, 10, 2048, 4, 16, 0));
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4, 16, 64)).partial_rotary_dim == 64);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4, 10, 2048, 4, 16, 64)).gdn_key_heads == 16);
    // A prefix above 255 must survive intact -- the direct check that 16 bits, not 8, were allocated
    // (the real model's head_dim is 256, so an 8-bit field would silently alias 256 to 0).
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(0, 0, 0, 0, 0, 256)).partial_rotary_dim == 256);
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(0, 0, 0, 0, 0, 0xffff)).partial_rotary_dim == 0xffff);
    // Every field packed at once still round-trips -- the word is now FULL (64 bits assigned), so the
    // next shape-neutral axis needs a third additive word rather than a repack.
    REQUIRE(sub0::arch_fingerprint2(0xff, 0xff, 0xffff, 0xff, 0xff, 0xffff) ==
            0xffff'ff'ff'ffff'ff'ffull);

    // MODEL_ARCH_ID folds this word in (see layout.hpp make_model_arch_id): two builds differing only in
    // a hypothetical GDN stride or MoE top-k must not collide, matching how it already handles
    // DEPTH_ATTN_STRIDE. Verified indirectly here since MODEL_ARCH_ID's inputs are compile-time constants
    // of THIS build -- the direct claim is just that ARCH_FINGERPRINT2 (which the id already mixes in) is
    // a real, distinguishing quantity, established above.
}

// --- Gated Residual -- DESIGN + Stage 0 skeleton only (docs/GATED_RESIDUAL.md) -------------------
// No forward op exists yet (layout.hpp's static_asserts refuse any nonzero HC_COUNT/HC_LOWRANK at
// compile time), so there is nothing here to run at production dims. What Stage 0 DOES add -- the
// USE_GATED_RESIDUAL gate, HC_WIDE, and the gr_param_delta() closed form -- is pure compile-time
// bookkeeping, pinned at TWO shapes per AGENTS.md S7, mirroring the GDN Stage 0 test just above.
TEST_CASE("Gated Residual PARAM_FLOATS delta is zero when off and strictly positive when on, at two "
          "shapes", "[layout][gr]") {
    // Shape 1: 8 layers, d_model 96 (the fast-iteration toy scale this thread's own precedent prefers).
    static_assert(sub0::gr_param_delta(8, 96, 0, 0) == 0);
    static_assert(sub0::gr_param_delta(8, 96, 4, 16) > 0);
    REQUIRE(sub0::gr_param_delta(8, 96, 0, 0) == 0);
    REQUIRE(sub0::gr_param_delta(8, 96, 4, 16) > 0);
    // Hand-computed at hc_count=4, hc_lowrank=16, d_model=96: wide=384.
    //   per_instance_with_inject = 384 + 2*384*16 + 384*4 = 384 + 12288 + 1536 = 14208
    //   top_instance             = 384 + 12288 = 12672
    //   total = 8 * 2 * 14208 + 12672 = 227328 + 12672 = 240000
    static_assert(sub0::gr_param_delta(8, 96, 4, 16) == 240000);
    REQUIRE(sub0::gr_param_delta(8, 96, 4, 16) == 240000);

    // Shape 2: 11 layers (ODD/ragged -- GR has no per-layer schedule to be ragged about, but this
    // confirms the closed form has no hidden even-layer-count assumption either), a different d_model.
    static_assert(sub0::gr_param_delta(11, 132, 0, 0) == 0);
    static_assert(sub0::gr_param_delta(11, 132, 4, 16) > 0);
    REQUIRE(sub0::gr_param_delta(11, 132, 0, 0) == 0);
    REQUIRE(sub0::gr_param_delta(11, 132, 4, 16) > 0);

    // hc_count == 1 is disallowed by layout.hpp's own static_assert once Stage 1 relaxes the Stage 0
    // hard clamp -- gr_param_delta() itself treats anything < 2 as "off" (0 delta), consistent with
    // USE_GATED_RESIDUAL's own (HC_COUNT >= 2) gate, so the two can never disagree about what "off" means.
    static_assert(sub0::gr_param_delta(8, 96, 1, 5) == 0);

    // ...and this build's own state agrees: written as an IMPLICATION (HC_COUNT == 0 -> ...), not an
    // unconditional claim, the same shape as the GDN Stage 0 test just above
    // (`GDN_FULL_ATTN_STRIDE > 0 || sub0::GDN_SCHEDULE.gdn_layers == 0`) -- this default-build test suite
    // always compiles at HC_COUNT == 0 (the only value the standard configure path ever produces), but an
    // ad-hoc GR-ON build (this thread's own two-scale/parity verification, docs/GATED_RESIDUAL.md S8)
    // must not fail to COMPILE this file just because it deliberately set HC_COUNT >= 2.
    static_assert(HC_COUNT != 0 || !sub0::USE_GATED_RESIDUAL);
    static_assert(HC_COUNT != 0 || sub0::HC_WIDE == D_MODEL);
    static_assert(HC_COUNT != 0 || sub0::gr_param_delta(N_LAYERS, D_MODEL, HC_COUNT, HC_LOWRANK) == 0);
    if constexpr (HC_COUNT == 0) {
        REQUIRE_FALSE(sub0::USE_GATED_RESIDUAL);
        REQUIRE(sub0::HC_WIDE == D_MODEL);
        REQUIRE(sub0::gr_param_delta(N_LAYERS, D_MODEL, HC_COUNT, HC_LOWRANK) == 0);
    } else {
        REQUIRE(sub0::USE_GATED_RESIDUAL);
        REQUIRE(sub0::HC_WIDE == HC_COUNT * D_MODEL);
        REQUIRE(sub0::gr_param_delta(N_LAYERS, D_MODEL, HC_COUNT, HC_LOWRANK) > 0);
    }
}

// MODEL_ARCH_ID folds HC_COUNT/HC_LOWRANK in unconditionally (see layout.hpp make_model_arch_id), the
// same "covers every axis" treatment NGRAM_MAX_N/NGRAM_TABLES_PER_ORDER/NGRAM_TABLE_SIZE already get --
// verified indirectly here since MODEL_ARCH_ID's inputs are compile-time constants of THIS build: the
// direct, testable claim is that GR_DIMS/gr_param_delta -- the values that mix -- are real, well-defined
// quantities (established above). `if constexpr` on HC_COUNT (not a hard-coded 0 assumption) so this
// also compiles and asserts something meaningful under an ad-hoc GR-ON build (docs/GATED_RESIDUAL.md S8's
// two-scale/parity verification), not just the default-suite's own neutral setting.
TEST_CASE("Gated Residual dims are inert and well-defined at the Stage 0 neutral setting", "[layout][gr]") {
    REQUIRE(sub0::GR_DIMS.hidden_size == D_MODEL);
    if constexpr (HC_COUNT == 0) {
        REQUIRE(sub0::GR_DIMS.hc_count == 0);
        REQUIRE(sub0::GR_DIMS.hc_lowrank == 0);
        REQUIRE(sub0::GR_DIMS.wide() == 0);   // "describes a shape nothing builds" -- see GR_DIMS's own comment
    } else {
        REQUIRE(sub0::GR_DIMS.hc_count == HC_COUNT);
        REQUIRE(sub0::GR_DIMS.hc_lowrank == HC_LOWRANK);
        REQUIRE(sub0::GR_DIMS.wide() == HC_COUNT * D_MODEL);
    }
}

// --- N-gram embeddings (docs/NGRAM_EMBEDDING.md) -------------------------------------------------
// The hashing math (ngram_num_embedders / ngram_vocab_dim / ngram_order_of / ngram_vocab_mod) is
// exposed as free functions of EXPLICIT parameters -- not folded into consteval array-builders closed
// over this build's own NGRAM_MAX_N/NGRAM_TABLES_PER_ORDER/NGRAM_TABLE_SIZE -- for the same reason
// DepthScheduleT's depth_schedule_for() above takes its parameters explicitly: it lets this test pin
// hand-verified values at hypothetical configs regardless of what THIS build was configured with.
TEST_CASE("n-gram hashing math matches hand-computed values at hypothetical configs", "[layout][ngram]") {
    // num_embedders = k * (max_n - 1): reference's `k * (n - 1)`.
    static_assert(sub0::ngram_num_embedders(0, 1) == 0);     // off
    static_assert(sub0::ngram_num_embedders(1, 4) == 0);     // max_n < 2 is also off
    static_assert(sub0::ngram_num_embedders(2, 1) == 1);     // bigrams only, 1 table
    static_assert(sub0::ngram_num_embedders(3, 2) == 4);     // bigrams+trigrams, 2 tables each
    static_assert(sub0::ngram_num_embedders(4, 3) == 9);     // bi/tri/4-grams, 3 tables each

    // Per-table vocab size: `m + index*2 + 1` (the reference's non-force-prime branch).
    static_assert(sub0::ngram_vocab_dim(1000, 0) == 1001);
    static_assert(sub0::ngram_vocab_dim(1000, 1) == 1003);
    static_assert(sub0::ngram_vocab_dim(1000, 3) == 1007);

    // Which highest order embedder `index` belongs to: index = (order-2)*k + j, inverted.
    // k=2 tables/order: indices 0,1 -> order 2 (bigram); 2,3 -> order 3 (trigram).
    static_assert(sub0::ngram_order_of(0, 2) == 2);
    static_assert(sub0::ngram_order_of(1, 2) == 2);
    static_assert(sub0::ngram_order_of(2, 2) == 3);
    static_assert(sub0::ngram_order_of(3, 2) == 3);

    // vocab_mod: power_mod = (power_mod * hash_base) % vocab_dim, accumulated `shift` times. Hand-
    // computed with small numbers so the modular arithmetic is checkable by inspection: hash_base=7,
    // table_size=4, index=0 -> vocab_dim = ngram_vocab_dim(4,0) = 5.
    //   shift=1: power = (1*7) % 5 = 2
    //   shift=2: power = (2*7) % 5 = 4
    //   shift=3: power = (4*7) % 5 = 3
    static_assert(sub0::ngram_vocab_dim(4, 0) == 5);
    static_assert(sub0::ngram_vocab_mod(4, 0, 7, 1) == 2);
    static_assert(sub0::ngram_vocab_mod(4, 0, 7, 2) == 4);
    static_assert(sub0::ngram_vocab_mod(4, 0, 7, 3) == 3);
    // A different table (index=1 -> vocab_dim=7) with the same hash_base gives a DIFFERENT sequence --
    // the whole point of k independent tables per order (independent hash functions via the modulus).
    static_assert(sub0::ngram_vocab_dim(4, 1) == 7);
    static_assert(sub0::ngram_vocab_mod(4, 1, 7, 1) == 0);   // (1*7) % 7 == 0
    static_assert(sub0::ngram_vocab_mod(4, 1, 7, 2) == 0);   // stays 0 once it hits 0 -- a real, if
                                                              // degenerate, property of this table choice
    REQUIRE(sub0::ngram_vocab_mod(4, 1, 7, 2) == 0);
}

// Stage 0: at the neutral setting (NGRAM_MAX_N == 0) the feature contributes ZERO new parameters and
// touches no PARAM_LAYOUT entry -- the same "bit-identical when off" contract depth attention's own
// Stage 0 test pins for ARCH_FINGERPRINT, applied here to the SHAPE axis instead (n-gram embeddings add
// real parameters, so PARAM_FLOATS/NUM_PARAMS -- not ARCH_FINGERPRINT -- is what discriminates a
// mismatched checkpoint; see layout.hpp's own NGRAM_EMBED section for why that classification is
// correct rather than an oversight).
TEST_CASE("n-gram embeddings are off by default and inert at the neutral setting", "[layout][ngram]") {
    // Written as an implication, NOT gated behind `if constexpr` -- a STATIC_REQUIRE inside an
    // untaken `if constexpr` branch at namespace/function scope still fires (see the positional-
    // encoding test above's own comment on exactly this trap), so these must hold regardless of the
    // build under test, collapsing to "true given false" when NGRAM_EMBED happens to be on.
    STATIC_REQUIRE((!sub0::NGRAM_EMBED) || sub0::NGRAM_NUM_EMBEDDERS > 0);
    STATIC_REQUIRE(sub0::NGRAM_EMBED || sub0::NGRAM_NUM_EMBEDDERS == 0);
    STATIC_REQUIRE(sub0::NGRAM_EMBED || sub0::NGRAM_EMB_DIM == 0);
    if constexpr (!sub0::NGRAM_EMBED) {
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
            REQUIRE(p.kind != sub0::PKind::NgramEmb);
            REQUIRE(p.kind != sub0::PKind::NgramProj);
        }
    }
}

// --- Mixture of Experts (docs/MOE.md) -------------------------------------------------------------
// moe_param_delta() is exposed as a free function of EXPLICIT parameters (not closed over this build's
// own NUM_EXPERTS/EXPERTS_PER_TOK) for the same reason ngram_num_embedders()/gr_param_delta() are: lets
// this test pin hand-verified values at hypothetical configs regardless of what THIS build was
// configured with. Hand-computed at n_layers=1, d_model=4, d_ff=8, num_experts=2 (the smallest "on"
// value):
//   router          = d_model*num_experts        = 4*2  = 8
//   per_expert      = 3*d_model*d_ff              = 3*4*8 = 96
//   experts total   = num_experts*per_expert      = 2*96 = 192
//   shared          = 3*d_model*d_ff              = 96
//   shared_gate     = d_model                     = 4
//   moe_layer_floats = 8+192+96+4                 = 300
//   dense (gated, 3 tensors)  = 3*d_model*d_ff     = 96   -> delta = 300-96  = 204
//   dense (plain, 4 tensors)  = 2*d_model*d_ff+d_ff+d_model = 64+8+4 = 76 -> delta = 300-76 = 224
TEST_CASE("Mixture of Experts param delta matches hand-computed values at hypothetical configs", "[layout][moe]") {
    static_assert(sub0::moe_param_delta(1, 4, 8, 0, 0, true) == 0);      // off (num_experts < 2)
    static_assert(sub0::moe_param_delta(1, 4, 8, 1, 1, true) == 0);      // 1 expert also off (< 2)
    static_assert(sub0::moe_param_delta(1, 4, 8, 2, 1, true) == 204);    // replacing a gated dense FFN
    static_assert(sub0::moe_param_delta(1, 4, 8, 2, 1, false) == 224);   // replacing a plain dense FFN
    static_assert(sub0::moe_param_delta(3, 4, 8, 2, 1, true) == 3 * 204);  // scales linearly in n_layers
    // Strictly positive for every num_experts >= 2 (docs/MOE.md S3b) -- checked at a larger, more
    // "production-shaped" hypothetical too, not just the minimal case above.
    static_assert(sub0::moe_param_delta(8, 96, 384, 8, 2, true) > 0);
    REQUIRE(sub0::moe_param_delta(1, 4, 8, 2, 1, true) == 204);
}

// MODEL_ARCH_ID folds NUM_EXPERTS in unconditionally and ARCH_FINGERPRINT2 folds EXPERTS_PER_TOK in
// (layout.hpp's make_model_arch_id/arch_fingerprint2) -- verified indirectly here since both are
// compile-time constants of THIS build: the direct, testable claim is that MOE_DIMS/moe_param_delta --
// the values that mix -- are real, well-defined quantities (established above).
TEST_CASE("MoE dims are inert and well-defined at the Stage 0/1 neutral setting", "[layout][moe]") {
    REQUIRE(sub0::MOE_DIMS.hidden_size == D_MODEL);
    REQUIRE(sub0::MOE_DIMS.d_ff == D_FF);
    if constexpr (NUM_EXPERTS == 0) {
        REQUIRE(sub0::MOE_DIMS.num_experts == 0);
        REQUIRE(sub0::MOE_DIMS.experts_per_tok == 0);
    } else {
        REQUIRE(sub0::MOE_DIMS.num_experts == NUM_EXPERTS);
        REQUIRE(sub0::MOE_DIMS.experts_per_tok == EXPERTS_PER_TOK);
    }
}

// Stage 0/1: at the neutral setting (NUM_EXPERTS == 0) the feature contributes ZERO new parameters and
// touches no PARAM_LAYOUT entry -- the same "bit-identical when off" contract every other mechanism in
// this thread pins. Written as an implication, not gated behind `if constexpr`, for the same
// STATIC_REQUIRE-inside-an-untaken-branch reason the n-gram test above documents.
TEST_CASE("Mixture of Experts is off by default and inert at the neutral setting", "[layout][moe]") {
    STATIC_REQUIRE(sub0::USE_MOE == (NUM_EXPERTS >= 2));
    STATIC_REQUIRE((!sub0::USE_MOE) || NUM_EXPERTS >= 2);
    if constexpr (!sub0::USE_MOE) {
        REQUIRE(sub0::moe_param_delta(N_LAYERS, D_MODEL, D_FF, NUM_EXPERTS, EXPERTS_PER_TOK, USE_GATED_FFN) == 0);
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
            REQUIRE(p.kind != sub0::PKind::MoeRouter);
            REQUIRE(p.kind != sub0::PKind::MoeGate);
            REQUIRE(p.kind != sub0::PKind::MoeUp);
            REQUIRE(p.kind != sub0::PKind::MoeDown);
            REQUIRE(p.kind != sub0::PKind::MoeSharedGate);
            REQUIRE(p.kind != sub0::PKind::MoeSharedUp);
            REQUIRE(p.kind != sub0::PKind::MoeSharedDown);
            REQUIRE(p.kind != sub0::PKind::MoeSharedGateProj);
        }
    }
}

// --- Qwen Sparse Attention (QSA) -- docs/QSA.md -------------------------------------------------
//
// qsa_param_delta() and qsa_schedule_for<LAYERS>() are both exposed as free functions of EXPLICIT
// parameters (not closed over this build's own QSA_INDEXER_*/GDN_FULL_ATTN_STRIDE) for the same reason
// gdn_schedule_for()/gr_param_delta()/moe_param_delta() are: AGENTS.md S7's own lesson (LoopSplit's
// odd-layer-count static_assert was invisible at one scale and instant at another) is only enforceable
// if hypothetical shapes -- including ones this binary was NOT built for -- can be asserted directly.

TEST_CASE("QSA layer schedule is a three-way classification, correct at an odd/non-dividing layer count",
          "[layout][qsa]") {
    using sub0::LayerMixer;
    // Even, dividing: 8 layers at stride 4 -> layers 3 and 7 are full attention (=> QSA when on),
    // everything else GDN. This is exactly the real model's own rule (verified against the real
    // config.json's layer_types array, docs/QSA.md S0), just at a smaller layer count.
    {
        constexpr auto s = sub0::qsa_schedule_for<8>(4, true);
        STATIC_REQUIRE(s[0] == LayerMixer::Gdn);
        STATIC_REQUIRE(s[1] == LayerMixer::Gdn);
        STATIC_REQUIRE(s[2] == LayerMixer::Gdn);
        STATIC_REQUIRE(s[3] == LayerMixer::Qsa);
        STATIC_REQUIRE(s[4] == LayerMixer::Gdn);
        STATIC_REQUIRE(s[7] == LayerMixer::Qsa);
        int qsa_n = 0; for (const LayerMixer m : s) if (m == LayerMixer::Qsa) ++qsa_n;
        REQUIRE(qsa_n == 2);
    }
    // ODD and NON-DIVIDING: 11 layers at stride 4. 4 does not divide 11, so the final partial group
    // (layers 8,9,10) has NO full-attention layer at all -- exactly the class of ragged-tail bug a
    // single-scale check cannot see. Layers 3 and 7 are QSA; layer 10 must NOT be.
    {
        constexpr auto s = sub0::qsa_schedule_for<11>(4, true);
        STATIC_REQUIRE(s[3]  == LayerMixer::Qsa);
        STATIC_REQUIRE(s[7]  == LayerMixer::Qsa);
        STATIC_REQUIRE(s[8]  == LayerMixer::Gdn);
        STATIC_REQUIRE(s[9]  == LayerMixer::Gdn);
        STATIC_REQUIRE(s[10] == LayerMixer::Gdn);
        int qsa_n = 0, gdn_n = 0;
        for (const LayerMixer m : s) { if (m == LayerMixer::Qsa) ++qsa_n; if (m == LayerMixer::Gdn) ++gdn_n; }
        REQUIRE(qsa_n == 2);
        REQUIRE(gdn_n == 9);
    }
    // ODD layer count with a stride that DOES divide it: 11 layers at stride 11 -> only the last layer.
    {
        constexpr auto s = sub0::qsa_schedule_for<11>(11, true);
        int qsa_n = 0; for (const LayerMixer m : s) if (m == LayerMixer::Qsa) ++qsa_n;
        REQUIRE(qsa_n == 1);
        STATIC_REQUIRE(s[10] == LayerMixer::Qsa);
    }
    // Stride 0 (this project's default): NO GDN layers at all, so EVERY layer is full attention -- and
    // therefore every layer is QSA when QSA is on, none when it is off.
    {
        constexpr auto on  = sub0::qsa_schedule_for<11>(0, true);
        constexpr auto off = sub0::qsa_schedule_for<11>(0, false);
        for (int l = 0; l < 11; ++l) {
            REQUIRE(on[static_cast<std::size_t>(l)]  == LayerMixer::Qsa);
            REQUIRE(off[static_cast<std::size_t>(l)] == LayerMixer::Attn);
        }
    }
    // qsa_on=false must never produce a Qsa entry at ANY stride -- the "zero effect when off" contract,
    // asserted rather than assumed.
    {
        constexpr auto s = sub0::qsa_schedule_for<11>(4, false);
        for (const LayerMixer m : s) REQUIRE(m != LayerMixer::Qsa);
    }
    // This build's own MIXER_SCHEDULE must reproduce the free function exactly, and must agree with the
    // pre-existing GDN_SCHEDULE on every layer (the invariant layout.hpp static_asserts).
    constexpr auto this_build = sub0::qsa_schedule_for<N_LAYERS>(GDN_FULL_ATTN_STRIDE, sub0::USE_QSA);
    for (int l = 0; l < N_LAYERS; ++l) {
        const std::size_t i = static_cast<std::size_t>(l);
        REQUIRE(sub0::MIXER_SCHEDULE[i] == this_build[i]);
        REQUIRE((sub0::MIXER_SCHEDULE[i] != LayerMixer::Gdn) == sub0::GDN_SCHEDULE.full_attn[i]);
    }
}

TEST_CASE("QSA param delta matches hand-computed values at hypothetical configs, at two shapes",
          "[layout][qsa]") {
    // Off is exactly zero at every shape -- the neutral-setting identity, not merely "small".
    STATIC_REQUIRE(sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 0, 0, 0, false) == 0);
    STATIC_REQUIRE(sub0::qsa_param_delta(11, 4, 132, 66, 33, true, 4, 1, 8, false) == 0);
    REQUIRE(sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 0, 0, 0, false) == 0);

    // Hand-computed, tiny, fully checkable shape: n_layers=1, gdn_stride=0 (=> 1 full-attention layer),
    // d_model=4, d_kv=4, d_head=2, qk_norm=true, idx_n=2, idx_kv=1, idx_hd=2.
    //   idx_qk_out = (2+1)*2 = 6
    //   attn_layer = 2*4*4 + 2*4*4 + 2*2         = 32 + 32 + 4  = 68
    //   qsa_layer  = 3*4*4 + 2*4*4 + 2*2 + 4*6 + 2*2 = 48+32+4+24+4 = 112
    //   delta      = 1 * (112 - 68) = 44
    STATIC_REQUIRE(sub0::qsa_param_delta(1, 0, 4, 4, 2, true, 2, 1, 2, true) == 44);
    REQUIRE(sub0::qsa_param_delta(1, 0, 4, 4, 2, true, 2, 1, 2, true) == 44);
    // Without QK-norm the attention baseline is 4 floats cheaper, so the delta is 4 LARGER (a QSA layer
    // always carries its own q_norm/k_norm -- docs/QSA.md S3b).
    STATIC_REQUIRE(sub0::qsa_param_delta(1, 0, 4, 4, 2, false, 2, 1, 2, true) == 48);
    // Scales linearly in the number of FULL-ATTENTION layers, not in n_layers: at stride 4, 8 layers
    // have exactly 2 full-attention layers, so the delta is 2x the per-layer value, not 8x.
    STATIC_REQUIRE(sub0::qsa_param_delta(8, 4, 4, 4, 2, true, 2, 1, 2, true) == 2 * 44);
    STATIC_REQUIRE(sub0::qsa_param_delta(8, 0, 4, 4, 2, true, 2, 1, 2, true) == 8 * 44);
    // ODD/non-dividing: 11 layers at stride 4 has exactly 2 full-attention layers (3 and 7), NOT 3 --
    // the same ragged-tail case the schedule test above pins, re-checked through the delta.
    STATIC_REQUIRE(sub0::qsa_param_delta(11, 4, 4, 4, 2, true, 2, 1, 2, true) == 2 * 44);

    // Strictly positive and monotonic in the indexer width at both of this thread's standard shapes.
    STATIC_REQUIRE(sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 4, 1, 8, true) > 0);
    STATIC_REQUIRE(sub0::qsa_param_delta(11, 0, 132, 66, 33, true, 4, 1, 8, true) > 0);
    REQUIRE(sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 8, 1, 8, true) >
            sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 4, 1, 8, true));
    REQUIRE(sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 4, 1, 16, true) >
            sub0::qsa_param_delta(8, 0, 96, 96, 48, true, 4, 1, 8, true));

    // This build's own delta is 0 exactly when QSA is off -- the direct neutral-setting claim.
    if constexpr (!sub0::USE_QSA) {
        REQUIRE(sub0::qsa_param_delta(N_LAYERS, GDN_FULL_ATTN_STRIDE, D_MODEL, sub0::D_KV, D_HEAD,
                                       USE_QK_NORM, QSA_INDEXER_N_HEADS, QSA_INDEXER_KV_HEADS,
                                       QSA_INDEXER_HEAD_DIM, sub0::USE_QSA) == 0);
    } else {
        REQUIRE(sub0::qsa_param_delta(N_LAYERS, GDN_FULL_ATTN_STRIDE, D_MODEL, sub0::D_KV, D_HEAD,
                                       USE_QK_NORM, QSA_INDEXER_N_HEADS, QSA_INDEXER_KV_HEADS,
                                       QSA_INDEXER_HEAD_DIM, sub0::USE_QSA) > 0);
    }
}

// At the neutral setting (every QSA_INDEXER_* == 0) the feature contributes ZERO new parameters and
// touches no PARAM_LAYOUT entry -- the same "bit-identical when off" contract every other mechanism in
// this thread pins. Written as an implication, not gated behind `if constexpr`, for the same
// STATIC_REQUIRE-inside-an-untaken-branch reason the n-gram/MoE tests above document.
TEST_CASE("QSA is off by default and inert at the neutral setting", "[layout][qsa]") {
    STATIC_REQUIRE(sub0::USE_QSA == (QSA_INDEXER_N_HEADS >= 1 && QSA_INDEXER_BUDGET >= 1));
    // QSA_DIMS is always a valid, never-divide-by-zero shape, even when nothing builds it.
    REQUIRE(sub0::QSA_DIMS.hidden_size == D_MODEL);
    REQUIRE(sub0::QSA_DIMS.n_heads == N_HEADS);
    REQUIRE(sub0::QSA_DIMS.head_dim == D_HEAD);
    REQUIRE(sub0::QSA_DIMS.n_kv_heads == N_KV_HEADS);
    REQUIRE(sub0::QSA_DIMS.rotary_dim == D_HEAD);
    // The _BUF forms are never degenerate, so decode-path scratch arrays are always valid bounds.
    REQUIRE(sub0::QSA_DIMS_BUF.idx_n_heads >= 1);
    REQUIRE(sub0::QSA_DIMS_BUF.idx_kv_heads >= 1);
    REQUIRE(sub0::QSA_DIMS_BUF.idx_head_dim >= 1);
    REQUIRE(sub0::QSA_DIMS_BUF.compress_ratio >= 1);
    REQUIRE(sub0::QSA_DIMS_BUF.budget >= 1);
    if constexpr (!sub0::USE_QSA) {
        REQUIRE(sub0::QSA_IDX_QK_OUT == 0);
        for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
            REQUIRE(p.kind != sub0::PKind::QsaQProj);
            REQUIRE(p.kind != sub0::PKind::QsaGateProj);
            REQUIRE(p.kind != sub0::PKind::QsaKProj);
            REQUIRE(p.kind != sub0::PKind::QsaVProj);
            REQUIRE(p.kind != sub0::PKind::QsaOProj);
            REQUIRE(p.kind != sub0::PKind::QsaQNorm);
            REQUIRE(p.kind != sub0::PKind::QsaKNorm);
            REQUIRE(p.kind != sub0::PKind::QsaIdxQkProj);
            REQUIRE(p.kind != sub0::PKind::QsaIdxQNorm);
            REQUIRE(p.kind != sub0::PKind::QsaIdxKNorm);
        }
    }
}

