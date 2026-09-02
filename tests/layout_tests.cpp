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
    constexpr int kAttnLayers = N_LAYERS - sub0::GDN_SCHEDULE.gdn_layers;
    constexpr int kAttnMixer  = 4 + (USE_QK_NORM ? 2 : 0);
    constexpr int kGdnMixer   = 9;
    constexpr int kFfnSlots   = USE_GATED_FFN ? 3 : 4;
    // Gated Residual (Stage 1, docs/GATED_RESIDUAL.md S3b): 4 slots per instance (GrHcNorm/MixDown/
    // MixUp/BlockInject), TWO full instances per layer (attn-wrapping, mlp-wrapping) plus one top-level
    // instance WITHOUT BlockInject (3 slots), appended once regardless of N_LAYERS. Zero at HC_COUNT==0.
    constexpr int kGrPerLayer = sub0::USE_GATED_RESIDUAL ? 2 * 4 : 0;
    constexpr int kGrTop      = sub0::USE_GATED_RESIDUAL ? 3 : 0;
    STATIC_REQUIRE(sub0::NUM_PARAMS == 1 + (sub0::HAS_POS_EMB ? 1 : 0)
                                        + N_LAYERS * (2 + kFfnSlots + kGrPerLayer)
                                        + kAttnLayers * kAttnMixer
                                        + sub0::GDN_SCHEDULE.gdn_layers * kGdnMixer
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
        const bool is_matrix =
            p.kind == PKind::Wq || p.kind == PKind::Wk || p.kind == PKind::Wv ||
            p.kind == PKind::Wo || p.kind == PKind::W1 || p.kind == PKind::W2 ||
            p.kind == PKind::Wg || p.kind == PKind::LmHead || p.kind == PKind::NgramProj ||
            p.kind == PKind::GdnInProjQkv || p.kind == PKind::GdnInProjZ ||
            p.kind == PKind::GdnInProjB || p.kind == PKind::GdnInProjA || p.kind == PKind::GdnOutProj ||
            p.kind == PKind::GrMixDown || p.kind == PKind::GrMixUp || p.kind == PKind::GrBlockInject;
        // AdamW weight decay applies only to the GEMM weight matrices.
        REQUIRE(p.decay == is_matrix);
        // Ternary quantization covers the block matrices but NOT the full-precision head/ngram-proj/
        // any GDN/GR projection (see the comment above for why both stay full precision for now).
        const bool is_gdn_matrix =
            p.kind == PKind::GdnInProjQkv || p.kind == PKind::GdnInProjZ ||
            p.kind == PKind::GdnInProjB || p.kind == PKind::GdnInProjA || p.kind == PKind::GdnOutProj;
        const bool is_gr_matrix =
            p.kind == PKind::GrMixDown || p.kind == PKind::GrMixUp || p.kind == PKind::GrBlockInject;
        const bool ternary_eligible = is_matrix && p.kind != PKind::LmHead && p.kind != PKind::NgramProj
                                      && !is_gdn_matrix && !is_gr_matrix;
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

// ARCH_FINGERPRINT2 must reproduce 0 at the only value GDN_FULL_ATTN_STRIDE can currently take, and
// must differ once a nonzero stride is EVER passed to the free function -- the exact same "neutral is
// bit-identical, non-neutral is distinguishable" property ARCH_FINGERPRINT's own DEPTH_ATTN_STRIDE test
// pins above. Calling arch_fingerprint2() directly (rather than requiring an actual GDN build) is what
// makes this testable at all while the static_assert in layout.hpp forbids ever compiling one.
TEST_CASE("arch fingerprint2 stays bit-identical when Gated DeltaNet is off", "[layout][gdn]") {
    REQUIRE(sub0::arch_fingerprint2(0) == 0);
    REQUIRE(sub0::ARCH_FINGERPRINT2 == 0);              // this build's own word: always 0 today
    REQUIRE(sub0::ARCH_FINGERPRINT2 == sub0::ARCH_FINGERPRINT2_LEGACY);

    // A nonzero stride MUST differ, or a (future) GDN checkpoint would load into a plain build and
    // silently compute something else -- there is no shape difference to catch it in general (a GDN
    // layer's parameter count is a different SHAPE from a softmax layer's, but nothing here proves that
    // holds at every possible dims/stride combination the way GQA's monotonic D_KV substitution does --
    // see docs/GATED_DELTANET.md's checkpoint-design section for the worked argument).
    REQUIRE(sub0::arch_fingerprint2(4) != sub0::arch_fingerprint2(0));
    REQUIRE(sub0::arch_fingerprint2(4) != sub0::arch_fingerprint2(3));

    // Round-trips through the decoder, so the diagnostic can name the mismatched value.
    REQUIRE(sub0::arch_axes2_of(sub0::arch_fingerprint2(4)).gdn_full_attn_stride == 4);

    // The reserved high 56 bits stay zero regardless of the low byte's value -- headroom for the NEXT
    // computation-changing, shape-neutral axis, so it does not repeat ARCH_FINGERPRINT's own scramble.
    REQUIRE((sub0::arch_fingerprint2(0xff) >> 8) == 0);

    // MODEL_ARCH_ID folds this word in (see layout.hpp make_model_arch_id): two builds differing only in
    // a hypothetical GDN stride must not collide, matching how it already handles DEPTH_ATTN_STRIDE.
    // Verified indirectly here since MODEL_ARCH_ID's inputs are compile-time constants of THIS build --
    // the direct claim is just that ARCH_FINGERPRINT2 (which the id already mixes in) is a real,
    // distinguishing quantity, established above.
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
