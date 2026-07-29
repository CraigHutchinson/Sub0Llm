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
    STATIC_REQUIRE(sub0::NUM_PARAMS == 1 + (sub0::HAS_POS_EMB ? 1 : 0)
                                        + (USE_GATED_FFN ? 9 : 10) * N_LAYERS
                                        + (USE_QK_NORM ? 2 * N_LAYERS : 0)
                                        + (USE_TIED_EMBEDDINGS ? 1 : 3));
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
    // Last tensor: ln_f when tied (no separate head slot -- see op_tied_head), else lm_bias.
    REQUIRE(sub0::PARAM_LAYOUT.back().kind == (USE_TIED_EMBEDDINGS ? PKind::LnF : PKind::LmBias));

    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
        const bool is_matrix =
            p.kind == PKind::Wq || p.kind == PKind::Wk || p.kind == PKind::Wv ||
            p.kind == PKind::Wo || p.kind == PKind::W1 || p.kind == PKind::W2 ||
            p.kind == PKind::Wg || p.kind == PKind::LmHead;
        // AdamW weight decay applies only to the GEMM weight matrices.
        REQUIRE(p.decay == is_matrix);
        // Ternary quantization covers the block matrices but NOT the full-precision head.
        const bool ternary_eligible = is_matrix && p.kind != PKind::LmHead;
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
