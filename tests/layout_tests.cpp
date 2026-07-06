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
    STATIC_REQUIRE(sub0::NUM_PARAMS == 2 + (USE_GATED_FFN ? 9 : 10) * N_LAYERS
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
    // SwiGLU-gated FFN is CPU-only too (no CUDA kernel yet) -- same invariant, enforced by the
    // configurator's --gated-ffn/--compute check and the backend_cuda.cu static_assert.
    STATIC_REQUIRE(!USE_GATED_FFN || COMPUTE_MODE == ComputeBackend::Cpu);
    // Tied embeddings: GPU support landed (launch_tied_head/launch_tied_head_bwd in backend_cuda.cu,
    // `if constexpr`-gated), so there is no CPU-only invariant left to assert for this axis -- it is
    // valid with any COMPUTE_MODE now.
    // QK-norm: GPU support landed the same way (qknorm_act_kernel/qknorm_backward_act_kernel in
    // backend_cuda.cu, `if constexpr`-gated) -- no CPU-only invariant left for this axis either.
    // A detected CUDA host has its arch + VRAM facts populated; an absent one zeroes them.
    STATIC_REQUIRE(!HAS_CUDA || (CUDA_ARCH > 0 && GPU_VRAM_MB > 0));
    STATIC_REQUIRE(HAS_CUDA || CUDA_ARCH == 0);
    // Shared/overflow memory is a non-negative MB figure (0 where there is no WDDM shared mem).
    STATIC_REQUIRE(GPU_SHARED_MEM_MB >= 0);
}

TEST_CASE("positional encoding facts are consistent", "[config]") {
    // RoPE (the default) injects RELATIVE position inside attention via a rotation of Q/K and
    // carries no gradient in its layout slot; Absolute uses a learned pos_emb table added to the
    // token embedding. Either way the scheme is one of the two enumerators, ROPE_THETA is a
    // positive frequency base, and the position table KEEPS its layout slot across schemes (kept
    // for checkpoint-format stability), so NUM_PARAMS is unchanged.
    STATIC_REQUIRE((POS_ENCODING == PosEncoding::Rope || POS_ENCODING == PosEncoding::Absolute));
    STATIC_REQUIRE(ROPE_THETA > 0.0f);
    STATIC_REQUIRE(sub0::PARAM_LAYOUT[1].kind == sub0::PKind::PosEmb);   // slot retained
}
