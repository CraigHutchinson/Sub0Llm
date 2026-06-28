// layout_tests.cpp — the constexpr parameter layout table and the baked compute-backend
// facts. These guard the shared "shape of the weights" truth (include/sub0/layout.hpp)
// that both the CPU and the future GPU backend address, and the configure-time GPU facts
// the engine compiles against.

#include <catch2/catch_test_macros.hpp>

#include "sub0/core.hpp"     // pulls sub0_config.hpp: HAS_CUDA, COMPUTE_MODE, CUDA_ARCH, ...
#include "sub0/layout.hpp"   // PARAM_LAYOUT, NUM_PARAMS, PARAM_FLOATS

#include <cstddef>

TEST_CASE("param layout is contiguous and totals PARAM_FLOATS", "[layout]") {
    STATIC_REQUIRE(sub0::NUM_PARAMS == 2 + 10 * N_LAYERS + 3);
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
    REQUIRE(sub0::PARAM_LAYOUT.back().kind  == PKind::LmBias);   // last tensor

    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) {
        const bool is_matrix =
            p.kind == PKind::Wq || p.kind == PKind::Wk || p.kind == PKind::Wv ||
            p.kind == PKind::Wo || p.kind == PKind::W1 || p.kind == PKind::W2 ||
            p.kind == PKind::LmHead;
        // AdamW weight decay applies only to the GEMM weight matrices.
        REQUIRE(p.decay == is_matrix);
        // Ternary quantization covers the block matrices but NOT the full-precision head.
        const bool ternary_eligible = is_matrix && p.kind != PKind::LmHead;
        REQUIRE(p.ternary == ternary_eligible);
    }
}

TEST_CASE("baked compute-backend facts are self-consistent", "[config]") {
    // BitNet/ternary is CPU-only: a ternary build must never select a GPU/HYBRID backend
    // (the same invariant the CMake guard and the backend_cuda.cu static_assert enforce).
    STATIC_REQUIRE(!USE_TERNARY || COMPUTE_MODE == ComputeBackend::Cpu);
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
