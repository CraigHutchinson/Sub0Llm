// tests/qwen4_real_axes/full48_totals.hpp -- the FULL 48-layer model's own layout totals, in both
// residency forms, as literals two independently-compiled translation units can assert against.
//
// THE SAME PROBLEM sub4_prefix.hpp SOLVES, one scale up. layout.hpp is closed over a single
// sub0_config.hpp, and MOE_QUANT_EXPERTS changes make_param_layout()'s output, so an all-f32 48-layer
// PARAM_LAYOUT and a quantized-resident 48-layer one cannot coexist in one binary any more than a
// 4-layer and a 48-layer one can. "The quantized-resident full-model target is the real model with
// exactly the routed experts moved out" therefore cannot be checked by comparing two layouts directly;
// it is checked by two builds each asserting the same hand-derived literals below.
//
// Every number here is DERIVED, with its arithmetic written out, not pasted from a compiler -- the same
// discipline sub4_prefix.hpp and qwen4_real_shape_tests.cpp's own census apply, and for the same
// reason: a change that moves one of these has to be understood, not merely re-recorded.
//
// WHO ASSERTS WHAT:
//   * tests/qwen4_real_shape_tests.cpp        -- compiled at N_LAYERS = 48 in BOTH residency forms
//                                                (targets sub0_qwen4_shape_tests / _q)
//   * tools/sub0llm-transplant.cpp            -- compiled at N_LAYERS = 48, quantized-resident
//                                                (target sub0llm-transplant-q48)

#pragma once

#include "sub4_prefix.hpp"

#include <cstddef>

namespace sub0::qwen4_full48 {

// --- the all-f32 form: the real model's own totals ------------------------------------------------
// These are qwen4_real_shape_tests.cpp's own two headline numbers, restated here so the quantized
// deduction below has a stated starting point rather than an implicit one. That file derives them
// independently, tensor class by tensor class; this header does not re-derive them, it names them.
//   NUM_PARAMS   = 1 (tok_emb) + 36 GDN layers x 1558 + 12 QSA layers x 1559 + 3 (GR exit) + 2 (head)
//   PARAM_FLOATS = docs/QWEN4_MEMORY_ORCHESTRATION.md S2f's real total, less the ln_f that does not
//                  exist in the real checkpoint (2,560 floats, one tensor)
inline constexpr int         NUM_PARAMS   = 74'802;
inline constexpr std::size_t PARAM_FLOATS = 125'711'064'960ull - 2'560ull;   // 125,711,062,400
static_assert(NUM_PARAMS == 1 + 36 * 1558 + 12 * 1559 + 3 + 2,
              "the 48-layer tensor count is the 3-GDN-then-1-QSA repeating unit, 12 times over");

// --- the quantized-resident form: the same model, routed experts in the S0Q1 sidecar ---------------
// Under MOE_QUANT_EXPERTS the 3 * NUM_EXPERTS routed-expert tensors per layer leave PARAM_LAYOUT
// entirely and nothing else changes (moe_quant.hpp's own header comment enumerates what STAYS: the
// dense-read router, the always-on shared expert, and every GR/GDN/QSA/embedding tensor). So the
// deduction is exactly N_LAYERS times sub4_prefix.hpp's own per-layer figures -- reused from there
// rather than re-spelled, so the 4-layer and 48-layer claims cannot drift apart.
inline constexpr int LAYERS = 48;
inline constexpr int         QUANT_NUM_PARAMS =
    NUM_PARAMS - LAYERS * qwen4_sub4::QUANT_EXPERT_TENSORS_PER_LAYER;                      // 1,074
inline constexpr std::size_t QUANT_PARAM_FLOATS =
    PARAM_FLOATS - LAYERS * qwen4_sub4::QUANT_EXPERT_FLOATS_PER_LAYER;             // 4,915,107,200
static_assert(QUANT_NUM_PARAMS == 1'074 && QUANT_PARAM_FLOATS == 4'915'107'200ull,
              "the quantized-resident 48-layer totals are a stated claim, not a compiler observation");

// The two on-disk sizes this implies, stated once so the tool, the operator and any reader agree.
// The f32 figure is why WP4e exists at all: 468 GiB is not a model this machine can hold, write, or
// even store comfortably -- it is 8x the total RAM and would be the largest file on the disk.
inline constexpr std::size_t PARAM_BYTES       = PARAM_FLOATS * 4;        // 502,844,249,600 == 468.3 GiB
inline constexpr std::size_t QUANT_PARAM_BYTES = QUANT_PARAM_FLOATS * 4;  //  19,660,428,800 ==  18.3 GiB

}  // namespace sub0::qwen4_full48
