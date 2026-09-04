// tests/qwen4_real_axes/sub4_prefix.hpp -- the shared claim that ties WP4c's 4-layer transplant target
// to WP4b's 48-layer real-shape gate.
//
// THE PROBLEM THIS SOLVES. layout.hpp is closed over one sub0_config.hpp, so a 48-layer PARAM_LAYOUT
// and a 4-layer one cannot coexist in a single translation unit (two definitions of sub0::PARAM_LAYOUT
// is an ODR violation -- qwen4_real_shape_tests.cpp's own header comment says why it is its own
// binary). So "the tool's 4-layer destination really is layers 0-3 of the real 48-layer model" cannot
// be checked by comparing the two layouts directly. It CAN be checked through two literals that both
// TUs assert independently, which is what this header is.
//
// Both numbers below are derived BY HAND from the real architecture, not pasted from a compiler, so a
// change that moves either has to be understood rather than merely re-recorded -- the same discipline
// qwen4_real_shape_tests.cpp's own census applies to the 48-layer totals.
//
// WHY THE PREFIX ENDS WHERE IT DOES. make_param_layout() emits, in order: tok_emb, then every layer,
// then the model-level Gated Residual exit collapse (3 tensors, no block_inject), then lm_head +
// lm_bias. (There is NO ln_f: the real Qwen4ExpTextModel has no final norm -- the exit collapse's own
// hc_norm is it -- so the layout emits none under USE_GATED_RESIDUAL. That removal is what changed
// PARAM_FLOATS/NUM_PARAMS below by exactly D_MODEL floats and one tensor.)
// So the first SUB4_PREFIX_TENSORS entries of ANY build at these axes with N_LAYERS >= 4 are
// exactly "tok_emb plus layers 0..3" -- the tail is what differs. Asserting that PARAM_LAYOUT[
// SUB4_PREFIX_TENSORS].off is the same float offset in a 4-layer build and a 48-layer build is
// therefore exactly the claim "the sub-stack is shape-identical to the real model's first four layers".

#pragma once

#include <cstddef>

namespace sub0::qwen4_sub4 {

// Per-layer tensor counts at these axes (mirrors qwen4_real_shape_tests.cpp's own derivation):
//   MoE per layer = router + 512*(gate,up,down) + shared triple + shared gate = 1 + 1536 + 3 + 1 = 1541
//   a GDN layer   = 4 GR(attn) + 9 GDN  + 4 GR(mlp) + 1541 = 1558
//   a QSA layer   = 4 GR(attn) + 10 QSA + 4 GR(mlp) + 1541 = 1559
// Layers 0..3 under GDN_FULL_ATTN_STRIDE = 4 are GDN, GDN, GDN, QSA.
inline constexpr int PREFIX_TENSORS = 1 + 3 * 1558 + 1559;          // 6,234

// The float offset those tensors occupy, tensor by tensor:
//   tok_emb                      248320 * 2560                              =    635,699,200
//   per layer, 2 GR instances    2 * (10240 + 2*10240*320 + 10240*4)        =     13,209,600
//   per layer, MoE               2560*512 + 512*4,915,200 + 4,915,200 + 2560 =  2,522,810,880
//   a GDN mixer                  (see the S2h-checked census in the shape test)  57,958,624
//   a QSA mixer                                                                  51,446,528
// so a GDN layer totals 2,593,979,104 and a QSA layer 2,587,467,008.
inline constexpr std::size_t PREFIX_FLOATS =
    635'699'200ull + 3ull * 2'593'979'104ull + 2'587'467'008ull;    // 11,005,103,520

// What a 4-layer build's OWN totals must be: the prefix plus the tail (3 GR-exit tensors + lm_head +
// lm_bias), which does not depend on N_LAYERS at all.
//   gr_top  = 10240 + 2*10240*320                                          =      6,563,840
//   lm_head + lm_bias = 2560*248320 + 248320                               =    635,947,520
// Was 6,240 tensors / 11,647,617,440 floats while an LnF slot still existed. Removing it drops both by
// exactly one tensor and D_MODEL = 2,560 floats -- nothing else moved, which is the check that the
// removal is the whole of the change.
inline constexpr int         NUM_PARAMS   = PREFIX_TENSORS + 5;                   // 6,239
inline constexpr std::size_t PARAM_FLOATS = PREFIX_FLOATS + 6'563'840ull + 635'947'520ull;
                                                                                   // 11,647,614,880
// The destination artifact's size, stated once so the tool and any reader agree on it: f32 throughout,
// which is this project's only parameter precision (core.hpp).
inline constexpr std::size_t PARAM_BYTES = PARAM_FLOATS * 4;        // 46,590,459,520 == 43.4 GiB

// --- WP4e: the same four layers with the routed experts kept quantized-resident -------------------
//
// Under MOE_QUANT_EXPERTS the 3*512 routed-expert tensors per layer leave PARAM_LAYOUT entirely (they
// live in the S0Q1 sidecar in their native encoding -- include/sub0/moe_quant.hpp), so the totals drop
// by exactly that much and by nothing else. Derived by hand here, like every number above, so the
// quantized-resident build's own layout is a stated claim the tool asserts rather than whatever the
// compiler happened to produce:
//   per layer, removed  3 * 512 tensors                                    =          1,536
//               floats  512 * (2*2560*640 + 640*2560) = 512 * 4,915,200    =  2,516,582,400
// Note what does NOT change: the router (2560*512), the shared expert's own triple and gate projection,
// and every GR/GDN/QSA/embedding tensor. That is the whole scope of WP4e.
inline constexpr int         QUANT_EXPERT_TENSORS_PER_LAYER = 3 * 512;              // 1,536
inline constexpr std::size_t QUANT_EXPERT_FLOATS_PER_LAYER  = 512ull * 4'915'200ull;  // 2,516,582,400
inline constexpr int         QUANT_PREFIX_TENSORS = PREFIX_TENSORS - 4 * QUANT_EXPERT_TENSORS_PER_LAYER;
                                                                                    // 90
inline constexpr std::size_t QUANT_PREFIX_FLOATS  = PREFIX_FLOATS - 4 * QUANT_EXPERT_FLOATS_PER_LAYER;
                                                                                    // 938,773,920
inline constexpr int         QUANT_NUM_PARAMS   = QUANT_PREFIX_TENSORS + 5;         // 95
inline constexpr std::size_t QUANT_PARAM_FLOATS =
    QUANT_PREFIX_FLOATS + 6'563'840ull + 635'947'520ull;                            // 1,581,285,280
inline constexpr std::size_t QUANT_PARAM_BYTES = QUANT_PARAM_FLOATS * 4;   // 6,325,141,120 == 5.89 GiB
static_assert(QUANT_NUM_PARAMS == 95 && QUANT_PARAM_FLOATS == 1'581'285'280ull,
              "the quantized-resident 4-layer totals are a stated claim, not a compiler observation");

}  // namespace sub0::qwen4_sub4
