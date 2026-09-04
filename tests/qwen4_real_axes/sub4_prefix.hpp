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
// then the model-level Gated Residual exit collapse (3 tensors, no block_inject), then ln_f + lm_head +
// lm_bias. So the first SUB4_PREFIX_TENSORS entries of ANY build at these axes with N_LAYERS >= 4 are
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

// What a 4-layer build's OWN totals must be: the prefix plus the tail (3 GR-exit tensors + ln_f +
// lm_head + lm_bias), which does not depend on N_LAYERS at all.
//   gr_top  = 10240 + 2*10240*320                                          =      6,563,840
//   ln_f + lm_head + lm_bias = 2560 + 2560*248320 + 248320                 =    635,950,080
inline constexpr int         NUM_PARAMS   = PREFIX_TENSORS + 6;                   // 6,240
inline constexpr std::size_t PARAM_FLOATS = PREFIX_FLOATS + 6'563'840ull + 635'950'080ull;
                                                                                   // 11,647,617,440
// The destination artifact's size, stated once so the tool and any reader agree on it: f32 throughout,
// which is this project's only parameter precision (core.hpp).
inline constexpr std::size_t PARAM_BYTES = PARAM_FLOATS * 4;        // 46,590,469,760 == 43.4 GiB

}  // namespace sub0::qwen4_sub4
