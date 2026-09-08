// qwen4_full48_quant_shape_tests.cpp -- WP5b's own compile-time gate: make_param_layout() at the REAL
// 48-layer Qwen3.8-Flash-Next axes with the routed experts kept quantized-resident must be the real
// model's layout with exactly those tensors removed, and nothing else moved.
//
// WHY A THIRD SHAPE TARGET, and not another case in qwen4_real_shape_tests.cpp. Same reason that file
// is its own binary rather than a case in sub0_tests, one level further in: layout.hpp is closed over a
// single sub0_config.hpp, and MOE_QUANT_EXPERTS changes what make_param_layout() emits, so the all-f32
// 48-layer PARAM_LAYOUT and the quantized-resident one are two different definitions of sub0::
// PARAM_LAYOUT and cannot share a translation unit. The sibling target (sub0_qwen4_shape_tests) owns
// the f32 form; this one owns the quantized-resident form; full48_totals.hpp carries the literals both
// of them, and tools/sub0llm-transplant.cpp's own 48-layer target, are held to.
//
// WHAT THIS RULES OUT, rather than merely exercises. The 48-layer quantized-resident layout is the
// destination of a ~58 GB two-file transplant that takes tens of minutes to write and cannot be
// meaningfully spot-checked afterwards. Every failure mode below is one that produces an artifact of a
// plausible size with its tensors in the wrong places:
//   * the routed experts not actually leaving the layout (or leaving from only some layers)
//   * the deduction being applied to the wrong per-layer figure -- e.g. dropping the shared expert or
//     the router along with the routed ones, both of which moe_quant.hpp explicitly keeps
//   * the sub-stack relationship silently breaking: the first four layers of THIS layout must still be
//     the 4-layer quantized target's own first four layers, or the WP4e artifact and this one are not
//     the same model truncated differently
//
// It links no engine library: nothing here runs. If it COMPILES, it passed -- the Catch2 case exists
// only so a run PRINTS the census.

#include <catch2/catch_test_macros.hpp>

#include "sub0/layout.hpp"
#include "full48_totals.hpp"
#include "sub4_prefix.hpp"

#include <cstddef>

static_assert(N_LAYERS == 48, "this target is the FULL model -- build it with SUB0_QWEN4_LAYERS=48");
static_assert(MOE_QUANT_EXPERTS,
              "this target is the quantized-resident form -- build it with SUB0_MOE_QUANT_EXPERTS=1");
static_assert(sub0::USE_MOE_QUANT, "MoE must be on for MOE_QUANT_EXPERTS to mean anything");

namespace {
consteval int count_kind(sub0::PKind k) {
    int n = 0;
    for (const sub0::ParamDesc& p : sub0::PARAM_LAYOUT) if (p.kind == k) ++n;
    return n;
}
}  // namespace

// --- the two headline numbers ----------------------------------------------------------------------
static_assert(sub0::NUM_PARAMS == sub0::qwen4_full48::QUANT_NUM_PARAMS,
              "the quantized-resident 48-layer tensor count disagrees with full48_totals.hpp's own "
              "hand-derived deduction from the real model's 74,802");
static_assert(sub0::PARAM_FLOATS == sub0::qwen4_full48::QUANT_PARAM_FLOATS,
              "the quantized-resident 48-layer float total disagrees with full48_totals.hpp's own "
              "hand-derived deduction from the real model's 125,711,062,400");

// --- what left, and what did NOT ---------------------------------------------------------------------
// The deduction is only correct if it removed exactly the ROUTED experts. moe_quant.hpp names three
// things that stay and why: the router (dense-read per token, and already F32 in the source file), the
// shared expert (always-on, one copy per layer), and every GR/GDN/QSA/embedding tensor. Each is checked
// by count, so "the deduction totalled correctly" cannot hide "it removed the wrong set".
static_assert(count_kind(sub0::PKind::MoeGate) == 0);
static_assert(count_kind(sub0::PKind::MoeUp)   == 0);
static_assert(count_kind(sub0::PKind::MoeDown) == 0);
static_assert(count_kind(sub0::PKind::MoeRouter)         == N_LAYERS, "the router stays resident");
static_assert(count_kind(sub0::PKind::MoeSharedGate)     == N_LAYERS, "the shared expert stays f32");
static_assert(count_kind(sub0::PKind::MoeSharedUp)       == N_LAYERS);
static_assert(count_kind(sub0::PKind::MoeSharedDown)     == N_LAYERS);
static_assert(count_kind(sub0::PKind::MoeSharedGateProj) == N_LAYERS);
static_assert(count_kind(sub0::PKind::GdnInProjQkv)  == 36, "36 GDN layers, unchanged by WP4e");
static_assert(count_kind(sub0::PKind::QsaQProj)      == 12, "12 QSA layers, unchanged by WP4e");
static_assert(count_kind(sub0::PKind::GrHcNorm)      == 2 * N_LAYERS + 1, "97 GR instances, unchanged");
static_assert(count_kind(sub0::PKind::GrBlockInject) == 2 * N_LAYERS, "the exit has no inject");
static_assert(count_kind(sub0::PKind::TokEmb) == 1 && count_kind(sub0::PKind::LmHead) == 1);
static_assert(count_kind(sub0::PKind::LnF) == 0, "the real model still has no final norm");
static_assert(count_kind(sub0::PKind::Ln1) == 0 && count_kind(sub0::PKind::Ln2) == 0);

// --- an independent statement of the same deduction --------------------------------------------------
// Adding the routed experts back must land exactly on the f32 form's own two totals -- which the
// SIBLING target asserts against its own independently-written per-tensor census. Two derivations, two
// binaries, one arithmetic identity: the quantized layout is the real layout minus 48 x 512 x 3
// tensors and 48 x 512 x 4,915,200 floats, and nothing else.
static_assert(sub0::NUM_PARAMS
                  + N_LAYERS * sub0::qwen4_sub4::QUANT_EXPERT_TENSORS_PER_LAYER
                  == sub0::qwen4_full48::NUM_PARAMS);
static_assert(sub0::PARAM_FLOATS
                  + N_LAYERS * sub0::qwen4_sub4::QUANT_EXPERT_FLOATS_PER_LAYER
                  == sub0::qwen4_full48::PARAM_FLOATS);

// --- the sub-stack relationship, still true in this residency form -----------------------------------
// sub4_prefix.hpp's QUANT_PREFIX_TENSORS/QUANT_PREFIX_FLOATS are what the 4-layer quantized transplant
// target (sub0llm-transplant-q) claims its layers 0-3 occupy. The first four layers of the FULL
// quantized layout must occupy exactly the same span, or the 4-layer artifact WP4e/WP4f validated is
// not a truncation of the artifact this work package produces -- which is the whole basis for carrying
// those stages' correctness forward to this scale.
static_assert(sub0::PARAM_LAYOUT[sub0::qwen4_sub4::QUANT_PREFIX_TENSORS].off
                  == sub0::qwen4_sub4::QUANT_PREFIX_FLOATS,
              "the full model's first four layers must occupy exactly the float span the 4-layer "
              "quantized-resident transplant target claims");
static_assert(sub0::PARAM_LAYOUT[sub0::qwen4_sub4::QUANT_PREFIX_TENSORS].kind == sub0::PKind::GrHcNorm,
              "entry QUANT_PREFIX_TENSORS is layer 4's leading GR hc_norm -- a real layer boundary");
static_assert(sub0::PARAM_LAYOUT[sub0::qwen4_sub4::QUANT_PREFIX_TENSORS - 1].kind
                  == sub0::PKind::MoeSharedGateProj,
              "and the entry before it is the last of layer 3's MoE block");

// --- the mixer schedule over all 48 layers -----------------------------------------------------------
// GDN_FULL_ATTN_STRIDE = 4 must produce the real model's own 48-entry layer_types array (docs/QSA.md S0
// verified that element by element). Checked here across the WHOLE stack rather than only its first
// four, because this is the first target in the repo whose layout actually spans all 48.
namespace {
consteval bool schedule_is_3gdn_then_qsa() {
    for (int l = 0; l < N_LAYERS; ++l) {
        const bool want_qsa = ((l + 1) % 4 == 0);
        const bool is_qsa = sub0::MIXER_SCHEDULE[static_cast<std::size_t>(l)] == sub0::LayerMixer::Qsa;
        if (is_qsa != want_qsa) return false;
    }
    return true;
}
}  // namespace
static_assert(schedule_is_3gdn_then_qsa(),
              "every 4th layer (3, 7, ... 47) is the full-attention/QSA layer, 12 of 48");

TEST_CASE("make_param_layout() at the real 48-layer axes, routed experts quantized-resident",
          "[qwen4][layout][realshape][moequant]") {
    WARN("full 48-layer quantized-resident: NUM_PARAMS = "
         << sub0::NUM_PARAMS << ", PARAM_FLOATS = " << sub0::PARAM_FLOATS << " ("
         << static_cast<double>(sub0::PARAM_FLOATS) * 4.0 / (1024.0 * 1024.0 * 1024.0)
         << " GiB as f32); the same model all-f32 would be " << sub0::qwen4_full48::NUM_PARAMS
         << " tensors / " << sub0::qwen4_full48::PARAM_FLOATS << " floats ("
         << static_cast<double>(sub0::qwen4_full48::PARAM_FLOATS) * 4.0 / (1024.0 * 1024.0 * 1024.0)
         << " GiB)");
    REQUIRE(sub0::NUM_PARAMS == sub0::qwen4_full48::QUANT_NUM_PARAMS);
    REQUIRE(sub0::PARAM_FLOATS == sub0::qwen4_full48::QUANT_PARAM_FLOATS);
}
