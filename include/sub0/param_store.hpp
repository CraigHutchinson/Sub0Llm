// sub0/param_store.hpp -- what element type the engine's shared parameter arena is made of, and how a
// kernel reads it. B24 phase 1, docs/BACKBONE_PRECISION.md S1.
//
// THE ONE DECISION THIS HEADER ENCODES. `PARAM_DTYPE` (configurator-baked, `--prec-param`) selects
// whether `g_param_data` is an array of `float` or of `bf16`. Everything downstream is derived from
// that single constant with `std::conditional_t`, so an F32 build's types are literally the types that
// were there before B24 -- `param_t` IS `float`, `ParamCPtr` IS `const float*` -- and every kernel this
// change touched compiles to the same code it did before (AGENTS.md S4: a new capability leaves the
// default build alone).
//
// WHY THE POINTER IS A PROXY RATHER THAN A PROMOTED BUFFER. docs/BACKBONE_PRECISION.md S1a's own
// recommendation was "promote-on-load": keep bf16 at rest, hand each op a promoted f32 view. That
// shape cannot be built as written, and the reason is arithmetic rather than taste. `Node::data` is a
// span over a WHOLE tensor, so a promoted f32 view of one is exactly as large as the f32 tensor it was
// supposed to replace: materialise them all and the resident footprint is unchanged (the win is gone),
// materialise one per access and decode -- which touches every weight of every layer every token, with
// no reuse to amortise against -- pays 2N bytes read + 4N written + 4N read back where it used to pay
// 4N read, i.e. it makes the DRAM traffic that S0 names as the target strictly WORSE. So the promote
// has to happen inside the innermost loop, where the promoted value lives in a register and never
// reaches memory at all. This header is how that is done without rewriting the loops: the loops become
// templates on their weight-pointer type, and `Bf16CPtr::operator[]` (bf16.hpp) IS the promote.
//
// SCOPE: bf16 parameter storage is an INFERENCE-side change. The optimizer keeps f32 gradients and f32
// AdamW moments (those arenas are unchanged); what a bf16 build changes is the precision of the master
// weight a step reads and writes back, which is a real and well-known training-quality regression and
// is not what phase 1 is for. `--prec-param 1` is documented and defaulted accordingly.
//
// `--prec-param 2` (FP8/E4M3, B33/B37, docs/BACKBONE_PRECISION.md S2d) -- READ THIS BEFORE CHOOSING IT.
// This is a REAL, HONEST NEGATIVE RESULT kept as a permanent, buildable, correctness-gated option, not
// a recommendation. Measured on the real 48-layer Qwen4-preview artifact, independently reproduced
// twice (S2d): a real ~4.63 GiB peak-memory win over BF16, traded for a ~40-60% decode THROUGHPUT
// SLOWDOWN (most likely `Fp8CPtr::operator[]`'s multi-branch exponent-remap widen costing more
// per-element CPU than the DRAM bytes it saves -- unlike bf16's branchless shift) AND a markedly worse
// quality floor (logits vs F32 reference L2-relative ~0.43, vs BF16's own ~0.199). **BF16 (`--prec-param
// 1`) remains the recommended reduced-precision choice; FP8 is not a faster or better default, it is a
// smaller one, with a real and currently-unmitigated cost on both axes that matter more here.** Choose
// it only if the ~4.63 GiB footprint reduction is worth those two costs for your own use case, or if you
// are picking this format up specifically to pursue the named-but-unbuilt fix (a branchless/lookup-table
// `fp8_widen`, `fp8.hpp`'s own comment) -- not by default and not without reading S2d first.

#pragma once

#include "sub0_config.hpp"   // generated: Dtype, PARAM_DTYPE
#include "sub0/bf16.hpp"
#include "sub0/fp8.hpp"
#include "sub0/model_file.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace sub0 {

static_assert(PARAM_DTYPE == Dtype::F32 || PARAM_DTYPE == Dtype::BF16 || PARAM_DTYPE == Dtype::FP8,
              "PARAM_DTYPE selects the parameter arena's element type; only F32, BF16 and FP8 are wired "
              "(the generated Dtype enum reserves F16/Q8/Q4 for a different, block-quantized mechanism "
              "that B24 Phase 2's own measurement decisively ruled out for this project -- see "
              "docs/BACKBONE_PRECISION.md S2c/S2d)");

inline constexpr bool PARAM_BF16 = (PARAM_DTYPE == Dtype::BF16);
inline constexpr bool PARAM_FP8  = (PARAM_DTYPE == Dtype::FP8);

// The arena's element type, and the pointer a kernel reads it through. Under F32 both are exactly what
// the code said before B24 existed.
using param_t   = std::conditional_t<PARAM_BF16, bf16, std::conditional_t<PARAM_FP8, fp8, float>>;
using ParamCPtr = std::conditional_t<PARAM_BF16, Bf16CPtr, std::conditional_t<PARAM_FP8, Fp8CPtr, const float*>>;

inline constexpr int PARAM_ELEM_BYTES = static_cast<int>(sizeof(param_t));
static_assert(PARAM_ELEM_BYTES == (PARAM_BF16 ? 2 : (PARAM_FP8 ? 1 : 4)));

// This build's element type as the FILE format names it (model_file.hpp's ParamDtype). The two enums
// are deliberately separate -- one is a build's compute-precision vocabulary, the other is an on-disk
// tag a tool with no generated config still has to write -- and this is the single place they meet.
inline constexpr ParamDtype PARAM_FILE_DTYPE =
    PARAM_BF16 ? ParamDtype::BF16 : (PARAM_FP8 ? ParamDtype::FP8 : ParamDtype::F32);

// Make a readable pointer out of the arena base. Written as a pair of ORDINARY OVERLOADS on the
// concrete pointee type, not as `if constexpr (PARAM_BF16)` inside one non-template function --
// `if constexpr` only defers checking a branch that is itself DEPENDENT on a template parameter; a
// branch built entirely from already-concrete types (which `const param_t*`/`ParamCPtr` are, since
// `param_t` is a fixed alias, not a template parameter) is ordinary code and is type-checked whether or
// not it is the taken branch -- `Bf16CPtr{base}` does not convert to `ParamCPtr == const float*` in an
// F32 build, and BOTH branches of a plain (or bool-NTTP-"templated" but not otherwise dependent)
// function are checked regardless. Caught building the sub4 BF16/F32 pair (docs/BACKBONE_PRECISION.md's
// own correctness-gate build) -- overload resolution sidesteps the whole issue: each overload only ever
// needs to be valid for ITS OWN pointee type, and exactly one of the two is ever actually called in a
// given build, but both are always independently well-formed C++.
[[nodiscard]] inline Bf16CPtr    param_cptr(const bf16* base) noexcept { return Bf16CPtr{base}; }
[[nodiscard]] inline Fp8CPtr     param_cptr(const fp8* base) noexcept { return Fp8CPtr{base}; }
[[nodiscard]] inline const float* param_cptr(const float* base) noexcept { return base; }

// Read/write ONE stored parameter as a float. These are the seams the optimizer and the initializer
// use; in the hot path a kernel goes through ParamCPtr's own indexing instead, which is the same shift
// with no function-call shape around it. Overloaded for the same reason param_cptr is above.
[[nodiscard]] inline float param_get(const bf16* base, std::size_t i) noexcept { return to_f32(base[i]); }
[[nodiscard]] inline float param_get(const fp8* base, std::size_t i) noexcept { return to_f32(base[i]); }
[[nodiscard]] inline float param_get(const float* base, std::size_t i) noexcept { return base[i]; }
inline void param_set(bf16* base, std::size_t i, float v) noexcept { base[i] = f32_narrow(v); }
inline void param_set(fp8* base, std::size_t i, float v) noexcept { base[i] = fp8_narrow(v); }
inline void param_set(float* base, std::size_t i, float v) noexcept { base[i] = v; }

}  // namespace sub0
