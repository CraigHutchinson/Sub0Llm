// sub0/simd_reduce.hpp -- B34 (docs/INDEPENDENT_REVIEW_BACKLOG.md): a shared multi-accumulator
// horizontal-reduction primitive for the hot dot-product-shaped loops in moe_math.hpp/gdn_math.hpp/
// qsa_math.hpp/gated_residual_math.hpp.
//
// THE PROBLEM THIS FIXES. A plain scalar reduction (`float s=0; for(i) s += a[i]*b[i];`) does NOT
// auto-vectorize under Clang even with -O3 -march=native and AVX2 already enabled (SUB0_NATIVE=ON):
// Clang's own remark is "cannot prove it is safe to reorder floating-point operations" -- a strict
// left-to-right float sum is not associative, so the compiler cannot know that summing it out of order
// (e.g. 8 lanes at a time, combined at the end) produces an "equivalent" result, and refuses to guess.
// Confirmed by compiling the exact loop shape standalone (see B34's own filing, docs/INDEPENDENT_REVIEW_
// BACKLOG.md) and independently re-confirmed here (see this file's own verification note below).
//
// THE FIX. Restructure the single scalar accumulator into N INDEPENDENT named scalar accumulators (not
// an array -- an array risks the compiler treating it as memory rather than N registers, defeating the
// point), each summing every Nth element; each accumulator's own chain is still a strict left-to-right
// sum (so each LANE's own partial sum is deterministic and reproducible), but the N chains have no data
// dependency on each other. The N partial sums are then horizontally combined (itself a tiny, fixed-
// size, reassociated sum) once at the end.
//
// ONE THING THE MULTI-ACCUMULATOR RESTRUCTURING ALONE DOES NOT DO, measured directly (not assumed) while
// building this file: it is STILL not sufficient by itself for LLVM's LoopVectorizer to pack the loop's
// OWN multiple trips into one SIMD instruction, because vectorizing ACROSS loop iterations means
// combining several copies of each lane's own running total -- which is itself a reassociation of that
// lane's sequential chain, so the vectorizer raises the exact same "cannot prove it is safe to reorder
// floating-point operations" refusal one level down. Confirmed empirically: the plain multi-accumulator
// loop below, compiled standalone with no further hint, remained fully scalar (0 `ymm` registers in the
// generated assembly) -- restructuring the accumulators is necessary but not sufficient on its own. What
// closes the gap is `#pragma clang loop vectorize(enable)` on the loop itself: a standard, portable,
// per-loop compiler directive (not a raw intrinsic, and not the blanket `-ffast-math` the compiler's own
// remark suggests, which would also relax NaN/Inf/associativity guarantees for code that never asked for
// it) that tells the vectorizer "reordering this SPECIFIC loop's accumulation is acceptable" -- exactly
// true here, since every accumulator is already an independent reduction by construction and the whole
// POINT of this file is that reassociating it is fine. With the pragma added, the SAME multi-accumulator
// loop below vectorizes for real: `-Rpass=loop-vectorize` reports "vectorized loop (vectorization width:
// 8, interleaved count: 1)" and the generated assembly is full of `ymm` registers (239 occurrences in the
// standalone probe that produced this finding). This project's build is Clang-only (CMakePresets.json's
// `base` preset hardcodes `clang`/`clang++` unconditionally, no GCC/MSVC-native fallback), so a
// Clang-specific pragma costs nothing in portability THIS project actually has; raw AVX2 intrinsics were
// tried-for-necessity first per this task's own brief and turned out to be unnecessary once this was
// found -- the portable (for this project) multi-accumulator-plus-pragma C++ form gets full vectorization
// on its own.
//
// LANE COUNT: 8, this host's real AVX2 width for `float` (one `__m256` = 8 lanes). Arrow Lake-HX (this
// project's own host, docs/host-cpu-arrow-lake-hx.md) has NO AVX-512, only AVX2/AVX-VNNI, so 16 lanes
// would just be two half-empty ymm-worth of work; `constexpr` per AGENTS.md S1/S2's compile-time-over-
// runtime preference, not a runtime-detected width -- this project always builds where it runs (see
// CMakePresets.json's own "native" preset comment).
//
// NUMERICAL IMPACT -- NOT bit-exact, honestly. Grouping-then-combining changes the ORDER additions
// happen in (float addition is not associative), so the last bits of the result can differ from the old
// strict-sequential sum -- expected and accepted per this session's own B24/B31 tolerance-not-bit-exact
// gate for reassociating changes; see B34's own backlog entry for the real measured logit-diff impact
// against the pure-reordering-should-cause-ULP-scale-noise expectation.
//
// THE ZERO-SKIP BRANCH. Several callers (expert_ffn_row_source's gate/up/down dot products, the shared-
// expert gate logit) used to skip terms where one operand is exactly 0.f (a real, common case after
// SiLU-gated sparsity -- see docs/MOE.md). A per-lane branch inside a vectorized loop would defeat
// vectorization again, so `dot()` below is UNCONDITIONALLY branch-free: 0.f * finite = 0.f exactly under
// IEEE754 (the only case this would NOT hold is one operand being NaN or +-Inf, which trained weights
// and post-SiLU activations do not produce in this engine -- forward() already asserts/relies on
// finiteness elsewhere), so dropping the skip changes nothing mathematically, only removes a branch that
// was never needed for a plain multiply-accumulate. Measured (B34's own filing): this project's SwiGLU
// activations are NOT densely-enough zero for the skip to have been a meaningful cost-avoidance in the
// first place at these dims -- see the backlog entry for the real measured fraction.
//
// No heap allocation (AGENTS.md S1): pure stack scalars, no buffers.
//
// B38 INTEGRATION (docs/INDEPENDENT_REVIEW_BACKLOG.md B38): this header is unchanged from B34's own
// `feature/b34-simd-unlock` except for the `*_choice<UseSimd>()` dispatchers added at the bottom. They
// are the ONE seam the four kernel files key their `USE_SIMD_REDUCE` toggle through: each kernel calls
// `simd::dot_choice<UseSimd>(...)`/`sumsq_choice<UseSimd>(...)`/`sum_choice<UseSimd>(...)` instead of
// hand-inlining either form, so the kernel's own surrounding logic (loop bounds, gating, tiling) is
// written exactly once and only the innermost reduction differs, selected via `if constexpr` inside the
// dispatcher rather than at every call site. The `UseSimd=false` arm is a plain strict left-to-right
// scalar loop -- deliberately NOT calling `dot_seq()` below (which is a *different, correctness-pinned*
// primitive reserved for `expert_ffn_row_source`'s three call sites regardless of this flag, see its own
// comment) -- chosen to be byte-for-byte the same accumulation shape the ORIGINAL pre-B34 scalar loops in
// moe_math.hpp/gdn_math.hpp/qsa_math.hpp/gated_residual_math.hpp used, so `USE_SIMD_REDUCE=false`
// reproduces today's `main` exactly (verified via the decode hash, docs/INDEPENDENT_REVIEW_BACKLOG.md B38).

#pragma once

#include <cstddef>

namespace sub0::simd {

// This host's real AVX2 lane count for float (docs/host-cpu-arrow-lake-hx.md: Arrow Lake-HX, AVX2/
// AVX-VNNI only, no AVX-512). A compile-time fact of the target ISA, not a tunable.
inline constexpr int kLanes = 8;

// out[o] = sum_i a[i]*b[i], for i in [0,n). Branch-free (see this file's header comment on the zero-skip
// case). `a`/`b` need not be aligned -- Clang's own vectorizer emits unaligned loads for a bare `float*`
// with no alignment assumption, which is what every caller here has (rows of a caller-owned scratch/
// resident-parameter arena, no guaranteed over-alignment).
// Templated on the two operand "pointer-like" types rather than hardcoded to `const float*`: several
// callers in this project pass a WEIGHT pointer whose type is a param-dtype proxy (e.g. `Bf16CPtr`,
// include/sub0/param_store.hpp) rather than a raw `float*` (B24's own templating, threaded through every
// `*_math.hpp` kernel). `operator[]` on a raw pointer and on a proxy both work here; the pragma-assisted
// vectorization above is verified (this file's own header note) for the raw-`float*`-on-both-sides case,
// which is what every hot call site actually is (routed-expert dequantized planes, resident f32
// activations) -- a proxy-typed call still computes the CORRECT value either way, it may just not
// vectorize as well, which is a performance-only difference, never a correctness one.
template <class A, class B>
inline float dot(A a, B b, int n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
    int i = 0;
    const int n8 = n - (n % kLanes);
    #pragma clang loop vectorize(enable)
    for (; i < n8; i += kLanes) {
        s0 += a[i + 0] * b[i + 0];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
    }
    float s = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// Strict left-to-right sum_i a[i]*b[i] -- deliberately NOT reordered. Exists for exactly one reason,
// found and fixed while verifying this file's own impact end-to-end (B34, docs/INDEPENDENT_REVIEW_
// BACKLOG.md): `moe_math.hpp`'s `expert_ffn_row_source` (decode's fused MoE resolve, forward_one's own
// path) and `expert_ffn_row` (forward()'s batched/shared-expert path) compute the SAME per-output value
// two DIFFERENT ways -- one a per-output dot product against the GGUF source layout, the other a
// scatter-accumulate against the transposed layout -- and B31's own bit-exactness argument for that pair
// depends on both summing the same terms in the same (ascending-index) order. `expert_ffn_row`'s scatter
// form cannot be multi-accumulator-restructured without losing the INPUT-major cache locality it was
// deliberately built for (see its own header comment) -- converting it to an output-major reduction would
// reintroduce the exact strided-read regression that shape was written to avoid. So `expert_ffn_row_source`
// uses THIS strict-order primitive instead of `dot()` for its three per-output calls, preserving
// forward()/forward_one() bit-exact parity for the routed-expert MoE path specifically (verified 0 again
// on the real 48-layer artifact after switching to this). Every OTHER hot reduction in this project has no
// such twin-implementation constraint (rms_norm/attn/softmax/indexer/hc_norm/the shared-expert gate logit
// are each called through exactly ONE implementation from both forward() and forward_one()), so `dot()`'s
// reordering is safe there and stays. Same branch-free reasoning as `dot()` (0.f*finite=0.f, so dropping
// the zero-skip is a pure no-op) -- only the ACCUMULATION ORDER differs from `dot()`, not the addend set.
//
// B38: used UNCONDITIONALLY at `expert_ffn_row_source`'s three call sites, regardless of USE_SIMD_REDUCE
// -- this correctness-pinned pairing is never touched by the build-time reduction-strategy toggle.
inline float dot_seq(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}

// sum_i a[i]*a[i] -- the same primitive specialised for a self-dot (RMS/L2-norm sum-of-squares). A
// distinct name rather than always writing dot(a,a,n) at call sites so those reads stay self-documenting;
// implemented in terms of dot() so there is exactly one copy of the accumulator shape.
inline float sumsq(const float* a, int n) { return dot(a, a, n); }

// Plain sum_i a[i], the same multi-accumulator shape, for reductions that are not a product (e.g.
// softmax's normalizer).
inline float sum(const float* a, int n) {
    float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
    int i = 0;
    const int n8 = n - (n % kLanes);
    #pragma clang loop vectorize(enable)
    for (; i < n8; i += kLanes) {
        s0 += a[i + 0]; s1 += a[i + 1]; s2 += a[i + 2]; s3 += a[i + 3];
        s4 += a[i + 4]; s5 += a[i + 5]; s6 += a[i + 6]; s7 += a[i + 7];
    }
    float s = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for (; i < n; ++i) s += a[i];
    return s;
}

// ---- B38: USE_SIMD_REDUCE dispatchers -----------------------------------------------------------
//
// The single seam every kernel file's reduction call site goes through. `UseSimd` is a compile-time
// bool (each kernel is instantiated with the generated `sub0::USE_SIMD_REDUCE` as an explicit template
// argument at its call site in src/backends/cpu/{backend,decode}.cpp -- these `*_math.hpp` headers stay
// engine-free/closed over explicit Dims per their own long-standing convention, so they take the flag as
// a template parameter rather than reading a global config constant directly). The `if constexpr` below
// means only ONE arm's code is ever emitted per instantiation -- no runtime branch, per AGENTS.md S2.
//
// UseSimd=false reproduces the ORIGINAL pre-B34 scalar loop shape exactly (strict left-to-right,
// zero-skip branches preserved where main had them -- callers keep their own skip logic, these
// dispatchers are unconditionally branch-free only in the UseSimd=true arm, matching dot()/sum()'s own
// contract), so USE_SIMD_REDUCE=false is bit-exact with today's `main`.

template <bool UseSimd, class A, class B>
inline float dot_choice(A a, B b, int n) {
    if constexpr (UseSimd) {
        return dot(a, b, n);
    } else {
        float s = 0.f;
        for (int i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    }
}

template <bool UseSimd>
inline float sumsq_choice(const float* a, int n) {
    if constexpr (UseSimd) {
        return sumsq(a, n);
    } else {
        float s = 0.f;
        for (int i = 0; i < n; ++i) s += a[i] * a[i];
        return s;
    }
}

template <bool UseSimd>
inline float sum_choice(const float* a, int n) {
    if constexpr (UseSimd) {
        return sum(a, n);
    } else {
        float s = 0.f;
        for (int i = 0; i < n; ++i) s += a[i];
        return s;
    }
}

}  // namespace sub0::simd
