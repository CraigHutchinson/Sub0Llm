// sub0/backbone_quant_dot.hpp -- O5 phase 1: fused int8-activation x native-quant-weight dot products
// for the BACKBONE (attention/GDN/QSA projections, Gated Residual, router, shared expert, lm_head,
// embeddings), mirroring moe_quant_dot.hpp's B35 shape for the routed experts. Design brief:
// docs/BACKBONE_NATIVE_QUANT.md.
//
// **PHASE 1 ONLY -- ISOLATED, NOT WIRED IN.** Nothing in src/ or tools/ includes this header yet. It
// exists to be tested and benchmarked in isolation; a later phase decides HOW (or whether) to dispatch
// to it from op_linear/decode's own GEMV call sites, after this design is reviewed. See the design doc's
// own "what this phase deliberately does not do" section.
//
// WHY THIS IS NOT JUST moe_quant_dot.hpp AGAIN. B35's three formats (IQ1_S/IQ2_XXS/IQ4_NL) are all
// PURE-SCALE codebook/nibble formats: `value = scale * (q + delta)`, one scale term. The backbone's
// three real quantized formats (docs/BACKBONE_NATIVE_QUANT.md's own census) are K-quants, which are
// AFFINE, not pure-scale, for two of the three:
//   Q8_0            value = d * q                                   (pure scale, no min)
//   Q4_K / Q5_K     value = d*sc*q - dmin*m   (per-32-element sub-block scale AND min)
//   Q6_K            value = d*sc*(q - 32)     (per-16-element sub-block scale, zero-centered -- an
//                                               affine form too, since the "-32" is a per-element
//                                               constant every bit as much as Q5_K's "-dmin*m" is)
// An affine weight needs an affine DOT: sum_j x[j]*(scale*q[j] + bias) = scale*sum_j(x[j]*q[j]) +
// bias*sum_j(x[j]). The second term is exactly why moeqd::ActBlocks already carries `gsum` (B35 needed
// it for IQ1_S's own additive delta) -- this header reuses `ActBlocks` and `gsum` completely unchanged,
// per this project's own "check whether an existing field already discriminates it" discipline
// (AGENTS.md S3) applied to a design choice rather than a file format: no second quantized-activation
// type was invented, this reuses moeqd's.
//
// GROUP = moeqd::GROUP = 32, REUSED, NOT REDERIVED. Q4_K/Q5_K's own native sub-block width IS 32 (8
// sub-blocks of 32 per 256-element super-block -- confirmed against gguf.hpp's dequantize_q4_k/q5_k,
// which is itself the single already-verified transcription of ggml's own decoder, AGENTS.md S5), so
// those two formats' sub-block boundary and moeqd's activation-group boundary coincide exactly, with no
// splitting needed at all -- see Q4KPlane/Q5KPlane::group() below. Q6_K's own native sub-block is 16,
// HALF of GROUP: Q6KPlane::group() below still decodes a full 32-element span per call (so the shared
// row walk stays uniform across all four formats), but internally treats it as two 16-wide sub-blocks,
// each with its own (scale, bias) pair -- see WeightGroup's own comment for why this generalizes
// cleanly rather than forking the row walk. This was decided by re-deriving Q6_K's actual bit layout
// from gguf.hpp's dequantize_q6_k (its interleaved four-strips-per-half indexing), not assumed to be a
// simple halving -- docs/BACKBONE_NATIVE_QUANT.md section 3 works the derivation in full.
//
// LAYOUT. Same DOT-not-AXPY reasoning as moe_quant_dot.hpp: a GGUF tensor is [out][in], contiguous over
// `in` (ne0), so `gemv_rows` below computes output row r (r in [row_lo, row_hi)) as one contiguous dot
// over the row's `row_elems` = `in` elements -- the natural threading axis is OUTPUT ROWS, and
// `gemv_rows`'s own [row_lo, row_hi) parameters are exactly that split, ready for a caller to hand
// non-overlapping row ranges to independent worker threads with no shared mutable state between them.
//
// NO RAW INTRINSICS IN THE PORTABLE PATH, deliberately: integer addition has no float-style
// reassociation barrier, so a plain loop already vectorizes under -O3 -march=native. An explicit AVX2
// path is provided alongside it so the two can be checked against each other exactly (integer
// arithmetic, no tolerance) and timed side by side.
//
// PERFORMANCE STATUS (phase 2a, 2026-09-22): the Q4_K/Q5_K/Q6_K "portable"/"group()-based AVX2" kernels
// above are kept as the correctness REFERENCE (Q4KPlane/Q5KPlane/Q6KPlane::group(), gemv_rows) -- they
// are no longer the fast path. The fast path is `detail::dot_row_q4_k_avx2`/`q5_k_avx2`/`q6_k_avx2`
// further down this file: one 256-element superblock at a time, d/dmin/the 8 sub-block (sc,m) pairs
// decoded ONCE per superblock via a branch-free bit-unpack (llama.cpp's own `utmp[4]` trick, re-derived
// onto this project's per-sub-block-float shape, AGENTS.md S5) instead of 8 branchy `k_scale_min` calls,
// and the nibble/high-bit unpack done via explicit AVX2 intrinsics instead of relying on the compiler to
// auto-vectorize a scalar loop -- which, per this file's own S12a design-doc writeup, it does
// inconsistently: Clang vectorized Q5KPlane::group()'s more complex unpack but NOT Q4KPlane::group()'s
// simpler one, a confirmed compiler heuristic quirk (checked via generated assembly), not a Q4_K-specific
// problem, and the actual cause of the "Q4_K 4x slower than Q5_K" anomaly S8a first measured -- NOT
// denormal/FTZ stalls, which were tested directly and ruled out (docs/BACKBONE_NATIVE_QUANT.md S12a).
//
// MEASURED RESULT (docs/BACKBONE_NATIVE_QUANT.md S12d-f, real Qwen3.8-Flash-Next shards, DRAM-streamed,
// this project's Arrow Lake-HX host): three optimization passes took Q4_K from phase 1's 1.1 GB/s/thread
// to ~10 GB/s, Q5_K from 4.3 to ~10, Q6_K from 5.3 to ~8, Q8_0 stayed ~13 -- roughly AT, not clearly past,
// the ~10 GB/s/thread compressed-byte gate (within this host's own +/-10-15% run-to-run noise), and NONE
// of the four formats reach the ~60 GB/s/8-thread aggregate gate (28-36 GB/s measured). The honest,
// complete picture is not "gate met" -- but it is not "gate missed" either: at 8 threads the streaming
// kernels beat the REAL `gemv::axpy` bf16 kernel on WALL-CLOCK time per output row for all four formats
// (15-93% faster) BECAUSE native bytes read are 1.9-3.6x fewer even though native's own GB/s is lower;
// at 1 thread this holds for Q4_K/Q5_K but not Q8_0/Q6_K. S12g names the concrete architectural reason
// (8 independent horizontal reductions per superblock, a consequence of keeping ActBlocks' per-32 float
// activation scale rather than adopting llama.cpp's own per-256 Q8_K-style scheme) and the 4th-pass lever
// this analysis found, not yet attempted. Parked per AGENTS.md S13, not reverted.
//
// NO HEAP ALLOCATION PER CALL (AGENTS.md S1): every kernel here reads only its caller-supplied spans and
// writes only its caller-supplied `out` pointer; no std::vector/new/malloc appears in any hot path below.

#pragma once

#include "sub0/gguf.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>

// --- O5 phase 2b-3 phase A: bbqd::Plane, defined BEFORE the moe_quant_dot.hpp include below -----------
//
// WHY THIS IS SPLIT OUT OF THE MAIN `namespace sub0::bbqd` BLOCK FURTHER DOWN. Plane is a pure POD view
// (four fields, no dependency on ActBlocks/moeqd at all -- see its own doc comment). Every `*_math.hpp`
// header (gdn_math.hpp, gated_residual_math.hpp, qsa_math.hpp, moe_math.hpp) needs it to stay
// storage-agnostic. moe_math.hpp is ALSO a transitive dependency of `#include "sub0/moe_quant_dot.hpp"`
// just below (moe_quant_dot.hpp includes moe_math.hpp for `moe::Dims`) -- so if Plane were declared only
// in the main block AFTER that include, a build with moe_math.hpp's own native-quant wiring (phase B)
// would form a real cycle: backbone_quant_dot.hpp -> moe_quant_dot.hpp -> moe_math.hpp ->
// backbone_quant_dot.hpp (for Plane), reaching moe_math.hpp's own `bbqd::Plane` use before this header's
// own namespace block had ever been opened. Declaring Plane here, before that include, breaks the cycle
// without touching moe_quant_dot.hpp (out of scope per this pass's own brief) or moe_math.hpp's own
// otherwise-natural single `#include "sub0/backbone_quant_dot.hpp"`.
namespace sub0::bbqd {

/** A format-agnostic view of one weight plane's bytes -- decouples the `*_math.hpp` headers (gdn_math.hpp,
 * gated_residual_math.hpp, qsa_math.hpp, moe_math.hpp) from the S0B1 sidecar's own storage format
 * (backbone_quant.hpp's `bbq::Desc`). Built once by the caller (decode's resolved-role table, O5 phase A)
 * from whatever storage format actually backs it; the math headers only ever see this. See `gemv_plane`'s
 * own Plane overload, further down this file, for the one seam every math header calls through.
 */
struct Plane {
    std::uint32_t                 type_raw  = 0;
    int                            n_rows    = 0;   ///< bbqd's own `n_rows` (the GEMV's output width)
    int                            row_elems = 0;   ///< bbqd's own `row_elems` (the GEMV's contraction width)
    std::span<const std::uint8_t> bytes{};          ///< the plane's own encoded payload, exactly as gemv_plane wants
};

}  // namespace sub0::bbqd

#include "sub0/moe_quant_dot.hpp"   // moeqd::ActBlocks, moeqd::GROUP -- reused verbatim, not re-derived

#if defined(__AVX2__)
#include <immintrin.h>
#define SUB0_BBQD_AVX2 1
#endif
// AVX-VNNI (`vpdpbusd`, VEX-encoded -- distinct from the AVX512VNNI form): this host (Arrow Lake-HX) has
// it without AVX-512 (confirmed via `clang++ -march=native -dM -E`, docs/BACKBONE_NATIVE_QUANT.md S14).
// `<immintrin.h>` pulls in `avxvnniintrin.h` unconditionally; the `_avx_epi32`-suffixed intrinsics it
// declares are gated by the `__AVXVNNI__` target-feature macro at the call site, not by a separate header
// guard, so this project's own SUB0_BBQD_VNNI mirrors that same macro rather than inventing a new test.
#if defined(__AVX2__) && defined(__AVXVNNI__)
#define SUB0_BBQD_VNNI 1
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace sub0::bbqd {

using moeqd::ActBlocks;
inline constexpr int GROUP = moeqd::GROUP;   // 32 -- see the file header for why this is the right width

/** Whether gemv_plane runs the streaming AVX2 kernels (detail::gemv_plane_avx2) or the fully-portable
 * ones (detail::gemv_plane_portable). Same convention as moeqd::kAvx2Kernels (moe_quant_dot.hpp) --
 * decided by the compiler's target ISA, which a SUB0_NATIVE build fixes at -march=native, not a runtime
 * or configurator knob: where AVX2 exists the vector kernel is strictly better (bit-exact against the
 * portable path, checked in tests/backbone_quant_dot_tests.cpp), so there is no choice to make at
 * runtime. The portable path is kept as the non-AVX2 fallback AND as the AVX2 kernels' own correctness
 * reference.
 */
#if defined(SUB0_BBQD_AVX2)
inline constexpr bool kAvx2Kernels = true;
#else
inline constexpr bool kAvx2Kernels = false;
#endif

/** Whether this translation unit was built with AVX-VNNI (`vpdpbusd`) available -- this host has it
 * without AVX-512 (docs/BACKBONE_NATIVE_QUANT.md S14's own `-march=native` macro dump). Records the ISA
 * fact for callers/tests (`detail::gemv_plane_super_vnni` only exists to call when this is true); it does
 * NOT drive `gemv_plane_super`'s own dispatch (see `detail::gemv_plane_super_dispatch`'s own comment) --
 * pass 4's own measurement found the VNNI kernels at rough PARITY with the plain-AVX2 ones here
 * (0.87x-1.09x across four runs, all three formats, mean ~0.95x -- a wash, not a clear win), so
 * availability and "is the default path" are deliberately two different questions here, unlike
 * `kAvx2Kernels` one ISA tier down.
 */
#if defined(SUB0_BBQD_VNNI)
inline constexpr bool kVnniKernels = true;
#else
inline constexpr bool kVnniKernels = false;
#endif

// --- one decoded 32-wide weight span, and the affine terms that fill it -----------------------------

/** GROUP consecutive weights of one row, decoded into the two-term affine form every format here needs:
 * `value[j] = scale_lo*q[j] + bias_lo` for j in [0,16) and `value[j] = scale_hi*q[j] + bias_hi` for
 * j in [16,32).
 *
 * A SINGLE (scale, bias) pair would suffice for Q8_0/Q4_K/Q5_K, whose native sub-block IS 32 wide (the
 * file header's own derivation) -- those three formats' unpackers below simply set `scale_hi=scale_lo`,
 * `bias_hi=bias_lo`. Q6_K's native sub-block is 16 wide, so it is the one format that actually needs two
 * distinct pairs within a single 32-wide span. Carrying two pairs unconditionally (rather than
 * specializing the shared row walk per format) keeps `gemv_rows` below written ONCE for all four
 * formats, at the cost of two redundant float reads for the three formats that do not need them -- a
 * cost this header's own microbenchmark measures rather than assumes is negligible.
 */
struct WeightGroup {
    std::array<std::int8_t, GROUP> q{};                        ///< the group's 32 decoded (signed) quants
    float scale_lo = 0.f, bias_lo = 0.f;                        ///< applies to q[0..16)
    float scale_hi = 0.f, bias_hi = 0.f;                        ///< applies to q[16..32)
};

// --- pass 4 (AGENTS.md S13, docs/BACKBONE_NATIVE_QUANT.md S12g/S14): a per-256 activation superblock ---
// --- for the K-quant kernels, so each superblock's per-sub-block INTEGER weight scale can fold into ----
// --- the integer accumulator the way llama.cpp's own block_q8_K-based kernels do -------------------------

/** Per-256-element ("superblock") quantized activation row, mirroring llama.cpp's own `block_q8_K`
 * (`D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-common.h`) field for field: one float scale per 256
 * elements (moeqd::ActBlocks carries one per 32), one int8 quant per element, and one int16 sum per
 * 16-element sub-block (`bsums`, llama.cpp's own name -- moeqd::ActBlocks' own `gsum` is per-32).
 *
 * WHY THIS TYPE EXISTS, NOT A CHANGE TO ActBlocks (the task's own framing, and AGENTS.md S3's "check
 * whether an existing field already discriminates it" cuts the OTHER way here): ActBlocks' per-32 float
 * scale is exactly what forecloses folding Q4_K/Q5_K/Q6_K's own per-sub-block scale into the integer
 * domain (docs/BACKBONE_NATIVE_QUANT.md S12g) -- a per-256 scale is what makes the fold possible, at the
 * cost of coarser activation quantization (measured in this header's own tests, and in the design doc's
 * own S13). Q8_0 keeps `ActBlocks` unchanged: its native block already IS 32 wide, matching ActBlocks'
 * own granularity exactly, so a coarser per-256 scale would only be WORSE for it with nothing to fold
 * (Q8_0 has no per-sub-block affine scale in the first place -- one `d` per 32 already is its whole
 * story). Used only by `dot_row_q{4,5,6}_k_super_*` below.
 *
 * @note Caller-owned and sized once (AGENTS.md S1), the same contract as `moeqd::ActBlocks::quantize`:
 *       allocates only when the width changes, so the steady state is allocation-free.
 */
struct ActSuper {
    std::vector<std::int8_t>  qs;      ///< n quantized values
    std::vector<float>        d;       ///< n/256 per-superblock scales
    std::vector<std::int16_t> bsums;   ///< n/16 per-16-element sums (llama.cpp's own `block_q8_K::bsums`)
    int                       n = 0;   ///< element count this instance is currently sized for

    /** Symmetric round-to-nearest int8 quantization, one scale per 256 elements -- re-derived onto this
     * project's caller-owned/reused-buffer convention from llama.cpp's own `quantize_row_q8_K_ref`
     * shape (AGENTS.md S5), not copied: the ROUNDING rule (divide by 127, not 128, so -128 never occurs)
     * matches `ActBlocks::quantize`'s own choice. The AVX2 K-quant path sums at most two unsigned
     * weight quants of 63 times signed activations of 127 in each 16-bit lane: 2*63*127 = 16002.
     * The VNNI path accumulates its products into 32-bit lanes.
     *
     * @note `n_elems` must be a multiple of 256 -- guaranteed by `super_fusable()` at the one call site.
     */
    void quantize(const float* x, int n_elems) {
        n = n_elems;
        const auto elems  = static_cast<std::size_t>(n_elems);
        const auto supers = static_cast<std::size_t>(n_elems / 256);
        const auto subs   = static_cast<std::size_t>(n_elems / 16);
        if (qs.size() != elems)    qs.assign(elems, 0);
        if (d.size() != supers)    d.assign(supers, 0.f);
        if (bsums.size() != subs)  bsums.assign(subs, 0);
        for (int s = 0, ns = n_elems / 256; s < ns; ++s) {
            const float* xs = x + static_cast<std::size_t>(s) * 256;
            float amax = 0.f;
            for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(xs[j]));
            const float inv = (amax > 0.f) ? (127.f / amax) : 0.f;
            std::int8_t* qsp = qs.data() + static_cast<std::size_t>(s) * 256;
            for (int sub = 0; sub < 16; ++sub) {
                std::int32_t sum = 0;
                for (int j = 0; j < 16; ++j) {
                    const int idx = sub * 16 + j;
                    const int q = std::clamp(static_cast<int>(std::lrintf(xs[idx] * inv)), -127, 127);
                    qsp[idx] = static_cast<std::int8_t>(q);
                    sum += q;
                }
                bsums[static_cast<std::size_t>(s) * 16 + static_cast<std::size_t>(sub)] =
                    static_cast<std::int16_t>(sum);
            }
            d[static_cast<std::size_t>(s)] = amax / 127.f;
        }
    }

    [[nodiscard]] bool ready_for(int row_elems) const {
        return row_elems > 0 && row_elems % 256 == 0 && n == row_elems &&
               qs.size() == static_cast<std::size_t>(row_elems) &&
               d.size() == static_cast<std::size_t>(row_elems / 256) &&
               bsums.size() == static_cast<std::size_t>(row_elems / 16);
    }
};

// --- the shared integer primitives, portable and AVX2 -------------------------------------------------

namespace detail {

/** Exact n-wide signed int8 dot product, n in {16, 32}. Portable: no compiler hints, no ISA assumption
 * -- this is the correctness REFERENCE every other path in this header is checked against, not an
 * optimization target itself. (moe_quant_dot.hpp's own dot_group found that a plain int8 loop already
 * vectorizes under -O3 -march=native; that remains true here too, but this specific function is kept
 * deliberately plain so it stays a trustworthy ground truth independent of what any given compiler does
 * with it on a given day.)
 */
[[nodiscard]] inline std::int32_t dot_n_portable(const std::int8_t* w, const std::int8_t* x, int n) {
    std::int32_t s = 0;
    for (int j = 0; j < n; ++j) s += static_cast<std::int32_t>(w[j]) * static_cast<std::int32_t>(x[j]);
    return s;
}

/** sum_j x[j] over n in {16, 32} -- the term an affine format's `bias` needs (see WeightGroup's own
 * comment). Portable reference, same reasoning as dot_n_portable.
 */
[[nodiscard]] inline std::int32_t sum_n_portable(const std::int8_t* x, int n) {
    std::int32_t s = 0;
    for (int j = 0; j < n; ++j) s += static_cast<std::int32_t>(x[j]);
    return s;
}

#if defined(__AVX2__)
/** Horizontal sum of the 8 int32 lanes of `v`. */
[[nodiscard]] inline std::int32_t hsum256_epi32(__m256i v) {
    const __m128i lo = _mm256_castsi256_si128(v);
    const __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/** 32-wide signed int8 dot product, explicit AVX2: sign-extend both 16-byte halves of each operand to
 * int16 (`_mm256_cvtepi8_epi16`), multiply-add adjacent pairs into int32 (`_mm256_madd_epi16` -- exact,
 * since |int8|*|int8| <= 127*127 fits an int16 product with headroom, and the pairwise add cannot
 * overflow int32 either), then horizontally reduce. This is the genuinely distinct "AVX2 path" this
 * header's own task brief asks for, separate from (not a replacement for) moe_quant_dot.hpp's own
 * "portable loop already vectorizes" finding -- see the file header comment.
 */
[[nodiscard]] inline std::int32_t dot32_avx2(const std::int8_t* w, const std::int8_t* x) {
    const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w));
    const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
    const __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw, 1));
    const __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
    const __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
    const __m256i p_lo = _mm256_madd_epi16(w_lo, x_lo);
    const __m256i p_hi = _mm256_madd_epi16(w_hi, x_hi);
    return hsum256_epi32(_mm256_add_epi32(p_lo, p_hi));
}

/** 16-wide signed int8 dot product, explicit AVX2/SSE4.1. `_mm_cvtepi8_epi16` sign-extends only the
 * LOW 8 bytes of its 128-bit input (its output is itself a full 128-bit register of 8 int16 lanes, so it
 * has no room for the other 8) -- a first version of this function loaded all 16 bytes but converted only
 * that low half, silently computing an 8-wide dot instead of 16-wide (caught by this header's own AVX2-
 * vs-portable differential test, not by inspection). Fixed by converting the high 8 bytes separately via
 * `_mm_srli_si128(v, 8)` (shift the upper half down into position) before its own `_mm_cvtepi8_epi16`.
 * SSE4.1 is always present alongside AVX2 on every x86-64 target this project builds for.
 */
[[nodiscard]] inline std::int32_t dot16_avx2(const std::int8_t* w, const std::int8_t* x) {
    const __m128i vw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w));
    const __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
    const __m128i w_lo = _mm_cvtepi8_epi16(vw);
    const __m128i x_lo = _mm_cvtepi8_epi16(vx);
    const __m128i w_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vw, 8));
    const __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    const __m128i p = _mm_add_epi32(_mm_madd_epi16(w_lo, x_lo), _mm_madd_epi16(w_hi, x_hi));
    __m128i s = _mm_add_epi32(p, _mm_shuffle_epi32(p, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/** sum_j x[j] over n in {16, 32}, explicit AVX2: sign-extend to int16, dot against an all-ones vector
 * (so `madd` produces the pairwise sum), horizontally reduce. Kept as a real vector reduction rather
 * than falling back to sum_n_portable, so the AVX2 arm's own timing in the microbenchmark reflects an
 * all-vector implementation, not a mixed one.
 */
[[nodiscard]] inline std::int32_t sum32_avx2(const std::int8_t* x) {
    const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
    const __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
    const __m256i p = _mm256_add_epi32(_mm256_madd_epi16(x_lo, ones), _mm256_madd_epi16(x_hi, ones));
    return hsum256_epi32(p);
}
[[nodiscard]] inline std::int32_t sum16_avx2(const std::int8_t* x) {
    const __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i x_lo = _mm_cvtepi8_epi16(vx);
    const __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));   // see dot16_avx2's own comment
    const __m128i p = _mm_add_epi32(_mm_madd_epi16(x_lo, ones), _mm_madd_epi16(x_hi, ones));
    __m128i s = _mm_add_epi32(p, _mm_shuffle_epi32(p, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtsi128_si32(s);
}
#endif  // __AVX2__

/// True only when this translation unit was actually compiled with AVX2 enabled -- callers use this to
/// decide whether requesting the AVX2 path is even meaningful, rather than getting a hard compile error
/// on a non-AVX2 build (SUB0_NATIVE=OFF). See gemv_rows<UseAvx2>'s own static_assert.
inline constexpr bool kHasAvx2 =
#if defined(__AVX2__)
    true;
#else
    false;
#endif

/// True only when this translation unit was compiled with AVX-VNNI enabled -- pass 4's own kernel
/// selection fact (docs/BACKBONE_NATIVE_QUANT.md S14), same convention as kHasAvx2.
inline constexpr bool kHasVnni =
#if defined(SUB0_BBQD_VNNI)
    true;
#else
    false;
#endif

}  // namespace detail

// --- per-format unpackers: EXACTLY the affine terms gguf.hpp's own (already S5-verified) scalar --------
// --- decoders compute, re-derived to a per-GROUP-span accessor rather than a whole-tensor walk ---------

/** Q8_0: `value = d*q`, one 34-byte block of 32 int8 quants -- pure scale, no bias. `row_elems % 32 ==
 * 0` is required (checked once by fusable(), never per group).
 */
struct Q8_0Plane {
    static constexpr int kSubW = 32;
    const std::uint8_t* plane = nullptr;   // non-owning; caller-owned bytes outlive every call

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 32) * 34;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk, sizeof d_bits);
        const float d = gguf::f16_to_f32(d_bits);
        WeightGroup wg;
        std::memcpy(wg.q.data(), blk + 2, GROUP);
        wg.scale_lo = wg.scale_hi = d;
        wg.bias_lo = wg.bias_hi = 0.f;
        return wg;
    }
};

/** Q4_K: `value = d*sc*q - dmin*m`, 8 sub-blocks of 32 per 256-element super-block, `sc`/`m` unpacked by
 * gguf::k_scale_min (reused verbatim -- AGENTS.md S3, one definition of the bit-packing). `row_elems %
 * 32 == 0` required; a 32-aligned `p` always lands on exactly one whole sub-block, so a single (scale,
 * bias) pair covers the whole group -- see the file header's own derivation.
 */
struct Q4KPlane {
    static constexpr int kSubW = 32;
    const std::uint8_t* plane = nullptr;   // non-owning; caller-owned bytes outlive every call

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 256) * 144;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        const float d = gguf::f16_to_f32(d_bits), dmin = gguf::f16_to_f32(dmin_bits);
        const std::uint8_t* scales = blk + 4;
        const int sub = static_cast<int>((p % 256) / 32);          // which of the 8 sub-blocks, 0..7
        const int is = sub;                                        // k_scale_min's own sub-block index
        const bool hi_nibble = (sub % 2) == 1;                     // odd sub-block reads the high nibble
        const std::uint8_t* ql = blk + 16 + (sub / 2) * 32;        // ql advances by 32 every TWO sub-blocks
        std::uint8_t sc = 0, m = 0;
        gguf::k_scale_min(is, scales, sc, m);
        WeightGroup wg;
        for (int l = 0; l < GROUP; ++l)
            wg.q[static_cast<std::size_t>(l)] =
                static_cast<std::int8_t>(hi_nibble ? (ql[l] >> 4) : (ql[l] & 0xF));
        wg.scale_lo = wg.scale_hi = d * static_cast<float>(sc);
        wg.bias_lo = wg.bias_hi = -dmin * static_cast<float>(m);
        return wg;
    }
};

/** Q5_K: identical to Q4_K's affine shape plus a per-element high bit from `qh` (`+16` when set) --
 * gguf.hpp's dequantize_q5_k is the reference this was re-derived from (AGENTS.md S5). Same 32-aligned/
 * single-sub-block-per-group property as Q4_K.
 */
struct Q5KPlane {
    static constexpr int kSubW = 32;
    const std::uint8_t* plane = nullptr;   // non-owning; caller-owned bytes outlive every call

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 256) * 176;
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        const float d = gguf::f16_to_f32(d_bits), dmin = gguf::f16_to_f32(dmin_bits);
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* qh = blk + 16;                         // 32 bytes, one high bit per element
        const int sub = static_cast<int>((p % 256) / 32);          // 0..7
        const int is = sub;
        const bool hi_nibble = (sub % 2) == 1;
        const std::uint8_t* ql = blk + 48 + (sub / 2) * 32;        // 128 bytes total, 32 per TWO sub-blocks
        // qh's own bit index advances by 2 every TWO sub-blocks (u1=1,u2=2 initially, <<=2 per pair --
        // gguf.hpp's own u1/u2 walk, re-expressed as a closed form for a single sub-block index).
        const std::uint8_t bit = static_cast<std::uint8_t>((hi_nibble ? 2 : 1) << (2 * (sub / 2)));
        std::uint8_t sc = 0, m = 0;
        gguf::k_scale_min(is, scales, sc, m);
        WeightGroup wg;
        for (int l = 0; l < GROUP; ++l) {
            const int nib = hi_nibble ? (ql[l] >> 4) : (ql[l] & 0xF);
            const int hi = (qh[l] & bit) ? 16 : 0;
            wg.q[static_cast<std::size_t>(l)] = static_cast<std::int8_t>(nib + hi);
        }
        wg.scale_lo = wg.scale_hi = d * static_cast<float>(sc);
        wg.bias_lo = wg.bias_hi = -dmin * static_cast<float>(m);
        return wg;
    }
};

/** Q6_K: `value = d*sc*(q-32)`, zero-centered, 16-element sub-blocks -- HALF of GROUP, so a single
 * group() call covers two sub-blocks with two independent (scale, bias) pairs. Re-derived from
 * gguf.hpp's dequantize_q6_k, whose own layout is genuinely interleaved (four 32-wide "strips" per
 * 128-element half, each strip's own elements y0+l/y0+l+32/y0+l+64/y0+l+96 for l in [0,32)) -- the file
 * header's own comment works the inversion from a 32-aligned contiguous output span back to (half,
 * strip, l) once; this is that derivation, implemented.
 *
 * A 32-aligned contiguous span of 32 output elements lands entirely within ONE (half, strip) pair (the
 * four strips partition each half into four disjoint contiguous 32-runs) -- checked, not assumed, by
 * the fact that gguf.hpp's own y0+l+32*strip indexing is itself strip-contiguous in l.
 */
struct Q6KPlane {
    static constexpr int kSubW = 16;
    const std::uint8_t* plane = nullptr;   // non-owning; caller-owned bytes outlive every call

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 256) * 210;
        const int e = static_cast<int>(p % 256);
        const int half = e / 128;
        const int e_in_half = e % 128;
        const int strip = e_in_half / 32;                          // 0..3
        const bool hi_nibble = (strip == 2 || strip == 3);
        const int ql_off = (strip == 1 || strip == 3) ? 32 : 0;
        const int shift = strip * 2;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = gguf::f16_to_f32(d_bits);
        const std::uint8_t* ql = blk + half * 64 + ql_off;          // 64 bytes/half, offset by strip parity
        const std::uint8_t* qh = blk + 128 + half * 32;             // 32 bytes/half
        const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);   // 8 int8/half
        // wg.q stores the RAW unsigned 6-bit code (0..63), NOT (code-32) -- the affine "-32" belongs
        // in `bias` alone. The two must not both carry it: an integer dot against x sums
        // `q_raw[j]*x[j]`, and the algebraic identity this whole header relies on is
        // `sum_j (q_raw[j]-32)*x[j] = sum_j q_raw[j]*x[j] - 32*sum_j x[j]` -- i.e. `isum - 32*gsum`,
        // which is exactly `wg.scale*isum + wg.bias*gsum` for `bias = -32*scale`. Baking `-32` into
        // `wg.q` AS WELL would apply it twice (caught by this header's own lossless-decode test: every
        // element came out offset by exactly `bias` before this was fixed).
        WeightGroup wg;
        for (int l = 0; l < GROUP; ++l) {
            const int nib = hi_nibble ? (ql[l] >> 4) : (ql[l] & 0xF);
            const int hi = (qh[l] >> shift) & 3;
            wg.q[static_cast<std::size_t>(l)] = static_cast<std::int8_t>(nib | (hi << 4));   // 0..63
        }
        wg.scale_lo = d * static_cast<float>(sc[2 * strip + 0]);
        wg.bias_lo  = -32.f * wg.scale_lo;
        wg.scale_hi = d * static_cast<float>(sc[2 * strip + 1]);
        wg.bias_hi  = -32.f * wg.scale_hi;
        return wg;
    }
};

// --- the shared row walk: one body, four unpackers, two integer-primitive choices --------------------

/** Can this GGML type/row-width pair be fused at all? Checked once per plane (fusable) or once per
 * dispatch call (gemv_plane), never per row/group.
 */
[[nodiscard]] inline bool fusable(std::uint32_t type_raw, int row_elems) {
    if (row_elems <= 0 || row_elems % GROUP != 0) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q8_0:
        case gguf::TensorType::Q4_K:
        case gguf::TensorType::Q5_K:
        case gguf::TensorType::Q6_K: return true;
        default:                     return false;
    }
}

/** Encoded byte length of `n_rows` rows of `row_elems` in `type_raw`, or 0 if unknown/misaligned. Same
 * contract as moeqd::plane_bytes -- delegated to gguf::block_spec, the ONE size table (AGENTS.md S3).
 */
[[nodiscard]] inline std::uint64_t plane_bytes(std::uint32_t type_raw, std::uint64_t elems) {
    const gguf::BlockSpec spec = gguf::block_spec(type_raw);
    if (spec.elems == 0 || elems % spec.elems != 0) return 0;
    return elems / spec.elems * spec.bytes;
}

/** One plane's GEMV over output rows [row_lo, row_hi): `out[r-row_lo] = sum_i plane[r*row_elems+i] *
 * x[i]`. This is the threading seam -- disjoint [row_lo, row_hi) ranges handed to independent worker
 * threads read disjoint regions of `plane` and write disjoint regions of `out`, with no shared mutable
 * state, so the result is bit-identical regardless of how many threads split the row range (the same
 * "threaded by output row, bit-exact across thread counts" property the design doc requires).
 *
 * `UseAvx2` is a compile-time choice, not a runtime CPU-feature branch (AGENTS.md S2) -- the caller
 * (a test or the microbenchmark, in this phase) selects it explicitly; a later wiring phase would bake
 * it from the same SUB0_NATIVE/-march=native build-time fact every other SIMD path here already uses.
 */
template <class Plane, bool UseAvx2>
inline void gemv_rows(const Plane& plane, int row_lo, int row_hi, int row_elems, const ActBlocks& x,
                       float* out) {
    static_assert(!UseAvx2 || detail::kHasAvx2,
                  "UseAvx2=true requires this translation unit to be compiled with AVX2 enabled "
                  "(SUB0_NATIVE=ON on this project's x86-64 hosts)");
    const int ng = row_elems / GROUP;
    for (int r = row_lo; r < row_hi; ++r) {
        const std::uint64_t base = static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems);
        float acc = 0.f;
        for (int g = 0; g < ng; ++g) {
            const std::size_t gi = static_cast<std::size_t>(g);
            const WeightGroup wg = plane.group(base + static_cast<std::uint64_t>(g) * GROUP);
            const std::int8_t* xq = x.qs.data() + gi * GROUP;
            if constexpr (Plane::kSubW == GROUP) {
                std::int32_t isum, gsum;
                if constexpr (UseAvx2) {
#if defined(__AVX2__)
                    isum = detail::dot32_avx2(wg.q.data(), xq);
#else
                    isum = 0;   // unreachable: static_assert above refuses this instantiation
#endif
                } else {
                    isum = detail::dot_n_portable(wg.q.data(), xq, GROUP);
                }
                gsum = x.gsum[gi];
                acc += x.scale[gi] * (wg.scale_lo * static_cast<float>(isum)
                                       + wg.bias_lo * static_cast<float>(gsum));
            } else {
                static_assert(Plane::kSubW * 2 == GROUP, "only a whole-group or half-group sub-block "
                                                          "width is supported");
                std::int32_t isum_lo, isum_hi, gsum_lo, gsum_hi;
                if constexpr (UseAvx2) {
#if defined(__AVX2__)
                    isum_lo = detail::dot16_avx2(wg.q.data(), xq);
                    isum_hi = detail::dot16_avx2(wg.q.data() + Plane::kSubW, xq + Plane::kSubW);
                    gsum_lo = detail::sum16_avx2(xq);
                    gsum_hi = detail::sum16_avx2(xq + Plane::kSubW);
#else
                    isum_lo = isum_hi = gsum_lo = gsum_hi = 0;   // unreachable
#endif
                } else {
                    isum_lo = detail::dot_n_portable(wg.q.data(), xq, Plane::kSubW);
                    isum_hi = detail::dot_n_portable(wg.q.data() + Plane::kSubW, xq + Plane::kSubW,
                                                      Plane::kSubW);
                    gsum_lo = detail::sum_n_portable(xq, Plane::kSubW);
                    gsum_hi = detail::sum_n_portable(xq + Plane::kSubW, Plane::kSubW);
                }
                acc += x.scale[gi] * (wg.scale_lo * static_cast<float>(isum_lo)
                                       + wg.bias_lo * static_cast<float>(gsum_lo)
                                       + wg.scale_hi * static_cast<float>(isum_hi)
                                       + wg.bias_hi * static_cast<float>(gsum_hi));
            }
        }
        out[static_cast<std::size_t>(r - row_lo)] = acc;
    }
}

/** Bounds/geometry validation shared by every entry point below, so "refused means untouched" holds
 * BEFORE any row is written -- checked once, up front, rather than redundantly inside a per-thread split
 * (which would let earlier threads write real output while a later thread's own range turned out to be
 * the one that was actually out of bounds; see gemv_plane<Threads>'s own comment).
 */
[[nodiscard]] inline bool plane_geometry_ok(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                            int n_rows, int row_elems, int row_lo, int row_hi) {
    if (row_lo < 0 || row_hi > n_rows || row_lo > row_hi) return false;
    const std::uint64_t need = plane_bytes(type_raw, static_cast<std::uint64_t>(n_rows)
                                                      * static_cast<std::uint64_t>(row_elems));
    return need != 0 && raw.size() >= need;
}

// =========================================================================================================
// Phase 2a -- streaming AVX2 kernels: one superblock at a time, scale/min hoisted once per 256 elements,
// nibble/high-bit unpack via explicit intrinsics rather than relying on the compiler to auto-vectorize a
// scalar loop (docs/BACKBONE_NATIVE_QUANT.md S12a: confirmed via generated assembly that Clang does this
// INCONSISTENTLY across near-identical unpack loops -- not something a caller can rely on).
//
// PRECONDITION, gated by the caller (gemv_plane_avx2 below), not asserted per-call: `row_elems % 256 ==
// 0`. True for every real backbone tensor these formats appear in (docs/BACKBONE_NATIVE_QUANT.md S2a's
// census: 2560/6144/10240/12288 all divide by 256); a hypothetical future tensor that does NOT divide
// evenly falls back to the portable-geometry `gemv_rows<Plane,true>` path instead, which handles any
// row_elems that is merely a multiple of GROUP=32 (fusable()'s own, weaker contract).
// =========================================================================================================

/** Q6_K's per-16-element activation sums (its `bias_lo`/`bias_hi` split needs a finer granularity than
 * `moeqd::ActBlocks::gsum`'s own per-32). Phase 1's `group()`-based path recomputed this from the raw
 * activation bytes on EVERY (row, group) pair via `sum16_avx2` -- an O(n_rows) redundant recompute of a
 * value that depends only on the activation column, never the row. `gemv_plane<Threads>` builds it ONCE
 * per call (i.e. once per activation row, shared across however many output rows that call handles and
 * however many threads split them) instead.
 *
 * O5 phase 2b-2b: PUBLIC (moved out of the AVX2-only `detail` namespace -- its own `build()` has no AVX2
 * dependency, `detail::sum_n_portable` is the portable scalar summer, gated only by `SUB0_BBQD_AVX2`
 * for physical-file locality, not by any real ISA requirement) and now CALLER-OWNED, REUSED scratch
 * (AGENTS.md S1's own "size once outside the hot loop, reuse across calls"): `gemv_plane<Threads>`'s own
 * `Gsum16*` parameter, when non-null, is built here (not per-thread, not per-call-site-internal) and
 * `build()`'s `v.assign(...)` only reallocates when the activation width changes, exactly like
 * `ActBlocks::quantize`'s own contract -- so a decode caller that owns one `Gsum16` per (thread, role)
 * across the whole generation pays the heap cost once, not every token. This closes the
 * `docs/BACKBONE_NATIVE_QUANT.md` S12i "TODO(phase 2b)" finding: decode's GDN out-proj wiring
 * (`src/backends/cpu/decode.cpp`) is this type's first real per-token caller.
 */
struct Gsum16 {
    std::vector<std::int32_t> v;   // v[2g+0] = sum(x.qs[g*32 : g*32+16)); v[2g+1] = sum of the other 16

    void build(const ActBlocks& x) {
        const auto groups = static_cast<std::size_t>(x.n / GROUP);
        v.assign(groups * 2, 0);
        for (std::size_t g = 0; g < groups; ++g) {
            v[2 * g + 0] = detail::sum_n_portable(x.qs.data() + g * GROUP, 16);
            v[2 * g + 1] = detail::sum_n_portable(x.qs.data() + g * GROUP + 16, 16);
        }
    }
};

#if defined(SUB0_BBQD_AVX2)
namespace detail {

/// 32-wide signed-weight x signed-activation dot, weight supplied as a REGISTER rather than a pointer
/// (dot32_avx2 above always re-loads from memory) -- the streaming kernels decode a weight group directly
/// into a register via the unpack helpers below and must not pay a store-then-reload round trip to reuse
/// dot32_avx2's own pointer-based signature.
[[nodiscard]] inline std::int32_t dot32_avx2_reg(__m256i vw, const std::int8_t* x) {
    const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x));
    const __m256i w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
    const __m256i w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw, 1));
    const __m256i x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
    const __m256i x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
    const __m256i p_lo = _mm256_madd_epi16(w_lo, x_lo);
    const __m256i p_hi = _mm256_madd_epi16(w_hi, x_hi);
    return hsum256_epi32(_mm256_add_epi32(p_lo, p_hi));
}

/// The 16-wide equivalent of dot32_avx2_reg, for Q6_K's own half-group (kSubW=16) split.
[[nodiscard]] inline std::int32_t dot16_avx2_reg(__m128i vw, const std::int8_t* x) {
    const __m128i vx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(x));
    const __m128i w_lo = _mm_cvtepi8_epi16(vw);
    const __m128i x_lo = _mm_cvtepi8_epi16(vx);
    const __m128i w_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vw, 8));   // see dot16_avx2's own comment
    const __m128i x_hi = _mm_cvtepi8_epi16(_mm_srli_si128(vx, 8));
    const __m128i p = _mm_add_epi32(_mm_madd_epi16(w_lo, x_lo), _mm_madd_epi16(w_hi, x_hi));
    __m128i s = _mm_add_epi32(p, _mm_shuffle_epi32(p, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(0, 1, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/// Low nibble of each of 32 bytes, as unsigned codes 0..15 -- a per-byte-safe AND, no cross-lane hazard.
[[nodiscard]] inline __m256i nibble_lo(__m256i bytes) noexcept {
    return _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));
}
/// High nibble of each of 32 bytes. The 16-bit-lane-shift idiom llama.cpp itself uses (quants.c's own
/// Q4_K/Q5_K AVX2 paths): for a 16-bit lane holding bytes [lo=b0, hi=b1], `(lane >> 4) & 0xFF` per byte
/// works out to `b0 >> 4` for the shifted lane's own low byte and `b1 >> 4` for its high byte, with no
/// cross-byte contamination -- re-derived bit by bit (docs/BACKBONE_NATIVE_QUANT.md S12b), not assumed
/// safe by analogy.
[[nodiscard]] inline __m256i nibble_hi(__m256i bytes) noexcept {
    return _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));
}

/// Q5_K's single `qh` high bit (`bit_idx` in [0,8), one bit per sub-block -- re-derived algebraically
/// from Q5KPlane::group()'s own `bit = (hi_nibble?2:1) << (2*(sub/2))` as `bit_idx == sub` exactly, not
/// copied), mapped to output bit 4 (value 16 or 0). Mask-then-compare rather than a runtime-count lane
/// shift: a variable shift risks the SAME cross-byte hazard nibble_hi's own comment works through, and
/// proving it safe for an arbitrary runtime shift is harder than for the two fixed shifts nibble_hi
/// needs -- mask-then-compare sidesteps the question entirely (AND and CMPEQ are always byte-safe).
[[nodiscard]] inline __m256i qh_bit_to_hi4(__m256i qh, int bit_idx) noexcept {
    const __m256i bitmask = _mm256_set1_epi8(static_cast<char>(1 << bit_idx));
    const __m256i is_set = _mm256_cmpeq_epi8(_mm256_and_si256(qh, bitmask), bitmask);
    return _mm256_and_si256(is_set, _mm256_set1_epi8(16));
}

/// Q6_K's 2-bit `qh` field at `shift` in {0,2,4,6} (`shift = strip*2`), mapped to output bits [4,5].
/// Mask first (byte-safe), THEN a compile-time-immediate shift selected by a 4-way switch -- each of the
/// four cases hand-verified not to cross a byte boundary the same way nibble_hi's comment does, rather
/// than trusting a single runtime-count shift to be safe for every case at once.
[[nodiscard]] inline __m256i qh_2bits_to_hi4(__m256i qh, int shift) noexcept {
    const __m256i masked = _mm256_and_si256(qh, _mm256_set1_epi8(static_cast<char>(3 << shift)));
    switch (shift) {
        case 0:  return _mm256_slli_epi16(masked, 4);
        case 2:  return _mm256_slli_epi16(masked, 2);
        case 4:  return masked;
        default: return _mm256_srli_epi16(masked, 2);   // shift == 6
    }
}

/** Q4_K/Q5_K's shared affine super-block state (`d`, `dmin`, the 8 sub-block `(sc, m)` pairs already
 * folded into `sc[i] = d*scale_i` / `m[i] = -dmin*min_i`), decoded ONCE per 256-element superblock.
 * Phase 1's `group()` recomputed this -- including two `f16_to_f32` calls and a `k_scale_min` call -- on
 * EVERY 32-element group, an 8x-redundant cost within one superblock; this is the actual fix, not the
 * vector unpack alone (docs/BACKBONE_NATIVE_QUANT.md S12a/S12b).
 */
struct KScaleTable {
    float d = 0.f, dmin = 0.f;
    std::array<float, 8> sc{}, m{};

    /// `blk` is a Q4_K (144-byte) or Q5_K (176-byte) superblock's own start -- both formats share the
    /// identical `d`/`dmin`/12-byte-packed-scales layout at the same offsets (AGENTS.md S5, re-checked
    /// against gguf.hpp's own dequantize_q4_k/dequantize_q5_k rather than assumed from the block sizes
    /// merely looking similar).
    ///
    /// AGENTS.md S13 pass 2: the first version of this function called `gguf::k_scale_min` 8 times, each
    /// with a data-dependent `if (j < 4)` branch -- ~55M superblocks/sec through this call at the pass-1
    /// measured throughput (docs/BACKBONE_NATIVE_QUANT.md S12), a real, unpredictable-branch cost. This
    /// unpacks all 8 (scale, min) pairs at once via the SAME branch-free bit-manipulation llama.cpp's own
    /// AVX2 `ggml_vec_dot_q4_K_q8_K`/`q5_K_q8_K` use (`utmp[4]`, `D:\Craig\llama.cpp-qwen4exp\ggml\src\
    /// ggml-cpu\arch\x86\quants.c`) rather than gguf::k_scale_min's own byte-at-a-time reference -- fetched
    /// from that source and re-derived, not assumed correct by resemblance, and checked bit-exact against
    /// `gguf::k_scale_min`'s own output for all 8 sub-block indices before being trusted (this file's own
    /// tests).
    void load(const std::uint8_t* blk) noexcept {
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        d = gguf::f16_to_f32(d_bits);
        dmin = gguf::f16_to_f32(dmin_bits);

        constexpr std::uint32_t kMask1 = 0x3f3f3f3fu, kMask2 = 0x0f0f0f0fu, kMask3 = 0x03030303u;
        std::uint32_t utmp[4] = {0, 0, 0, 0};
        std::memcpy(utmp, blk + 4, 12);   // fills utmp[0..2]; utmp[3] is derived below
        utmp[3] = ((utmp[2] >> 4) & kMask2) | (((utmp[1] >> 6) & kMask3) << 4);
        const std::uint32_t uaux = utmp[1] & kMask1;
        utmp[1] = (utmp[2] & kMask2) | (((utmp[0] >> 6) & kMask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kMask1;
        // utmp[0]/utmp[1]'s 4 bytes each are sub-block scales 0..3 / 4..7; utmp[2]/utmp[3]'s are the
        // matching mins -- exactly the byte layout `_mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3..0]))`
        // consumes in the AVX2 reference, read here with plain byte indexing since this project's
        // per-32-group ActBlocks (unlike llama.cpp's per-256 block_q8_K) still needs each sub-block's
        // scale/min as an individually-addressable float, not a single vector register.
        const auto* sc_bytes = reinterpret_cast<const std::uint8_t*>(utmp);
        const auto* m_bytes = reinterpret_cast<const std::uint8_t*>(utmp) + 8;
        for (int i = 0; i < 8; ++i) {
            sc[static_cast<std::size_t>(i)] = d * static_cast<float>(sc_bytes[i]);
            m[static_cast<std::size_t>(i)] = -dmin * static_cast<float>(m_bytes[i]);
        }
    }
};

// Gsum16 moved to public sub0::bbqd scope (below, before this AVX2-only block) in O5 phase 2b-2b, so its
// type is nameable -- and usable as caller-owned, reused scratch -- from a build without SUB0_BBQD_AVX2
// too. See its own doc comment there for what changed and why.

/** Q4_K, one row, streaming. Requires `row_base % 256 == 0 && row_elems % 256 == 0` (the caller's own
 * job to check once, not per row). Processes 64 elements (2 sub-blocks) per AVX2 chunk from one 32-byte
 * `ql` load -- llama.cpp's own iteration shape (`ggml_vec_dot_q4_K_q8_K`'s AVX2 path,
 * D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-cpu\arch\x86\quants.c), re-derived onto this project's own
 * per-32 `ActBlocks` rather than copied (AGENTS.md S5): llama.cpp folds its per-sub-block scale into the
 * INTEGER accumulator because its own activation is quantized per-256 (`block_q8_K`, one scale for the
 * whole superblock); this engine's activation is quantized per-32 (`moeqd::ActBlocks`, kept unchanged --
 * see the file header's own S12b note on why per-32 was kept rather than adopting a Q8_K-style per-256
 * scheme), so the scale fold stays a per-32-group float FMA here, in the exact same sequential order
 * (sub-block 0, 1, 2, ..., 7) the OLD `group()`-based `gemv_rows<Q4KPlane,true>` already used. This
 * changes HOW `isum` is computed (vectorized unpack instead of a scalar loop + a non-inlined `group()`
 * call), never the arithmetic expression or its evaluation order -- which is exactly why it is checked,
 * and found, bit-identical to `gemv_rows<Q4KPlane,true>` (tests/backbone_quant_dot_tests.cpp).
 */
[[nodiscard]] inline float dot_row_q4_k_avx2(const std::uint8_t* plane, std::uint64_t row_base,
                                             int row_elems, const ActBlocks& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 144;
        KScaleTable t;
        t.load(blk);
        const std::uint8_t* ql = blk + 16;
        const int g0 = s * 8;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const int gi_lo = g0 + sub_lo, gi_hi = g0 + sub_hi;
            const __m256i wlo = nibble_lo(raw), whi = nibble_hi(raw);
            const std::int32_t isum_lo =
                dot32_avx2_reg(wlo, x.qs.data() + static_cast<std::size_t>(gi_lo) * GROUP);
            const std::int32_t isum_hi =
                dot32_avx2_reg(whi, x.qs.data() + static_cast<std::size_t>(gi_hi) * GROUP);
            acc += x.scale[static_cast<std::size_t>(gi_lo)] *
                   (t.sc[static_cast<std::size_t>(sub_lo)] * static_cast<float>(isum_lo) +
                    t.m[static_cast<std::size_t>(sub_lo)] *
                        static_cast<float>(x.gsum[static_cast<std::size_t>(gi_lo)]));
            acc += x.scale[static_cast<std::size_t>(gi_hi)] *
                   (t.sc[static_cast<std::size_t>(sub_hi)] * static_cast<float>(isum_hi) +
                    t.m[static_cast<std::size_t>(sub_hi)] *
                        static_cast<float>(x.gsum[static_cast<std::size_t>(gi_hi)]));
        }
    }
    return acc;
}

/** Q5_K, one row, streaming -- identical shape to dot_row_q4_k_avx2 plus the `qh` high-bit OR
 * (qh_bit_to_hi4). Same precondition, same accumulation order, same bit-exactness contract.
 */
[[nodiscard]] inline float dot_row_q5_k_avx2(const std::uint8_t* plane, std::uint64_t row_base,
                                             int row_elems, const ActBlocks& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 176;
        KScaleTable t;
        t.load(blk);
        const std::uint8_t* qh_base = blk + 16;
        const std::uint8_t* ql = blk + 48;
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));
        const int g0 = s * 8;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const int gi_lo = g0 + sub_lo, gi_hi = g0 + sub_hi;
            const __m256i nlo = nibble_lo(raw), nhi = nibble_hi(raw);
            const __m256i hlo = qh_bit_to_hi4(qhv, sub_lo), hhi = qh_bit_to_hi4(qhv, sub_hi);
            const __m256i wlo = _mm256_or_si256(nlo, hlo), whi = _mm256_or_si256(nhi, hhi);
            const std::int32_t isum_lo =
                dot32_avx2_reg(wlo, x.qs.data() + static_cast<std::size_t>(gi_lo) * GROUP);
            const std::int32_t isum_hi =
                dot32_avx2_reg(whi, x.qs.data() + static_cast<std::size_t>(gi_hi) * GROUP);
            acc += x.scale[static_cast<std::size_t>(gi_lo)] *
                   (t.sc[static_cast<std::size_t>(sub_lo)] * static_cast<float>(isum_lo) +
                    t.m[static_cast<std::size_t>(sub_lo)] *
                        static_cast<float>(x.gsum[static_cast<std::size_t>(gi_lo)]));
            acc += x.scale[static_cast<std::size_t>(gi_hi)] *
                   (t.sc[static_cast<std::size_t>(sub_hi)] * static_cast<float>(isum_hi) +
                    t.m[static_cast<std::size_t>(sub_hi)] *
                        static_cast<float>(x.gsum[static_cast<std::size_t>(gi_hi)]));
        }
    }
    return acc;
}

/** Q6_K, one row, streaming. `gsum16` must already be built (Gsum16::build) against THIS row's `x`;
 * the caller builds it once per `gemv_plane_avx2` call, not once per row. Same precondition and
 * bit-exactness contract as the other two streaming kernels.
 */
[[nodiscard]] inline float dot_row_q6_k_avx2(const std::uint8_t* plane, std::uint64_t row_base,
                                             int row_elems, const ActBlocks& x,
                                             const Gsum16& gsum16) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 210;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = gguf::f16_to_f32(d_bits);
        const int g0 = s * 8;
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            const std::uint8_t* qh_base = blk + 128 + half * 32;
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));
            for (int strip = 0; strip < 4; ++strip) {
                const int gi = g0 + half * 4 + strip;
                const bool hi_nibble = (strip == 2 || strip == 3);
                const int ql_off = (strip == 1 || strip == 3) ? 32 : 0;
                const std::uint8_t* qlc = blk + half * 64 + ql_off;
                const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
                const __m256i nib = hi_nibble ? nibble_hi(raw) : nibble_lo(raw);
                const __m256i hi2 = qh_2bits_to_hi4(qhv, strip * 2);
                const __m256i wfull = _mm256_or_si256(nib, hi2);   // raw 6-bit code, 0..63
                const __m128i wlo = _mm256_castsi256_si128(wfull);
                const __m128i whi = _mm256_extracti128_si256(wfull, 1);
                const std::int32_t isum_lo =
                    dot16_avx2_reg(wlo, x.qs.data() + static_cast<std::size_t>(gi) * GROUP);
                const std::int32_t isum_hi =
                    dot16_avx2_reg(whi, x.qs.data() + static_cast<std::size_t>(gi) * GROUP + 16);
                const float scale_lo = d * static_cast<float>(sc[2 * strip + 0]);
                const float scale_hi = d * static_cast<float>(sc[2 * strip + 1]);
                const float bias_lo = -32.f * scale_lo, bias_hi = -32.f * scale_hi;
                const std::int32_t gsum_lo = gsum16.v[2 * static_cast<std::size_t>(gi) + 0];
                const std::int32_t gsum_hi = gsum16.v[2 * static_cast<std::size_t>(gi) + 1];
                acc += x.scale[static_cast<std::size_t>(gi)] *
                       (scale_lo * static_cast<float>(isum_lo) + bias_lo * static_cast<float>(gsum_lo) +
                        scale_hi * static_cast<float>(isum_hi) + bias_hi * static_cast<float>(gsum_hi));
            }
        }
    }
    return acc;
}

}  // namespace detail
#endif  // SUB0_BBQD_AVX2

// =========================================================================================================
// Pass 4 (AGENTS.md S13, docs/BACKBONE_NATIVE_QUANT.md S12g/S14): the per-256 activation superblock
// (ActSuper, above) lets Q4_K/Q5_K/Q6_K fold each superblock's per-sub-block INTEGER scale into the
// integer accumulator BEFORE any float conversion, the way llama.cpp's own AVX2 K-quant kernels do
// (D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-cpu\arch\x86\quants.c, ggml_vec_dot_q{4,5,6}_K_q8_K) --
// collapsing 8 (Q4_K/Q5_K) or 16 (Q6_K) horizontal int32->float reductions per superblock (S12's own
// streaming kernels above, one per sub-block) into ONE. Three implementations, all checked bit-exact
// against each other in tests/backbone_quant_dot_tests.cpp (integer arithmetic is associative and
// commutative exactly -- no rounding survives a reordering, so any two of these disagreeing is a bug, the
// same discipline S12's own kernels already established):
//   - *_super_portable: the correctness reference, no ISA assumption, available on every build.
//   - *_super_avx2: llama.cpp's own `maddubs_epi16` (unsigned weight x signed activation, one 16-bit
//     intermediate per byte-pair) + `madd_epi16` (scale fold, pairs summed to int32) shape, re-derived
//     onto this file's own per-sub-block loop structure (not llama.cpp's shuffle-broadcast batching --
//     see this section's own comment on why) rather than copied line for line (AGENTS.md S5).
//   - *_super_vnni: `vpdpbusd` (AVX-VNNI, `_mm256_dpbusd_avx_epi32`/`_mm_dpbusd_avx_epi32`, VEX-encoded --
//     this host has it WITHOUT AVX-512), which fuses maddubs+madd into one instruction operating directly
//     on int32 lanes -- the lever the task brief named explicitly, measured rather than assumed to win.
//
// WHY THIS FILE DOES NOT REPLICATE llama.cpp's OWN `get_scale_shuffle_k4` BATCHING. The reference batches
// TWO sub-blocks' worth of scale broadcast into one `_mm256_shuffle_epi8` against a combined 8-scale
// register, processing 64 raw elements per outer-loop iteration. This file's existing S12 kernels (and
// every Plane::group() before them) already walk ONE sub-block-worth of elements per unpack call
// (`nibble_lo`/`nibble_hi` decode exactly one sub-block's 32-wide nibble span at a time); re-deriving the
// fold onto that SAME per-sub-block granularity means a plain `_mm256_set1_epi16`/`_mm256_set1_epi32`
// broadcast of the (already individually-addressable) raw scale suffices -- no shuffle table needed, at
// the cost of one broadcast instruction per sub-block instead of amortizing two sub-blocks per shuffle.
// This was a deliberate simplicity-over-micro-optimization call, re-derivable from measurement if the
// broadcast overhead turns out to matter (it did not, per this section's own S14 pass numbers).
//
// OVERFLOW, worked through per format rather than assumed safe by analogy to S12's own per-32 bound:
//   Q4_K/Q5_K: per-element product <= 31 (Q5's widest nibble+hibit) * 127 = 3937; a 32-wide sub-block dot
//     <= 32*3937 = 125,984; scaled by sc (<=63) <= 7,936,992; summed over 8 sub-blocks (worst-case same
//     sign) <= 63,495,936 -- three orders of magnitude below INT32_MAX (2,147,483,647). The min term is
//     smaller still (m<=63, bsum_pair<=32*127=4064, summed over 8 <= 2,048,256).
//   Q6_K: per-element product <= 63 (raw 6-bit code) * 127 = 8001; a 16-wide sub-block dot <= 128,016;
//     scaled by a SIGNED sc (|sc|<=127, not 63 -- Q6_K's own scale field is int8, unlike Q4_K/Q5_K's 6-bit
//     unsigned one) <= 16,257,858 per sub-block; summed over 16 sub-blocks <= 260,125,728, still an order
//     of magnitude below INT32_MAX. `q8sclsub` (sc*bsum summed over 16, then *32) <= 127*2032*16*32 =
//     132,096,512, same margin.
// Every accumulator below is a plain `std::int32_t`/`__m256i`/`__m128i` of int32 lanes on this basis, not
// a wider type -- checked here once rather than re-justified at each call site.
// =========================================================================================================

namespace detail {

/** Q4_K/Q5_K's shared per-superblock affine state, RAW (the small unsigned 6-bit `sc`/`m` codes
 * themselves, NOT pre-multiplied by `d`/`dmin` into a float the way `KScaleTable` above does for the
 * per-32-activation streaming kernels). The fused per-256 kernels below need the raw codes so they can be
 * multiplied into the INTEGER accumulator per sub-block and fold `d`/`dmin` in as a single float multiply
 * only once, at the very end of the superblock -- llama.cpp's own fold (this section's header comment).
 * Shares the identical branch-free `utmp[4]` bit-unpack `KScaleTable::load` already uses (verified there
 * against `gguf::k_scale_min`, tests/backbone_quant_dot_tests.cpp) -- re-derived here rather than reusing
 * that function directly because it multiplies by `d`/`dmin` inline, which this struct deliberately does
 * not want (AGENTS.md S3: the existing field/function does not discriminate what THIS struct needs).
 */
struct KScaleRaw {
    float d = 0.f, dmin = 0.f;
    std::array<std::uint8_t, 8> sc{}, m{};   // raw 6-bit codes, 0..63

    void load(const std::uint8_t* blk) noexcept {
        std::uint16_t d_bits = 0, dmin_bits = 0;
        std::memcpy(&d_bits, blk, 2);
        std::memcpy(&dmin_bits, blk + 2, 2);
        d = gguf::f16_to_f32(d_bits);
        dmin = gguf::f16_to_f32(dmin_bits);

        constexpr std::uint32_t kMask1 = 0x3f3f3f3fu, kMask2 = 0x0f0f0f0fu, kMask3 = 0x03030303u;
        std::uint32_t utmp[4] = {0, 0, 0, 0};
        std::memcpy(utmp, blk + 4, 12);
        utmp[3] = ((utmp[2] >> 4) & kMask2) | (((utmp[1] >> 6) & kMask3) << 4);
        const std::uint32_t uaux = utmp[1] & kMask1;
        utmp[1] = (utmp[2] & kMask2) | (((utmp[0] >> 6) & kMask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kMask1;
        const auto* sc_bytes = reinterpret_cast<const std::uint8_t*>(utmp);
        const auto* m_bytes = reinterpret_cast<const std::uint8_t*>(utmp) + 8;
        for (int i = 0; i < 8; ++i) {
            sc[static_cast<std::size_t>(i)] = sc_bytes[i];
            m[static_cast<std::size_t>(i)] = m_bytes[i];
        }
    }
};

/** Q4_K, one row, per-256-activation streaming, PORTABLE (the correctness reference for
 * `*_super_avx2` / `*_super_vnni` below). Same byte layout as `Q4KPlane`/`dot_row_q4_k_avx2` (blk+16 is `ql`,
 * `sub` selects nibble half exactly as there); the only difference from `dot_row_q4_k_avx2` is WHICH
 * activation type is read and WHERE the float conversion happens -- once per superblock here, once per
 * sub-block there.
 */
[[nodiscard]] inline float dot_row_q4_k_super_portable(const std::uint8_t* plane, std::uint64_t row_base,
                                                        int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 144;
        KScaleRaw t;
        t.load(blk);
        const std::uint8_t* ql = blk + 16;
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        std::int32_t isum = 0, isum_min = 0;
        for (int sub = 0; sub < 8; ++sub) {
            const bool hi_nibble = (sub % 2) == 1;
            const std::uint8_t* qlc = ql + (sub / 2) * 32;
            std::array<std::int8_t, 32> w{};
            for (int l = 0; l < 32; ++l)
                w[static_cast<std::size_t>(l)] =
                    static_cast<std::int8_t>(hi_nibble ? (qlc[l] >> 4) : (qlc[l] & 0xF));
            const std::int32_t dot = dot_n_portable(w.data(), xs + sub * 32, 32);
            isum += static_cast<std::int32_t>(t.sc[static_cast<std::size_t>(sub)]) * dot;
            const std::int32_t bsum_pair =
                static_cast<std::int32_t>(bsums[2 * sub]) + static_cast<std::int32_t>(bsums[2 * sub + 1]);
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub)]) * bsum_pair;
        }
        const float d_combined = x.d[static_cast<std::size_t>(s)] * t.d;
        const float dmin_combined = -x.d[static_cast<std::size_t>(s)] * t.dmin;
        acc += d_combined * static_cast<float>(isum) + dmin_combined * static_cast<float>(isum_min);
    }
    return acc;
}

/** Q5_K, one row, per-256-activation streaming, PORTABLE. Identical shape to
 * dot_row_q4_k_super_portable plus the `qh` high-bit OR, same layout as `Q5KPlane`/`dot_row_q5_k_avx2`.
 */
[[nodiscard]] inline float dot_row_q5_k_super_portable(const std::uint8_t* plane, std::uint64_t row_base,
                                                        int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 176;
        KScaleRaw t;
        t.load(blk);
        const std::uint8_t* qh = blk + 16;
        const std::uint8_t* ql = blk + 48;
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        std::int32_t isum = 0, isum_min = 0;
        for (int sub = 0; sub < 8; ++sub) {
            const bool hi_nibble = (sub % 2) == 1;
            const std::uint8_t* qlc = ql + (sub / 2) * 32;
            const std::uint8_t bit = static_cast<std::uint8_t>((hi_nibble ? 2 : 1) << (2 * (sub / 2)));
            std::array<std::int8_t, 32> w{};
            for (int l = 0; l < 32; ++l) {
                const int nib = hi_nibble ? (qlc[l] >> 4) : (qlc[l] & 0xF);
                const int hi = (qh[l] & bit) ? 16 : 0;
                w[static_cast<std::size_t>(l)] = static_cast<std::int8_t>(nib + hi);
            }
            const std::int32_t dot = dot_n_portable(w.data(), xs + sub * 32, 32);
            isum += static_cast<std::int32_t>(t.sc[static_cast<std::size_t>(sub)]) * dot;
            const std::int32_t bsum_pair =
                static_cast<std::int32_t>(bsums[2 * sub]) + static_cast<std::int32_t>(bsums[2 * sub + 1]);
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub)]) * bsum_pair;
        }
        const float d_combined = x.d[static_cast<std::size_t>(s)] * t.d;
        const float dmin_combined = -x.d[static_cast<std::size_t>(s)] * t.dmin;
        acc += d_combined * static_cast<float>(isum) + dmin_combined * static_cast<float>(isum_min);
    }
    return acc;
}

/** Q6_K, one row, per-256-activation streaming, PORTABLE. Uses llama.cpp's own `q8sclsub` shape (one
 * combined `-32*sum_sub(scale[sub]*bsum[sub])` term per superblock, computed ONCE, subtracted from the
 * raw isum before the single float multiply) rather than per-sub-block bias floats -- Q6_K has no
 * separate "min" the way Q4_K/Q5_K do, only the constant zero-point, so this is the whole affine
 * correction, not half of it. Same byte layout as `Q6KPlane`/`dot_row_q6_k_avx2`.
 *
 * @note The output element order within a superblock (half, strip, [lo 16, hi 16)) is, worked through
 *       once here: strictly sequential in the ROW's own linear index -- e = half*128 + strip*32 + l, and
 *       `wg.q[0..16)`/`[16..32)` at a given (half,strip) map to sequential e-ranges too, so the activation
 *       sub-block index `sc`/`bsums` need (i = half*8 + 2*strip + {0 for l<16, 1 for l>=16}) equals the
 *       SAME sequential 16-wide index the ActSuper row itself uses (`x.bsums[i]`, `xs[16*i .. 16*i+16)`).
 *       Checked by this file's own tests, not merely asserted here.
 */
[[nodiscard]] inline float dot_row_q6_k_super_portable(const std::uint8_t* plane, std::uint64_t row_base,
                                                        int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 210;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = gguf::f16_to_f32(d_bits);
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        std::int32_t q8sclsub = 0;
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            for (int k = 0; k < 8; ++k)
                q8sclsub += static_cast<std::int32_t>(sc[k]) *
                            static_cast<std::int32_t>(bsums[half * 8 + k]);
        }
        q8sclsub *= 32;

        std::int32_t isum = 0;
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            const std::uint8_t* qh = blk + 128 + half * 32;
            for (int strip = 0; strip < 4; ++strip) {
                const bool hi_nibble = (strip == 2 || strip == 3);
                const int ql_off = (strip == 1 || strip == 3) ? 32 : 0;
                const int shift = strip * 2;
                const std::uint8_t* qlc = blk + half * 64 + ql_off;
                std::array<std::int8_t, 32> w{};
                for (int l = 0; l < 32; ++l) {
                    const int nib = hi_nibble ? (qlc[l] >> 4) : (qlc[l] & 0xF);
                    const int hi = (qh[l] >> shift) & 3;
                    w[static_cast<std::size_t>(l)] = static_cast<std::int8_t>(nib | (hi << 4));   // 0..63
                }
                const int e_base = half * 128 + strip * 32;
                const std::int32_t dot_lo = dot_n_portable(w.data(), xs + e_base, 16);
                const std::int32_t dot_hi = dot_n_portable(w.data() + 16, xs + e_base + 16, 16);
                isum += static_cast<std::int32_t>(sc[2 * strip + 0]) * dot_lo;
                isum += static_cast<std::int32_t>(sc[2 * strip + 1]) * dot_hi;
            }
        }
        const float d_combined = x.d[static_cast<std::size_t>(s)] * d;
        acc += d_combined * static_cast<float>(isum - q8sclsub);
    }
    return acc;
}

}  // namespace detail

#if defined(SUB0_BBQD_AVX2)
namespace detail {

/** Q4_K, one row, per-256-activation streaming, AVX2 (`maddubs_epi16` + `madd_epi16` fold -- this
 * section's own header comment). One sub-block (32 elements) per iteration: `nibble_lo`/`nibble_hi`
 * (already defined above, reused verbatim) unpack the unsigned nibble byte, `_mm256_maddubs_epi16`
 * multiplies it against the signed activation (16 int16 lanes, each already a pairwise sum), then
 * `_mm256_madd_epi16` against a `sc[sub]` broadcast folds the raw scale in AND reduces to 8 int32 lanes
 * -- accumulated into a running `sumi`/`smin` across all 8 sub-blocks, with exactly ONE `hsum256_epi32`
 * per superblock (down from S12's own 8).
 */
[[nodiscard]] inline float dot_row_q4_k_super_avx2(const std::uint8_t* plane, std::uint64_t row_base,
                                                    int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 144;
        KScaleRaw t;
        t.load(blk);
        const std::uint8_t* ql = blk + 16;
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        __m256i sumi = _mm256_setzero_si256();
        std::int32_t isum_min = 0;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const __m256i wlo = nibble_lo(raw), whi = nibble_hi(raw);
            const __m256i xlo =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_lo * 32));
            const __m256i xhi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_hi * 32));
            const __m256i plo = _mm256_maddubs_epi16(wlo, xlo);
            const __m256i phi = _mm256_maddubs_epi16(whi, xhi);
            const __m256i scale_lo = _mm256_set1_epi16(static_cast<short>(t.sc[static_cast<std::size_t>(sub_lo)]));
            const __m256i scale_hi = _mm256_set1_epi16(static_cast<short>(t.sc[static_cast<std::size_t>(sub_hi)]));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale_lo, plo));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale_hi, phi));
            const std::int32_t bp_lo = static_cast<std::int32_t>(bsums[2 * sub_lo]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_lo + 1]);
            const std::int32_t bp_hi = static_cast<std::int32_t>(bsums[2 * sub_hi]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_hi + 1]);
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_lo)]) * bp_lo;
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_hi)]) * bp_hi;
        }
        const std::int32_t isum = hsum256_epi32(sumi);
        const float d_combined = x.d[static_cast<std::size_t>(s)] * t.d;
        const float dmin_combined = -x.d[static_cast<std::size_t>(s)] * t.dmin;
        acc += d_combined * static_cast<float>(isum) + dmin_combined * static_cast<float>(isum_min);
    }
    return acc;
}

/** Q5_K, one row, per-256-activation streaming, AVX2. Same shape as dot_row_q4_k_super_avx2 plus the
 * `qh` high-bit OR (`qh_bit_to_hi4`, already defined above).
 */
[[nodiscard]] inline float dot_row_q5_k_super_avx2(const std::uint8_t* plane, std::uint64_t row_base,
                                                    int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 176;
        KScaleRaw t;
        t.load(blk);
        const std::uint8_t* qh_base = blk + 16;
        const std::uint8_t* ql = blk + 48;
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));

        __m256i sumi = _mm256_setzero_si256();
        std::int32_t isum_min = 0;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const __m256i nlo = nibble_lo(raw), nhi = nibble_hi(raw);
            const __m256i hlo = qh_bit_to_hi4(qhv, sub_lo), hhi = qh_bit_to_hi4(qhv, sub_hi);
            const __m256i wlo = _mm256_or_si256(nlo, hlo), whi = _mm256_or_si256(nhi, hhi);
            const __m256i xlo =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_lo * 32));
            const __m256i xhi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_hi * 32));
            const __m256i plo = _mm256_maddubs_epi16(wlo, xlo);
            const __m256i phi = _mm256_maddubs_epi16(whi, xhi);
            const __m256i scale_lo = _mm256_set1_epi16(static_cast<short>(t.sc[static_cast<std::size_t>(sub_lo)]));
            const __m256i scale_hi = _mm256_set1_epi16(static_cast<short>(t.sc[static_cast<std::size_t>(sub_hi)]));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale_lo, plo));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scale_hi, phi));
            const std::int32_t bp_lo = static_cast<std::int32_t>(bsums[2 * sub_lo]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_lo + 1]);
            const std::int32_t bp_hi = static_cast<std::int32_t>(bsums[2 * sub_hi]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_hi + 1]);
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_lo)]) * bp_lo;
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_hi)]) * bp_hi;
        }
        const std::int32_t isum = hsum256_epi32(sumi);
        const float d_combined = x.d[static_cast<std::size_t>(s)] * t.d;
        const float dmin_combined = -x.d[static_cast<std::size_t>(s)] * t.dmin;
        acc += d_combined * static_cast<float>(isum) + dmin_combined * static_cast<float>(isum_min);
    }
    return acc;
}

/** Q6_K, one row, per-256-activation streaming, AVX2. Builds the full 32-wide raw 6-bit code the same
 * way dot_row_q6_k_avx2 above does (`nibble_lo`/`nibble_hi`/`qh_2bits_to_hi4`), then splits into its two
 * 16-wide (scale_lo, scale_hi) halves via `_mm256_castsi256_si128`/`_mm256_extracti128_si256` and folds
 * each half's own scale via 128-bit `maddubs`+`madd`, accumulating all 16 sub-blocks into one running
 * `__m128i` before a single 4-wide horizontal reduction per superblock.
 */
[[nodiscard]] inline float dot_row_q6_k_super_avx2(const std::uint8_t* plane, std::uint64_t row_base,
                                                    int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 210;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = gguf::f16_to_f32(d_bits);
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        std::int32_t q8sclsub = 0;
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            for (int k = 0; k < 8; ++k)
                q8sclsub += static_cast<std::int32_t>(sc[k]) *
                            static_cast<std::int32_t>(bsums[half * 8 + k]);
        }
        q8sclsub *= 32;

        __m128i sumi = _mm_setzero_si128();
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            const std::uint8_t* qh_base = blk + 128 + half * 32;
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));
            for (int strip = 0; strip < 4; ++strip) {
                const bool hi_nibble = (strip == 2 || strip == 3);
                const int ql_off = (strip == 1 || strip == 3) ? 32 : 0;
                const std::uint8_t* qlc = blk + half * 64 + ql_off;
                const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
                const __m256i nib = hi_nibble ? nibble_hi(raw) : nibble_lo(raw);
                const __m256i hi2 = qh_2bits_to_hi4(qhv, strip * 2);
                const __m256i wfull = _mm256_or_si256(nib, hi2);   // raw 6-bit code, 0..63
                const __m128i wlo = _mm256_castsi256_si128(wfull);
                const __m128i whi = _mm256_extracti128_si256(wfull, 1);
                const int e_base = half * 128 + strip * 32;
                const __m128i xlo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xs + e_base));
                const __m128i xhi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xs + e_base + 16));
                const __m128i plo = _mm_maddubs_epi16(wlo, xlo);
                const __m128i phi = _mm_maddubs_epi16(whi, xhi);
                const __m128i scale_lo = _mm_set1_epi16(static_cast<short>(sc[2 * strip + 0]));
                const __m128i scale_hi = _mm_set1_epi16(static_cast<short>(sc[2 * strip + 1]));
                sumi = _mm_add_epi32(sumi, _mm_madd_epi16(scale_lo, plo));
                sumi = _mm_add_epi32(sumi, _mm_madd_epi16(scale_hi, phi));
            }
        }
        __m128i h = _mm_add_epi32(sumi, _mm_shuffle_epi32(sumi, _MM_SHUFFLE(1, 0, 3, 2)));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 1, 0, 1)));
        const std::int32_t isum = _mm_cvtsi128_si32(h);
        const float d_combined = x.d[static_cast<std::size_t>(s)] * d;
        acc += d_combined * static_cast<float>(isum - q8sclsub);
    }
    return acc;
}

}  // namespace detail
#endif  // SUB0_BBQD_AVX2

#if defined(SUB0_BBQD_VNNI)
namespace detail {

/** Q4_K, one row, per-256-activation streaming, AVX-VNNI. Replaces dot_row_q4_k_super_avx2's
 * `maddubs_epi16`+`madd_epi16` PAIR with a SINGLE `_mm256_dpbusd_avx_epi32` (unsigned weight nibble x
 * signed activation, fused multiply-add straight to int32, no int16 intermediate) -- the lever the task
 * brief named explicitly. `dpbusd`'s raw 8-lane output is the sub-block's own dot product distributed
 * across lanes (not yet scaled); `_mm256_mullo_epi32` folds the raw `sc[sub]` scale into each lane
 * BEFORE accumulating, so the algebra (and therefore the exact int32 total) is identical to the AVX2
 * path's `madd_epi16(scale_broadcast, maddubs_result)` -- distributivity of integer multiplication over
 * addition makes the two bit-exact (tests/backbone_quant_dot_tests.cpp), not merely close.
 */
[[nodiscard]] inline float dot_row_q4_k_super_vnni(const std::uint8_t* plane, std::uint64_t row_base,
                                                    int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 144;
        KScaleRaw t;
        t.load(blk);
        const std::uint8_t* ql = blk + 16;
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        __m256i sumi = _mm256_setzero_si256();
        std::int32_t isum_min = 0;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const __m256i wlo = nibble_lo(raw), whi = nibble_hi(raw);
            const __m256i xlo =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_lo * 32));
            const __m256i xhi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_hi * 32));
            const __m256i raw_lo = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wlo, xlo);
            const __m256i raw_hi = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), whi, xhi);
            const __m256i scale_lo = _mm256_set1_epi32(static_cast<int>(t.sc[static_cast<std::size_t>(sub_lo)]));
            const __m256i scale_hi = _mm256_set1_epi32(static_cast<int>(t.sc[static_cast<std::size_t>(sub_hi)]));
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(raw_lo, scale_lo));
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(raw_hi, scale_hi));
            const std::int32_t bp_lo = static_cast<std::int32_t>(bsums[2 * sub_lo]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_lo + 1]);
            const std::int32_t bp_hi = static_cast<std::int32_t>(bsums[2 * sub_hi]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_hi + 1]);
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_lo)]) * bp_lo;
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_hi)]) * bp_hi;
        }
        const std::int32_t isum = hsum256_epi32(sumi);
        const float d_combined = x.d[static_cast<std::size_t>(s)] * t.d;
        const float dmin_combined = -x.d[static_cast<std::size_t>(s)] * t.dmin;
        acc += d_combined * static_cast<float>(isum) + dmin_combined * static_cast<float>(isum_min);
    }
    return acc;
}

/** Q5_K, one row, per-256-activation streaming, AVX-VNNI. Same relationship to dot_row_q5_k_super_avx2
 * that dot_row_q4_k_super_vnni has to dot_row_q4_k_super_avx2.
 */
[[nodiscard]] inline float dot_row_q5_k_super_vnni(const std::uint8_t* plane, std::uint64_t row_base,
                                                    int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 176;
        KScaleRaw t;
        t.load(blk);
        const std::uint8_t* qh_base = blk + 16;
        const std::uint8_t* ql = blk + 48;
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;
        const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));

        __m256i sumi = _mm256_setzero_si256();
        std::int32_t isum_min = 0;
        for (int half = 0; half < 4; ++half) {
            const std::uint8_t* qlc = ql + half * 32;
            const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
            const int sub_lo = 2 * half, sub_hi = 2 * half + 1;
            const __m256i nlo = nibble_lo(raw), nhi = nibble_hi(raw);
            const __m256i hlo = qh_bit_to_hi4(qhv, sub_lo), hhi = qh_bit_to_hi4(qhv, sub_hi);
            const __m256i wlo = _mm256_or_si256(nlo, hlo), whi = _mm256_or_si256(nhi, hhi);
            const __m256i xlo =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_lo * 32));
            const __m256i xhi =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(xs + sub_hi * 32));
            const __m256i raw_lo = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), wlo, xlo);
            const __m256i raw_hi = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), whi, xhi);
            const __m256i scale_lo = _mm256_set1_epi32(static_cast<int>(t.sc[static_cast<std::size_t>(sub_lo)]));
            const __m256i scale_hi = _mm256_set1_epi32(static_cast<int>(t.sc[static_cast<std::size_t>(sub_hi)]));
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(raw_lo, scale_lo));
            sumi = _mm256_add_epi32(sumi, _mm256_mullo_epi32(raw_hi, scale_hi));
            const std::int32_t bp_lo = static_cast<std::int32_t>(bsums[2 * sub_lo]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_lo + 1]);
            const std::int32_t bp_hi = static_cast<std::int32_t>(bsums[2 * sub_hi]) +
                                       static_cast<std::int32_t>(bsums[2 * sub_hi + 1]);
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_lo)]) * bp_lo;
            isum_min += static_cast<std::int32_t>(t.m[static_cast<std::size_t>(sub_hi)]) * bp_hi;
        }
        const std::int32_t isum = hsum256_epi32(sumi);
        const float d_combined = x.d[static_cast<std::size_t>(s)] * t.d;
        const float dmin_combined = -x.d[static_cast<std::size_t>(s)] * t.dmin;
        acc += d_combined * static_cast<float>(isum) + dmin_combined * static_cast<float>(isum_min);
    }
    return acc;
}

/** Q6_K, one row, per-256-activation streaming, AVX-VNNI. Same relationship to dot_row_q6_k_super_avx2
 * that the other two *_super_vnni kernels have to their *_super_avx2 counterparts, using the 128-bit
 * `_mm_dpbusd_avx_epi32` for each 16-wide sub-block.
 */
[[nodiscard]] inline float dot_row_q6_k_super_vnni(const std::uint8_t* plane, std::uint64_t row_base,
                                                    int row_elems, const ActSuper& x) noexcept {
    float acc = 0.f;
    const std::uint64_t first_super = row_base / 256;
    const int ns = row_elems / 256;
    for (int s = 0; s < ns; ++s) {
        const std::uint8_t* blk = plane + (first_super + static_cast<std::uint64_t>(s)) * 210;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk + 208, 2);
        const float d = gguf::f16_to_f32(d_bits);
        const std::int8_t* xs = x.qs.data() + static_cast<std::size_t>(s) * 256;
        const std::int16_t* bsums = x.bsums.data() + static_cast<std::size_t>(s) * 16;

        std::int32_t q8sclsub = 0;
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            for (int k = 0; k < 8; ++k)
                q8sclsub += static_cast<std::int32_t>(sc[k]) *
                            static_cast<std::int32_t>(bsums[half * 8 + k]);
        }
        q8sclsub *= 32;

        __m128i sumi = _mm_setzero_si128();
        for (int half = 0; half < 2; ++half) {
            const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192 + half * 8);
            const std::uint8_t* qh_base = blk + 128 + half * 32;
            const __m256i qhv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qh_base));
            for (int strip = 0; strip < 4; ++strip) {
                const bool hi_nibble = (strip == 2 || strip == 3);
                const int ql_off = (strip == 1 || strip == 3) ? 32 : 0;
                const std::uint8_t* qlc = blk + half * 64 + ql_off;
                const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(qlc));
                const __m256i nib = hi_nibble ? nibble_hi(raw) : nibble_lo(raw);
                const __m256i hi2 = qh_2bits_to_hi4(qhv, strip * 2);
                const __m256i wfull = _mm256_or_si256(nib, hi2);
                const __m128i wlo = _mm256_castsi256_si128(wfull);
                const __m128i whi = _mm256_extracti128_si256(wfull, 1);
                const int e_base = half * 128 + strip * 32;
                const __m128i xlo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xs + e_base));
                const __m128i xhi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xs + e_base + 16));
                const __m128i raw_lo = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), wlo, xlo);
                const __m128i raw_hi = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), whi, xhi);
                const __m128i scale_lo = _mm_set1_epi32(static_cast<int>(sc[2 * strip + 0]));
                const __m128i scale_hi = _mm_set1_epi32(static_cast<int>(sc[2 * strip + 1]));
                sumi = _mm_add_epi32(sumi, _mm_mullo_epi32(raw_lo, scale_lo));
                sumi = _mm_add_epi32(sumi, _mm_mullo_epi32(raw_hi, scale_hi));
            }
        }
        __m128i h = _mm_add_epi32(sumi, _mm_shuffle_epi32(sumi, _MM_SHUFFLE(1, 0, 3, 2)));
        h = _mm_add_epi32(h, _mm_shuffle_epi32(h, _MM_SHUFFLE(0, 1, 0, 1)));
        const std::int32_t isum = _mm_cvtsi128_si32(h);
        const float d_combined = x.d[static_cast<std::size_t>(s)] * d;
        acc += d_combined * static_cast<float>(isum - q8sclsub);
    }
    return acc;
}

}  // namespace detail
#endif  // SUB0_BBQD_VNNI

/** Can this format/row-width pair use the per-256 ActSuper path at all? Q8_0 is deliberately excluded
 * (ActSuper's own doc comment: no fold to gain, only a coarser scale to lose). `row_elems % 256 == 0` is
 * REQUIRED here, not merely preferred the way `fusable()`'s `% GROUP` is -- ActSuper's own granularity IS
 * 256, so there is no coarser-fallback path the way S12's `gemv_plane_avx2` falls back to `gemv_rows` for
 * an unaligned row; every real backbone K-quant tensor satisfies this (docs/BACKBONE_NATIVE_QUANT.md
 * S2a's census: 2560/6144/10240/12288 all divide by 256).
 */
[[nodiscard]] inline bool super_fusable(std::uint32_t type_raw, int row_elems) {
    if (row_elems <= 0 || row_elems % 256 != 0) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q4_K:
        case gguf::TensorType::Q5_K:
        case gguf::TensorType::Q6_K: return true;
        default:                     return false;
    }
}

namespace detail {

/** The portable entry point for the ActSuper path -- available on every build, the correctness reference
 * for `*_avx2` / `*_vnni` below (tests/backbone_quant_dot_tests.cpp).
 */
[[nodiscard]] inline bool gemv_plane_super_portable(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                                    int n_rows, int row_elems, const ActSuper& x, float* out,
                                                    int row_lo, int row_hi) {
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    if (!super_fusable(type_raw, row_elems)) return false;
    if (!x.ready_for(row_elems)) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q4_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q4_k_super_portable(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q5_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q5_k_super_portable(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q6_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q6_k_super_portable(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        default:
            return false;
    }
}

#if defined(SUB0_BBQD_AVX2)
[[nodiscard]] inline bool gemv_plane_super_avx2(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                                int n_rows, int row_elems, const ActSuper& x, float* out,
                                                int row_lo, int row_hi) {
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    if (!super_fusable(type_raw, row_elems)) return false;
    if (!x.ready_for(row_elems)) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q4_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q4_k_super_avx2(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q5_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q5_k_super_avx2(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q6_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q6_k_super_avx2(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        default:
            return false;
    }
}
#endif  // SUB0_BBQD_AVX2

#if defined(SUB0_BBQD_VNNI)
[[nodiscard]] inline bool gemv_plane_super_vnni(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                                int n_rows, int row_elems, const ActSuper& x, float* out,
                                                int row_lo, int row_hi) {
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    if (!super_fusable(type_raw, row_elems)) return false;
    if (!x.ready_for(row_elems)) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q4_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q4_k_super_vnni(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q5_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q5_k_super_vnni(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q6_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q6_k_super_vnni(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        default:
            return false;
    }
}
#endif  // SUB0_BBQD_VNNI

/// gemv_plane_super<Threads>'s own kernel choice: plain AVX2, else portable. NOT VNNI-first, even though
/// `kVnniKernels` is true on this host -- measured, not assumed (docs/BACKBONE_NATIVE_QUANT.md S14's own
/// pass-4 table): across four independent 1-thread runs x three formats (12 measurements),
/// `dot_row_q{4,5,6}_k_super_vnni` sat at rough PARITY with `dot_row_q{4,5,6}_k_super_avx2` -- ratios
/// 0.87x-1.09x, mean ~0.95x, i.e. no clear win either direction, not a strong regression. The mechanistic
/// reading: `_mm256_dpbusd_avx_epi32`'s fusion (replacing `maddubs_epi16`+`madd_epi16`) is spent again by
/// the separate `_mm256_mullo_epi32` the scale-fold still needs -- `madd_epi16` already does "multiply
/// and reduce to int32" in one instruction, so VNNI's dpbusd-then-mullo is not actually fewer ops here,
/// and `vpmulld` is typically the higher-latency of the two paths' extra instructions on x86. Given a
/// wash, plain AVX2 is preferred as the default: it needs no `-mavxvnni` availability and is the simpler
/// path. `gemv_plane_super_vnni` stays in the tree, individually callable and tested
/// (tests/backbone_quant_dot_tests.cpp), for a future re-measurement rather than reverted (AGENTS.md
/// S13 -- the per-256 ActSuper MECHANISM is a clear win either way; only its VNNI variant is a wash).
[[nodiscard]] inline bool gemv_plane_super_dispatch(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                                    int n_rows, int row_elems, const ActSuper& x, float* out,
                                                    int row_lo, int row_hi) {
#if defined(SUB0_BBQD_AVX2)
    if constexpr (kAvx2Kernels)
        return gemv_plane_super_avx2(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi);
#endif
    return gemv_plane_super_portable(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi);
}

}  // namespace detail

/** The ActSuper-path counterpart of `gemv_plane<Threads>` -- same contract (threaded by output row,
 * bit-exact across any split, refused means untouched), but for the per-256-activation Q4_K/Q5_K/Q6_K
 * kernels above rather than `moeqd::ActBlocks`. Q8_0 is not reachable through this entry point
 * (`super_fusable` refuses it) -- callers needing Q8_0 use `gemv_plane` unchanged.
 *
 * @return false if the format is unfusable at this granularity, `raw` is too short, or the row range is
 *         out of bounds -- checked BEFORE any thread starts, same as `gemv_plane`.
 */
template <int Threads = 1>
[[nodiscard]] inline bool gemv_plane_super(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                           int n_rows, int row_elems, const ActSuper& x, float* out,
                                           int row_lo = 0, int row_hi = -1) {
    static_assert(Threads >= 1, "Threads must be >= 1");
    if (row_hi < 0) row_hi = n_rows;
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    if (!super_fusable(type_raw, row_elems)) return false;
    if (!x.ready_for(row_elems)) return false;
    if constexpr (Threads == 1) {
        return detail::gemv_plane_super_dispatch(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi);
    } else {
#if defined(_OPENMP)
        if (!omp_in_parallel()) {
            bool ok = true;
            const int total = row_hi - row_lo;
            const int per = (total + Threads - 1) / Threads;
            #pragma omp parallel for num_threads(Threads) schedule(static)
            for (int t = 0; t < Threads; ++t) {
                const int lo = row_lo + std::min(total, t * per);
                const int hi = row_lo + std::min(total, (t + 1) * per);
                if (lo < hi) {
                    const bool arm_ok = detail::gemv_plane_super_dispatch(
                        type_raw, raw, n_rows, row_elems, x, out + (lo - row_lo), lo, hi);
                    if (!arm_ok) {
                        #pragma omp atomic write
                        ok = false;
                    }
                }
            }
            return ok;
        }
#endif
        return detail::gemv_plane_super_dispatch(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi);
    }
}

namespace detail {

/** The fully-portable entry point: `gemv_rows<Plane,false>` for every format, unconditionally -- the
 * correctness reference every other path here is checked against (tests/backbone_quant_dot_tests.cpp),
 * and the only path available on a non-AVX2 build.
 */
[[nodiscard]] inline bool gemv_plane_portable(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                              int n_rows, int row_elems, const ActBlocks& x, float* out,
                                              int row_lo, int row_hi) {
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q8_0:
            gemv_rows<Q8_0Plane, false>(Q8_0Plane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        case gguf::TensorType::Q4_K:
            gemv_rows<Q4KPlane, false>(Q4KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        case gguf::TensorType::Q5_K:
            gemv_rows<Q5KPlane, false>(Q5KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        case gguf::TensorType::Q6_K:
            gemv_rows<Q6KPlane, false>(Q6KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        default:
            return false;
    }
}

#if defined(SUB0_BBQD_AVX2)
/** The fast entry point. Q8_0 always runs `gemv_rows<Q8_0Plane,true>` (its native block already IS one
 * GROUP -- no superblock to stream, and it already cleared the throughput gate in phase 1, S8a). Q4_K/
 * Q5_K/Q6_K run the streaming kernels above when `row_elems % 256 == 0` (every real backbone tensor,
 * S2a); otherwise they fall back to `gemv_rows<Plane,true>`, which accepts any `row_elems` that is merely
 * a multiple of GROUP=32 -- correctness is never conditional on the 256 alignment, only which kernel
 * gets used.
 */
// `gsum16_hint`: caller-owned, ALREADY-BUILT Q6_K per-16 sums (gemv_plane<Threads>'s own job to build it
// exactly once, before any thread starts -- see that function's comment). nullptr means "build a local,
// one-call one" -- the pre-O5-2b-2b behavior, kept for callers with no persistent scratch to hand in
// (tests, the microbenchmark).
[[nodiscard]] inline bool gemv_plane_avx2(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                          int n_rows, int row_elems, const ActBlocks& x, float* out,
                                          int row_lo, int row_hi, const Gsum16* gsum16_hint = nullptr) {
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    const auto type = static_cast<gguf::TensorType>(type_raw);
    if (type == gguf::TensorType::Q8_0) {
        gemv_rows<Q8_0Plane, true>(Q8_0Plane{raw.data()}, row_lo, row_hi, row_elems, x, out);
        return true;
    }
    if (row_elems % 256 != 0) {
        switch (type) {
            case gguf::TensorType::Q4_K:
                gemv_rows<Q4KPlane, true>(Q4KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
                return true;
            case gguf::TensorType::Q5_K:
                gemv_rows<Q5KPlane, true>(Q5KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
                return true;
            case gguf::TensorType::Q6_K:
                gemv_rows<Q6KPlane, true>(Q6KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
                return true;
            default:
                return false;
        }
    }
    switch (type) {
        case gguf::TensorType::Q4_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q4_k_avx2(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q5_K:
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q5_k_avx2(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x);
            return true;
        case gguf::TensorType::Q6_K: {
            // O5 phase 2b-2b: closes the TODO(phase 2b) this line used to carry (docs/BACKBONE_NATIVE_
            // QUANT.md S12i). `gsum16_hint`, when the caller supplied one, is ALREADY built for this
            // call's own `x` -- built exactly once in gemv_plane<Threads>, before any thread split, never
            // per row and never per thread. Falling back to a local one-call build keeps every OTHER
            // caller (tests, the microbenchmark) byte-for-byte unchanged.
            Gsum16 local_gsum16;
            const Gsum16* gsum16 = gsum16_hint;
            if (!gsum16) { local_gsum16.build(x); gsum16 = &local_gsum16; }
            for (int r = row_lo; r < row_hi; ++r)
                out[static_cast<std::size_t>(r - row_lo)] = dot_row_q6_k_avx2(
                    raw.data(), static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems),
                    row_elems, x, *gsum16);
            return true;
        }
        default:
            return false;
    }
}
#endif  // SUB0_BBQD_AVX2

/// The kernel gemv_plane<Threads> runs for one row range: gemv_plane_avx2 where the target has AVX2
/// (kAvx2Kernels), else gemv_plane_portable. Mirrors moeqd::detail::gemv_best's own convention exactly.
/// `gsum16_hint`: forwarded to gemv_plane_avx2 unchanged (see its own comment); gemv_plane_portable never
/// uses Gsum16 at all (its Q6_K path is the group()-based one, not the streaming one), so this parameter
/// is simply unused on that arm -- [[maybe_unused]] would be redundant here since it is still named and
/// read on the AVX2 arm.
[[nodiscard]] inline bool gemv_plane_dispatch(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                              int n_rows, int row_elems, const ActBlocks& x, float* out,
                                              int row_lo, int row_hi, const Gsum16* gsum16_hint = nullptr) {
#if defined(SUB0_BBQD_AVX2)
    if constexpr (kAvx2Kernels)
        return gemv_plane_avx2(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi, gsum16_hint);
#endif
    (void)gsum16_hint;
    return gemv_plane_portable(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi);
}

}  // namespace detail

/** One plane's GEMV over output rows [row_lo, row_hi) (default: the full range), dispatched on the
 * plane's own `type_raw` (never its role -- unsloth's per-tensor mixed quantization, the same reasoning
 * moe_quant_dot.hpp's own gemv_plane documents) and auto-selecting the AVX2 or portable kernel from the
 * compiled target (`bbqd::kAvx2Kernels`) -- the same convention `moeqd::gemv_plane`/`gemv_best` already
 * establish, not a caller-chosen template flag (AGENTS.md S2: there is no runtime OR call-site choice to
 * make here, only a compile-time ISA fact).
 *
 * @tparam Threads  compile-time fan-out (1 = serial, the default). `Threads > 1` splits [row_lo, row_hi)
 *         into `Threads` contiguous ranges across a persistent OpenMP team, mirroring `gemv::axpy`'s own
 *         shape exactly, including its `omp_in_parallel()` nesting guard (a call made from inside an
 *         existing OpenMP team runs serially rather than nesting). Each range reads disjoint plane bytes
 *         and writes a disjoint `out` slice with no shared mutable state, so the result is bit-identical
 *         regardless of thread count (tests/backbone_quant_dot_tests.cpp's own "threading" case).
 * @return false if the format is unfusable, `raw` is shorter than the geometry requires, or the row range
 *         is out of bounds -- `out` is left completely untouched in every refusal case, checked BEFORE
 *         any thread starts (plane_geometry_ok), not merely before this thread's own portion.
 * @param gsum16 O5 phase 2b-2b: optional CALLER-OWNED scratch for Q6_K's per-16 activation sums
 *        (docs/BACKBONE_NATIVE_QUANT.md S12i's own TODO). nullptr (the default) preserves every existing
 *        caller's exact behavior (a local, one-call `Gsum16` built inside the AVX2 kernel, same as before
 *        this parameter existed). When non-null, THIS function builds it ONCE, here, BEFORE any thread
 *        starts -- never per row, never per thread -- and every thread/row-range below only READS it
 *        (build() already finished, serially, by the time the parallel region opens), so sharing one
 *        instance across the whole call is safe. `build()` reuses `gsum16->v`'s existing capacity when
 *        the activation width has not changed (the same "assign only reallocates on a real size change"
 *        contract `ActBlocks::quantize` already has), so a caller that owns one `Gsum16` per (thread,
 *        role) across a whole decode generation pays its heap cost once, not every token -- the AGENTS.md
 *        S1 fix this parameter exists for. Passed even for a non-Q6_K plane (build() is then simply
 *        unused work downstream, not incorrect) -- cheap enough (O(row_elems/16)) that the caller need
 *        not special-case which roles are Q6_K before deciding whether to pass one.
 */
template <int Threads = 1>
[[nodiscard]] inline bool gemv_plane(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                     int n_rows, int row_elems, const ActBlocks& x, float* out,
                                     int row_lo = 0, int row_hi = -1, Gsum16* gsum16 = nullptr) {
    static_assert(Threads >= 1, "Threads must be >= 1");
    if (row_hi < 0) row_hi = n_rows;
    if (!plane_geometry_ok(type_raw, raw, n_rows, row_elems, row_lo, row_hi)) return false;
    if (gsum16) gsum16->build(x);   // once, serially, before any thread below can read it
    const Gsum16* gsum16_hint = gsum16;
    if constexpr (Threads == 1) {
        return detail::gemv_plane_dispatch(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi,
                                           gsum16_hint);
    } else {
#if defined(_OPENMP)
        if (!omp_in_parallel()) {
            bool ok = true;
            const int total = row_hi - row_lo;
            const int per = (total + Threads - 1) / Threads;
            #pragma omp parallel for num_threads(Threads) schedule(static)
            for (int t = 0; t < Threads; ++t) {
                const int lo = row_lo + std::min(total, t * per);
                const int hi = row_lo + std::min(total, (t + 1) * per);
                if (lo < hi) {
                    // gemv_plane_dispatch writes out[r - row_lo_of_THIS_call] for r in [lo, hi) -- offset
                    // the pointer by (lo - row_lo) so that lands at the SAME absolute out[r - row_lo] a
                    // single-threaded call would have used, since every thread shares one `out` buffer.
                    // gsum16_hint is read-only here (its own build() above already finished before this
                    // parallel region opened), so every thread sharing the one pointer is safe.
                    const bool arm_ok = detail::gemv_plane_dispatch(
                        type_raw, raw, n_rows, row_elems, x, out + (lo - row_lo), lo, hi, gsum16_hint);
                    if (!arm_ok) {
                        #pragma omp atomic write
                        ok = false;
                    }
                }
            }
            return ok;
        }
#endif
        return detail::gemv_plane_dispatch(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi,
                                           gsum16_hint);
    }
}

// --- O5 phase 2b-3, phase A: the format-agnostic plane view's GEMV overload ----------------------------
//
// `Plane` itself is declared near the TOP of this file, before the `moe_quant_dot.hpp` include -- see
// that declaration's own comment for why (a real include cycle once moe_math.hpp also needs it, phase B).
// WHY Plane EXISTS AT ALL. Before this pass, gdn_math.hpp (and, as of phase B, gated_residual_math.hpp/
// qsa_math.hpp/moe_math.hpp) included backbone_quant.hpp directly and stored a `const bbq::Desc*` +
// its raw byte span in their own `Native` structs -- coupling every math header that wants a native-quant
// weight source to the S0B1 SIDECAR'S OWN on-disk format (bbq::Desc's role/layer/off/bytes fields, none of
// which a GEMV kernel needs). That is a real layering inversion: the math headers are this project's
// storage-agnostic forward-math core (this file's own header comment, "engine-free"), and backbone_quant.hpp
// is one PARTICULAR storage format for one particular residency strategy. A future second storage form
// (e.g. a resident dequantized cache, or a different sidecar shape) would have had to either grow a second
// Native variant per math header or force every math header to learn a second storage type.
//
// This is the fix: a plane is fully described by the four fields `gemv_plane(type_raw, raw, n_rows,
// row_elems, ...)` already takes as loose parameters -- so naming that tuple once, here, in the KERNEL
// header (not the storage header), lets every math header depend on `backbone_quant_dot.hpp` alone. The
// caller (decode.cpp, which already depends on both bbq::Store for loading and bbqd for kernels) is the
// one place that converts a `bbq::Desc` (or any other future storage format) into a `Plane`.

/** `gemv_plane<Threads>` over a `Plane` view, otherwise identical to the loose-parameter overload above
 * (same defaults, same row-range/threading/Gsum16 contract) -- see that overload's own doc comment for the
 * full behavior. This is the ONE overload every `*_math.hpp` Native struct calls; the loose-parameter form
 * stays available for existing test/benchmark callers.
 */
template <int Threads = 1>
[[nodiscard]] inline bool gemv_plane(const Plane& p, const ActBlocks& x, float* out,
                                     int row_lo = 0, int row_hi = -1, Gsum16* gsum16 = nullptr) {
    return gemv_plane<Threads>(p.type_raw, p.bytes, p.n_rows, p.row_elems, x, out, row_lo, row_hi, gsum16);
}

}  // namespace sub0::bbqd
