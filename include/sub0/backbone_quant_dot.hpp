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
// PERFORMANCE STATUS: compute-bound, NOT yet a win. Only the integer DOT is vectorized; the per-group
// UNPACK (group() below) is element-by-element and dominates. Measured against the real bf16
// gemv::axpy at integration (docs/BACKBONE_NATIVE_QUANT.md S8a): Q8_0 ~26 GB/s/thread, but Q6_K ~5.3,
// Q5_K ~4.3, Q4_K ~1.1 -- slower than bf16 at one thread despite 2.4-3.2x fewer bytes. Phase 2's first
// gate is a streaming unpack, >= 10 GB/s of compressed bytes per thread.
//
// NO HEAP ALLOCATION PER CALL (AGENTS.md S1): every kernel here reads only its caller-supplied spans and
// writes only its caller-supplied `out` pointer; no std::vector/new/malloc appears in any hot path below.

#pragma once

#include "sub0/gguf.hpp"
#include "sub0/moe_quant_dot.hpp"   // moeqd::ActBlocks, moeqd::GROUP -- reused verbatim, not re-derived

#include <array>
#include <cstdint>
#include <cstring>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sub0::bbqd {

using moeqd::ActBlocks;
inline constexpr int GROUP = moeqd::GROUP;   // 32 -- see the file header for why this is the right width

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

/** One plane's whole GEMV (all rows), dispatched on the plane's own `type_raw` -- never on its role
 * (unsloth's per-tensor mixed quantization, the same reasoning moe_quant_dot.hpp's own gemv_plane
 * documents). `row_lo`/`row_hi` default to the full range; pass a sub-range directly for threading.
 *
 * @return false if the format is unfusable or `raw` is shorter than the geometry requires -- `out` is
 *         left untouched in either case, matching moeqd::gemv_plane's own contract.
 */
template <bool UseAvx2 = false>
[[nodiscard]] inline bool gemv_plane(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                     int n_rows, int row_elems, const ActBlocks& x, float* out,
                                     int row_lo = 0, int row_hi = -1) {
    if (row_hi < 0) row_hi = n_rows;
    if (row_lo < 0 || row_hi > n_rows || row_lo > row_hi) return false;
    const std::uint64_t need = plane_bytes(type_raw, static_cast<std::uint64_t>(n_rows)
                                                      * static_cast<std::uint64_t>(row_elems));
    if (need == 0 || raw.size() < need) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::Q8_0:
            gemv_rows<Q8_0Plane, UseAvx2>(Q8_0Plane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        case gguf::TensorType::Q4_K:
            gemv_rows<Q4KPlane, UseAvx2>(Q4KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        case gguf::TensorType::Q5_K:
            gemv_rows<Q5KPlane, UseAvx2>(Q5KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        case gguf::TensorType::Q6_K:
            gemv_rows<Q6KPlane, UseAvx2>(Q6KPlane{raw.data()}, row_lo, row_hi, row_elems, x, out);
            return true;
        default:
            return false;
    }
}

}  // namespace sub0::bbqd
