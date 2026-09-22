// sub0/moe_quant_dot.hpp -- B35: fused quantized dot products taken DIRECTLY against the `.moeq`
// sidecar's own native GGUF bytes, for decode's routed-expert resolve. Design brief:
// docs/MOE_QUANT_DOT.md.
//
// WHAT THIS REPLACES, AND WHY IT DOES NOT CONTRADICT B24 PHASE 2. Today's decode resolve
// (moeq::ExpertCacheSource + moe::expert_ffn_row_source, B31) reads an expert's 1.55 MiB of encoded
// bytes, WRITES 18.75 MiB of f32 planes, READS those 18.75 MiB back exactly once inside the FFN, then
// discards them: 39.05 MiB of DRAM traffic to consume 1.55 MiB of actual weight. B24 Phase 2 measured
// "dequantize once into a resident buffer" beating "dequantize inline per read" by 5-30x -- but at the
// BACKBONE's dense access density, where that buffer is re-read every token and the dequant cost
// amortizes over all those reads. Decode's resolve pool has a hit rate that is PROVABLY zero
// (src/backends/cpu/internal.hpp's MOE_DECODE_SLOTS comment derives it: a token's top-k indices are
// distinct by construction, the cache key includes the layer, and 480 resolves/token round-robin any
// pool several times over). With nothing to amortize against, the materialization is pure overhead and
// the ledger inverts. docs/MOE_QUANT_DOT.md S2 is the full argument.
//
// WHAT IT COMPUTES. Each projection is a GEMV whose weights are unpacked in registers, 32 elements at a
// time, and multiplied against an int8-quantized copy of the activation row; both operands stay
// integers through the multiply-accumulate and only the per-group scale re-enters float. Technique read
// from -- not vendored out of -- llama.cpp's `ggml_vec_dot_iq1_s_q8_K` / `ggml_vec_dot_iq2_xxs_q8_K` /
// `ggml_vec_dot_iq4_nl_q8_0` (D:\Craig\llama.cpp-qwen4exp, ggml/src/ggml-cpu/arch/x86/quants.c), with
// AGENTS.md S5's "re-derive the reference's conventions onto this project's own" applied literally: the
// block layouts stay gguf.hpp's single definition (AGENTS.md S3), and every unpack below was
// cross-read against gguf.hpp's OWN dequantize_iq1_s/dequantize_iq2_xxs/dequantize_iq4_nl AND against
// ggml-quants.c's dequantize_row_* scalar references. All three formats' layouts agreed exactly.
//
// TWO KERNELS: PORTABLE, AND AVX2 INTRINSICS (O1). B35 shipped portable-only, on the measured grounds
// that the integer inner dot auto-vectorizes (`vpmovsxbw`/`vpmaddwd` for two formats,
// `vpmovsxbd`/`vpmulld`/`vphaddd` for the third) -- see detail::dot_group. That was true and
// answered the wrong question: "does the dot vectorize" is not "is the loop the right shape". The
// post-B35 roofline (docs/optimization/roofline_post_b35.md) put the fused path at 2.2% of the AVX2 int8
// ceiling, because every 32-element group still ends in a horizontal reduction, a convert and a serial
// scalar accumulate. detail::gemv_avx2 keeps the accumulator in a vector for the whole row, which a
// portable loop cannot express without the float reassociation Clang refuses (B34/B38). The layout
// knowledge is NOT duplicated: each unpacker's bit fields are read in one `fields()` function that both
// materialisations share; only the final register form differs. The portable kernel stays as the
// non-AVX2 fallback and as the AVX2 kernel's test reference (docs/optimization/opportunities/O1_*.md).
//
// NOT BIT-EXACT, BY CONSTRUCTION -- one genuinely new error source, with no precedent in this codebase.
// The WEIGHT side adds none: the integer path is algebraically the same expression gguf::to_f32
// evaluates, with the per-group scale factored out of the sum. The ACTIVATION side does: x is quantized
// to int8 per 32-element group before the dot. tests/moe_quant_tests.cpp's "B35 fused quantized dot"
// case isolates exactly that term against real sidecar-format bytes, so the end-to-end logit diff can
// be explained rather than merely observed.
//
// GROUP SIZE 32, NOT q8_K's 256. llama.cpp quantizes the activation into `block_q8_K` (one scale per
// 256 elements) because its kernels serve a general [n x k] GEMM whose k is a multiple of 256. This
// engine's real shapes are not: the down projection's rows are d_ff = 640 elements, so row j begins at
// element j*640 -- a HALF-block offset inside a 256-element IQ1_S/IQ2_XXS super-block for every odd j.
// All three formats are internally structured in 32-element groups, though (IQ1_S's eight per-32 `qh`
// scales, IQ2_XXS's eight per-32 `aux32` pairs, IQ4_NL's blocks ARE 32), so 32 is the granularity at
// which this engine's real row shapes and every format simultaneously align: 2560 % 32 == 0 and
// 640 % 32 == 0. A per-32 activation scale is also strictly more accurate than a per-256 one, which is
// a free consequence rather than the goal.
//
// NO HEAP ALLOCATION PER CALL (AGENTS.md S1): ActBlocks is caller-owned and sized once; every kernel
// writes only into caller-supplied buffers and its own stack.

#pragma once

#include "sub0/gguf.hpp"
#include "sub0/gguf_quant_tables.hpp"
#include "sub0/moe_math.hpp"
#include "sub0/moe_quant.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define SUB0_MOEQD_AVX2 1
#endif

namespace sub0::moeqd {

/** Whether gemv_plane runs the AVX2 kernels (detail::gemv_avx2) or the portable ones (detail::gemv).
 *
 * Decided by the compiler's target ISA, which a SUB0_NATIVE build fixes at `-march=native`. There is no
 * runtime or configurator knob because there is no choice to make: where AVX2 exists the vector kernel
 * is the one to run. The portable kernel is kept, not deleted -- it is the fallback for non-AVX2
 * targets AND the reference the AVX2 kernel is tested against (tests/moe_quant_tests.cpp, "O1").
 */
#if defined(SUB0_MOEQD_AVX2)
inline constexpr bool kAvx2Kernels = true;
#else
inline constexpr bool kAvx2Kernels = false;
#endif

/// The one granularity at which this engine's real row shapes (hidden_size 2560, d_ff 640) and all
/// three sidecar formats' internal sub-block structure simultaneously align -- see the file header.
inline constexpr int GROUP = 32;

// --- the quantized activation row -----------------------------------------------------------------

/** One row of activations as int8 groups of GROUP elements, each carrying its own float scale and its
 * own int32 sum-of-quants.
 *
 * The sum is not an optimization: IQ1_S's decoded value is `dl * (grid + delta)` with a per-group
 * delta, so its dot product needs `sum_j q_j` as a separate term -- the same job llama.cpp's
 * `block_q8_K::bsums` does at its own coarser granularity.
 *
 * @note Caller-owned and sized once (AGENTS.md S1); quantize() allocates only on the first call for a
 *       given width, and decode reuses one instance per layer for the whole run.
 */
struct ActBlocks {
    std::vector<std::int8_t>  qs;       ///< n quantized values
    std::vector<float>        scale;    ///< n / GROUP group scales
    std::vector<std::int32_t> gsum;     ///< n / GROUP group sums of qs
    int                       n = 0;    ///< element count this instance is currently sized for

    /** Symmetric round-to-nearest int8 quantization of `n_elems` floats, one scale per GROUP.
     *
     * The only way to put an ActBlocks into a usable state -- there is deliberately no separate
     * `resize()`, so the object is never sized-but-stale.
     *
     * @note The divisor is 127, not 128, so the encoding is symmetric and -128 never occurs. That
     *       bound is what keeps the products below in range for a 16-bit intermediate (127*127*2 =
     *       32258 < 32767), which is the same headroom argument llama.cpp's own kernels rely on.
     * @note `n_elems` must be a multiple of GROUP -- guaranteed by fusable() at the one call site.
     * @note Allocates only when the width changes, so the steady state is allocation-free
     *       (AGENTS.md S1): decode reuses one instance per layer for the whole run.
     */
    void quantize(const float* x, int n_elems) {
        n = n_elems;
        const auto elems  = static_cast<std::size_t>(n_elems);
        const auto groups = static_cast<std::size_t>(n_elems / GROUP);
        if (qs.size() != elems)     qs.assign(elems, 0);
        if (scale.size() != groups) scale.assign(groups, 0.f);
        if (gsum.size() != groups)  gsum.assign(groups, 0);
        for (int g = 0, ng = n_elems / GROUP; g < ng; ++g) {
            const float* xg = x + static_cast<std::size_t>(g) * GROUP;
            float amax = 0.f;
            for (int j = 0; j < GROUP; ++j) amax = std::max(amax, std::fabs(xg[j]));
            const float inv = (amax > 0.f) ? (127.f / amax) : 0.f;
            std::int8_t* qg = qs.data() + static_cast<std::size_t>(g) * GROUP;
            std::int32_t sum = 0;
            for (int j = 0; j < GROUP; ++j) {
                const int q = std::clamp(static_cast<int>(std::lrintf(xg[j] * inv)), -127, 127);
                qg[j] = static_cast<std::int8_t>(q);
                sum += q;
            }
            scale[static_cast<std::size_t>(g)] = amax / 127.f;
            gsum[static_cast<std::size_t>(g)]  = sum;
        }
    }
};

// --- signs table, DERIVED rather than transcribed --------------------------------------------------

/** IQ2_XXS's eight per-group signs as +1/-1 int8 bytes, indexed by the format's own 7-bit sign index.
 *
 * gguf.hpp's KSIGNS_IQ2XS already owns the expansion of that index into a bitmask (its 8th sign is the
 * parity bit); this derives the byte form FROM it at compile time rather than transcribing ggml's
 * second table `keven_signs_q2xs`, so there stays exactly one definition of what the sign bits mean
 * (AGENTS.md S3).
 */
inline constexpr std::array<std::uint64_t, 128> make_signs64() {
    std::array<std::uint64_t, 128> t{};
    for (int k = 0; k < 128; ++k) {
        std::uint64_t w = 0;
        for (int j = 0; j < 8; ++j) {
            const std::uint64_t b = (gguf::KSIGNS_IQ2XS[k] & gguf::KMASK_IQ2XS[j]) ? 0xFFu : 0x01u;
            w |= b << (8 * j);
        }
        t[static_cast<std::size_t>(k)] = w;
    }
    return t;
}
inline constexpr std::array<std::uint64_t, 128> SIGNS64 = make_signs64();

// --- one decoded weight group, and the three things that produce one -------------------------------

/** GROUP consecutive weights of one plane, decoded into the integer form the shared dot consumes.
 *
 * `scale` and `delta` reproduce the format's own `value = scale * (q + delta)` relation exactly, so
 * folding them out of the inner loop is an algebraic identity rather than an approximation.
 */
struct WeightGroup {
    std::array<std::int8_t, GROUP> q{};
    float scale = 0.f;
    float delta = 0.f;   ///< IQ1_S's per-group additive offset; unused (and never read) elsewhere
};

#if defined(SUB0_MOEQD_AVX2)
/** The AVX2 form of one decoded group: 32 weights as UNSIGNED magnitudes plus a separate sign source.
 *
 * Split this way because it is the operand shape `vpmaddubsw` wants (unsigned x signed): the kernel
 * moves each weight's sign onto the activation with `vpsignb` and multiplies magnitudes. IQ2_XXS stores
 * exactly this split natively (grid magnitudes + a sign table), so it hands both over as-is -- no negate
 * pass at all, which is the sign-fold the O1 brief asked for. IQ1_S and IQ4_NL decode to signed weights
 * and pass `mag = |w|`, `sgn = w`.
 */
struct WeightGroupV {
    __m256i mag;           ///< 32 x uint8 |w|
    __m256i sgn;           ///< 32 x int8 carrying w's sign in its own sign; value otherwise unused
    float   scale = 0.f;
    float   delta = 0.f;   ///< IQ1_S only, as WeightGroup::delta
};
#endif

/** Unpacks one GROUP of IQ1_S: 256 elements per 50-byte block = f16 d, 32 grid-index low bytes, then
 * eight u16 `qh` each packing three high grid-index bits per 8-element sub-group at [3l+2:3l], a 3-bit
 * scale at [14:12], and the delta's sign at bit 15. The grid is a ternary codebook, so `q` is already
 * the decoded integer and the whole per-element value is `scale * (q + delta)`.
 */
struct Iq1SPlane {
    static constexpr bool kHasDelta = true;
    const std::uint8_t* plane = nullptr;   // non-owning; the sidecar mapping outlives every resolve

    /// The group's bit fields: the ONE place this format's layout is read. group() and group_v() differ
    /// only in how they materialise the result.
    struct Fields {
        std::uint16_t       d_bits = 0;
        std::uint16_t       qh     = 0;
        const std::uint8_t* qs     = nullptr;   // non-owning; into `plane`
    };
    [[nodiscard]] Fields fields(std::uint64_t p) const noexcept {
        const std::uint8_t* blk = plane + (p / 256) * 50;
        const int ib = static_cast<int>((p % 256) / 32);
        Fields f;
        f.qs = blk + 2 + 4 * ib;
        std::memcpy(&f.d_bits, blk, sizeof f.d_bits);
        std::memcpy(&f.qh, blk + 34 + 2 * ib, sizeof f.qh);
        return f;
    }
    /// Eight int8 weights, little-endian.
    [[nodiscard]] static std::uint64_t grid(const Fields& f, int l) noexcept {
        return gguf::IQ1S_GRID[f.qs[l] | (((f.qh >> (3 * l)) & 7) << 8)];
    }
    [[nodiscard]] static float scale(const Fields& f, float d) noexcept {
        return d * static_cast<float>(2 * ((f.qh >> 12) & 7) + 1);
    }
    [[nodiscard]] static float delta(const Fields& f) noexcept {
        return (f.qh & 0x8000) ? -gguf::IQ1S_DELTA : gguf::IQ1S_DELTA;
    }

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const Fields f = fields(p);
        WeightGroup wg;
        wg.scale = scale(f, gguf::f16_to_f32(f.d_bits));
        wg.delta = delta(f);
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t g = grid(f, l);
            std::memcpy(wg.q.data() + 8 * l, &g, sizeof g);
        }
        return wg;
    }

#if defined(SUB0_MOEQD_AVX2)
    [[nodiscard]] WeightGroupV group_v(std::uint64_t p) const noexcept {
        const Fields f = fields(p);
        const auto ll = [](std::uint64_t v) { return static_cast<long long>(v); };
        const __m256i w = _mm256_set_epi64x(ll(grid(f, 3)), ll(grid(f, 2)), ll(grid(f, 1)), ll(grid(f, 0)));
        return {_mm256_sign_epi8(w, w), w, scale(f, _cvtsh_ss(f.d_bits)), delta(f)};
    }
#endif
};

/** Unpacks one GROUP of IQ2_XXS: 256 elements per 66-byte block = f16 d, then eight groups of two u32.
 * The first u32's four bytes are grid indices; the second's top nibble is the group scale and its low
 * 28 bits are four 7-bit sign indices. The grid stores unsigned MAGNITUDES (only 0x08/0x19/0x2b occur,
 * so the signed product below cannot overflow), with the signs carried separately.
 */
struct Iq2XxsPlane {
    static constexpr bool kHasDelta = false;
    const std::uint8_t* plane = nullptr;   // non-owning; the sidecar mapping outlives every resolve

    /// The group's bit fields: the ONE place this format's layout is read (see Iq1SPlane::Fields).
    struct Fields {
        std::uint16_t d_bits = 0;
        std::uint32_t aux[2] = {0, 0};   ///< [0]: four grid-index bytes; [1]: scale nibble + 4 x 7-bit signs
    };
    [[nodiscard]] Fields fields(std::uint64_t p) const noexcept {
        const std::uint8_t* blk = plane + (p / 256) * 66;
        const int ib32 = static_cast<int>((p % 256) / 32);
        Fields f;
        std::memcpy(&f.d_bits, blk, sizeof f.d_bits);
        std::memcpy(f.aux, blk + 2 + 8 * ib32, sizeof f.aux);
        return f;
    }
    /// Eight unsigned magnitudes, little-endian.
    [[nodiscard]] static std::uint64_t grid(const Fields& f, int l) noexcept {
        return gguf::IQ2XXS_GRID[(f.aux[0] >> (8 * l)) & 0xFFu];
    }
    /// Eight +1/-1 bytes, little-endian.
    [[nodiscard]] static std::uint64_t signs(const Fields& f, int l) noexcept {
        return SIGNS64[(f.aux[1] >> (7 * l)) & 127u];
    }
    [[nodiscard]] static float scale(const Fields& f, float d) noexcept {
        return d * (0.5f + static_cast<float>(f.aux[1] >> 28)) * 0.25f;
    }

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const Fields f = fields(p);
        WeightGroup wg;
        wg.scale = scale(f, gguf::f16_to_f32(f.d_bits));
        std::array<std::int8_t, GROUP> sign{};
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t g = grid(f, l), s = signs(f, l);
            std::memcpy(wg.q.data() + 8 * l, &g, sizeof g);
            std::memcpy(sign.data() + 8 * l, &s, sizeof s);
        }
        for (int j = 0; j < GROUP; ++j) wg.q[j] = static_cast<std::int8_t>(wg.q[j] * sign[j]);
        return wg;
    }

#if defined(SUB0_MOEQD_AVX2)
    /// The format's own magnitude/sign split, handed straight to the kernel: no negate pass.
    [[nodiscard]] WeightGroupV group_v(std::uint64_t p) const noexcept {
        const Fields f = fields(p);
        const auto ll = [](std::uint64_t v) { return static_cast<long long>(v); };
        return {_mm256_set_epi64x(ll(grid(f, 3)), ll(grid(f, 2)), ll(grid(f, 1)), ll(grid(f, 0))),
                _mm256_set_epi64x(ll(signs(f, 3)), ll(signs(f, 2)), ll(signs(f, 1)), ll(signs(f, 0))),
                scale(f, _cvtsh_ss(f.d_bits)), 0.f};
    }
#endif
};

/** Unpacks one GROUP of IQ4_NL, whose blocks ARE 32 elements: f16 d plus 16 packed nibble pairs. Byte
 * j holds element j in its LOW nibble and element j+16 in its HIGH nibble -- NOT adjacent pairs, the
 * same non-obvious split gguf.hpp's own decoder calls out.
 */
struct Iq4NlPlane {
    static constexpr bool kHasDelta = false;
    const std::uint8_t* plane = nullptr;   // non-owning; the sidecar mapping outlives every resolve

    /// The group's fields: the ONE place this format's layout is read (see Iq1SPlane::Fields).
    struct Fields {
        std::uint16_t       d_bits = 0;
        const std::uint8_t* qn     = nullptr;   ///< non-owning; the block's 16 packed nibble-pair bytes
    };
    [[nodiscard]] Fields fields(std::uint64_t p) const noexcept {
        const std::uint8_t* blk = plane + (p / 32) * 18;
        Fields f;
        f.qn = blk + 2;
        std::memcpy(&f.d_bits, blk, sizeof f.d_bits);
        return f;
    }

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const Fields f = fields(p);
        const std::uint8_t* qn = f.qn;

        WeightGroup wg;
        wg.scale = gguf::f16_to_f32(f.d_bits);
        for (int j = 0; j < 16; ++j) {
            wg.q[static_cast<std::size_t>(j)]      = gguf::KVALUES_IQ4NL[qn[j] & 0xF];
            wg.q[static_cast<std::size_t>(j) + 16] = gguf::KVALUES_IQ4NL[qn[j] >> 4];
        }
        return wg;
    }

#if defined(SUB0_MOEQD_AVX2)
    /// Both nibble halves decoded by ONE `vpshufb` against the 16-entry codebook broadcast to both
    /// 128-bit lanes: low nibbles land in bytes 0-15 (elements 0-15), high nibbles in 16-31.
    [[nodiscard]] WeightGroupV group_v(std::uint64_t p) const noexcept {
        const Fields f = fields(p);
        const __m128i raw  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f.qn));
        const __m128i m4   = _mm_set1_epi8(0x0F);
        const __m256i idx  = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), _mm_and_si128(raw, m4));
        const __m256i book = _mm256_broadcastsi128_si256(
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(gguf::KVALUES_IQ4NL)));
        const __m256i w = _mm256_shuffle_epi8(book, idx);
        return {_mm256_sign_epi8(w, w), w, _cvtsh_ss(f.d_bits), 0.f};
    }
#endif
};

// --- the shared seam: one GEMV body, one integer MAC, three unpackers ------------------------------

namespace detail {

/** Exact 32-element signed int8 dot product -- the one place the inner multiply-accumulate is written.
 *
 * Integer, so there is no reassociation barrier (unlike the float reductions B34/B38 had to coax) and
 * Clang vectorizes this on its own at every call site. Verified in the generated assembly rather than
 * assumed, and the two shapes it picks are both real vector code: where the weight bytes reach it
 * through memory (IQ2_XXS, IQ4_NL) it emits `vpmovsxbw`+`vpmaddwd`, 16 lanes per multiply; where they
 * arrive as four register-resident qwords (IQ1_S) it emits `vpmovsxbd`+`vpmulld`+`vphaddd` instead, 8
 * lanes per multiply. `#pragma clang loop vectorize(enable)` was tried on this loop and changed
 * neither -- there is no scalar case to rescue, so it is not carried.
 */
[[nodiscard]] inline int dot_group(const std::int8_t* w, const std::int8_t* q) {
    int s = 0;
    for (int j = 0; j < GROUP; ++j) s += static_cast<int>(w[j]) * static_cast<int>(q[j]);
    return s;
}

/** One plane's whole GEMV: `out[r] = sum_i plane[r * row_elems + i] * x[i]` for r in [0, n_rows).
 *
 * The only thing that varies between the three formats is `Plane::group()`; the row walk, the integer
 * MAC, the scale fold and the delta term are written once, here. `Plane::kHasDelta` compiles the delta
 * term away entirely for the two formats that do not have one, so sharing the body costs them nothing.
 *
 * Accumulation is group-ascending, matching the element order a sequential f32 dot would use; each
 * group's integer part is exact, so the only float rounding is the one accumulate per group.
 */
template <class Plane>
void gemv(const Plane& plane, int n_rows, int row_elems, const ActBlocks& x, float* out) {
    const int ng = row_elems / GROUP;
    for (int r = 0; r < n_rows; ++r) {
        const std::uint64_t base = static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems);
        float acc = 0.f;
        for (int g = 0; g < ng; ++g) {
            const std::size_t gi = static_cast<std::size_t>(g);
            const WeightGroup wg = plane.group(base + static_cast<std::uint64_t>(g) * GROUP);
            const int isum = dot_group(wg.q.data(), x.qs.data() + gi * GROUP);
            float term = static_cast<float>(isum);
            if constexpr (Plane::kHasDelta) term += wg.delta * static_cast<float>(x.gsum[gi]);
            acc += x.scale[gi] * wg.scale * term;
        }
        out[r] = acc;
    }
}

#if defined(SUB0_MOEQD_AVX2)
/// Horizontal sum of eight floats.
[[nodiscard]] inline float hsum(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

/** gemv()'s computation with the accumulator held in a vector register for the whole row.
 *
 * WHY, when the portable dot already vectorizes: every 32-element group there ends in a horizontal
 * reduction to one int, an int->float convert and a serial scalar accumulate -- per-group overhead on
 * the order of the 32 MACs themselves, which is how the fused path sat at 2.2% of the AVX2 int8 ceiling
 * (docs/optimization/roofline_post_b35.md). Here a group's partial sums stay as eight int32 lanes, are
 * scaled as a vector and fold into a float vector accumulator; the horizontal reduction happens once
 * per ROW.
 *
 * The MAC is the vpmaddubsw shape llama.cpp's AVX2 IQ kernels use: each weight's sign moves onto the
 * activation (`vpsignb`), then unsigned magnitude x signed activation. In range for the reason
 * ActBlocks::quantize documents -- |w|, |q| <= 127, so a pair sum is <= 32258 and the saturating int16
 * add never saturates. Every int32 lane is exact and converts to float exactly (< 2^24).
 *
 * NOT bit-identical to gemv(): per-group float products are summed per lane and then across lanes,
 * not group-sequentially. The difference is float reassociation only; test "O1" bounds it.
 */
template <class Plane>
void gemv_avx2(const Plane& plane, int n_rows, int row_elems, const ActBlocks& x, float* out) noexcept {
    const int ng = row_elems / GROUP;
    const __m256i ones = _mm256_set1_epi16(1);
    for (int r = 0; r < n_rows; ++r) {
        const std::uint64_t base = static_cast<std::uint64_t>(r) * static_cast<std::uint64_t>(row_elems);
        __m256 acc  = _mm256_setzero_ps();
        float  dacc = 0.f;
        for (int g = 0; g < ng; ++g) {
            const std::size_t gi = static_cast<std::size_t>(g);
            const WeightGroupV wg = plane.group_v(base + static_cast<std::uint64_t>(g) * GROUP);
            const __m256i q   = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.qs.data() + gi * GROUP));
            const __m256i p16 = _mm256_maddubs_epi16(wg.mag, _mm256_sign_epi8(q, wg.sgn));
            const __m256i p32 = _mm256_madd_epi16(p16, ones);
            const float   s   = x.scale[gi] * wg.scale;
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(s), acc);
            if constexpr (Plane::kHasDelta) dacc += s * wg.delta * static_cast<float>(x.gsum[gi]);
        }
        out[r] = hsum(acc) + dacc;
    }
}
#endif

/// The kernel gemv_plane runs: gemv_avx2 where the target has it (kAvx2Kernels), else gemv.
template <class Plane>
void gemv_best(const Plane& plane, int n_rows, int row_elems, const ActBlocks& x, float* out) {
#if defined(SUB0_MOEQD_AVX2)
    gemv_avx2(plane, n_rows, row_elems, x, out);
#else
    gemv(plane, n_rows, row_elems, x, out);
#endif
}

}  // namespace detail

/** Can this plane be fused at all?
 *
 * Checked once per plane, never per row: an unsupported format or a row width that is not a multiple
 * of GROUP has no fused form, and the caller must report that rather than compute something wrong.
 */
[[nodiscard]] inline bool fusable(std::uint32_t type_raw, int row_elems) {
    if (row_elems <= 0 || row_elems % GROUP != 0) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::IQ1_S:
        case gguf::TensorType::IQ2_XXS:
        case gguf::TensorType::IQ4_NL:  return true;
        default:                        return false;
    }
}

/** How many encoded bytes a plane of `n_rows * row_elems` elements occupies in `type_raw`.
 *
 * The kernels index the plane by arithmetic on the element position, so nothing inside them can notice
 * a span that is shorter than the geometry claims -- it would simply read past the end. This is what
 * lets the one caller check that once, up front.
 *
 * @return 0 if the type is unknown or the geometry is not a whole number of blocks.
 */
[[nodiscard]] inline std::uint64_t plane_bytes(std::uint32_t type_raw, std::uint64_t elems) {
    const gguf::BlockSpec spec = gguf::block_spec(type_raw);
    if (spec.elems == 0 || elems % spec.elems != 0) return 0;
    return elems / spec.elems * spec.bytes;
}

/** One plane's GEMV against its encoded bytes, dispatched on the plane's OWN type_raw.
 *
 * Never on its role: the sidecar carries per-layer mixed quantization (moe_quant.hpp's header comment),
 * so gate and up in the SAME layer may be different formats. The dispatch happens once per plane, so
 * the row loop inside stays monomorphic.
 *
 * @return false if the format is not one this path can fuse, or if `raw` is shorter than the geometry
 *         requires -- the caller must treat either as fatal, and `out` is left untouched.
 */
[[nodiscard]] inline bool gemv_plane(std::uint32_t type_raw, std::span<const std::uint8_t> raw,
                                     int n_rows, int row_elems, const ActBlocks& x, float* out) {
    const std::uint64_t need = plane_bytes(type_raw, static_cast<std::uint64_t>(n_rows)
                                                      * static_cast<std::uint64_t>(row_elems));
    if (need == 0 || raw.size() < need) return false;
    switch (static_cast<gguf::TensorType>(type_raw)) {
        case gguf::TensorType::IQ1_S:
            detail::gemv_best(Iq1SPlane{raw.data()}, n_rows, row_elems, x, out);
            return true;
        case gguf::TensorType::IQ2_XXS:
            detail::gemv_best(Iq2XxsPlane{raw.data()}, n_rows, row_elems, x, out);
            return true;
        case gguf::TensorType::IQ4_NL:
            detail::gemv_best(Iq4NlPlane{raw.data()}, n_rows, row_elems, x, out);
            return true;
        default:
            return false;
    }
}

/** One encoded plane: the sidecar's own descriptor for it, plus the bytes themselves. */
struct EncodedPlane {
    moeq::Desc                    desc{};   ///< format (`type_raw`) and the GGUF DECLARED extents
    std::span<const std::uint8_t> bytes;    ///< non-owning: the sidecar mapping, or a staged I/O buffer
};

/** One routed expert's three planes, named.
 *
 * Bundled rather than passed as three loose (descriptor, bytes) pairs: the three are positionally
 * interchangeable in a parameter list, and a swapped pair produces finite, plausibly-scaled, wrong
 * output that no tolerance check would catch. Naming them at the construction site is the cheapest
 * place to make that mistake impossible.
 */
struct ExpertPlanes {
    EncodedPlane gate, up, down;
};

namespace detail {

/// Is `p` readable as `n_rows` rows of `row_elems` in its own declared format and geometry?
[[nodiscard]] inline bool plane_ok(const EncodedPlane& p, int n_rows, int row_elems) {
    return fusable(p.desc.type_raw, row_elems)
           && p.desc.in_f == static_cast<std::uint32_t>(row_elems)
           && p.desc.out_f == static_cast<std::uint32_t>(n_rows);
}

}  // namespace detail

/** moe::expert_ffn_row_source's computation -- `out = down(silu(gate(x)) * up(x))` -- with the
 * dequantize step removed entirely: every GEMV runs straight against the encoded planes.
 *
 * @param xq  the row's activation, quantized ONCE by the caller and reused across all selected experts
 *            of this layer (docs/MOE_QUANT_DOT.md S4 -- the hoist is real: this never re-quantizes it).
 * @param pq  caller-owned scratch for the down projection's input, which is per-expert and so must be
 *            quantized here: d_ff = 640 elements against d_ff*hidden_size dot terms, ~0.04% of the work.
 * @return false if any plane's format is unfusable, its declared geometry disagrees with `d`, or its
 *         byte span is shorter than that geometry needs. Nothing is written in any of those cases.
 *
 * @note Plane geometry, re-derived rather than assumed (AGENTS.md S5): a Desc's in_f/out_f are the GGUF
 *       DECLARED extents with ne[0] fastest-varying, so element (out o, in i) lives at o*in_f + i. For
 *       gate/up that is o*hidden_size + i with o over d_ff; for down it is j*d_ff + o with j over
 *       hidden_size. Identical to what moe::expert_ffn_row_source already consumes -- the same
 *       traversal, with the decode moved inside the dot. Checked against the descriptors rather than
 *       trusted, because a sidecar built at different axes would otherwise decode plausible garbage.
 * @note Gate and up are two separate full passes rather than the interleaved one
 *       expert_ffn_row_source uses. Each plane is then read as one contiguous stream, and the reused
 *       vector (x, 10 KiB at the real axes) stays L1-resident across both either way.
 */
[[nodiscard]] inline bool expert_ffn_row_quant(const moe::Dims& d, const ActBlocks& xq,
                                               const ExpertPlanes& planes, float* out,
                                               float* pre_scratch, float* g_scratch, ActBlocks& pq) {
    if (xq.n != d.hidden_size) return false;
    if (!detail::plane_ok(planes.gate, d.d_ff, d.hidden_size)) return false;
    if (!detail::plane_ok(planes.up, d.d_ff, d.hidden_size)) return false;
    if (!detail::plane_ok(planes.down, d.hidden_size, d.d_ff)) return false;

    if (!gemv_plane(planes.gate.desc.type_raw, planes.gate.bytes, d.d_ff, d.hidden_size, xq, g_scratch))
        return false;
    if (!gemv_plane(planes.up.desc.type_raw, planes.up.bytes, d.d_ff, d.hidden_size, xq, pre_scratch))
        return false;
    for (int o = 0; o < d.d_ff; ++o) pre_scratch[o] = moe::detail::silu(g_scratch[o]) * pre_scratch[o];

    pq.quantize(pre_scratch, d.d_ff);
    return gemv_plane(planes.down.desc.type_raw, planes.down.bytes, d.hidden_size, d.d_ff, pq, out);
}

}  // namespace sub0::moeqd
