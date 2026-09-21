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
// NO RAW INTRINSICS, DELIBERATELY -- and this is a measurement, not a preference. B34/B38 found that a
// float reduction does not auto-vectorize because Clang will not reassociate floating-point addition.
// INTEGER addition has no such barrier, so the portable form of the inner loop below vectorizes on its
// own: compiled at `-O3 -march=native`, the 32-element int8 dot emits `vpmovsxbw`/`vpmaddwd` (the same
// shape the hand-written AVX2 reference builds out of `_mm256_maddubs_epi16`) for two of the three
// unpackers and `vpmovsxbd`/`vpmulld`/`vphaddd` for the third -- vector code either way, with no scalar
// fallback to rescue. Raw intrinsics would buy at most the narrower multiply on one format, and would
// cost a second, unshared copy of the kernel per format. See detail::dot_group's own comment.
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

namespace sub0::moeqd {

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

/** Unpacks one GROUP of IQ1_S: 256 elements per 50-byte block = f16 d, 32 grid-index low bytes, then
 * eight u16 `qh` each packing three high grid-index bits per 8-element sub-group at [3l+2:3l], a 3-bit
 * scale at [14:12], and the delta's sign at bit 15. The grid is a ternary codebook, so `q` is already
 * the decoded integer and the whole per-element value is `scale * (q + delta)`.
 */
struct Iq1SPlane {
    static constexpr bool kHasDelta = true;
    const std::uint8_t* plane = nullptr;   // non-owning; the sidecar mapping outlives every resolve

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 256) * 50;
        const int ib = static_cast<int>((p % 256) / 32);
        std::uint16_t d_bits = 0, qh = 0;
        std::memcpy(&d_bits, blk, sizeof d_bits);
        std::memcpy(&qh, blk + 34 + 2 * ib, sizeof qh);
        const std::uint8_t* qs = blk + 2 + 4 * ib;

        WeightGroup wg;
        wg.scale = gguf::f16_to_f32(d_bits) * static_cast<float>(2 * ((qh >> 12) & 7) + 1);
        wg.delta = (qh & 0x8000) ? -gguf::IQ1S_DELTA : gguf::IQ1S_DELTA;
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t grid = gguf::IQ1S_GRID[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
            std::memcpy(wg.q.data() + 8 * l, &grid, sizeof grid);   // eight int8, little-endian
        }
        return wg;
    }
};

/** Unpacks one GROUP of IQ2_XXS: 256 elements per 66-byte block = f16 d, then eight groups of two u32.
 * The first u32's four bytes are grid indices; the second's top nibble is the group scale and its low
 * 28 bits are four 7-bit sign indices. The grid stores unsigned MAGNITUDES (only 0x08/0x19/0x2b occur,
 * so the signed product below cannot overflow), with the signs carried separately.
 */
struct Iq2XxsPlane {
    static constexpr bool kHasDelta = false;
    const std::uint8_t* plane = nullptr;   // non-owning; the sidecar mapping outlives every resolve

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 256) * 66;
        const int ib32 = static_cast<int>((p % 256) / 32);
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk, sizeof d_bits);
        std::uint32_t aux[2] = {0, 0};
        std::memcpy(aux, blk + 2 + 8 * ib32, sizeof aux);
        const auto* aux8 = reinterpret_cast<const std::uint8_t*>(aux);

        WeightGroup wg;
        wg.scale = gguf::f16_to_f32(d_bits) * (0.5f + static_cast<float>(aux[1] >> 28)) * 0.25f;
        std::array<std::int8_t, GROUP> sign{};
        for (int l = 0; l < 4; ++l) {
            const std::uint64_t grid  = gguf::IQ2XXS_GRID[aux8[l]];
            const std::uint64_t signs = SIGNS64[(aux[1] >> (7 * l)) & 127];
            std::memcpy(wg.q.data() + 8 * l, &grid, sizeof grid);
            std::memcpy(sign.data() + 8 * l, &signs, sizeof signs);
        }
        for (int j = 0; j < GROUP; ++j) wg.q[j] = static_cast<std::int8_t>(wg.q[j] * sign[j]);
        return wg;
    }
};

/** Unpacks one GROUP of IQ4_NL, whose blocks ARE 32 elements: f16 d plus 16 packed nibble pairs. Byte
 * j holds element j in its LOW nibble and element j+16 in its HIGH nibble -- NOT adjacent pairs, the
 * same non-obvious split gguf.hpp's own decoder calls out.
 */
struct Iq4NlPlane {
    static constexpr bool kHasDelta = false;
    const std::uint8_t* plane = nullptr;   // non-owning; the sidecar mapping outlives every resolve

    [[nodiscard]] WeightGroup group(std::uint64_t p) const {
        const std::uint8_t* blk = plane + (p / 32) * 18;
        std::uint16_t d_bits = 0;
        std::memcpy(&d_bits, blk, sizeof d_bits);
        const std::uint8_t* qn = blk + 2;

        WeightGroup wg;
        wg.scale = gguf::f16_to_f32(d_bits);
        for (int j = 0; j < 16; ++j) {
            wg.q[static_cast<std::size_t>(j)]      = gguf::KVALUES_IQ4NL[qn[j] & 0xF];
            wg.q[static_cast<std::size_t>(j) + 16] = gguf::KVALUES_IQ4NL[qn[j] >> 4];
        }
        return wg;
    }
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
            detail::gemv(Iq1SPlane{raw.data()}, n_rows, row_elems, x, out);
            return true;
        case gguf::TensorType::IQ2_XXS:
            detail::gemv(Iq2XxsPlane{raw.data()}, n_rows, row_elems, x, out);
            return true;
        case gguf::TensorType::IQ4_NL:
            detail::gemv(Iq4NlPlane{raw.data()}, n_rows, row_elems, x, out);
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
