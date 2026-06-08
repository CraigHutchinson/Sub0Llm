#include "sub0llm/backends/cpu/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(SUB0LLM_AVX512) || defined(SUB0LLM_AVX2)
#  include <immintrin.h>
#endif

// ── int8 dot-product: VNNI dispatch ─────────────────────────────────────────────
// `vpdpbusd` does an unsigned×signed 4-byte dot accumulated into int32 in ONE
// instruction, replacing the maddubs(int16) + madd(int16→int32) two-op chain. The
// win on a memory-bound GEMV is not the saved ALU op — it's INSTRUCTION DENSITY:
// fewer uops per consumed weight byte lets each core keep more loads in flight, so
// the per-core load-issue throttle (not the DRAM bus, which a pure-read probe shows
// has headroom) eases. This is the lever behind llama.cpp's ~7% multi-thread edge.
//   - Arrow Lake / Alder Lake (hybrid, no AVX-512): AVX-VNNI via __AVXVNNI__,
//     intrinsic _mm256_dpbusd_avx_epi32 (VEX encoding).
//   - Ice Lake-SP / Sapphire Rapids (AVX-512): AVX512-VNNI+VL, _mm256_dpbusd_epi32.
// Define SUB0LLM_DISABLE_VNNI (e.g. -DSUB0LLM_DISABLE_VNNI) to force the maddubs
// fallback — used for the controlled A/B that isolates the VNNI+prefetch speedup.
#if (defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)) && !defined(SUB0LLM_DISABLE_VNNI)
#  if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#    define SUB0LLM_DPBUSD(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#    define SUB0LLM_HAVE_VNNI 1
#  elif defined(__AVXVNNI__)
#    define SUB0LLM_DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#    define SUB0LLM_HAVE_VNNI 1
#  endif
#endif

namespace sub0llm::backend::cpu {

// ── f16 <-> f32 ────────────────────────────────────────────────────────────────

float f16_to_f32(uint16_t h) noexcept {
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    if (exp == 0) {
        const float val = std::ldexp(static_cast<float>(mant), -24);
        return sign ? -val : val;
    }
    if (exp == 31) {
        const float inf = std::numeric_limits<float>::infinity();
        return sign ? -inf : inf;
    }
    const uint32_t bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    float val;
    std::memcpy(&val, &bits, 4);
    return val;
}

uint16_t f32_to_f16(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        // Subnormal/underflow: flush small magnitudes (Q8 scales are positive normals).
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t h = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) ++h;  // round to nearest
        return static_cast<uint16_t>(sign | h);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow → inf
    uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    if (mant & 0x1000u) ++h;  // round to nearest
    return h;
}

// ── quantize / dequantize ──────────────────────────────────────────────────────

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t k) noexcept {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * QK8_0;
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(xb[j]));
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        for (int j = 0; j < QK8_0; ++j)
            y[i].qs[j] = static_cast<int8_t>(std::lrint(xb[j] * id));
    }
}

void dequantize_row_q8_0(const BlockQ8_0* x, float* y, int64_t k) noexcept {
    const int64_t nb = k / QK8_0;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(x[i].d);
        for (int j = 0; j < QK8_0; ++j)
            y[i * QK8_0 + j] = d * static_cast<float>(x[i].qs[j]);
    }
}

// ── SIMD horizontal sums + fast f16 scale ──────────────────────────────────────
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
// Hardware f16→f32 (F16C) — one instruction, branchless. The per-block scale
// conversion is on the hot path of every Q8 dot; the portable f16_to_f32 above
// has subnormal/inf branches that dominate memory-bound GEMVs.
#if defined(__F16C__)
static inline float f16_fast(uint16_t h) noexcept {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}
#else
static inline float f16_fast(uint16_t h) noexcept { return f16_to_f32(h); }
#endif
static inline float hsum_ps_avx(__m256 v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}
// Kept SIMD primitive: int32 horizontal sum. The current Q8 dot accumulates as
// float (one reduction per row) so this isn't on the hot path today, but it is a
// hard-won, correct building block for future int-domain kernels (e.g. an
// int-accumulate variant, block-summed bias, or a Q4 path). [[maybe_unused]]
// keeps it without tripping -Wunused-function.
[[maybe_unused]] static inline int32_t hsum_epi32_avx(__m256i v) noexcept {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s  = _mm_add_epi32(lo, hi);
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}
#endif

// ── (A) dequant-on-the-fly dot: Q8 weight row · f32 activation ──────────────────

float dot_q8_0_f32(const BlockQ8_0* w, const float* x, int64_t k) noexcept {
    const int64_t nb = k / QK8_0;
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const __m256 vd = _mm256_set1_ps(f16_fast(w[i].d));
        const float* xb = x + i * QK8_0;
        for (int g = 0; g < QK8_0; g += 8) {           // 8 int8 at a time
            const __m128i q8  = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w[i].qs + g));
            const __m256  qf  = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            const __m256  wf  = _mm256_mul_ps(qf, vd);  // dequantized weights
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xb + g), acc);
        }
    }
    return hsum_ps_avx(acc);
#else
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(w[i].d);
        const float* xb = x + i * QK8_0;
        for (int j = 0; j < QK8_0; ++j)
            acc += d * static_cast<float>(w[i].qs[j]) * xb[j];
    }
    return acc;
#endif
}

// ── (A') dequant-on-the-fly to an f16 intermediate, then f32 dot ────────────────
//
// Completes the "dequant target" grid: expand each Q8 weight to f32, ROUND to f16,
// convert back to f32, then FMA. For a GEMV (each weight used once) this is pure
// overhead vs (A) — there is no reuse to amortize the narrower intermediate — so it
// must be ≤ (A). The f16 intermediate only pays off in a BATCHED/TILED matmul where
// a dequantized tile is reused across many activation columns and the half-width
// scratch halves cache pressure (a separate, batched experiment).
float dot_q8_0_f16(const BlockQ8_0* w, const float* x, int64_t k) noexcept {
    const int64_t nb = k / QK8_0;
#if defined(__F16C__) && (defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512))
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const __m256 vd = _mm256_set1_ps(f16_fast(w[i].d));
        const float* xb = x + i * QK8_0;
        for (int g = 0; g < QK8_0; g += 8) {
            const __m128i q8   = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w[i].qs + g));
            const __m256  wf32 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8)), vd);
            // round to f16 and back — the "dequant to f16" step.
            const __m128i wf16 = _mm256_cvtps_ph(wf32, _MM_FROUND_TO_NEAREST_INT);
            const __m256  wf   = _mm256_cvtph_ps(wf16);
            acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xb + g), acc);
        }
    }
    return hsum_ps_avx(acc);
#else
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = f16_to_f32(w[i].d);
        for (int j = 0; j < QK8_0; ++j)
            acc += f16_to_f32(f32_to_f16(d * static_cast<float>(w[i].qs[j]))) * x[i * QK8_0 + j];
    }
    return acc;
#endif
}

void matvec_q8_0_f16(const BlockQ8_0* W, const float* x, float* y,
                     int64_t M, int64_t K) noexcept {
    const int64_t nb = K / QK8_0;
    for (int64_t m = 0; m < M; ++m) y[m] = dot_q8_0_f16(W + m * nb, x, K);
}

// ── (B) quantized int8 dot: Q8 weight row · Q8 activation row ────────────────────

float dot_q8_0_q8_0(const BlockQ8_0* a, const BlockQ8_0* b, int64_t nblocks) noexcept {
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    // Software-prefetch the weight stream `a` a few blocks ahead. `a` is the big
    // sequential read (one full weight row, and rows are contiguous in W, so this
    // also warms the start of the next row across the boundary); `b` (activation)
    // is tiny and already hot in L1. 8 blocks ≈ 272 B ≈ 4–5 cache lines ahead keeps
    // the load stream from stalling at row edges, especially on the weaker-prefetch
    // E-cores. An out-of-range hint on the final row is harmless (prefetch never faults).
    constexpr int64_t kPrefetchAhead = 8;
#  if !SUB0LLM_HAVE_VNNI
    const __m256i ones = _mm256_set1_epi16(1);
#  endif
    // Accumulate (int32 partial dots × per-block scale) as floats in a vector and
    // hsum ONCE per row — avoids a horizontal integer reduction every block.
    __m256 facc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nblocks; ++i) {
        _mm_prefetch(reinterpret_cast<const char*>(a + i + kPrefetchAhead), _MM_HINT_T0);
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a[i].qs));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b[i].qs));
        // signed×signed via |a| (unsigned) × (b·sign(a)) (signed): vpdpbusd wants an
        // unsigned first operand, so fold a's sign into b and take |a|.
        const __m256i ax  = _mm256_sign_epi8(va, va);
        const __m256i sy  = _mm256_sign_epi8(vb, va);
#  if SUB0LLM_HAVE_VNNI
        const __m256i d32 = SUB0LLM_DPBUSD(_mm256_setzero_si256(), ax, sy);  // 8 int32
#  else
        const __m256i d16 = _mm256_maddubs_epi16(ax, sy);
        const __m256i d32 = _mm256_madd_epi16(d16, ones);   // 8 int32 partials
#  endif
        const __m256  sc  = _mm256_set1_ps(f16_fast(a[i].d) * f16_fast(b[i].d));
        facc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d32), sc, facc);
    }
    return hsum_ps_avx(facc);
#else
    float acc = 0.0f;
    for (int64_t i = 0; i < nblocks; ++i) {
        int32_t s = 0;
        for (int j = 0; j < QK8_0; ++j)
            s += static_cast<int32_t>(a[i].qs[j]) * static_cast<int32_t>(b[i].qs[j]);
        acc += f16_to_f32(a[i].d) * f16_to_f32(b[i].d) * static_cast<float>(s);
    }
    return acc;
#endif
}

// ── f16 weights: store + dequant-on-the-fly dot (F16C) ──────────────────────────

void quantize_row_f16(const float* x, uint16_t* y, int64_t k) noexcept {
    for (int64_t i = 0; i < k; ++i) y[i] = f32_to_f16(x[i]);
}

float dot_f16_f32(const uint16_t* w, const float* x, int64_t k) noexcept {
#if defined(__F16C__) && (defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512))
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= k; i += 8) {
        const __m128i h  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i));
        const __m256  wf = _mm256_cvtph_ps(h);     // 8 f16 → 8 f32, one instruction
        acc = _mm256_fmadd_ps(wf, _mm256_loadu_ps(x + i), acc);
    }
    float r = hsum_ps_avx(acc);
    for (; i < k; ++i) r += f16_to_f32(w[i]) * x[i];
    return r;
#else
    float r = 0.0f;
    for (int64_t i = 0; i < k; ++i) r += f16_to_f32(w[i]) * x[i];
    return r;
#endif
}

void matvec_f16_f32(const uint16_t* W, const float* x, float* y,
                    int64_t M, int64_t K) noexcept {
    for (int64_t m = 0; m < M; ++m)
        y[m] = dot_f16_f32(W + m * K, x, K);
}

// ── GEMV wrappers ────────────────────────────────────────────────────────────────

void matvec_q8_0_f32(const BlockQ8_0* W, const float* x, float* y,
                     int64_t M, int64_t K) noexcept {
    const int64_t nb = K / QK8_0;
    for (int64_t m = 0; m < M; ++m)
        y[m] = dot_q8_0_f32(W + m * nb, x, K);
}

void matvec_q8_0_q8_0(const BlockQ8_0* W, const float* x, float* y,
                      int64_t M, int64_t K, BlockQ8_0* xq) noexcept {
    const int64_t nb = K / QK8_0;
    quantize_row_q8_0(x, xq, K);                 // quantize activation ONCE
    for (int64_t m = 0; m < M; ++m)
        y[m] = dot_q8_0_q8_0(W + m * nb, xq, nb);
}

void matmul_q8_0_q8_0(const BlockQ8_0* W, const BlockQ8_0* Xq, float* Y,
                      int64_t M, int64_t K, int64_t T) noexcept {
    const int64_t nb = K / QK8_0;
    // Rows outer, columns inner: W + m*nb is loaded once and reused for all T columns
    // (it stays resident in L1/L2 across the inner loop), so W is streamed from RAM a
    // single time instead of once per column.
    for (int64_t m = 0; m < M; ++m) {
        const BlockQ8_0* wr = W + m * nb;
        float* yr = Y + m * T;
        for (int64_t t = 0; t < T; ++t)
            yr[t] = dot_q8_0_q8_0(wr, Xq + t * nb, nb);
    }
}

} // namespace sub0llm::backend::cpu
