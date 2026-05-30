#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

// Pull in SIMD intrinsics if the build selected them.
#if defined(SUB0LLM_AVX512)
#  include <immintrin.h>
#  define SIMD_WIDTH 16u
#elif defined(SUB0LLM_AVX2)
#  include <immintrin.h>
#  define SIMD_WIDTH 8u
#else
#  define SIMD_WIDTH 1u
#endif

namespace sub0llm::backend::cpu {

// ── Internal helpers ──────────────────────────────────────────────────────────

inline constexpr std::size_t simd_prefix(std::size_t n) noexcept {
    return n & ~(SIMD_WIDTH - 1u);
}

// ── Fast vectorised exp approximation ────────────────────────────────────────
// Degree-5 Horner polynomial for exp(r) where r = x - n*ln2, |r| <= ln2/2.
// Decomposition: exp(x) = 2^n * exp(r).  Max relative error: ~1e-6.
// Used by sigmoid_f32 to enable SIMD throughput without libm or -ffast-math.
#if defined(SUB0LLM_AVX512)
static inline __m512 fast_exp_avx512(__m512 x) noexcept {
    // Clamp to avoid overflow / denormals.
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.3f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.7f));
    // Compute n = floor(x / ln2).
    const __m512 log2e = _mm512_set1_ps(1.44269504089f);
    __m512 z  = _mm512_fmadd_ps(x, log2e, _mm512_set1_ps(0.5f));
    __m512i n = _mm512_cvttps_epi32(_mm512_sub_ps(z, _mm512_set1_ps(0.5f)));
    __m512  nf = _mm512_cvtepi32_ps(n);
    // r = x - n * ln2  (two-constant subtraction for accuracy)
    __m512 r = _mm512_fnmadd_ps(nf, _mm512_set1_ps(0.693359375f), x);
    r         = _mm512_fnmadd_ps(nf, _mm512_set1_ps(-2.12194440e-4f), r);
    // Horner polynomial for exp(r):
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(1.6666664760e-1f));
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(r, p, _mm512_set1_ps(1.0f));
    // Scale by 2^n via IEEE exponent field.
    __m512i ni = _mm512_add_epi32(n, _mm512_set1_epi32(127));
    ni = _mm512_slli_epi32(ni, 23);
    return _mm512_mul_ps(p, _mm512_castsi512_ps(ni));
}
#elif defined(SUB0LLM_AVX2)
static inline __m256 fast_exp_avx2(__m256 x) noexcept {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.7f));
    const __m256 log2e = _mm256_set1_ps(1.44269504089f);
    __m256 z  = _mm256_fmadd_ps(x, log2e, _mm256_set1_ps(0.5f));
    __m256i n = _mm256_cvttps_epi32(_mm256_sub_ps(z, _mm256_set1_ps(0.5f)));
    __m256  nf = _mm256_cvtepi32_ps(n);
    __m256 r = _mm256_fnmadd_ps(nf, _mm256_set1_ps(0.693359375f), x);
    r         = _mm256_fnmadd_ps(nf, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.6666664760e-1f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f));
    p = _mm256_fmadd_ps(r, p, _mm256_set1_ps(1.0f));
    __m256i ni = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    ni = _mm256_slli_epi32(ni, 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(ni));
}
#endif

// ── Element-wise binary ───────────────────────────────────────────────────────

void add_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#elif defined(SUB0LLM_AVX2)
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] + b[i];
}

void sub_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#elif defined(SUB0LLM_AVX2)
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] - b[i];
}

void mul_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_mul_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#elif defined(SUB0LLM_AVX2)
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] * b[i];
}

void div_f32(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        _mm512_storeu_ps(out + i,
            _mm512_div_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
    }
#elif defined(SUB0LLM_AVX2)
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_div_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
#endif
    for (; i < n; ++i) out[i] = a[i] / b[i];
}

// ── Scalar broadcast ─────────────────────────────────────────────────────────

void add_scalar_f32(const float* __restrict__ a, float s, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 vs = _mm512_set1_ps(s);
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i), vs));
#elif defined(SUB0LLM_AVX2)
    const __m256 vs = _mm256_set1_ps(s);
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vs));
#endif
    for (; i < n; ++i) out[i] = a[i] + s;
}

void mul_scalar_f32(const float* __restrict__ a, float s, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 vs = _mm512_set1_ps(s);
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_loadu_ps(a + i), vs));
#elif defined(SUB0LLM_AVX2)
    const __m256 vs = _mm256_set1_ps(s);
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vs));
#endif
    for (; i < n; ++i) out[i] = a[i] * s;
}

// ── Activations ──────────────────────────────────────────────────────────────

void relu_f32(const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 vz = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        _mm512_storeu_ps(out + i, _mm512_max_ps(_mm512_loadu_ps(in + i), vz));
#elif defined(SUB0LLM_AVX2)
    const __m256 vz = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        _mm256_storeu_ps(out + i, _mm256_max_ps(_mm256_loadu_ps(in + i), vz));
#endif
    for (; i < n; ++i) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

void neg_f32(const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 vz = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        _mm512_storeu_ps(out + i, _mm512_sub_ps(vz, _mm512_loadu_ps(in + i)));
#elif defined(SUB0LLM_AVX2)
    const __m256 vz = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        _mm256_storeu_ps(out + i, _mm256_sub_ps(vz, _mm256_loadu_ps(in + i)));
#endif
    for (; i < n; ++i) out[i] = -in[i];
}

// Sigmoid: 1/(1+exp(-x)).  Uses fast polynomial exp for SIMD throughput;
// scalar tail falls back to std::exp (accurate) so boundary elements are exact.
void sigmoid_f32(const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 vz  = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        __m512 x   = _mm512_loadu_ps(in + i);
        __m512 neg = _mm512_sub_ps(vz, x);
        __m512 e   = fast_exp_avx512(neg);
        _mm512_storeu_ps(out + i, _mm512_div_ps(one, _mm512_add_ps(one, e)));
    }
#elif defined(SUB0LLM_AVX2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 vz  = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        __m256 x   = _mm256_loadu_ps(in + i);
        __m256 neg = _mm256_sub_ps(vz, x);
        __m256 e   = fast_exp_avx2(neg);
        _mm256_storeu_ps(out + i, _mm256_div_ps(one, _mm256_add_ps(one, e)));
    }
#endif
    for (; i < n; ++i) out[i] = 1.0f / (1.0f + std::exp(-in[i]));
}

// Scalar math ops — leave exp/log/sqrt as scalar (transcendental accuracy required
// for log-softmax, cross-entropy; fast approximation is only safe for sigmoid gate).
void exp_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::exp(in[i]);   }
void log_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::log(in[i]);   }
void sqrt_f32   (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::sqrt(in[i]);  }
void abs_f32    (const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::abs(in[i]);   }

// SiLU: out[i] = in[i] * sigmoid(in[i]).
void silu_f32(const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 vz  = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        __m512 x   = _mm512_loadu_ps(in + i);
        __m512 e   = fast_exp_avx512(_mm512_sub_ps(vz, x));
        __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(x, sig));
    }
#elif defined(SUB0LLM_AVX2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 vz  = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        __m256 x   = _mm256_loadu_ps(in + i);
        __m256 e   = fast_exp_avx2(_mm256_sub_ps(vz, x));
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(x, sig));
    }
#endif
    for (; i < n; ++i) {
        const float x = in[i];
        out[i] = x / (1.0f + std::exp(-x));
    }
}

// GELU: out[i] = in[i] * sigmoid(z), where z = kK*(in[i] + kC*in[i]^3).
// Tanh identity: 0.5*(1+tanh(k)) = sigmoid(2k), so gelu(x) = x*sigmoid(kK*(x+kC*x³)).
void gelu_f32(const float* __restrict__ in, float* __restrict__ out, std::size_t n) noexcept {
    // kK = 2*sqrt(2/pi) so that sigmoid(kK*k_arg) = 0.5*(1+tanh(sqrt(2/pi)*k_arg))
    constexpr float kK = 1.5957691216f;
    constexpr float kC = 0.044715f;
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 vkK = _mm512_set1_ps(kK);
    const __m512 vkC = _mm512_set1_ps(kC);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 vz  = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        __m512 x     = _mm512_loadu_ps(in + i);
        __m512 x2    = _mm512_mul_ps(x, x);
        // inner = x + kC*x^3  =  fmadd(kC*x^2, x, x)
        __m512 inner = _mm512_fmadd_ps(_mm512_mul_ps(vkC, x2), x, x);
        __m512 z     = _mm512_mul_ps(vkK, inner);
        __m512 e     = fast_exp_avx512(_mm512_sub_ps(vz, z));
        __m512 sig   = _mm512_div_ps(one, _mm512_add_ps(one, e));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(x, sig));
    }
#elif defined(SUB0LLM_AVX2)
    const __m256 vkK = _mm256_set1_ps(kK);
    const __m256 vkC = _mm256_set1_ps(kC);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 vz  = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        __m256 x     = _mm256_loadu_ps(in + i);
        __m256 x2    = _mm256_mul_ps(x, x);
        __m256 inner = _mm256_fmadd_ps(_mm256_mul_ps(vkC, x2), x, x);
        __m256 z     = _mm256_mul_ps(vkK, inner);
        __m256 e     = fast_exp_avx2(_mm256_sub_ps(vz, z));
        __m256 sig   = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(x, sig));
    }
#endif
    for (; i < n; ++i) {
        const float x = in[i];
        const float z = kK * (x + kC * x * x * x);
        out[i] = x / (1.0f + std::exp(-z));
    }
}

// SiLU backward: grad_in[i] = grad_out[i] * sig*(1 + x*(1-sig)), sig = sigmoid(x).
void silu_backward_f32(const float* __restrict__ grad_out, const float* __restrict__ x,
                       float* __restrict__ grad_in, std::size_t n) noexcept {
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 vz  = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        __m512 xi  = _mm512_loadu_ps(x + i);
        __m512 gi  = _mm512_loadu_ps(grad_out + i);
        __m512 e   = fast_exp_avx512(_mm512_sub_ps(vz, xi));
        __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, e));
        // dp = sig * (1 + x*(1-sig)) = sig + x*sig*(1-sig)
        __m512 dp  = _mm512_fmadd_ps(_mm512_mul_ps(xi, sig),
                                      _mm512_sub_ps(one, sig), sig);
        _mm512_storeu_ps(grad_in + i, _mm512_mul_ps(gi, dp));
    }
#elif defined(SUB0LLM_AVX2)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 vz  = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        __m256 xi  = _mm256_loadu_ps(x + i);
        __m256 gi  = _mm256_loadu_ps(grad_out + i);
        __m256 e   = fast_exp_avx2(_mm256_sub_ps(vz, xi));
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        __m256 dp  = _mm256_fmadd_ps(_mm256_mul_ps(xi, sig),
                                      _mm256_sub_ps(one, sig), sig);
        _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(gi, dp));
    }
#endif
    for (; i < n; ++i) {
        const float sig = 1.0f / (1.0f + std::exp(-x[i]));
        grad_in[i] = grad_out[i] * sig * (1.0f + x[i] * (1.0f - sig));
    }
}

// GELU backward: grad_in[i] = grad_out[i] * gelu'(x[i]).
// gelu'(x) = sig(z) + x * sig(z)*(1-sig(z)) * z', where z = kK*(x+kC*x³), z' = kK*(1+3*kC*x²).
void gelu_backward_f32(const float* __restrict__ grad_out, const float* __restrict__ x,
                       float* __restrict__ grad_in, std::size_t n) noexcept {
    constexpr float kK  = 1.5957691216f;
    constexpr float kC  = 0.044715f;
    constexpr float k3C = 3.0f * kC * kK;   // kK*3*kC — coefficient for z' constant term
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    const __m512 vkK  = _mm512_set1_ps(kK);
    const __m512 vkC  = _mm512_set1_ps(kC);
    const __m512 vk3C = _mm512_set1_ps(k3C);
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 vz   = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        __m512 xi   = _mm512_loadu_ps(x + i);
        __m512 gi   = _mm512_loadu_ps(grad_out + i);
        __m512 x2   = _mm512_mul_ps(xi, xi);
        __m512 z    = _mm512_mul_ps(vkK, _mm512_fmadd_ps(_mm512_mul_ps(vkC, x2), xi, xi));
        __m512 e    = fast_exp_avx512(_mm512_sub_ps(vz, z));
        __m512 sig  = _mm512_div_ps(one, _mm512_add_ps(one, e));
        // z' = kK*(1 + 3*kC*x²) = kK + k3C*x²
        __m512 zp   = _mm512_fmadd_ps(vk3C, x2, vkK);
        // gelu'(x) = sig + x*sig*(1-sig)*z'
        __m512 dp   = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_mul_ps(xi, sig),
                                                      _mm512_sub_ps(one, sig)),
                                       zp, sig);
        _mm512_storeu_ps(grad_in + i, _mm512_mul_ps(gi, dp));
    }
#elif defined(SUB0LLM_AVX2)
    const __m256 vkK  = _mm256_set1_ps(kK);
    const __m256 vkC  = _mm256_set1_ps(kC);
    const __m256 vk3C = _mm256_set1_ps(k3C);
    const __m256 one  = _mm256_set1_ps(1.0f);
    const __m256 vz   = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        __m256 xi   = _mm256_loadu_ps(x + i);
        __m256 gi   = _mm256_loadu_ps(grad_out + i);
        __m256 x2   = _mm256_mul_ps(xi, xi);
        __m256 z    = _mm256_mul_ps(vkK, _mm256_fmadd_ps(_mm256_mul_ps(vkC, x2), xi, xi));
        __m256 e    = fast_exp_avx2(_mm256_sub_ps(vz, z));
        __m256 sig  = _mm256_div_ps(one, _mm256_add_ps(one, e));
        __m256 zp   = _mm256_fmadd_ps(vk3C, x2, vkK);
        __m256 dp   = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_mul_ps(xi, sig),
                                                      _mm256_sub_ps(one, sig)),
                                       zp, sig);
        _mm256_storeu_ps(grad_in + i, _mm256_mul_ps(gi, dp));
    }
#endif
    for (; i < n; ++i) {
        const float xi = x[i];
        const float z  = kK * (xi + kC * xi * xi * xi);
        const float sig = 1.0f / (1.0f + std::exp(-z));
        const float zp  = kK + k3C * xi * xi;
        grad_in[i] = grad_out[i] * (sig + xi * sig * (1.0f - sig) * zp);
    }
}

// ── Softmax ───────────────────────────────────────────────────────────────────

// Numerically-stable row-wise softmax using fast polynomial exp.
void softmax_rows_f32(const float* __restrict__ in, float* __restrict__ out,
                      std::size_t rows, std::size_t cols) noexcept {
    for (std::size_t r = 0; r < rows; ++r) {
        const float* rin  = in  + r * cols;
        float*       rout = out + r * cols;
        const float  mx   = max_f32(rin, cols);
        std::size_t  i    = 0;
        float        s    = 0.0f;
#if defined(SUB0LLM_AVX512)
        const __m512 vmx = _mm512_set1_ps(mx);
        __m512 vacc = _mm512_setzero_ps();
        for (const std::size_t end = simd_prefix(cols); i < end; i += 16) {
            __m512 e = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(rin + i), vmx));
            _mm512_storeu_ps(rout + i, e);
            vacc = _mm512_add_ps(vacc, e);
        }
        s = _mm512_reduce_add_ps(vacc);
#elif defined(SUB0LLM_AVX2)
        const __m256 vmx = _mm256_set1_ps(mx);
        __m256 vacc = _mm256_setzero_ps();
        for (const std::size_t end = simd_prefix(cols); i < end; i += 8) {
            __m256 e = fast_exp_avx2(_mm256_sub_ps(_mm256_loadu_ps(rin + i), vmx));
            _mm256_storeu_ps(rout + i, e);
            vacc = _mm256_add_ps(vacc, e);
        }
        __m128 hi  = _mm256_extractf128_ps(vacc, 1);
        __m128 lo  = _mm256_castps256_ps128(vacc);
        __m128 sum = _mm_add_ps(hi, lo);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        s = _mm_cvtss_f32(sum);
#endif
        for (; i < cols; ++i) {
            rout[i] = std::exp(rin[i] - mx);
            s += rout[i];
        }
        // Inline normalization: calling mul_scalar_f32(rout, s, rout, cols) would
        // pass the same pointer for two __restrict__ params — undefined behaviour.
        const float inv_s = 1.0f / s;
        for (std::size_t k = 0; k < cols; ++k) rout[k] *= inv_s;
    }
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

void rms_norm_fwd_f32(const float* __restrict__ x, const float* __restrict__ w,
                      float* __restrict__ x_norm, float* __restrict__ inv_rms,
                      float* __restrict__ out,
                      std::size_t T, std::size_t D, float eps) noexcept {
    const float invD = 1.0f / static_cast<float>(D);
    const std::size_t Ds = simd_prefix(D);
    for (std::size_t t = 0; t < T; ++t) {
        const float* xt  = x     + t * D;
        float*       xnt = x_norm + t * D;
        float*       ot  = out   + t * D;
        float ss = 0.0f;
        std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
        __m512 vacc = _mm512_setzero_ps();
        for (; j < Ds; j += 16) {
            __m512 v = _mm512_loadu_ps(xt + j);
            vacc = _mm512_fmadd_ps(v, v, vacc);
        }
        ss = _mm512_reduce_add_ps(vacc);
#elif defined(SUB0LLM_AVX2)
        __m256 vacc = _mm256_setzero_ps();
        for (; j < Ds; j += 8) {
            __m256 v = _mm256_loadu_ps(xt + j);
            vacc = _mm256_fmadd_ps(v, v, vacc);
        }
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 s  = _mm_add_ps(hi, lo);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        ss = _mm_cvtss_f32(s);
#endif
        for (; j < D; ++j) ss += xt[j] * xt[j];
        const float ir = 1.0f / std::sqrt(ss * invD + eps);
        inv_rms[t] = ir;
        j = 0;
#if defined(SUB0LLM_AVX512)
        const __m512 vir = _mm512_set1_ps(ir);
        for (; j < Ds; j += 16) {
            __m512 xn = _mm512_mul_ps(_mm512_loadu_ps(xt + j), vir);
            _mm512_storeu_ps(xnt + j, xn);
            _mm512_storeu_ps(ot  + j, _mm512_mul_ps(_mm512_loadu_ps(w + j), xn));
        }
#elif defined(SUB0LLM_AVX2)
        const __m256 vir = _mm256_set1_ps(ir);
        for (; j < Ds; j += 8) {
            __m256 xn = _mm256_mul_ps(_mm256_loadu_ps(xt + j), vir);
            _mm256_storeu_ps(xnt + j, xn);
            _mm256_storeu_ps(ot  + j, _mm256_mul_ps(_mm256_loadu_ps(w + j), xn));
        }
#endif
        for (; j < D; ++j) { xnt[j] = xt[j]*ir; ot[j] = w[j]*xnt[j]; }
    }
}

void rms_norm_bwd_x_f32(const float* __restrict__ g,
                          const float* __restrict__ x_norm,
                          const float* __restrict__ inv_rms,
                          const float* __restrict__ w,
                          float* __restrict__ gx,
                          std::size_t T, std::size_t D) noexcept {
    const float invD = 1.0f / static_cast<float>(D);
    const std::size_t Ds = simd_prefix(D);
    for (std::size_t t = 0; t < T; ++t) {
        const float* gt  = g     + t * D;
        const float* xnt = x_norm + t * D;
        float*       gxt = gx   + t * D;
        const float  ir  = inv_rms[t];
        // sigma = dot(w * g[t,:], x_norm[t,:])
        float sigma = 0.0f;
        std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
        __m512 vacc = _mm512_setzero_ps();
        for (; j < Ds; j += 16)
            vacc = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_loadu_ps(w+j), _mm512_loadu_ps(gt+j)),
                                    _mm512_loadu_ps(xnt+j), vacc);
        sigma = _mm512_reduce_add_ps(vacc);
#elif defined(SUB0LLM_AVX2)
        __m256 vacc = _mm256_setzero_ps();
        for (; j < Ds; j += 8)
            vacc = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_loadu_ps(w+j), _mm256_loadu_ps(gt+j)),
                                    _mm256_loadu_ps(xnt+j), vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 s  = _mm_add_ps(hi, lo);
        s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
        sigma = _mm_cvtss_f32(s);
#endif
        for (; j < D; ++j) sigma += w[j] * gt[j] * xnt[j];
        // gx[t,j] = ir*w[j]*g[t,j] − ir*sigma*invD*x_norm[t,j]
        const float c2 = ir * sigma * invD;
        j = 0;
#if defined(SUB0LLM_AVX512)
        const __m512 vir = _mm512_set1_ps(ir);
        const __m512 vc2 = _mm512_set1_ps(c2);
        for (; j < Ds; j += 16) {
            __m512 h = _mm512_mul_ps(_mm512_loadu_ps(w+j), _mm512_loadu_ps(gt+j));
            _mm512_storeu_ps(gxt+j, _mm512_fnmadd_ps(vc2, _mm512_loadu_ps(xnt+j),
                                                       _mm512_mul_ps(vir, h)));
        }
#elif defined(SUB0LLM_AVX2)
        const __m256 vir = _mm256_set1_ps(ir);
        const __m256 vc2 = _mm256_set1_ps(c2);
        for (; j < Ds; j += 8) {
            __m256 h = _mm256_mul_ps(_mm256_loadu_ps(w+j), _mm256_loadu_ps(gt+j));
            _mm256_storeu_ps(gxt+j, _mm256_fnmadd_ps(vc2, _mm256_loadu_ps(xnt+j),
                                                       _mm256_mul_ps(vir, h)));
        }
#endif
        for (; j < D; ++j) gxt[j] = ir * w[j] * gt[j] - c2 * xnt[j];
    }
}

void rms_norm_bwd_w_f32(const float* __restrict__ g,
                          const float* __restrict__ x_norm,
                          float* gw,   // read-modified-write: no __restrict__
                          std::size_t T, std::size_t D) noexcept {
    std::memset(gw, 0, D * sizeof(float));
    const std::size_t Ds = simd_prefix(D);
    for (std::size_t t = 0; t < T; ++t) {
        const float* gt  = g      + t * D;
        const float* xnt = x_norm + t * D;
        std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
        for (; j < Ds; j += 16) {
            __m512 acc = _mm512_loadu_ps(gw + j);
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(gt+j), _mm512_loadu_ps(xnt+j), acc);
            _mm512_storeu_ps(gw + j, acc);
        }
#elif defined(SUB0LLM_AVX2)
        for (; j < Ds; j += 8) {
            __m256 acc = _mm256_loadu_ps(gw + j);
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(gt+j), _mm256_loadu_ps(xnt+j), acc);
            _mm256_storeu_ps(gw + j, acc);
        }
#endif
        for (; j < D; ++j) gw[j] += gt[j] * xnt[j];
    }
}

// ── Embedding backward ────────────────────────────────────────────────────────

void embed_bwd_f32(const float* __restrict__ g_out, const int32_t* __restrict__ idx,
                   float* __restrict__ g_w, std::size_t N, std::size_t D) noexcept {
    const std::size_t Ds = simd_prefix(D);
    for (std::size_t i = 0; i < N; ++i) {
        const float* src = g_out + i * D;
        float*       dst = g_w + static_cast<std::size_t>(idx[i]) * D;
        std::size_t j = 0;
#if defined(SUB0LLM_AVX512)
        for (; j < Ds; j += 16)
            _mm512_storeu_ps(dst+j,
                _mm512_add_ps(_mm512_loadu_ps(dst+j), _mm512_loadu_ps(src+j)));
#elif defined(SUB0LLM_AVX2)
        for (; j < Ds; j += 8)
            _mm256_storeu_ps(dst+j,
                _mm256_add_ps(_mm256_loadu_ps(dst+j), _mm256_loadu_ps(src+j)));
#endif
        for (; j < D; ++j) dst[j] += src[j];
    }
}

// ── Reductions ────────────────────────────────────────────────────────────────

float sum_f32(const float* __restrict__ in, std::size_t n) noexcept {
    float acc = 0.0f;
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    __m512 vacc = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        vacc = _mm512_add_ps(vacc, _mm512_loadu_ps(in + i));
    acc = _mm512_reduce_add_ps(vacc);
#elif defined(SUB0LLM_AVX2)
    __m256 vacc = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        vacc = _mm256_add_ps(vacc, _mm256_loadu_ps(in + i));
    __m128 hi  = _mm256_extractf128_ps(vacc, 1);
    __m128 lo  = _mm256_castps256_ps128(vacc);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    acc = _mm_cvtss_f32(sum);
#endif
    for (; i < n; ++i) acc += in[i];
    return acc;
}

float max_f32(const float* __restrict__ in, std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    std::size_t i = 0;
    float result;
#if defined(SUB0LLM_AVX512)
    __m512 vmax = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        vmax = _mm512_max_ps(vmax, _mm512_loadu_ps(in + i));
    result = _mm512_reduce_max_ps(vmax);
#elif defined(SUB0LLM_AVX2)
    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(in + i));
    // Horizontal max: fold 256-bit into 128-bit then reduce.
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 mx = _mm_max_ps(hi, lo);
    mx = _mm_max_ps(mx, _mm_movehl_ps(mx, mx));
    mx = _mm_max_ps(mx, _mm_movehdup_ps(mx));
    result = _mm_cvtss_f32(mx);
#else
    result = -std::numeric_limits<float>::infinity();
#endif
    for (; i < n; ++i) result = result < in[i] ? in[i] : result;
    return result;
}

float min_f32(const float* __restrict__ in, std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    std::size_t i = 0;
    float result;
#if defined(SUB0LLM_AVX512)
    __m512 vmin = _mm512_set1_ps(std::numeric_limits<float>::infinity());
    for (const std::size_t end = simd_prefix(n); i < end; i += 16)
        vmin = _mm512_min_ps(vmin, _mm512_loadu_ps(in + i));
    result = _mm512_reduce_min_ps(vmin);
#elif defined(SUB0LLM_AVX2)
    __m256 vmin = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    for (const std::size_t end = simd_prefix(n); i < end; i += 8)
        vmin = _mm256_min_ps(vmin, _mm256_loadu_ps(in + i));
    __m128 hi = _mm256_extractf128_ps(vmin, 1);
    __m128 lo = _mm256_castps256_ps128(vmin);
    __m128 mn = _mm_min_ps(hi, lo);
    mn = _mm_min_ps(mn, _mm_movehl_ps(mn, mn));
    mn = _mm_min_ps(mn, _mm_movehdup_ps(mn));
    result = _mm_cvtss_f32(mn);
#else
    result = std::numeric_limits<float>::infinity();
#endif
    for (; i < n; ++i) result = result > in[i] ? in[i] : result;
    return result;
}

// L2 norm using FMA dot-product accumulation.
float norm_f32(const float* __restrict__ in, std::size_t n) noexcept {
    float acc = 0.0f;
    std::size_t i = 0;
#if defined(SUB0LLM_AVX512)
    __m512 vacc = _mm512_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 16) {
        const __m512 v = _mm512_loadu_ps(in + i);
        vacc = _mm512_fmadd_ps(v, v, vacc);
    }
    acc = _mm512_reduce_add_ps(vacc);
#elif defined(SUB0LLM_AVX2)
    __m256 vacc = _mm256_setzero_ps();
    for (const std::size_t end = simd_prefix(n); i < end; i += 8) {
        const __m256 v = _mm256_loadu_ps(in + i);
        vacc = _mm256_fmadd_ps(v, v, vacc);
    }
    __m128 hi  = _mm256_extractf128_ps(vacc, 1);
    __m128 lo  = _mm256_castps256_ps128(vacc);
    __m128 sum = _mm_add_ps(hi, lo);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    acc = _mm_cvtss_f32(sum);
#endif
    for (; i < n; ++i) acc += in[i] * in[i];
    return std::sqrt(acc);
}

} // namespace sub0llm::backend::cpu
