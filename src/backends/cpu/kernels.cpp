#include "kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
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

// Number of elements in the SIMD-processed prefix (rounded down to SIMD_WIDTH).
inline constexpr std::size_t simd_prefix(std::size_t n) noexcept {
    return n & ~(SIMD_WIDTH - 1u);
}

// ── Element-wise binary ───────────────────────────────────────────────────────

void add_f32(const float* a, const float* b, float* out, std::size_t n) noexcept {
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

void sub_f32(const float* a, const float* b, float* out, std::size_t n) noexcept {
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

void mul_f32(const float* a, const float* b, float* out, std::size_t n) noexcept {
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

void div_f32(const float* a, const float* b, float* out, std::size_t n) noexcept {
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

void add_scalar_f32(const float* a, float s, float* out, std::size_t n) noexcept {
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

void mul_scalar_f32(const float* a, float s, float* out, std::size_t n) noexcept {
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

void relu_f32(const float* in, float* out, std::size_t n) noexcept {
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

void neg_f32(const float* in, float* out, std::size_t n) noexcept {
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

// Scalar math ops — no SIMD benefit due to transcendental cost.
void exp_f32    (const float* in, float* out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::exp(in[i]);   }
void log_f32    (const float* in, float* out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::log(in[i]);   }
void sqrt_f32   (const float* in, float* out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::sqrt(in[i]);  }
void abs_f32    (const float* in, float* out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=std::abs(in[i]);   }
void sigmoid_f32(const float* in, float* out, std::size_t n) noexcept { for (std::size_t i=0;i<n;++i) out[i]=1.0f/(1.0f+std::exp(-in[i])); }

// ── Reductions ────────────────────────────────────────────────────────────────

float sum_f32(const float* in, std::size_t n) noexcept {
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
    // Horizontal add: sum the 8 lanes.
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

float max_f32(const float* in, std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    return *std::max_element(in, in + n);
}

float min_f32(const float* in, std::size_t n) noexcept {
    if (n == 0) return 0.0f;
    return *std::min_element(in, in + n);
}

float norm_f32(const float* in, std::size_t n) noexcept {
    float s = 0.0f;
    for (std::size_t i = 0; i < n; ++i) s += in[i] * in[i];
    return std::sqrt(s);
}

} // namespace sub0llm::backend::cpu
