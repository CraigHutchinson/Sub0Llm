// sub0llm-qbench (Ch27) — profile Q8 weight-consumption strategies.
//
// "Which is best may vary per model — profile, don't assume." For a GEMV
// y[M] = W[M,K]·x[K] this times three kernels at equal vectorization quality:
//   f32         — weights in f32 (current model path): 4 B/weight, f32 SIMD FMA
//   dequant-otf — Q8 weights, expand to f32 per block then f32 FMA: 1.06 B/weight
//   quantized   — Q8 weights, quantize x once, int8 maddubs dots: 1.06 B/weight
// and reports GFLOP/s, RAM/weight, and accuracy vs the f32 result. Run the model
// preset to see how the winner shifts across the shapes a real forward issues.

#include "sub0llm/backends/cpu/quant.hpp"
#ifdef SUB0LLM_CUDA
#  include "backends/cuda/backend.hpp"   // device Q8 matmul bench (validates vs CPU)
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#if defined(SUB0LLM_AVX512) || defined(SUB0LLM_AVX2)
#  include <immintrin.h>
#endif

namespace {
using namespace sub0llm::backend::cpu;

// f32 GEMV baseline, hand-vectorized to the same quality as the Q8 kernels so the
// comparison isolates representation, not kernel effort.
float dot_f32(const float* __restrict w, const float* __restrict x, int64_t k) noexcept {
#if defined(SUB0LLM_AVX2) || defined(SUB0LLM_AVX512)
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= k; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_loadu_ps(x + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc), hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float r = _mm_cvtss_f32(s);
    for (; i < k; ++i) r += w[i] * x[i];
    return r;
#else
    float r = 0.0f;
    for (int64_t i = 0; i < k; ++i) r += w[i] * x[i];
    return r;
#endif
}

void fill(std::vector<float>& v, uint32_t seed) {
    uint32_t s = seed | 1u;
    for (float& x : v) { s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 0.2f; }
}

double now_s() { return std::chrono::duration<double>(
    std::chrono::steady_clock::now().time_since_epoch()).count(); }

template<class F> double time_it(int reps, F&& f) {
    for (int w = 0; w < 3; ++w) f();        // warm-up
    const double t0 = now_s();
    for (int r = 0; r < reps; ++r) f();
    return now_s() - t0;
}

void bench_shape(const char* label, int64_t M, int64_t K, int reps) {
    std::vector<float> W(static_cast<std::size_t>(M * K)), x(static_cast<std::size_t>(K));
    std::vector<float> y0(static_cast<std::size_t>(M)), y1(static_cast<std::size_t>(M)), y2(static_cast<std::size_t>(M));
    fill(W, 2); fill(x, 3);

    const int64_t nb = K / QK8_0;
    std::vector<BlockQ8_0> Wq(static_cast<std::size_t>(M * nb)), xq(static_cast<std::size_t>(nb));
    std::vector<uint16_t> Wh(static_cast<std::size_t>(M * K));
    std::vector<float> yh(static_cast<std::size_t>(M));
    for (int64_t m = 0; m < M; ++m) {
        quantize_row_q8_0(W.data() + m * K, Wq.data() + m * nb, K);
        quantize_row_f16(W.data() + m * K, Wh.data() + m * K, K);
    }

    std::vector<float> y3(static_cast<std::size_t>(M));
    auto f_f32 = [&] { for (int64_t m = 0; m < M; ++m) y0[static_cast<std::size_t>(m)] = dot_f32(W.data() + m * K, x.data(), K); };
    auto f_f16 = [&] { matvec_f16_f32(Wh.data(), x.data(), yh.data(), M, K); };
    auto f_dq  = [&] { matvec_q8_0_f32(Wq.data(), x.data(), y1.data(), M, K); };
    auto f_d16 = [&] { matvec_q8_0_f16(Wq.data(), x.data(), y3.data(), M, K); };
    auto f_qq  = [&] { matvec_q8_0_q8_0(Wq.data(), x.data(), y2.data(), M, K, xq.data()); };

    const double t0 = time_it(reps, f_f32);
    const double th = time_it(reps, f_f16);
    const double t1 = time_it(reps, f_dq);
    const double t3 = time_it(reps, f_d16);
    const double t2 = time_it(reps, f_qq);

    // accuracy vs f32 baseline: relative RMS over the whole output vector
    // (robust — individual outputs sit near zero with random data, which makes
    // per-element relative error explode meaninglessly).
    auto rel_rms = [&](const std::vector<float>& y) {
        double se = 0.0, sr = 0.0;
        for (int64_t m = 0; m < M; ++m) {
            const double ref = y0[static_cast<std::size_t>(m)];
            const double d = y[static_cast<std::size_t>(m)] - ref;
            se += d * d; sr += ref * ref;
        }
        return std::sqrt(se / (sr + 1e-12));
    };

    const double gf = 2.0 * static_cast<double>(M) * static_cast<double>(K) * reps / 1e9;
    const double mb_f32 = static_cast<double>(M * K * 4) / (1024.0 * 1024.0);
    const double mb_f16 = static_cast<double>(M * K * 2) / (1024.0 * 1024.0);
    const double mb_q8  = static_cast<double>(M * nb * 34) / (1024.0 * 1024.0);

    std::printf("\n%s  (M=%lld K=%lld, %d reps)\n", label,
                static_cast<long long>(M), static_cast<long long>(K), reps);
    std::printf("  weight RAM:  f32 %.1f   f16 %.1f   Q8 %.1f MiB\n", mb_f32, mb_f16, mb_q8);
    std::printf("  %-14s %8.2f GFLOP/s   %7.2f ms   1.00x  (4 B/w, baseline)\n",
                "f32",         gf / t0, t0 / reps * 1e3);
    std::printf("  %-14s %8.2f GFLOP/s   %7.2f ms   %.2fx  (2 B/w)  relRMS %.2e\n",
                "f16",         gf / th, th / reps * 1e3, t0 / th, rel_rms(yh));
    std::printf("  %-14s %8.2f GFLOP/s   %7.2f ms   %.2fx  (1.06 B/w) relRMS %.2e\n",
                "Q8 deq->f32",  gf / t1, t1 / reps * 1e3, t0 / t1, rel_rms(y1));
    std::printf("  %-14s %8.2f GFLOP/s   %7.2f ms   %.2fx  (1.06 B/w) relRMS %.2e\n",
                "Q8 deq->f16",  gf / t3, t3 / reps * 1e3, t0 / t3, rel_rms(y3));
    std::printf("  %-14s %8.2f GFLOP/s   %7.2f ms   %.2fx  (1.06 B/w) relRMS %.2e\n",
                "Q8 quantized",gf / t2, t2 / reps * 1e3, t0 / t2, rel_rms(y2));
}

// Batched GEMM: per-column GEMVs (W re-streamed per column) vs row-reuse matmul
// (W streamed once). Same FLOPs and pre-quantized inputs — isolates cache locality.
void bench_batch(const char* label, int64_t M, int64_t K, int64_t T, int reps) {
    std::vector<float> W(static_cast<std::size_t>(M * K)), X(static_cast<std::size_t>(T * K));
    fill(W, 2); fill(X, 3);
    const int64_t nb = K / QK8_0;
    std::vector<BlockQ8_0> Wq(static_cast<std::size_t>(M * nb)), Xq(static_cast<std::size_t>(T * nb));
    for (int64_t m = 0; m < M; ++m) quantize_row_q8_0(W.data() + m * K, Wq.data() + m * nb, K);
    for (int64_t t = 0; t < T; ++t) quantize_row_q8_0(X.data() + t * K, Xq.data() + t * nb, K);
    std::vector<float> Y(static_cast<std::size_t>(M * T));

    // per-column: t outer, m inner → W streamed once PER column (T× the W traffic)
    auto f_cols = [&] {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t m = 0; m < M; ++m)
                Y[static_cast<std::size_t>(m * T + t)] =
                    dot_q8_0_q8_0(Wq.data() + m * nb, Xq.data() + t * nb, nb);
    };
    auto f_batch = [&] { matmul_q8_0_q8_0(Wq.data(), Xq.data(), Y.data(), M, K, T); };

    // Hoisted: pre-decode the T·nb activation block scales ONCE (reused across all M rows),
    // vs the inline f16-decode the plain matmul redoes per row. Scale decode is O(T·nb) —
    // negligible beside the O(M·T·nb) GEMM — so it is realistically amortized; time only the
    // row loop to isolate the inner-kernel win.
    std::vector<float> xsd(static_cast<std::size_t>(T * nb));
    decode_q8_block_scales(Xq.data(), xsd.data(), T * nb);
    auto f_hoist = [&] {
        for (int64_t m = 0; m < M; ++m)
            gemm_row_q8_0(Wq.data() + m * nb, Xq.data(), xsd.data(), Y.data() + m * T, nb, T);
    };

    const double tc = time_it(reps, f_cols);
    const double tb = time_it(reps, f_batch);
    const double th = time_it(reps, f_hoist);
    const double gf = 2.0 * static_cast<double>(M) * static_cast<double>(K) * static_cast<double>(T) * reps / 1e9;
    std::printf("\n%s  (M=%lld K=%lld T=%lld, %d reps)\n", label,
                static_cast<long long>(M), static_cast<long long>(K), static_cast<long long>(T), reps);
    std::printf("  per-column GEMVs (W re-streamed): %7.2f GFLOP/s\n", gf / tc);
    std::printf("  batched matmul   (W reused/tile): %7.2f GFLOP/s   %.2fx\n", gf / tb, tc / tb);
    std::printf("  + hoisted act-scale decode:       %7.2f GFLOP/s   %.2fx vs batched\n",
                gf / th, tb / th);

#ifdef SUB0LLM_CUDA
    // Same Q8 weights/activations on the GPU (dp4a int8 dot). The result must match the CPU
    // matmul within Q8 tolerance — different backend, different float-accumulation order, so
    // compare by relative RMS rather than bitwise. Y currently holds the CPU result.
    std::vector<float> Yg(static_cast<std::size_t>(M * T));
    auto rel_to_cpu = [&] {
        double se = 0.0, sr = 0.0;
        for (std::size_t i = 0; i < static_cast<std::size_t>(M * T); ++i) {
            const double d = static_cast<double>(Yg[i]) - static_cast<double>(Y[i]);
            se += d * d;  sr += static_cast<double>(Y[i]) * static_cast<double>(Y[i]);
        }
        return std::sqrt(se / (sr + 1e-12));
    };
    const char* names[2] = {"CUDA Q8 dp4a   (CUDA cores)", "CUDA Q8 IMMA   (tensor cores)"};
    for (int variant = 0; variant < 2; ++variant) {
        const double tgpu = sub0llm::backend::cuda::matmul_q8_0_bench(
            Wq.data(), Xq.data(), Yg.data(),
            static_cast<int>(M), static_cast<int>(K), static_cast<int>(T), reps, variant);
        std::printf("  %-29s   %7.2f GFLOP/s   %.2fx vs CPU batched   relRMS %.2e\n",
                    names[variant], gf / tgpu, tb / tgpu, rel_to_cpu());
    }
#endif
}

} // namespace

int main(int argc, char** argv) {
    int64_t M = 0, K = 0, T = 0; int reps = 200; std::string preset = "qwen3";
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if      (a == "--M")      M = std::stoll(next());
        else if (a == "--K")      K = std::stoll(next());
        else if (a == "--batch")  T = std::stoll(next());
        else if (a == "--reps")   reps = std::stoi(next());
        else if (a == "--preset") preset = next();
        else if (a == "-h" || a == "--help") {
            std::printf("usage: sub0llm-qbench [--M N --K N] [--batch T] [--reps N] [--preset qwen3|gemma]\n");
            return 0;
        }
    }

#if defined(SUB0LLM_AVX512)
    std::printf("[qbench] SIMD: AVX-512\n");
#elif defined(SUB0LLM_AVX2)
    std::printf("[qbench] SIMD: AVX2\n");
#else
    std::printf("[qbench] SIMD: scalar\n");
#endif

    if (T > 0) {  // batched/tiling experiment: W-reuse across T columns
        std::printf("=== batched GEMM (prompt processing, T=%lld): tiling/locality ===\n",
                    static_cast<long long>(T));
        if (M > 0 && K > 0) bench_batch("custom", M, K, T, reps);
        else {
            bench_batch("ffn gate/up   ",  3072, 1024, T, reps);
            bench_batch("ffn down      ",  1024, 3072, T, reps);
            bench_batch("lm head       ", 151936, 1024, T, std::max(1, reps / 8));
        }
        return 0;
    }

    if (M > 0 && K > 0) { bench_shape("custom", M, K, reps); return 0; }

    if (preset == "gemma") {
        std::printf("=== Gemma-4-12B representative GEMVs ===\n");
        bench_shape("attn q_proj   ",  4096, 3840, reps);   // 16*256 x 3840
        bench_shape("attn kv_proj  ",  2048, 3840, reps);   //  8*256 x 3840
        bench_shape("ffn gate/up   ", 15360, 3840, reps);
        bench_shape("ffn down      ",  3840, 15360, reps);
        bench_shape("lm head       ", 262144, 3840, reps / 4 + 1);
    } else {
        std::printf("=== Qwen3-0.6B GEMVs ===\n");
        bench_shape("attn q_proj   ",  2048, 1024, reps);
        bench_shape("attn kv_proj  ",  1024, 1024, reps);
        bench_shape("ffn gate/up   ",  3072, 1024, reps);
        bench_shape("ffn down      ",  1024, 3072, reps);
        bench_shape("lm head       ", 151936, 1024, reps / 4 + 1);
    }
    return 0;
}
