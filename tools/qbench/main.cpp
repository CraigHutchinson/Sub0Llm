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
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
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
    const char* names[3] = {"CUDA Q8 dp4a   (CUDA cores)", "CUDA Q8 IMMA   (tensor cores)",
                            "CUDA Q8 GEMV-aligned (T=1)"};
    const int nvar = (T == 1) ? 3 : 2;   // aligned GEMV is a decode (T=1) kernel
    for (int variant = 0; variant < nvar; ++variant) {
        const double tgpu = sub0llm::backend::cuda::matmul_q8_0_bench(
            Wq.data(), Xq.data(), Yg.data(),
            static_cast<int>(M), static_cast<int>(K), static_cast<int>(T), reps, variant);
        std::printf("  %-29s   %7.2f GFLOP/s   %.2fx vs CPU batched   relRMS %.2e\n",
                    names[variant], gf / tgpu, tb / tgpu, rel_to_cpu());
    }
#endif
}

#ifdef SUB0LLM_CUDA
// Validate the GPU layer sub-kernels (rmsnorm / rope / geglu / flash-attn decode) against an
// inline CPU reference that mirrors gemma.cpp. Different backend + float-accumulation order, so
// compare by relative RMS (not bitwise) — same gate as the Q8 matmul rows above.
void bench_layer_kernels() {
    namespace cuda = sub0llm::backend::cuda;
    auto rel_rms = [](const std::vector<float>& a, const std::vector<float>& b) {
        double se = 0.0, sr = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            se += d * d;  sr += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        return std::sqrt(se / (sr + 1e-12));
    };
    std::printf("\n=== GPU layer sub-kernel validation (relRMS vs CPU reference) ===\n");

    // ── RMSNorm (with weight, and Gemma V no-weight) ───────────────────────────────────────
    {
        const std::size_t D = 3840;  const float eps = 1e-6f;
        std::vector<float> x(D), w(D), gw(D), gn(D), cw(D), cn(D);
        fill(x, 11); fill(w, 12);
        auto cpu_rms = [&](const float* wt, std::vector<float>& out) {
            double ss = 0.0; for (std::size_t i = 0; i < D; ++i) ss += double(x[i]) * x[i];
            const float inv = 1.0f / std::sqrt(float(ss / double(D)) + eps);
            for (std::size_t i = 0; i < D; ++i) out[i] = x[i] * inv * (wt ? wt[i] : 1.0f);
        };
        cpu_rms(w.data(), cw);          cuda::rmsnorm_dev(x.data(), w.data(), gw.data(), int(D), eps);
        cpu_rms(nullptr,  cn);          cuda::rmsnorm_dev(x.data(), nullptr,  gn.data(), int(D), eps);
        std::printf("  rmsnorm (D=%zu, weighted)   relRMS %.2e\n", D, rel_rms(gw, cw));
        std::printf("  rmsnorm (D=%zu, no-weight)  relRMS %.2e\n", D, rel_rms(gn, cn));
    }

    // ── NEOX RoPE (local dh=256 base 1e4 no-ff, and global dh=512 base 1e6 with freq_factors) ─
    {
        auto rope_cpu = [](std::vector<float> v, int dh, int pos, float base, const float* ff) {
            const int half = dh / 2;
            for (int i = 0; i < half; ++i) {
                float theta = float(pos) * std::pow(base, -2.0f * float(i) / float(dh));
                if (ff) theta /= ff[size_t(i)];
                const float c = std::cos(theta), s = std::sin(theta);
                const float a = v[size_t(i)], b = v[size_t(i + half)];
                v[size_t(i)] = a * c - b * s;  v[size_t(i + half)] = a * s + b * c;
            }
            return v;
        };
        for (auto [dh, base, use_ff] : std::initializer_list<std::tuple<int,float,bool>>{
                 {256, 1e4f, false}, {512, 1e6f, true}}) {
            const std::size_t dz = static_cast<std::size_t>(dh), hz = dz / 2;
            std::vector<float> x(dz), ff(hz), g(dz);
            fill(x, 21); for (int i = 0; i < dh / 2; ++i) ff[size_t(i)] = 1.0f + 0.5f * float(i) / float(dh / 2);
            const int pos = 137;
            const std::vector<float> c = rope_cpu(x, dh, pos, base, use_ff ? ff.data() : nullptr);
            cuda::rope_neox_dev(x.data(), g.data(), dh, pos, base, use_ff ? ff.data() : nullptr);
            std::printf("  rope_neox (dh=%d base=%.0e %s)  relRMS %.2e\n",
                        dh, base, use_ff ? "ff" : "no-ff", rel_rms(g, c));
        }
    }

    // ── GeGLU (gelu_tanh(gate) ⊙ up) ───────────────────────────────────────────────────────
    {
        const std::size_t dff = 15360;
        std::vector<float> gate(dff), up(dff), g(dff), c(dff);
        fill(gate, 31); fill(up, 32);
        auto gelu = [](float xv) { constexpr float kK = 1.5957691216f, kC = 0.044715f;
            const float z = kK * (xv + kC * xv * xv * xv); return xv / (1.0f + std::exp(-z)); };
        for (std::size_t i = 0; i < dff; ++i) c[i] = gelu(gate[i]) * up[i];
        cuda::geglu_dev(gate.data(), up.data(), g.data(), int(dff));
        std::printf("  geglu (dff=%zu)             relRMS %.2e\n", dff, rel_rms(g, c));
    }

    // ── Flash-attention decode (scale=1.0 online softmax over kvlen) ───────────────────────
    {
        for (auto [dhi, kvleni] : std::initializer_list<std::pair<int,int>>{{256, 300}, {512, 1024}}) {
            const std::size_t dh = static_cast<std::size_t>(dhi), kvlen = static_cast<std::size_t>(kvleni);
            std::vector<float> q(dh), Kc(kvlen * dh), Vc(kvlen * dh), g(dh), c(dh), sc(kvlen);
            fill(q, 41); fill(Kc, 42); fill(Vc, 43);
            float mx = -1e30f;
            for (std::size_t t = 0; t < kvlen; ++t) {
                float d = 0.0f; for (std::size_t i = 0; i < dh; ++i) d += q[i] * Kc[t * dh + i];
                sc[t] = d; mx = std::max(mx, d);
            }
            float sum = 0.0f; for (std::size_t t = 0; t < kvlen; ++t) { sc[t] = std::exp(sc[t] - mx); sum += sc[t]; }
            const float inv = 1.0f / sum;
            for (std::size_t i = 0; i < dh; ++i) c[i] = 0.0f;
            for (std::size_t t = 0; t < kvlen; ++t)
                for (std::size_t i = 0; i < dh; ++i) c[i] += sc[t] * inv * Vc[t * dh + i];
            cuda::flash_attn_decode_dev(q.data(), Kc.data(), Vc.data(), g.data(), dhi, kvleni);
            std::printf("  flash_attn_decode (dh=%d kvlen=%d)  relRMS %.2e\n", dhi, kvleni, rel_rms(g, c));
        }
    }

    // ── Device Q8 activation quantizer (mirrors quantize_row_q8_0) ─────────────────────────
    {
        const int n = 3840;  const std::size_t nb = static_cast<std::size_t>(n) / QK8_0;
        std::vector<float> x(static_cast<std::size_t>(n));  fill(x, 51);
        std::vector<BlockQ8_0> cq(nb), gq(nb);
        quantize_row_q8_0(x.data(), cq.data(), n);
        cuda::quantize_q8_dev(x.data(), gq.data(), n);
        std::size_t blk_exact = 0;
        for (std::size_t b = 0; b < nb; ++b) {
            bool same = (cq[b].d == gq[b].d);
            for (int j = 0; j < QK8_0 && same; ++j) same = (cq[b].qs[j] == gq[b].qs[j]);
            blk_exact += same ? 1 : 0;
        }
        std::vector<float> cd(static_cast<std::size_t>(n)), gd(static_cast<std::size_t>(n));
        dequantize_row_q8_0(cq.data(), cd.data(), n);
        dequantize_row_q8_0(gq.data(), gd.data(), n);
        std::printf("  quantize_q8 (n=%d)         relRMS %.2e   blocks bit-exact %zu/%zu\n",
                    n, rel_rms(gd, cd), blk_exact, nb);
    }
}

// Validate ONE full Gemma layer's on-device decode (gemma_layer_decode_dev) against a faithful
// scalar CPU reference that mirrors GemmaModel::forward_one's per-layer body. Same Q8 weights,
// norms, KV prefill and input on both sides → differences are only float-accumulation order +
// GPU cos/sin rounding. Exercises the whole pipeline (norm/quantize/Q8 matmul/rope/flash-attn/
// geglu/residuals) wired together, for a local (dh256, 8 kv) and a global (dh512, MQA, V=rawK,
// freq_factors) layer at a position with pre-filled KV (window attention + cache store).
void bench_gpu_layer() {
    namespace cuda = sub0llm::backend::cuda;
    auto rel_rms = [](const std::vector<float>& a, const std::vector<float>& b) {
        double se = 0.0, sr = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double d = double(a[i]) - double(b[i]);  se += d * d;  sr += double(b[i]) * double(b[i]);
        }
        return std::sqrt(se / (sr + 1e-12));
    };
    auto rmsn = [](const float* x, const float* w, float* y, int n, float eps) {
        double ss = 0.0; for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
        const float inv = 1.0f / std::sqrt(float(ss / n) + eps);
        for (int i = 0; i < n; ++i) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
    };
    auto rope = [](float* x, int dh, int pos, float base, const float* ff) {
        const int half = dh / 2;
        for (int i = 0; i < half; ++i) {
            float th = float(pos) * std::pow(base, -2.0f * float(i) / float(dh));
            if (ff) th /= ff[i];
            const float c = std::cos(th), s = std::sin(th), a = x[i], b = x[i + half];
            x[i] = a * c - b * s;  x[i + half] = a * s + b * c;
        }
    };
    auto gelu = [](float xv) { constexpr float kK = 1.5957691216f, kC = 0.044715f;
        const float z = kK * (xv + kC * xv * xv * xv); return xv / (1.0f + std::exp(-z)); };

    const int D = 3840, dff = 15360;  const float eps = 1e-6f, out_scale = 0.95f;
    const int max_pos = 64, pos = 40;
    struct Cfg { const char* name; int dh, nH, nKV, window; float base; bool has_wv, ff; };
    const Cfg cfgs[2] = {{"local ", 256, 16, 8, 1024, 1e4f, true, false},
                         {"global", 512, 16, 1,    0, 1e6f, false, true}};
    std::printf("\n=== GPU single Gemma layer vs CPU reference (relRMS) ===\n");

    for (const Cfg& cfg : cfgs) {
        const int dh = cfg.dh, nH = cfg.nH, nKV = cfg.nKV, qM = nH * dh, kvM = nKV * dh;
        const int group = nH / nKV;
        const auto Z = [](long long n) { return static_cast<std::size_t>(n); };

        // shared Q8 weights (random f32 → quantize once), shared by CPU + GPU
        auto mkQ = [&](int Mr, int Kr, uint32_t seed) {
            std::vector<float> W(Z(1LL * Mr * Kr));  fill(W, seed);
            std::vector<BlockQ8_0> Wq(Z(1LL * Mr * (Kr / QK8_0)));
            for (int m = 0; m < Mr; ++m)
                quantize_row_q8_0(W.data() + Z(1LL * m * Kr), Wq.data() + Z(1LL * m * (Kr / QK8_0)), Kr);
            return Wq;
        };
        const auto wq = mkQ(qM, D, 1), wk = mkQ(kvM, D, 2), wv = mkQ(kvM, D, 3), wo = mkQ(D, qM, 4),
                   gate = mkQ(dff, D, 5), up = mkQ(dff, D, 6), down = mkQ(D, dff, 7);
        // norms (f32) — Gemma bakes (1+w), so ~1.0
        auto mkN = [&](int n, uint32_t s) { std::vector<float> v(Z(n)); fill(v, s); for (auto& z : v) z += 1.0f; return v; };
        const auto an = mkN(D, 11), pan = mkN(D, 12), fn = mkN(D, 13), pfn = mkN(D, 14),
                   kn = mkN(dh, 16);
        // Gemma folds the query pre-attention scaling (~1/sqrt(head_dim)) into q_norm at GGUF
        // conversion, so real queries are small and attn scores are O(1). Emulate that here —
        // a ~1.0 q_norm would leave scores huge (esp. dh=512), peaking softmax and amplifying
        // the CPU/GPU score-dot summation-order difference into the output (not a kernel bug).
        auto qn = mkN(dh, 15);  { const float qs = 1.0f / std::sqrt(float(dh)); for (auto& z : qn) z *= qs; }
        std::vector<float> ffv(Z(dh / 2));  for (int i = 0; i < dh / 2; ++i) ffv[Z(i)] = 1.0f + 0.5f * float(i) / float(dh / 2);
        const float* ffp = cfg.ff ? ffv.data() : nullptr;

        std::vector<float> kc(Z(1LL * nKV * max_pos * dh)), vc(Z(1LL * nKV * max_pos * dh)), x(Z(D));
        fill(kc, 71); fill(vc, 72); fill(x, 73);

        // ── CPU reference: one layer (mirrors forward_one) ─────────────────────────────────
        std::vector<float> kcC = kc, vcC = vc, outC(Z(D)), h(Z(D));
        rmsn(x.data(), an.data(), h.data(), D, eps);
        std::vector<float> qv(Z(qM)), kcur(Z(kvM)), vcur(Z(kvM));
        { std::vector<BlockQ8_0> xq(Z(D / QK8_0));
          matvec_q8_0_q8_0(wq.data(), h.data(), qv.data(),   qM,  D, xq.data());
          matvec_q8_0_q8_0(wk.data(), h.data(), kcur.data(), kvM, D, xq.data());
          if (cfg.has_wv) matvec_q8_0_q8_0(wv.data(), h.data(), vcur.data(), kvM, D, xq.data());
          else vcur = kcur; }
        for (int hd = 0; hd < nH; ++hd) { float* p = qv.data() + hd * dh; rmsn(p, qn.data(), p, dh, eps); rope(p, dh, pos, cfg.base, ffp); }
        for (int g = 0; g < nKV; ++g) {
            float* kh = kcur.data() + g * dh; float* vh = vcur.data() + g * dh;
            rmsn(kh, kn.data(), kh, dh, eps); rope(kh, dh, pos, cfg.base, ffp); rmsn(vh, nullptr, vh, dh, eps);
            const std::size_t bo = Z((1LL * g * max_pos + pos) * dh);
            std::copy(kh, kh + dh, kcC.begin() + std::ptrdiff_t(bo));
            std::copy(vh, vh + dh, vcC.begin() + std::ptrdiff_t(bo));
        }
        const int kv_lo = cfg.window > 0 ? std::max(0, pos - cfg.window + 1) : 0, kvlen = pos - kv_lo + 1;
        std::vector<float> ac(Z(qM));
        for (int hd = 0; hd < nH; ++hd) {
            const float* qh = qv.data() + hd * dh; const int g = hd / group;
            const float* Kb = kcC.data() + Z(1LL * g * max_pos * dh);
            const float* Vb = vcC.data() + Z(1LL * g * max_pos * dh);
            std::vector<float> sc(Z(kvlen));  float mx = -1e30f;
            for (int t = kv_lo; t <= pos; ++t) { float d = dot_f32(qh, Kb + Z(1LL * t * dh), dh); sc[Z(t - kv_lo)] = d; mx = std::max(mx, d); }
            float sum = 0.0f; for (auto& s : sc) { s = std::exp(s - mx); sum += s; }  const float inv = 1.0f / sum;
            float* oh = ac.data() + hd * dh; for (int d = 0; d < dh; ++d) oh[d] = 0.0f;
            for (int t = kv_lo; t <= pos; ++t) { const float w = sc[Z(t - kv_lo)] * inv; const float* vt = Vb + Z(1LL * t * dh); for (int d = 0; d < dh; ++d) oh[d] += w * vt[d]; }
        }
        std::vector<float> ao(Z(D));
        { std::vector<BlockQ8_0> acq(Z(qM / QK8_0)); matvec_q8_0_q8_0(wo.data(), ac.data(), ao.data(), D, qM, acq.data()); }
        rmsn(ao.data(), pan.data(), ao.data(), D, eps);
        std::vector<float> aout(Z(D)); for (int i = 0; i < D; ++i) aout[Z(i)] = x[Z(i)] + ao[Z(i)];
        rmsn(aout.data(), fn.data(), h.data(), D, eps);
        std::vector<float> gb(Z(dff)), ub(Z(dff));
        { std::vector<BlockQ8_0> xq(Z(D / QK8_0));
          matvec_q8_0_q8_0(gate.data(), h.data(), gb.data(), dff, D, xq.data());
          matvec_q8_0_q8_0(up.data(),   h.data(), ub.data(), dff, D, xq.data()); }
        for (int i = 0; i < dff; ++i) gb[Z(i)] = gelu(gb[Z(i)]) * ub[Z(i)];
        std::vector<float> ffo(Z(D));
        { std::vector<BlockQ8_0> gq(Z(dff / QK8_0)); matvec_q8_0_q8_0(down.data(), gb.data(), ffo.data(), D, dff, gq.data()); }
        rmsn(ffo.data(), pfn.data(), ffo.data(), D, eps);
        for (int i = 0; i < D; ++i) outC[Z(i)] = (aout[Z(i)] + ffo[Z(i)]) * out_scale;

        // ── GPU: same inputs through gemma_layer_decode_dev ────────────────────────────────
        std::vector<float> kcG = kc, vcG = vc, outG(Z(D));
        cuda::GpuLayerDesc L{};
        L.D = D; L.d_ff = dff; L.dh = dh; L.n_head = nH; L.n_kv_head = nKV; L.window = cfg.window;
        L.eps = eps; L.rope_base = cfg.base; L.out_scale = out_scale; L.has_wv = cfg.has_wv;
        L.wq = wq.data(); L.wk = wk.data(); L.wv = cfg.has_wv ? wv.data() : nullptr; L.wo = wo.data();
        L.gate = gate.data(); L.up = up.data(); L.down = down.data();
        L.attn_norm = an.data(); L.post_attn_norm = pan.data(); L.ffn_norm = fn.data();
        L.post_ffw_norm = pfn.data(); L.q_norm = qn.data(); L.k_norm = kn.data(); L.rope_freqs = ffp;
        cuda::gemma_layer_decode_dev(L, x.data(), pos, kcG.data(), vcG.data(), max_pos, outG.data());

        // compare layer output + the K/V slot written at `pos`
        const std::size_t slot = Z((1LL * 0 * max_pos + pos) * dh);
        std::vector<float> kSlotC(kcC.begin() + std::ptrdiff_t(slot), kcC.begin() + std::ptrdiff_t(slot) + dh);
        std::vector<float> kSlotG(kcG.begin() + std::ptrdiff_t(slot), kcG.begin() + std::ptrdiff_t(slot) + dh);
        std::printf("  %s layer (dh=%d nKV=%d %s)  out relRMS %.2e   K@pos relRMS %.2e\n",
                    cfg.name, dh, nKV, cfg.has_wv ? "wv" : "V=K", rel_rms(outG, outC), rel_rms(kSlotG, kSlotC));
    }
}

// One synthetic layer that OWNS its weights/norms; desc points into them (heap buffers, stable).
struct SynthLayer {
    std::vector<BlockQ8_0> wq, wk, wv, wo, gate, up, down;
    std::vector<float> an, pan, fn, pfn, qn, kn, ff;
    sub0llm::backend::cuda::GpuLayerDesc desc{};
};
std::unique_ptr<SynthLayer> make_synth_layer(int D, int dff, int dh, int nH, int nKV, int window,
                                             float base, bool has_wv, bool use_ff, uint32_t seed) {
    auto S = std::make_unique<SynthLayer>();
    const auto Z = [](long long n) { return static_cast<std::size_t>(n); };
    auto mkQ = [&](std::vector<BlockQ8_0>& dst, int Mr, int Kr, uint32_t s) {
        std::vector<float> W(Z(1LL * Mr * Kr));  fill(W, s);
        dst.resize(Z(1LL * Mr * (Kr / QK8_0)));
        for (int m = 0; m < Mr; ++m) quantize_row_q8_0(W.data() + Z(1LL * m * Kr), dst.data() + Z(1LL * m * (Kr / QK8_0)), Kr);
    };
    const int qM = nH * dh, kvM = nKV * dh;
    mkQ(S->wq, qM, D, seed + 1); mkQ(S->wk, kvM, D, seed + 2); if (has_wv) mkQ(S->wv, kvM, D, seed + 3);
    mkQ(S->wo, D, qM, seed + 4); mkQ(S->gate, dff, D, seed + 5); mkQ(S->up, dff, D, seed + 6); mkQ(S->down, D, dff, seed + 7);
    auto mkN = [&](std::vector<float>& dst, int n, uint32_t s) { dst.resize(Z(n)); fill(dst, s); for (auto& z : dst) z += 1.0f; };
    mkN(S->an, D, seed + 11); mkN(S->pan, D, seed + 12); mkN(S->fn, D, seed + 13); mkN(S->pfn, D, seed + 14);
    mkN(S->qn, dh, seed + 15); { const float qs = 1.0f / std::sqrt(float(dh)); for (auto& z : S->qn) z *= qs; }
    mkN(S->kn, dh, seed + 16);
    S->ff.resize(Z(dh / 2)); for (int i = 0; i < dh / 2; ++i) S->ff[Z(i)] = 1.0f + 0.5f * float(i) / float(dh / 2);
    auto& d = S->desc;
    d.D = D; d.d_ff = dff; d.dh = dh; d.n_head = nH; d.n_kv_head = nKV; d.window = window;
    d.eps = 1e-6f; d.rope_base = base; d.out_scale = 0.95f; d.has_wv = has_wv;
    d.wq = S->wq.data(); d.wk = S->wk.data(); d.wv = has_wv ? S->wv.data() : nullptr; d.wo = S->wo.data();
    d.gate = S->gate.data(); d.up = S->up.data(); d.down = S->down.data();
    d.attn_norm = S->an.data(); d.post_attn_norm = S->pan.data(); d.ffn_norm = S->fn.data();
    d.post_ffw_norm = S->pfn.data(); d.q_norm = S->qn.data(); d.k_norm = S->kn.data();
    d.rope_freqs = use_ff ? S->ff.data() : nullptr;
    return S;
}

// Equivalence check: the resident GemmaGpuLayers (weights uploaded once, KV + activations device-
// resident across layers) must reproduce the per-call gemma_layer_decode_dev form (host KV, layer
// chained on the host) BIT-for-bit — same kernels, same order, only data-movement timing differs.
// Runs a 3-layer (local/global/local) stack for several decode steps and compares final outputs.
void bench_gpu_resident() {
    namespace cuda = sub0llm::backend::cuda;
    auto rel_rms = [](const std::vector<float>& a, const std::vector<float>& b) {
        double se = 0.0, sr = 0.0; for (std::size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - double(b[i]); se += d * d; sr += double(b[i]) * double(b[i]); }
        return std::sqrt(se / (sr + 1e-12));
    };
    const int D = 3840, dff = 15360, max_pos = 64, ntok = 5;
    const auto Z = [](long long n) { return static_cast<std::size_t>(n); };
    std::vector<std::unique_ptr<SynthLayer>> S;
    S.push_back(make_synth_layer(D, dff, 256, 16, 8, 1024, 1e4f, true,  false, 100));
    S.push_back(make_synth_layer(D, dff, 512, 16, 1,    0, 1e6f, false, true,  200));
    S.push_back(make_synth_layer(D, dff, 256, 16, 8, 1024, 1e4f, true,  false, 300));
    std::vector<cuda::GpuLayerDesc> descs;  for (auto& s : S) descs.push_back(s->desc);
    const int L = int(descs.size());

    std::vector<std::vector<float>> kcA(Z(L)), vcA(Z(L));   // (A) host KV per layer
    for (int l = 0; l < L; ++l) { const int kv = descs[Z(l)].n_kv_head * max_pos * descs[Z(l)].dh; kcA[Z(l)].assign(Z(kv), 0.0f); vcA[Z(l)].assign(Z(kv), 0.0f); }
    cuda::GemmaGpuLayers gpu(descs, max_pos);               // (B) resident

    std::printf("\n=== GPU resident layers vs per-call form (relRMS, expect ~0) ===\n");
    double worst = 0.0;
    for (int t = 0; t < ntok; ++t) {
        std::vector<float> x(Z(D));  fill(x, 900u + uint32_t(t));
        std::vector<float> xa = x, outa(Z(D));             // (A) chain per-call form on the host
        for (int l = 0; l < L; ++l) {
            cuda::gemma_layer_decode_dev(descs[Z(l)], xa.data(), t, kcA[Z(l)].data(), vcA[Z(l)].data(), max_pos, outa.data());
            xa = outa;
        }
        std::vector<float> outb(Z(D));                     // (B) resident form
        gpu.decode(x.data(), t, outb.data());
        const double r = rel_rms(outb, outa);  worst = std::max(worst, r);
        std::printf("  token %d (pos %d)  relRMS %.2e\n", t, t, r);
    }
    std::printf("  worst over %d tokens: %.2e %s\n", ntok, worst, worst < 1e-6 ? "(identical)" : "(MISMATCH)");
}
#endif

} // namespace

int main(int argc, char** argv) {
    int64_t M = 0, K = 0, T = 0; int reps = 200; std::string preset = "qwen3";
    bool layers = false, gpu_layer = false, gpu_resident = false;
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&] { return std::string(argv[++i]); };
        if      (a == "--M")      M = std::stoll(next());
        else if (a == "--K")      K = std::stoll(next());
        else if (a == "--batch")  T = std::stoll(next());
        else if (a == "--reps")   reps = std::stoi(next());
        else if (a == "--preset") preset = next();
        else if (a == "--layers") layers = true;
        else if (a == "--gpu-layer") gpu_layer = true;
        else if (a == "--gpu-resident") gpu_resident = true;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: sub0llm-qbench [--M N --K N] [--batch T] [--reps N] "
                        "[--preset qwen3|gemma] [--layers] [--gpu-layer] [--gpu-resident]\n");
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

    if (layers || gpu_layer || gpu_resident) {  // GPU validation vs CPU reference
#ifdef SUB0LLM_CUDA
        if (layers)       bench_layer_kernels();
        if (gpu_layer)    bench_gpu_layer();
        if (gpu_resident) bench_gpu_resident();
#else
        std::printf("[qbench] --layers/--gpu-layer/--gpu-resident need a CUDA build (cuda preset)\n");
#endif
        return 0;
    }

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
