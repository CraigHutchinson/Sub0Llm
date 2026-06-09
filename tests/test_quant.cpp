// test_quant.cpp (Ch27) — Q8_0 kernels must match an f32 reference, or any speed
// they offer is meaningless. Gate correctness before profiling.

#include "sub0llm/backends/cpu/quant.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

using namespace sub0llm::backend::cpu;

namespace {

std::vector<float> make_vec(int64_t n, uint32_t seed, float scale = 1.0f) {
    std::vector<float> v(static_cast<std::size_t>(n));
    uint32_t s = seed | 1u;
    for (float& x : v) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x = (static_cast<float>(s >> 8) / 16777216.0f - 0.5f) * 2.0f * scale;
    }
    return v;
}

float ref_dot(const float* a, const float* b, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
}

} // namespace

TEST_CASE("f16 round-trips representative scales", "[quant]") {
    for (float v : {1.0f, 0.5f, 0.0078125f, 3.14159f, 1e-3f, 255.0f}) {
        const float r = f16_to_f32(f32_to_f16(v));
        REQUIRE_THAT(r, Catch::Matchers::WithinRel(v, 1e-3f));
    }
}

TEST_CASE("Q8_0 quantize/dequantize round-trips within quant step", "[quant]") {
    const int64_t k = 256;
    const auto x = make_vec(k, 7, 4.0f);
    std::vector<BlockQ8_0> q(static_cast<std::size_t>(k / QK8_0));
    std::vector<float> y(static_cast<std::size_t>(k));
    quantize_row_q8_0(x.data(), q.data(), k);
    dequantize_row_q8_0(q.data(), y.data(), k);

    // Per block the error is bounded by the block's step = amax/127.
    for (int64_t b = 0; b < k / QK8_0; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = std::max(amax, std::fabs(x[static_cast<std::size_t>(b * QK8_0 + j)]));
        const float step = amax / 127.0f;
        for (int j = 0; j < QK8_0; ++j) {
            const auto i = static_cast<std::size_t>(b * QK8_0 + j);
            REQUIRE(std::fabs(y[i] - x[i]) <= step + 1e-6f);
        }
    }
}

TEST_CASE("dequant-on-the-fly dot matches f32 reference", "[quant]") {
    const int64_t k = 1024;
    const auto w = make_vec(k, 11, 1.0f);
    const auto x = make_vec(k, 23, 1.0f);
    std::vector<BlockQ8_0> wq(static_cast<std::size_t>(k / QK8_0));
    quantize_row_q8_0(w.data(), wq.data(), k);

    const float got = dot_q8_0_f32(wq.data(), x.data(), k);
    const float ref = ref_dot(w.data(), x.data(), k);
    // Only the weights are quantized → relative error ~ 1/127.
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(ref, 0.02f));
}

TEST_CASE("quantized int8 dot matches f32 reference", "[quant]") {
    const int64_t k = 1024;
    const auto a = make_vec(k, 31, 1.0f);
    const auto b = make_vec(k, 41, 1.0f);
    std::vector<BlockQ8_0> aq(static_cast<std::size_t>(k / QK8_0)), bq(static_cast<std::size_t>(k / QK8_0));
    quantize_row_q8_0(a.data(), aq.data(), k);
    quantize_row_q8_0(b.data(), bq.data(), k);

    const float got = dot_q8_0_q8_0(aq.data(), bq.data(), k / QK8_0);
    const float ref = ref_dot(a.data(), b.data(), k);
    // Both operands quantized → looser tolerance.
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(ref, 0.04f));
}

TEST_CASE("Q8 dequant-to-f16 dot matches f32 reference", "[quant]") {
    const int64_t k = 1024;
    const auto w = make_vec(k, 17, 1.0f);
    const auto x = make_vec(k, 19, 1.0f);
    std::vector<BlockQ8_0> wq(static_cast<std::size_t>(k / QK8_0));
    quantize_row_q8_0(w.data(), wq.data(), k);
    // f16 intermediate must not degrade accuracy below the Q8-direct path.
    const float a = dot_q8_0_f16(wq.data(), x.data(), k);
    const float b = dot_q8_0_f32(wq.data(), x.data(), k);
    REQUIRE_THAT(a, Catch::Matchers::WithinRel(b, 0.01f));
}

TEST_CASE("f16 weight dot matches f32 reference", "[quant]") {
    const int64_t k = 1024;
    const auto w = make_vec(k, 13, 1.0f);
    const auto x = make_vec(k, 29, 1.0f);
    std::vector<uint16_t> wh(static_cast<std::size_t>(k));
    quantize_row_f16(w.data(), wh.data(), k);

    const float got = dot_f16_f32(wh.data(), x.data(), k);
    const float ref = ref_dot(w.data(), x.data(), k);
    // f16 weights → ~1e-3 relative error (10-bit mantissa), much tighter than Q8.
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(ref, 0.005f));
}

TEST_CASE("batched Q8 GEMM matches per-column GEMV", "[quant]") {
    const int64_t M = 32, K = 256, T = 5;
    const auto W = make_vec(M * K, 3, 1.0f);
    const auto X = make_vec(T * K, 9, 1.0f);
    const int64_t nb = K / QK8_0;
    std::vector<BlockQ8_0> Wq(static_cast<std::size_t>(M * nb)), Xq(static_cast<std::size_t>(T * nb));
    for (int64_t m = 0; m < M; ++m) quantize_row_q8_0(W.data() + m * K, Wq.data() + m * nb, K);
    for (int64_t t = 0; t < T; ++t) quantize_row_q8_0(X.data() + t * K, Xq.data() + t * nb, K);

    std::vector<float> Y(static_cast<std::size_t>(M * T));
    matmul_q8_0_q8_0(Wq.data(), Xq.data(), Y.data(), M, K, T);

    // Y[m,t] must equal the single GEMV dot of weight row m with activation col t.
    for (int64_t m = 0; m < M; ++m)
        for (int64_t t = 0; t < T; ++t) {
            const float ref = dot_q8_0_q8_0(Wq.data() + m * nb, Xq.data() + t * nb, nb);
            REQUIRE_THAT(Y[static_cast<std::size_t>(m * T + t)], Catch::Matchers::WithinAbs(ref, 1e-4f));
        }
}

TEST_CASE("Q8 GEMV strategies agree with f32 matmul", "[quant]") {
    const int64_t M = 40, K = 512;
    const auto W = make_vec(M * K, 5, 1.0f);   // row-major (M,K)
    const auto x = make_vec(K, 99, 1.0f);

    std::vector<BlockQ8_0> Wq(static_cast<std::size_t>(M * (K / QK8_0)));
    for (int64_t m = 0; m < M; ++m)
        quantize_row_q8_0(W.data() + m * K, Wq.data() + m * (K / QK8_0), K);

    std::vector<float> y_dq(static_cast<std::size_t>(M)), y_qq(static_cast<std::size_t>(M));
    std::vector<BlockQ8_0> xq(static_cast<std::size_t>(K / QK8_0));
    matvec_q8_0_f32(Wq.data(), x.data(), y_dq.data(), M, K);
    matvec_q8_0_q8_0(Wq.data(), x.data(), y_qq.data(), M, K, xq.data());

    for (int64_t m = 0; m < M; ++m) {
        const float ref = ref_dot(W.data() + m * K, x.data(), K);
        REQUIRE_THAT(y_dq[static_cast<std::size_t>(m)], Catch::Matchers::WithinAbs(ref, 0.05f * std::fabs(ref) + 0.5f));
        REQUIRE_THAT(y_qq[static_cast<std::size_t>(m)], Catch::Matchers::WithinAbs(ref, 0.08f * std::fabs(ref) + 0.5f));
    }
}

// ── f32 SIMD reduction primitives (Ch27) ───────────────────────────────────────
// The SIMD bulk + scalar-remainder helpers must match a scalar reference across sizes
// that exercise the remainder tail (not just multiples of 8).

TEST_CASE("sum_squares_f32 matches scalar reference", "[quant][simd]") {
    for (int64_t n : {1, 7, 8, 9, 31, 256, 257, 3840}) {
        const auto x = make_vec(n, 11);
        float ref = 0.0f;
        for (int64_t i = 0; i < n; ++i) ref += x[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        REQUIRE_THAT(sum_squares_f32(x.data(), n),
                     Catch::Matchers::WithinRel(ref, 1e-4f));
    }
}

TEST_CASE("dot_f32 matches scalar reference", "[quant][simd]") {
    for (int64_t n : {1, 7, 8, 9, 31, 256, 257, 3840}) {
        const auto a = make_vec(n, 3), b = make_vec(n, 7);
        REQUIRE_THAT(dot_f32(a.data(), b.data(), n),
                     Catch::Matchers::WithinRel(ref_dot(a.data(), b.data(), n), 1e-4f));
    }
}

TEST_CASE("axpy_f32 matches scalar reference", "[quant][simd]") {
    for (int64_t n : {1, 7, 8, 31, 256, 257}) {
        const auto x = make_vec(n, 5);
        auto y0 = make_vec(n, 9);
        auto y1 = y0;
        const float a = 1.75f;
        axpy_f32(a, x.data(), y1.data(), n);
        for (int64_t i = 0; i < n; ++i) {
            const float ref = y0[static_cast<std::size_t>(i)] + a * x[static_cast<std::size_t>(i)];
            REQUIRE_THAT(y1[static_cast<std::size_t>(i)], Catch::Matchers::WithinAbs(ref, 1e-5f));
        }
    }
}

TEST_CASE("argmax_f32 matches std max element first-occurrence", "[quant][simd]") {
    for (int64_t n : {1, 5, 8, 9, 33, 1000}) {
        auto x = make_vec(n, 17);
        int64_t ref = 0;
        for (int64_t i = 1; i < n; ++i)
            if (x[static_cast<std::size_t>(i)] > x[static_cast<std::size_t>(ref)]) ref = i;
        REQUIRE(argmax_f32(x.data(), n) == ref);
    }
    // ties resolve to the first index
    std::vector<float> t{1.0f, 3.0f, 3.0f, 2.0f, 3.0f};
    REQUIRE(argmax_f32(t.data(), 5) == 1);
}

// gemm_row_q8_0 (4-column register-blocked prefill kernel) must equal T independent
// dot_q8_0_q8_0 calls — the tiling only reorders weight-side work, not the math.
TEST_CASE("gemm_row_q8_0 matches per-column dot_q8_0_q8_0", "[quant][gemm]") {
    const int64_t K = 256, nb = K / QK8_0;
    for (int64_t T : {1, 3, 4, 5, 8, 13}) {
        const auto wf = make_vec(K, 21, 2.0f);
        std::vector<BlockQ8_0> w(static_cast<std::size_t>(nb));
        quantize_row_q8_0(wf.data(), w.data(), K);

        std::vector<BlockQ8_0> Xq(static_cast<std::size_t>(T * nb));
        for (int64_t t = 0; t < T; ++t) {
            const auto xf = make_vec(K, static_cast<uint32_t>(100 + t), 1.5f);
            quantize_row_q8_0(xf.data(), Xq.data() + t * nb, K);
        }
        std::vector<float> tiled(static_cast<std::size_t>(T));
        gemm_row_q8_0(w.data(), Xq.data(), tiled.data(), nb, T);
        for (int64_t t = 0; t < T; ++t) {
            const float ref = dot_q8_0_q8_0(w.data(), Xq.data() + t * nb, nb);
            REQUIRE_THAT(tiled[static_cast<std::size_t>(t)],
                         Catch::Matchers::WithinAbs(ref, 1e-3f));
        }
    }
}
