#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/ops.hpp"

using namespace sub0llm;
using namespace sub0llm::ops;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static constexpr float kEps = 1e-4f;

// ── CPU SIMD path: results must match scalar reference ─────────────────────────
// These tests run regardless of SIMD level - they verify the kernel output,
// not the implementation path.  The SIMD path is selected transparently by
// the build system.

TEST_CASE("add_f32 - 1M elements correctness", "[backends][cpu]") {
    constexpr std::size_t N = 1024 * 1024;
    Tensor a = ones({static_cast<std::int64_t>(N)});
    Tensor b = ones({static_cast<std::int64_t>(N)});
    Tensor c = add(a, b);
    auto sp = c.data_as<float>();
    for (float v : sp) REQUIRE_THAT(v, WithinAbs(2.0f, kEps));
}

TEST_CASE("add_f32 - non-multiple-of-8 (tail handling)", "[backends][cpu]") {
    // 13 elements: 8 SIMD + 5 scalar tail
    Tensor a = ones({13});
    Tensor b = ones({13});
    Tensor c = add(a, b);
    for (float v : c.data_as<float>())
        REQUIRE_THAT(v, WithinAbs(2.0f, kEps));
}

TEST_CASE("mul_f32 - 1M elements correctness", "[backends][cpu]") {
    constexpr std::size_t N = 1024 * 1024;
    Tensor a = ones({static_cast<std::int64_t>(N)});
    Tensor b = ones({static_cast<std::int64_t>(N)});
    Tensor c = mul(a, b);
    for (float v : c.data_as<float>()) REQUIRE_THAT(v, WithinAbs(1.0f, kEps));
}

TEST_CASE("mul_scalar_f32 - correctness", "[backends][cpu]") {
    Tensor a = ones({64});
    Tensor b = mul(a, 3.14f);
    for (float v : b.data_as<float>()) REQUIRE_THAT(v, WithinAbs(3.14f, kEps));
}

TEST_CASE("relu - SIMD zeroes negatives", "[backends][cpu]") {
    constexpr std::size_t N = 256;
    Tensor t({static_cast<std::int64_t>(N)}, DType::Float32);
    auto sp = t.data_as<float>();
    for (std::size_t i = 0; i < N; ++i)
        sp[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    Tensor r = relu(t);
    auto out = r.data_as<float>();
    for (std::size_t i = 0; i < N; ++i)
        REQUIRE_THAT(out[i], WithinAbs(i % 2 == 0 ? 1.0f : 0.0f, kEps));
}

TEST_CASE("sum_f32 - SIMD horizontal add", "[backends][cpu]") {
    constexpr std::int64_t N = 1024;
    Tensor t = ones({N});
    REQUIRE_THAT(sum(t), WithinAbs(static_cast<float>(N), kEps));
}

TEST_CASE("sum_f32 - non-multiple-of-8 tail", "[backends][cpu]") {
    Tensor t = ones({13});
    REQUIRE_THAT(sum(t), WithinAbs(13.0f, kEps));
}

// ── Blocked matmul correctness ─────────────────────────────────────────────────

TEST_CASE("matmul_blocked - identity matrix", "[backends][cpu][matmul]") {
    // A × I == A
    const std::int64_t N = 64;
    Tensor A = randn({N, N});
    Tensor I = zeros({N, N});
    auto sp = I.data_as<float>();
    for (std::int64_t i = 0; i < N; ++i)
        sp[static_cast<std::size_t>(i * N + i)] = 1.0f;

    Tensor B = matmul(A, I);
    auto sa = A.data_as<float>();
    auto sb = B.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(N * N); ++i)
        REQUIRE_THAT(sb[i], WithinAbs(sa[i], 1e-4f));
}

TEST_CASE("matmul_blocked - (128,64)x(64,128)", "[backends][cpu][matmul]") {
    Tensor A = ones({128, 64});
    Tensor B = ones({64, 128});
    Tensor C = matmul(A, B);  // each cell = 64

    REQUIRE(C.shape(0) == 128);
    REQUIRE(C.shape(1) == 128);
    for (float v : C.data_as<float>())
        REQUIRE_THAT(v, WithinAbs(64.0f, 1e-3f));
}

TEST_CASE("matmul_blocked - non-square (37,51)x(51,29)", "[backends][cpu][matmul]") {
    Tensor A = ones({37, 51});
    Tensor B = ones({51, 29});
    Tensor C = matmul(A, B);  // each cell = 51

    REQUIRE(C.shape(0) == 37);
    REQUIRE(C.shape(1) == 29);
    for (float v : C.data_as<float>())
        REQUIRE_THAT(v, WithinAbs(51.0f, 1e-2f));
}

TEST_CASE("matmul_blocked - (256,256)x(256,256) GFLOP sanity", "[backends][cpu][matmul]") {
    Tensor A = randn({256, 256});
    Tensor B = randn({256, 256});
    Tensor C = matmul(A, B);
    REQUIRE(C.numel() == 256 * 256);
    // Just check it completes and produces finite values.
    for (float v : C.data_as<float>())
        REQUIRE(std::isfinite(v));
}

// ── Device dispatch guard ─────────────────────────────────────────────────────

TEST_CASE("to() CPU no-op still correct", "[backends][dispatch]") {
    Tensor a = randn({16});
    Tensor b = a.to(Device::cpu());
    auto sa = a.data_as<float>();
    auto sb = b.data_as<float>();
    for (std::size_t i = 0; i < 16; ++i)
        REQUIRE_THAT(sb[i], WithinAbs(sa[i], 0.0f));
}

TEST_CASE("to() CUDA throws without CUDA build", "[backends][dispatch]") {
#ifndef SUB0LLM_CUDA
    Tensor a = ones({4});
    REQUIRE_THROWS_AS(a.to(Device::cuda()), std::runtime_error);
#else
    SUCCEED("CUDA build - skipping no-CUDA guard test");
#endif
}

// Stage 4 Phase 1: the CUDA softmax kernel must match the CPU reference. Gated on the CUDA
// build (runs on the GPU); on a CPU build the dispatch never reaches the device branch.
TEST_CASE("softmax - CUDA matches CPU reference", "[backends][cuda][device]") {
#ifdef SUB0LLM_CUDA
    const int64_t rows = 8, cols = 1000;   // vocab-ish row width
    Tensor x = randn({rows, cols});
    {   // widen the dynamic range so the max-subtract path matters
        auto xs = x.data_as<float>();
        for (int64_t i = 0; i < rows * cols; ++i) xs[static_cast<std::size_t>(i)] *= 6.0f;
    }
    const Tensor cpu_y = ops::softmax(x, -1);

    const Tensor x_dev = x.to(Device::cuda());
    const Tensor y_dev = ops::softmax(x_dev, -1);
    REQUIRE(y_dev.device().is_cuda());
    const Tensor gpu_y = y_dev.to(Device::cpu());

    const auto cy = cpu_y.data_as<float>();
    const auto gy = gpu_y.data_as<float>();
    double se = 0.0, sref = 0.0;
    for (int64_t i = 0; i < rows * cols; ++i) {
        const double d = static_cast<double>(gy[static_cast<std::size_t>(i)]) -
                         static_cast<double>(cy[static_cast<std::size_t>(i)]);
        se   += d * d;
        sref += static_cast<double>(cy[static_cast<std::size_t>(i)]) *
                static_cast<double>(cy[static_cast<std::size_t>(i)]);
    }
    const double rel_rms = std::sqrt(se / sref);
    REQUIRE(rel_rms < 1e-4);

    for (int64_t r = 0; r < rows; ++r) {   // each GPU row is a valid distribution
        double s = 0.0;
        for (int64_t c = 0; c < cols; ++c) s += gy[static_cast<std::size_t>(r * cols + c)];
        REQUIRE_THAT(s, WithinAbs(1.0, 1e-4));
    }
#else
    SUCCEED("CPU build - CUDA softmax parity is exercised on the cuda preset");
#endif
}
