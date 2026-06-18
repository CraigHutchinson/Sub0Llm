#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/ops.hpp"
#include "backends/cpu/kernels.hpp"        // CPU reference kernels (rms_norm parity)
#include "backends/cuda/backend.hpp"       // CUDA kernels under test

using namespace sub0llm;
using namespace sub0llm::ops;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static constexpr float kEps = 1e-4f;

// Relative RMS error between two same-numel f32 tensors — the parity metric the CUDA kernel
// tests share. eps guards the all-zeros reference (sref==0).
[[maybe_unused]] static double rel_rms(const Tensor& a, const Tensor& b) {
    const auto ap = a.data_as<float>();
    const auto bp = b.data_as<float>();
    double se = 0.0, sref = 0.0;
    for (std::size_t i = 0; i < ap.size(); ++i) {
        const double d = static_cast<double>(ap[i]) - static_cast<double>(bp[i]);
        se += d * d;  sref += static_cast<double>(bp[i]) * static_cast<double>(bp[i]);
    }
    return std::sqrt(se / (sref + 1e-30));
}

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

// Stage 4 Phase 2: the CUDA rms_norm training kernels (fwd, bwd_x, bwd_w) must match the CPU
// reference. Kernel-level parity (H2D via Tensor::to → launch → D2H), gated on the CUDA build.
TEST_CASE("rms_norm - CUDA training kernels match CPU reference", "[backends][cuda][device]") {
#ifdef SUB0LLM_CUDA
    const int64_t T = 5, D = 130;     // D not a multiple of the warp/block width (tail coverage)
    const float   eps = 1e-5f;
    const std::size_t Tn = static_cast<std::size_t>(T), Dn = static_cast<std::size_t>(D);

    Tensor x = randn({T, D});
    Tensor w = randn({D});
    Tensor g = randn({T, D});         // upstream gradient for the backward kernels

    // ── CPU reference ──
    Tensor xn_c = zeros({T, D}), ir_c = zeros({T}), out_c = zeros({T, D});
    backend::cpu::rms_norm_fwd_f32(
        x.data_as<float>().data(), w.data_as<float>().data(),
        xn_c.data_as<float>().data(), ir_c.data_as<float>().data(),
        out_c.data_as<float>().data(), Tn, Dn, eps);
    Tensor gx_c = zeros({T, D});
    backend::cpu::rms_norm_bwd_x_f32(
        g.data_as<float>().data(), xn_c.data_as<float>().data(),
        ir_c.data_as<float>().data(), w.data_as<float>().data(),
        gx_c.data_as<float>().data(), Tn, Dn);
    Tensor gw_c = zeros({D});
    backend::cpu::rms_norm_bwd_w_f32(
        g.data_as<float>().data(), xn_c.data_as<float>().data(),
        gw_c.data_as<float>().data(), Tn, Dn);

    // ── CUDA (device-resident; results copied back to host for comparison) ──
    const Tensor xd = x.to(Device::cuda()), wd = w.to(Device::cuda()), gd = g.to(Device::cuda());
    Tensor xn_d = zeros({T, D}, DType::Float32, Device::cuda());
    Tensor ir_d = zeros({T},    DType::Float32, Device::cuda());
    Tensor out_d = zeros({T, D}, DType::Float32, Device::cuda());
    backend::cuda::rms_norm_fwd(
        reinterpret_cast<const float*>(xd.raw_ptr()), reinterpret_cast<const float*>(wd.raw_ptr()),
        reinterpret_cast<float*>(xn_d.raw_ptr()), reinterpret_cast<float*>(ir_d.raw_ptr()),
        reinterpret_cast<float*>(out_d.raw_ptr()), static_cast<int>(T), static_cast<int>(D), eps);

    Tensor gx_d = zeros({T, D}, DType::Float32, Device::cuda());
    backend::cuda::rms_norm_bwd_x(
        reinterpret_cast<const float*>(gd.raw_ptr()), reinterpret_cast<const float*>(xn_d.raw_ptr()),
        reinterpret_cast<const float*>(ir_d.raw_ptr()), reinterpret_cast<const float*>(wd.raw_ptr()),
        reinterpret_cast<float*>(gx_d.raw_ptr()), static_cast<int>(T), static_cast<int>(D));

    Tensor gw_d = zeros({D}, DType::Float32, Device::cuda());
    backend::cuda::rms_norm_bwd_w(
        reinterpret_cast<const float*>(gd.raw_ptr()), reinterpret_cast<const float*>(xn_d.raw_ptr()),
        reinterpret_cast<float*>(gw_d.raw_ptr()), static_cast<int>(T), static_cast<int>(D));

    REQUIRE(rel_rms(out_d.to(Device::cpu()), out_c) < 1e-4);
    REQUIRE(rel_rms(xn_d.to(Device::cpu()),  xn_c)  < 1e-4);
    REQUIRE(rel_rms(ir_d.to(Device::cpu()),  ir_c)  < 1e-4);
    REQUIRE(rel_rms(gx_d.to(Device::cpu()),  gx_c)  < 1e-4);
    REQUIRE(rel_rms(gw_d.to(Device::cpu()),  gw_c)  < 1e-4);
#else
    SUCCEED("CPU build - CUDA rms_norm parity is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 3: ops::matmul_tb (C = Aᵀ·B, the weight-gradient op) must dispatch to CUDA and
// match the CPU reference. Dims are deliberately non-multiples of the 16×16 tile (tail coverage).
TEST_CASE("matmul_tb - CUDA matches CPU reference", "[backends][cuda][device]") {
#ifdef SUB0LLM_CUDA
    const int64_t M = 70, K = 34, N = 50;   // A(M,K), B(M,N) → C(K,N); none a multiple of TILE=16
    Tensor a = randn({M, K});
    Tensor b = randn({M, N});

    const Tensor c_cpu = matmul_tb(a, b);
    const Tensor c_gpu = matmul_tb(a.to(Device::cuda()), b.to(Device::cuda())).to(Device::cpu());

    REQUIRE(c_cpu.shape() == Tensor::Shape{K, N});
    REQUIRE(c_gpu.shape() == c_cpu.shape());
    REQUIRE(rel_rms(c_gpu, c_cpu) < 1e-4);
#else
    SUCCEED("CPU build - CUDA matmul_tb parity is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 5: silu forward (ops::silu dispatch) and backward kernel must match CPU. The CPU
// SIMD path uses a fast-sigmoid approximation vs the GPU's expf, so the tolerance is looser than
// the exact ops.
TEST_CASE("silu - CUDA fwd+bwd match CPU reference", "[backends][cuda][device]") {
#ifdef SUB0LLM_CUDA
    const int64_t n = 1000;
    Tensor x = randn({n});
    Tensor g = randn({n});      // upstream gradient

    const Tensor y_cpu = ops::silu(x);
    const Tensor y_gpu = ops::silu(x.to(Device::cuda())).to(Device::cpu());
    REQUIRE(rel_rms(y_gpu, y_cpu) < 2e-3);

    Tensor gi_cpu = zeros({n});
    backend::cpu::silu_backward_f32(g.data_as<float>().data(), x.data_as<float>().data(),
                                    gi_cpu.data_as<float>().data(), static_cast<std::size_t>(n));
    const Tensor xd = x.to(Device::cuda()), gd = g.to(Device::cuda());
    Tensor gi_d = zeros({n}, DType::Float32, Device::cuda());
    backend::cuda::silu_bwd(reinterpret_cast<const float*>(gd.raw_ptr()),
                            reinterpret_cast<const float*>(xd.raw_ptr()),
                            reinterpret_cast<float*>(gi_d.raw_ptr()), static_cast<int>(n));
    REQUIRE(rel_rms(gi_d.to(Device::cpu()), gi_cpu) < 2e-3);
#else
    SUCCEED("CPU build - CUDA silu parity is exercised on the cuda preset");
#endif
}

// Stage 4 Phase 5: embedding-scatter backward — repeated indices (N>V) exercise the atomicAdd
// accumulation that multiple tokens sharing a row require.
TEST_CASE("embed_bwd - CUDA scatter matches CPU reference", "[backends][cuda][device]") {
#ifdef SUB0LLM_CUDA
    const int64_t V = 20, D = 16, N = 50;
    Tensor g = randn({N, D});
    Tensor idx({N}, DType::Int32);
    { auto ip = idx.data_as<int32_t>();
      for (int64_t i = 0; i < N; ++i) ip[static_cast<std::size_t>(i)] = static_cast<int32_t>(i % V); }

    Tensor gw_cpu = zeros({V, D});
    backend::cpu::embed_bwd_f32(g.data_as<float>().data(), idx.data_as<int32_t>().data(),
                                gw_cpu.data_as<float>().data(),
                                static_cast<std::size_t>(N), static_cast<std::size_t>(D));

    const Tensor gd = g.to(Device::cuda()), idd = idx.to(Device::cuda());
    Tensor gw_d = zeros({V, D}, DType::Float32, Device::cuda());
    backend::cuda::embed_bwd(reinterpret_cast<const float*>(gd.raw_ptr()),
                             reinterpret_cast<const int*>(idd.raw_ptr()),
                             reinterpret_cast<float*>(gw_d.raw_ptr()),
                             static_cast<int>(N), static_cast<int>(D));
    REQUIRE(rel_rms(gw_d.to(Device::cpu()), gw_cpu) < 1e-4);
#else
    SUCCEED("CPU build - CUDA embed_bwd parity is exercised on the cuda preset");
#endif
}
