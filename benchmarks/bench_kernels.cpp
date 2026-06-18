// Micro-benchmark for sub0llm CPU kernels.
// Measures throughput (GB/s or GFLOPs/s) for the hot paths in the training loop.
//
// Usage:
//   ./build/bin/bench_kernels [--n <size>] [--iters <N>]
//
// Build with the native release build for meaningful numbers:
//   cmake --build build-native --parallel
//   ./build-native/bin/bench_kernels

#include "sub0llm/core/ops.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/nn/optimizer.hpp"
#include "sub0llm/autograd/ops.hpp"
#include "sub0llm/autograd/variable.hpp"
#ifdef SUB0LLM_CUDA
#  include "backends/cuda/backend.hpp"   // device op benches (kernel-only timing + parity)
#  include "backends/cpu/kernels.hpp"    // CPU reference (rms_norm fwd) for the parity column
#endif

#include <chrono>
#include <cmath>
#include <format>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace sub0llm;

// Prevents the compiler from eliminating pure reduction calls whose return values
// would otherwise be discarded and optimised away under Release+LTO.
static volatile float g_sink = 0.0f;

// ── Timing utility ────────────────────────────────────────────────────────────

struct BenchResult {
    double ms_per_iter;
    double gflops;       // only meaningful for matmul; -1 for element-wise
    double gbps;         // memory bandwidth (read + write)
};

template<typename Fn>
static BenchResult time_it(const std::string& name, std::size_t n_floats,
                            int64_t flops_per_call, int iters, Fn fn)
{
    // Warmup
    for (int i = 0; i < std::min(iters / 10, 5); ++i) fn();

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    const auto t1 = std::chrono::steady_clock::now();

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    const double bytes = static_cast<double>(n_floats) * 4.0;  // float32
    const double gbps  = (bytes / 1e9) / (ms * 1e-3);
    const double gflops = flops_per_call > 0
        ? static_cast<double>(flops_per_call) / 1e9 / (ms * 1e-3) : -1.0;

    if (gflops > 0)
        std::cout << std::format("  {:32s}  {:8.3f} ms  {:7.2f} GFLOPs/s  {:6.1f} GB/s\n",
            name, ms, gflops, gbps);
    else
        std::cout << std::format("  {:32s}  {:8.3f} ms  {:6.1f} GB/s\n",
            name, ms, gbps);
    return {ms, gflops, gbps};
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

static void bench_element_wise(std::size_t N, int iters)
{
    std::cout << "\n--- Element-wise ops (N=" << N << ") ---\n";

    Tensor a = zeros({static_cast<int64_t>(N)});
    Tensor b = zeros({static_cast<int64_t>(N)});
    auto ap = a.data_as<float>();
    auto bp = b.data_as<float>();
    for (std::size_t i = 0; i < N; ++i) { ap[i] = 1.0f + static_cast<float>(i) * 0.001f; bp[i] = 2.0f; }

    time_it("add_f32",     N * 2, -1, iters, [&]{ (void)ops::add(a, b); });
    time_it("mul_f32",     N * 2, -1, iters, [&]{ (void)ops::mul(a, b); });
    time_it("relu_f32",    N * 2, -1, iters, [&]{ (void)ops::relu(a);   });
    time_it("sigmoid_f32", N * 2, -1, iters, [&]{ (void)ops::sigmoid(a); });
    time_it("silu_f32",    N * 2, -1, iters, [&]{ (void)ops::silu(a);   });
    time_it("exp_f32",     N * 2, -1, iters, [&]{ (void)ops::exp(a);    });
    time_it("log_f32",     N * 2, -1, iters, [&]{ (void)ops::log(a);    });
}

static void bench_reductions(std::size_t N, int iters)
{
    std::cout << "\n--- Reductions (N=" << N << ") ---\n";

    Tensor a = zeros({static_cast<int64_t>(N)});
    auto ap = a.data_as<float>();
    for (std::size_t i = 0; i < N; ++i) ap[i] = static_cast<float>(i) * 0.001f - 50.0f;

    time_it("sum_f32",  N, -1, iters, [&]{ g_sink += ops::sum(a);  });
    time_it("max_f32",  N, -1, iters, [&]{ g_sink += ops::max(a);  });
    time_it("min_f32",  N, -1, iters, [&]{ g_sink += ops::min(a);  });
    time_it("norm_f32", N, -1, iters, [&]{ g_sink += ops::norm(a); });
}

static void bench_matmul(int iters)
{
    std::cout << "\n--- Matmul (row-major, C = A*B) ---\n";

    struct Spec { int64_t M, K, N; std::string label; };
    const Spec specs[] = {
        {32,  32,  32,  "32x32x32    (attention T=32, D=32)"},
        {32,  128, 128, "32x128x128  (proj, D=128)"},
        {32,  341, 128, "32x341x128  (SwiGLU down, d_ff=341)"},
        {128, 128, 128, "128x128x128 (medium square)"},
        {256, 256, 256, "256x256x256 (large square)"},
    };

    for (const auto& s : specs) {
        Tensor A = zeros({s.M, s.K});
        Tensor B = zeros({s.K, s.N});
        const int64_t flops = 2 * s.M * s.K * s.N;
        const std::size_t n_floats = static_cast<std::size_t>(s.M * s.K + s.K * s.N + s.M * s.N);
        time_it(s.label, n_floats, flops, iters, [&]{ (void)ops::matmul(A, B); });
    }
}

static void bench_copy_strided(int iters)
{
    std::cout << "\n--- copy_strided_2d (transpose) ---\n";

    // Typical weight shapes seen during training.
    struct Spec { int64_t M, N; std::string label; };
    const Spec specs[] = {
        {128,  128,  "128×128  (D=128 proj weight)"},
        {341,  128,  "341×128  (SwiGLU gate weight)"},
        {512,  512,  "512×512  (D=512 proj weight)"},
        {1024, 1024, "1024×1024 (D=1024 proj weight)"},
    };
    for (const auto& s : specs) {
        Tensor A = zeros({s.M, s.N});
        const std::size_t n_floats = static_cast<std::size_t>(s.M * s.N) * 2;
        time_it(s.label, n_floats, -1, iters,
            [&]{ (void)A.transpose(0, 1).contiguous(); });
    }
}

static void bench_autograd_forward(int iters)
{
    std::cout << "\n--- Autograd forward (T=32, D=128) ---\n";

    using namespace autograd;

    const int64_t T = 32, D = 128;
    Tensor x_t = zeros({T, D});
    Tensor w_t = zeros({D});
    auto xs = x_t.data_as<float>();
    auto ws = w_t.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i) xs[i] = static_cast<float>(i) * 0.001f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i) ws[i] = 1.0f;

    Variable x(x_t, true);
    Variable w(w_t, true);
    // Pre-create weight matrix so the benchmark measures only the matmul wrapper,
    // not the zeros({D,D}) allocation (which is a benchmark artefact).
    Tensor wmat = zeros({D, D});
    Variable wv(wmat, true);
    const std::size_t n = static_cast<std::size_t>(T * D);

    time_it("rms_norm fwd",    n * 2, -1, iters, [&]{ (void)rms_norm(x, w); });
    time_it("softmax fwd",     n * 2, -1, iters, [&]{ (void)softmax(x); });
    time_it("silu fwd",        n * 2, -1, iters, [&]{ (void)silu(x); });
    time_it("gelu fwd",        n * 2, -1, iters, [&]{ (void)gelu(x); });
    time_it("matmul (32x128x128)", static_cast<std::size_t>(T*(D+D)+D*D), 2*T*D*D, iters,
        [&]{ (void)matmul(x, wv); });
}

// ── Device (CPU vs CUDA) op benchmarks ──────────────────────────────────────────
//
// The per-op CPU-vs-GPU harness for the Stage-4 CUDA kernel work: for each shape it times the
// CPU reference and the device KERNEL (preallocated buffers + GPU events, no per-iter cudaMalloc),
// prints both + the speedup + the parity relRMS (a perf number is only trustworthy once the
// kernel matches CPU). Each new kernel (rms_norm, matmul_tb, rope, …) adds a row here.
static void bench_device_ops(int iters)
{
#ifdef SUB0LLM_CUDA
    std::cout << "\n--- Device softmax: CPU vs CUDA (rows × cols) ---\n";
    struct Spec { int64_t rows, cols; std::string label; };
    const Spec specs[] = {
        {64,   1025,  "64×1025    (T=64, vocab≈1024)"},
        {256,  1025,  "256×1025   (T=256 prefill)"},
        {1024, 4097,  "1024×4097  (long ctx, vocab≈4096)"},
    };
    for (const auto& s : specs) {
        const std::size_t n = static_cast<std::size_t>(s.rows * s.cols);
        Tensor x = zeros({s.rows, s.cols});
        auto xp = x.data_as<float>();
        for (std::size_t i = 0; i < n; ++i)
            xp[i] = 6.0f * (static_cast<float>((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f);

        // CPU reference (timed + kept for parity).
        Tensor cpu_y = ops::softmax(x, -1);
        const auto cpu_t = time_it(s.label + "  [CPU]", n * 2, -1, iters,
                                   [&]{ (void)ops::softmax(x, -1); });

        // CUDA kernel-only time (seconds for `iters` launches) + parity vs CPU.
        std::vector<float> gpu_y(n);
        const double gpu_secs = sub0llm::backend::cuda::softmax_rows_bench(
            xp.data(), gpu_y.data(), static_cast<int>(s.rows), static_cast<int>(s.cols), iters);
        const double gpu_ms = gpu_secs * 1e3 / iters;
        const double gpu_gbps = (static_cast<double>(n) * 4.0 / 1e9) / (gpu_ms * 1e-3);

        double se = 0.0, sref = 0.0;
        const auto cy = cpu_y.data_as<float>();
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(gpu_y[i]) - static_cast<double>(cy[i]);
            se += d * d;  sref += static_cast<double>(cy[i]) * static_cast<double>(cy[i]);
        }
        const double rel_rms = std::sqrt(se / sref);
        std::cout << std::format(
            "  {:32s}  {:8.3f} ms  {:6.1f} GB/s  [CUDA]   {:5.2f}x vs CPU   relRMS {:.2e}\n",
            s.label + "  [CUDA]", gpu_ms, gpu_gbps, cpu_t.ms_per_iter / gpu_ms, rel_rms);
    }

    std::cout << "\n--- Device rms_norm fwd: CPU vs CUDA (T × D) ---\n";
    struct RSpec { int64_t T, D; std::string label; };
    const RSpec rspecs[] = {
        {64,   256,  "64×256     (T=64, D=256 founded)"},
        {256,  1024, "256×1024   (T=256, D=1024)"},
        {1024, 4096, "1024×4096  (long ctx, wide D)"},
    };
    for (const auto& s : rspecs) {
        const std::size_t n = static_cast<std::size_t>(s.T * s.D);
        Tensor x = zeros({s.T, s.D});
        Tensor w = zeros({s.D});
        auto xp = x.data_as<float>();  auto wp = w.data_as<float>();
        for (std::size_t i = 0; i < n; ++i)
            xp[i] = static_cast<float>((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f;
        for (int64_t j = 0; j < s.D; ++j) wp[static_cast<std::size_t>(j)] = 1.0f + 0.01f * static_cast<float>(j % 7);

        Tensor xn_c = zeros({s.T, s.D}), ir_c = zeros({s.T}), out_c = zeros({s.T, s.D});
        const auto cpu_t = time_it(s.label + "  [CPU]", n * 2, -1, iters, [&]{
            sub0llm::backend::cpu::rms_norm_fwd_f32(
                xp.data(), wp.data(), xn_c.data_as<float>().data(),
                ir_c.data_as<float>().data(), out_c.data_as<float>().data(),
                static_cast<std::size_t>(s.T), static_cast<std::size_t>(s.D), 1e-5f);
        });

        std::vector<float> gpu_out(n);
        const double gpu_secs = sub0llm::backend::cuda::rms_norm_fwd_bench(
            xp.data(), wp.data(), gpu_out.data(),
            static_cast<int>(s.T), static_cast<int>(s.D), 1e-5f, iters);
        const double gpu_ms = gpu_secs * 1e3 / iters;
        const double gpu_gbps = (static_cast<double>(n) * 4.0 / 1e9) / (gpu_ms * 1e-3);

        double se = 0.0, sref = 0.0;
        const auto oc = out_c.data_as<float>();
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(gpu_out[i]) - static_cast<double>(oc[i]);
            se += d * d;  sref += static_cast<double>(oc[i]) * static_cast<double>(oc[i]);
        }
        const double rel_rms = std::sqrt(se / (sref + 1e-30));
        std::cout << std::format(
            "  {:32s}  {:8.3f} ms  {:6.1f} GB/s  [CUDA]   {:5.2f}x vs CPU   relRMS {:.2e}\n",
            s.label + "  [CUDA]", gpu_ms, gpu_gbps, cpu_t.ms_per_iter / gpu_ms, rel_rms);
    }

    std::cout << "\n--- Device matmul_tb (Aᵀ·B weight grad): CPU vs CUDA (M·K·N) ---\n";
    struct MSpec { int64_t M, K, N; std::string label; };
    const MSpec mspecs[] = {
        {256,  256, 256,  "256·256·256   (T=256, D=256)"},
        {256,  256, 1024, "256·256·1024  (D_in=256, d_ff=1024)"},
        {1024, 512, 512,  "1024·512·512  (T=1024, D=512)"},
    };
    for (const auto& s : mspecs) {
        Tensor A = zeros({s.M, s.K});
        Tensor B = zeros({s.M, s.N});
        auto ap = A.data_as<float>();  auto bp = B.data_as<float>();
        for (std::size_t i = 0; i < ap.size(); ++i) ap[i] = static_cast<float>((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f;
        for (std::size_t i = 0; i < bp.size(); ++i) bp[i] = static_cast<float>((i * 40503u) & 0xFFFF) / 32768.0f - 1.0f;
        const int64_t flops = 2 * s.K * s.N * s.M;
        const std::size_t nf = static_cast<std::size_t>(s.M * s.K + s.M * s.N + s.K * s.N);

        Tensor C_cpu = ops::matmul_tb(A, B);
        const auto cpu_t = time_it(s.label + "  [CPU]", nf, flops, iters, [&]{ (void)ops::matmul_tb(A, B); });

        std::vector<float> gpu_c(static_cast<std::size_t>(s.K * s.N));
        const double gpu_secs = sub0llm::backend::cuda::matmul_tb_bench(
            ap.data(), bp.data(), gpu_c.data(),
            static_cast<int>(s.M), static_cast<int>(s.N), static_cast<int>(s.K), iters);
        const double gpu_ms = gpu_secs * 1e3 / iters;
        const double gpu_gflops = static_cast<double>(flops) / 1e9 / (gpu_ms * 1e-3);

        double se = 0.0, sref = 0.0;
        const auto cc = C_cpu.data_as<float>();
        for (std::size_t i = 0; i < gpu_c.size(); ++i) {
            const double d = static_cast<double>(gpu_c[i]) - static_cast<double>(cc[i]);
            se += d * d;  sref += static_cast<double>(cc[i]) * static_cast<double>(cc[i]);
        }
        const double rel_rms = std::sqrt(se / (sref + 1e-30));
        std::cout << std::format(
            "  {:32s}  {:8.3f} ms  {:7.2f} GFLOPs/s  [CUDA]  {:5.2f}x vs CPU  relRMS {:.2e}\n",
            s.label + "  [CUDA]", gpu_ms, gpu_gflops, cpu_t.ms_per_iter / gpu_ms, rel_rms);
    }

    std::cout << "\n--- Device rope fwd: CPU vs CUDA (T × Dh) ---\n";
    struct PSpec { int64_t T, Dh; std::string label; };
    const PSpec pspecs[] = {
        {256,  64,  "256×64     (T=256, head_dim=64)"},
        {1024, 128, "1024×128   (long ctx, Dh=128)"},
    };
    for (const auto& s : pspecs) {
        const int64_t D2 = s.Dh / 2;
        const std::size_t n = static_cast<std::size_t>(s.T * s.Dh);
        Tensor x = zeros({s.T, s.Dh}), cosT = zeros({s.T, D2}), sinT = zeros({s.T, D2});
        auto xp = x.data_as<float>(), cp = cosT.data_as<float>(), sp = sinT.data_as<float>();
        for (std::size_t i = 0; i < n; ++i) xp[i] = static_cast<float>((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f;
        for (int64_t k = 0; k < s.T * D2; ++k) {
            const float th = 0.01f * static_cast<float>(k % 257);
            cp[static_cast<std::size_t>(k)] = std::cos(th);  sp[static_cast<std::size_t>(k)] = std::sin(th);
        }
        Tensor out_c = zeros({s.T, s.Dh});
        auto op = out_c.data_as<float>();
        const auto cpu_t = time_it(s.label + "  [CPU]", n * 2, -1, iters, [&]{
            for (int64_t t = 0; t < s.T; ++t)
                for (int64_t i = 0; i < D2; ++i) {
                    const float c = cp[t * D2 + i], sn = sp[t * D2 + i];
                    const float x0 = xp[t * s.Dh + i], x1 = xp[t * s.Dh + i + D2];
                    op[t * s.Dh + i]      = x0 * c - x1 * sn;
                    op[t * s.Dh + i + D2] = x0 * sn + x1 * c;
                }
        });

        std::vector<float> gpu_out(n);
        const double gpu_secs = sub0llm::backend::cuda::rope_fwd_bench(
            xp.data(), cp.data(), sp.data(), gpu_out.data(),
            static_cast<int>(s.T), static_cast<int>(s.Dh), iters);
        const double gpu_ms = gpu_secs * 1e3 / iters;
        const double gpu_gbps = (static_cast<double>(n) * 4.0 / 1e9) / (gpu_ms * 1e-3);

        double se = 0.0, sref = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = static_cast<double>(gpu_out[i]) - static_cast<double>(op[i]);
            se += d * d;  sref += static_cast<double>(op[i]) * static_cast<double>(op[i]);
        }
        const double rel_rms = std::sqrt(se / (sref + 1e-30));
        std::cout << std::format(
            "  {:32s}  {:8.3f} ms  {:6.1f} GB/s  [CUDA]   {:5.2f}x vs CPU   relRMS {:.2e}\n",
            s.label + "  [CUDA]", gpu_ms, gpu_gbps, cpu_t.ms_per_iter / gpu_ms, rel_rms);
    }

    std::cout << "\n--- Device silu fwd: CPU vs CUDA (N) ---\n";
    for (const std::size_t SN : {std::size_t{1} << 16, std::size_t{1} << 20}) {
        Tensor x = zeros({static_cast<int64_t>(SN)});
        auto xp = x.data_as<float>();
        for (std::size_t i = 0; i < SN; ++i) xp[i] = static_cast<float>((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f;
        const std::string label = std::format("N={}", SN);
        Tensor y_cpu = ops::silu(x);
        const auto cpu_t = time_it(label + "  [CPU]", SN * 2, -1, iters, [&]{ (void)ops::silu(x); });

        std::vector<float> gpu_y(SN);
        const double gpu_secs = sub0llm::backend::cuda::silu_fwd_bench(
            xp.data(), gpu_y.data(), static_cast<int>(SN), iters);
        const double gpu_ms = gpu_secs * 1e3 / iters;
        const double gpu_gbps = (static_cast<double>(SN) * 4.0 / 1e9) / (gpu_ms * 1e-3);

        double se = 0.0, sref = 0.0;
        const auto yc = y_cpu.data_as<float>();
        for (std::size_t i = 0; i < SN; ++i) {
            const double d = static_cast<double>(gpu_y[i]) - static_cast<double>(yc[i]);
            se += d * d;  sref += static_cast<double>(yc[i]) * static_cast<double>(yc[i]);
        }
        const double rel_rms = std::sqrt(se / (sref + 1e-30));
        std::cout << std::format(
            "  {:32s}  {:8.3f} ms  {:6.1f} GB/s  [CUDA]   {:5.2f}x vs CPU   relRMS {:.2e}\n",
            label + "  [CUDA]", gpu_ms, gpu_gbps, cpu_t.ms_per_iter / gpu_ms, rel_rms);
    }

    std::cout << "\n--- Device scale (x*alpha): CPU vs CUDA (N) ---\n";
    for (const std::size_t SN : {std::size_t{1} << 20}) {
        Tensor x = zeros({static_cast<int64_t>(SN)});
        auto xp = x.data_as<float>();
        for (std::size_t i = 0; i < SN; ++i) xp[i] = static_cast<float>((i * 2654435761u) & 0xFFFF) / 32768.0f - 1.0f;
        const float alpha = 1.5f;
        const std::string label = std::format("N={}", SN);
        Tensor y_cpu = ops::mul(x, alpha);
        const auto cpu_t = time_it(label + "  [CPU]", SN * 2, -1, iters, [&]{ (void)ops::mul(x, alpha); });

        std::vector<float> gpu_y(SN);
        const double gpu_secs = sub0llm::backend::cuda::mul_scalar_bench(
            xp.data(), alpha, gpu_y.data(), static_cast<int>(SN), iters);
        const double gpu_ms = gpu_secs * 1e3 / iters;
        const double gpu_gbps = (static_cast<double>(SN) * 4.0 / 1e9) / (gpu_ms * 1e-3);

        double se = 0.0, sref = 0.0;
        const auto yc = y_cpu.data_as<float>();
        for (std::size_t i = 0; i < SN; ++i) {
            const double d = static_cast<double>(gpu_y[i]) - static_cast<double>(yc[i]);
            se += d * d;  sref += static_cast<double>(yc[i]) * static_cast<double>(yc[i]);
        }
        const double rel_rms = std::sqrt(se / (sref + 1e-30));
        std::cout << std::format(
            "  {:32s}  {:8.3f} ms  {:6.1f} GB/s  [CUDA]   {:5.2f}x vs CPU   relRMS {:.2e}\n",
            label + "  [CUDA]", gpu_ms, gpu_gbps, cpu_t.ms_per_iter / gpu_ms, rel_rms);
    }
#else
    (void)iters;
    std::cout << "\n--- Device ops: CPU-only build (configure --preset cuda for GPU rows) ---\n";
#endif
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::size_t N     = 1 << 20;  // 1M elements
    int         iters = 100;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n"     && i + 1 < argc) N     = static_cast<std::size_t>(std::stoi(argv[++i]));
        if (arg == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    }

    std::cout << std::format("sub0llm kernel benchmarks  (N={}, iters={})\n", N, iters);

    bench_element_wise(N, iters);
    bench_reductions(N, iters);
    bench_matmul(iters);
    bench_copy_strided(iters);
    bench_autograd_forward(iters);
    bench_device_ops(iters);

    std::cout << "\nDone.\n";
    return 0;
}
