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

    time_it("add_f32",     N * 2, -1, iters, [&]{ ops::add(a, b); });
    time_it("mul_f32",     N * 2, -1, iters, [&]{ ops::mul(a, b); });
    time_it("relu_f32",    N * 2, -1, iters, [&]{ ops::relu(a);   });
    time_it("sigmoid_f32", N * 2, -1, iters, [&]{ ops::sigmoid(a); });
    time_it("silu_f32",    N * 2, -1, iters, [&]{ ops::silu(a);   });
    time_it("exp_f32",     N * 2, -1, iters, [&]{ ops::exp(a);    });
    time_it("log_f32",     N * 2, -1, iters, [&]{ ops::log(a);    });
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
        time_it(s.label, n_floats, flops, iters, [&]{ ops::matmul(A, B); });
    }
}

static void bench_autograd_forward(int iters)
{
    std::cout << "\n--- Autograd forward (T=32, D=128) ---\n";

    using namespace autograd;

    const int64_t T = 32, D = 128;
    Tensor x_t = zeros({T, D});
    Tensor w_t = zeros({D});
    Tensor b_t = zeros({D});
    auto xs = x_t.data_as<float>();
    auto ws = w_t.data_as<float>();
    for (std::size_t i = 0; i < static_cast<std::size_t>(T * D); ++i) xs[i] = static_cast<float>(i) * 0.001f;
    for (std::size_t i = 0; i < static_cast<std::size_t>(D); ++i) ws[i] = 1.0f;

    Variable x(x_t, true);
    Variable w(w_t, true);
    Variable b(b_t, true);
    const std::size_t n = static_cast<std::size_t>(T * D);

    time_it("rms_norm fwd",    n * 2, -1, iters, [&]{ rms_norm(x, w); });
    time_it("softmax fwd",     n * 2, -1, iters, [&]{ softmax(x); });
    time_it("silu fwd",        n * 2, -1, iters, [&]{ silu(x); });
    time_it("gelu fwd",        n * 2, -1, iters, [&]{ gelu(x); });
    time_it("matmul (32x128x128)", static_cast<std::size_t>(T*(D+D)+D*D), 2*T*D*D, iters,
        [&]{ Tensor wmat = zeros({D, D}); Variable wv(wmat, true); matmul(x, wv); });
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
    bench_autograd_forward(iters);

    std::cout << "\nDone.\n";
    return 0;
}
