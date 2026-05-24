// Chapter 02 — Compute Backends
//
// Ch01 gave us correct, readable tensor ops.  Ch02 makes them fast and
// portable across hardware by layering a three-tier dispatch:
//
//   ops::add(a, b)
//     │
//     ├─ a.device().is_cuda()     → backend::cuda::add()    (cuBLAS/custom kernels)
//     ├─ a.device().is_openvino() → backend::openvino::add() (oneDNN, stub here)
//     └─ CPU                      → backend::cpu::add_f32()
//                                      ├─ SUB0LLM_AVX512 → 16-wide  _mm512_add_ps
//                                      ├─ SUB0LLM_AVX2   →  8-wide  _mm256_add_ps
//                                      └─ else           → scalar loop
//
// Topics covered:
//   • SIMD fundamentals: registers, lanes, throughput vs latency
//   • AVX2 element-wise kernels and the SIMD tail pattern
//   • Cache-blocked matrix multiply with AVX2 FMA
//   • CUDA memory model: host/device, cudaMalloc, kernel launch
//   • Tensor::to(device) — zero-copy on same device, DMA transfer otherwise
//   • Runtime-performance comparison: naive vs blocked vs SIMD matmul

#include <chrono>
#include <format>
#include <iostream>

#include "sub0llm/compat/print.hpp"
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/ops.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;
using namespace sub0llm::ops;
using clk = std::chrono::high_resolution_clock;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ── Benchmark helper ──────────────────────────────────────────────────────────
// Returns microseconds per call (average over `reps` repetitions).
template<typename F>
double bench_us(F fn, int reps = 5) {
    fn(); // warm up
    auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) fn();
    auto t1 = clk::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 02: Compute Backends", version_string);

    // ── 1. SIMD fundamentals ──────────────────────────────────────────────────
    section("1. SIMD fundamentals");

    std::println("AVX2:   {} compile-time detection: {}",
        "SUB0LLM_AVX2",
#ifdef SUB0LLM_AVX2
        "ON"
#else
        "OFF"
#endif
    );
    std::println("AVX512: {} compile-time detection: {}",
        "SUB0LLM_AVX512",
#ifdef SUB0LLM_AVX512
        "ON"
#else
        "OFF"
#endif
    );
    std::println("");
    std::println("With AVX2: one _mm256_add_ps adds 8 floats per clock cycle.");
    std::println("A 256-element vector add takes ~32 iterations (256/8) instead of 256.");

    // ── 2. Element-wise ops via SIMD dispatch ──────────────────────────────────
    section("2. Element-wise ops — SIMD dispatch");

    const std::size_t N = 1024 * 1024; // 1M floats = 4 MB
    Tensor a = randn({static_cast<std::int64_t>(N)});
    Tensor b = randn({static_cast<std::int64_t>(N)});

    const double add_us = bench_us([&]{ auto c = add(a, b); (void)c; });
    const double mul_us = bench_us([&]{ auto c = mul(a, b); (void)c; });

    std::println("add(1M floats): {:.1f} µs  ({:.1f} GB/s effective)",
        add_us, (3.0 * N * 4) / (add_us * 1e-6) / 1e9);
    std::println("mul(1M floats): {:.1f} µs  ({:.1f} GB/s effective)",
        mul_us, (3.0 * N * 4) / (mul_us * 1e-6) / 1e9);

    // ── 3. Cache-blocked matmul ────────────────────────────────────────────────
    section("3. Cache-blocked matmul with AVX2 FMA");

    // Small: fits in cache — shows pure compute throughput
    {
        Tensor A = randn({256, 256});
        Tensor B = randn({256, 256});
        const double us = bench_us([&]{ auto C = matmul(A, B); (void)C; });
        const double gflops = 2.0 * 256 * 256 * 256 / (us * 1e-6) / 1e9;
        std::println("matmul(256×256): {:.1f} µs  {:.2f} GFLOP/s", us, gflops);
    }

    // Medium: slightly larger than L1 — tests blocking effectiveness
    {
        Tensor A = randn({512, 512});
        Tensor B = randn({512, 512});
        const double us = bench_us([&]{ auto C = matmul(A, B); (void)C; }, 3);
        const double gflops = 2.0 * 512 * 512 * 512 / (us * 1e-6) / 1e9;
        std::println("matmul(512×512): {:.1f} µs  {:.2f} GFLOP/s", us, gflops);
    }

    // ── 4. Device transfer ─────────────────────────────────────────────────────
    section("4. Tensor::to(device)");

    Tensor cpu_t = randn({4, 4});
    std::println("Original device: {}", cpu_t.device().str());
    std::println("Tensor::to(cpu)  — same device, shallow copy: {}",
        cpu_t.to(Device::cpu()).device().str());

#ifdef SUB0LLM_CUDA
    Tensor gpu_t = cpu_t.to(Device::cuda(0));
    std::println("After .to(cuda:0): {}", gpu_t.device().str());
    Tensor back  = gpu_t.to(Device::cpu());
    // Verify round-trip
    auto orig = cpu_t.data_as<float>();
    auto rt   = back.data_as<float>();
    bool ok = true;
    for (std::size_t i = 0; i < orig.size(); ++i)
        if (orig[i] != rt[i]) { ok = false; break; }
    std::println("Round-trip CPU→GPU→CPU correct: {}", ok ? "YES" : "NO");
#else
    std::println("CUDA not compiled in — rebuild with -DSUB0LLM_ENABLE_CUDA=ON");
    std::println("(Tensor::to(Device::cuda()) throws a descriptive error in this build)");
#endif

    // ── 5. Dispatch path demo ──────────────────────────────────────────────────
    section("5. Dispatch path");

    std::println("ops::add dispatches based on tensor.device():");
    std::println("  CPU  tensor → backend::cpu::add_f32 (SIMD or scalar)");
    std::println("  CUDA tensor → backend::cuda::add    (custom kernel / cuBLAS)");
    std::println("  OVINO tensor→ backend::openvino::add (oneDNN, Ch14)");
    std::println("");
    std::println("The API ops::add(a,b) is unchanged regardless of device.");
    std::println("This is the design pattern used by llama.cpp's ggml backend system.");

    // ── 6. What's next ────────────────────────────────────────────────────────
    section("What's next — Ch03: Tokenization");
    std::println(
        "With fast tensor ops in place, Ch03 builds the text processing pipeline:\n"
        "  • Byte-pair encoding (BPE) tokenizer from scratch\n"
        "  • GPT-2 vocabulary loading from JSON\n"
        "  • Unicode normalization and special token handling\n"
        "  • Encode/decode roundtrip tests with real text"
    );

    return 0;
}
