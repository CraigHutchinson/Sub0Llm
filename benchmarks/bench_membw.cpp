// Memory-bandwidth ceiling probe (STREAM-style) for sub0llm.
//
// Establishes the THREE numbers that bracket LLM decode throughput on this exact
// machine, so the "we hit 71 GB/s vs 102 GB/s on paper" claim is grounded in a
// measured ceiling rather than a spec sheet:
//
//   1. THEORETICAL  — JEDEC peak from the RAM spec (you pass it via --peak).
//   2. EXERCISED MAX — what raw memory traffic actually achieves here (this tool).
//   3. LLM REAL      — single-token decode GB/s (from sub0llm-gemma -t N).
//
// Single-token decode is a *pure streaming read* with zero arithmetic reuse, so the
// most LLM-relevant figure is the READ kernel below (pure loads). Copy/Triad are the
// classic STREAM kernels (load+store / 2 load + 1 store) for cross-referencing against
// published STREAM numbers and for the prompt/training (write-heavy) regimes.
//
// Build with the native release preset for meaningful numbers:
//   cmake --build build-native --target bench_membw
//   ./build-native/bin/bench_membw --peak 102
//
// Flags:
//   --mib  N    per-array size in MiB (default 256; must be >> LLC to be DRAM-bound)
//   --iters N   timed repetitions per thread count (default 30)
//   --max-threads N  cap the thread sweep (default = hardware_concurrency)
//   --peak GBPS theoretical JEDEC peak, for the % column (default 0 = unknown)

#include <algorithm>
#include <atomic>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <iostream>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__) || defined(__i386__)
#  include <immintrin.h>
#  define MEMBW_HAVE_AVX 1
#else
#  define MEMBW_HAVE_AVX 0
#endif

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace {

// Pin the calling thread to one logical CPU — mirrors src/nn/gemma.cpp so the probe
// measures the same threading regime the model actually runs under (no OS migration
// thrashing the per-core caches).
void pin_thread_to_cpu(int cpu) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << (cpu & 63));
#else
    (void)cpu;
#endif
}

float* alloc_floats(std::size_t n) {
    constexpr std::size_t kAlign = 64;
    void* p = ::operator new[](n * sizeof(float), std::align_val_t{kAlign});
    return static_cast<float*>(p);
}
void free_floats(float* p) {
    ::operator delete[](p, std::align_val_t{64});
}

// ── Kernels over [lo, hi) ─────────────────────────────────────────────────────
// Each returns nothing meaningful except the READ kernel, which returns a partial
// sum so the loads can't be optimised away.

float kernel_read(const float* a, std::size_t lo, std::size_t hi) {
#if MEMBW_HAVE_AVX
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    std::size_t i = lo;
    for (; i + 32 <= hi; i += 32) {
        s0 = _mm256_add_ps(s0, _mm256_load_ps(a + i +  0));
        s1 = _mm256_add_ps(s1, _mm256_load_ps(a + i +  8));
        s2 = _mm256_add_ps(s2, _mm256_load_ps(a + i + 16));
        s3 = _mm256_add_ps(s3, _mm256_load_ps(a + i + 24));
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, s0);
    float acc = 0.0f;
    for (float v : tmp) acc += v;
    for (; i < hi; ++i) acc += a[i];
    return acc;
#else
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    std::size_t i = lo;
    for (; i + 4 <= hi; i += 4) { a0 += a[i]; a1 += a[i+1]; a2 += a[i+2]; a3 += a[i+3]; }
    float acc = a0 + a1 + a2 + a3;
    for (; i < hi; ++i) acc += a[i];
    return acc;
#endif
}

// Copy: b = a   (1 read + 1 write per element)
void kernel_copy(const float* a, float* b, std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) b[i] = a[i];
}

// Triad: a = b + s*c   (2 read + 1 write per element) — classic STREAM kernel.
void kernel_triad(float* a, const float* b, const float* c, float s,
                  std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) a[i] = b[i] + s * c[i];
}

enum class Kernel { Read, Copy, Triad };

struct SweepResult { int threads; double gbps; };

// Run one kernel across `nthreads` pinned threads for `iters` timed reps; returns GB/s
// counting the canonical STREAM byte traffic for that kernel.
double measure(Kernel k, int nthreads, std::size_t n, int iters,
               float* A, float* B, float* C, volatile float* sink) {
    std::barrier sync(nthreads);
    std::atomic<double> elapsed_s{0.0};

    const auto worker = [&](int tid) {
        pin_thread_to_cpu(tid);
        const auto nt = static_cast<std::size_t>(nthreads);
        const std::size_t lo = (n * static_cast<std::size_t>(tid)) / nt;
        const std::size_t hi = (n * static_cast<std::size_t>(tid + 1)) / nt;

        float local = 0.0f;
        // Warmup (also faults-in / settles the pages on first touch).
        for (int w = 0; w < 3; ++w) {
            switch (k) {
                case Kernel::Read:  local += kernel_read(A, lo, hi); break;
                case Kernel::Copy:  kernel_copy(A, B, lo, hi); break;
                case Kernel::Triad: kernel_triad(A, B, C, 3.0f, lo, hi); break;
            }
        }

        sync.arrive_and_wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) {
            switch (k) {
                case Kernel::Read:  local += kernel_read(A, lo, hi); break;
                case Kernel::Copy:  kernel_copy(A, B, lo, hi); break;
                case Kernel::Triad: kernel_triad(A, B, C, 3.0f, lo, hi); break;
            }
        }
        sync.arrive_and_wait();
        if (tid == 0) {
            const auto t1 = std::chrono::steady_clock::now();
            elapsed_s.store(std::chrono::duration<double>(t1 - t0).count(),
                            std::memory_order_relaxed);
        }
        *sink += local;  // defeat dead-code elimination of the read sums
    };

    std::vector<std::thread> ts;
    ts.reserve(static_cast<std::size_t>(nthreads) - 1);
    for (int t = 1; t < nthreads; ++t) ts.emplace_back(worker, t);
    worker(0);
    for (auto& t : ts) t.join();

    const double secs = elapsed_s.load(std::memory_order_relaxed);
    const double arrays = (k == Kernel::Read) ? 1.0 : (k == Kernel::Copy ? 2.0 : 3.0);
    const double bytes = arrays * static_cast<double>(n) * sizeof(float)
                       * static_cast<double>(iters);
    return (bytes / 1e9) / secs;
}

int parse_int(std::string_view s, int fallback) {
    int v = fallback;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}
double parse_double(std::string_view s, double fallback) {
    try { return std::stod(std::string(s)); } catch (...) { return fallback; }
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t per_array_mib = 256;
    int iters = 30;
    int max_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (max_threads <= 0) max_threads = 1;
    double peak = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&]() -> std::string_view { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--mib")              per_array_mib = static_cast<std::size_t>(parse_int(next(), 256));
        else if (a == "--iters")       iters = parse_int(next(), 30);
        else if (a == "--max-threads") max_threads = parse_int(next(), max_threads);
        else if (a == "--peak")        peak = parse_double(next(), 0.0);
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: bench_membw [--mib N] [--iters N] [--max-threads N] [--peak GBPS]\n";
            return 0;
        }
    }

    const std::size_t n = (per_array_mib * 1024 * 1024) / sizeof(float);

    float* A = alloc_floats(n);
    float* B = alloc_floats(n);
    float* C = alloc_floats(n);
    for (std::size_t i = 0; i < n; ++i) { A[i] = 1.0f; B[i] = 2.0f; C[i] = 3.0f; }
    volatile float sink = 0.0f;

    std::cout << std::format(
        "Memory-bandwidth probe — {} MiB/array, {} timed reps, up to {} threads\n",
        per_array_mib, iters, max_threads);
    std::cout << "Read = pure loads (the LLM single-token-decode-relevant ceiling)\n";
    std::cout << "Copy = 1R+1W, Triad = 2R+1W (classic STREAM)\n\n";

    std::cout << std::format("  {:>7s} {:>12s} {:>12s} {:>12s}\n",
                             "threads", "Read GB/s", "Copy GB/s", "Triad GB/s");

    double best_read = 0.0;
    int best_read_threads = 1;

    // Sweep 1, 2, 4, 8, then every 4 up to max, plus max itself.
    std::vector<int> sweep;
    for (int t = 1; t <= max_threads; t = (t < 8 ? t * 2 : t + 4)) sweep.push_back(t);
    if (sweep.empty() || sweep.back() != max_threads) sweep.push_back(max_threads);

    for (int t : sweep) {
        const double r  = measure(Kernel::Read,  t, n, iters, A, B, C, &sink);
        const double cp = measure(Kernel::Copy,  t, n, iters, A, B, C, &sink);
        const double tr = measure(Kernel::Triad, t, n, iters, A, B, C, &sink);
        if (r > best_read) { best_read = r; best_read_threads = t; }
        std::cout << std::format("  {:7d} {:12.1f} {:12.1f} {:12.1f}\n", t, r, cp, tr);
    }

    std::cout << "\n";
    std::cout << std::format("EXERCISED MAX (read):  {:.1f} GB/s  @ {} threads\n",
                             best_read, best_read_threads);
    if (peak > 0.0) {
        std::cout << std::format("THEORETICAL (JEDEC):   {:.1f} GB/s\n", peak);
        std::cout << std::format("Efficiency:            {:.0f}% of paper peak\n",
                                 100.0 * best_read / peak);
    }
    std::cout << "LLM REAL: compare against sub0llm-gemma -t N (tok/s x model GB/token)\n";

    (void)sink;
    free_floats(A);
    free_floats(B);
    free_floats(C);
    return 0;
}
