// Memory-bandwidth ceiling probe + Q8 decode-ladder diagnostic for sub0llm.
//
// Single-token LLM decode is a pure streaming read of the weights (zero arithmetic
// reuse), so its hard ceiling is DRAM bandwidth ÷ model bytes. But real decode on this
// machine tops out ~30% BELOW the exercised bus maximum — so *something between a raw
// memory read and the real kernel* is the bottleneck. This probe walks a LADDER from
// pure memory to the actual int8 GEMV so we can see exactly which rung we fall off:
//
//   1. read      — pure f32 loads (the classic STREAM read; the absolute ceiling).
//   2. copy/triad— STREAM 1R+1W / 2R+1W, for cross-referencing published numbers.
//   3. q8_stream — read the Q8 weight bytes (34 B blocks) with ~no math: isolates any
//                  cost of the Q8 byte layout / 34-byte stride vs aligned f32.
//   4. q8_gemv   — the REAL dot_q8_0_q8_0 kernel over a big weight buffer, activation
//                  reused across rows (exactly the decode hot path): adds the int8
//                  compute-per-byte. If this drops below q8_stream, we're compute/
//                  load-issue bound, not bus bound — and ILP (register blocking, VNNI)
//                  is the lever, not memory layout.
//
// Each rung is thread-swept and core-pinned (mirroring src/nn/gemma.cpp's pool), with a
// per-core-efficiency column and a 2/4-core readout (a cleaner "theoretical max" target
// than 20+ cores, below the bus wall). A drift recheck re-measures `read` at the end:
// Arrow Lake-HX throttles under sustained load, so a >5% drop flags thermal bias.
//
// Build (native release for meaningful numbers):
//   cmake --build build-native --target bench_membw
//   ./build-native/bin/bench_membw --peak 102
//
// Flags:
//   --mib N      per-array size in MiB (default 256; must be >> LLC to be DRAM-bound)
//   --iters N    timed repetitions per thread count (default 30)
//   --max-threads N  cap the thread sweep (default = hardware_concurrency)
//   --k N        GEMV inner dim K for q8_gemv (default 3840 = Gemma D; rows = buf/K)
//   --peak GBPS  theoretical JEDEC peak, for the % column (default 0 = unknown)

#include "sub0llm/backends/cpu/quant.hpp"

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

namespace cpu = sub0llm::backend::cpu;
using cpu::BlockQ8_0;
constexpr int QK = cpu::QK8_0;   // 32 weights / block, 34 bytes / block

// Pin the calling thread to one logical CPU — mirrors src/nn/gemma.cpp so the probe
// measures the same threading regime the model actually runs under (no OS migration
// thrashing the per-core caches).
void pin_thread_to_cpu(int cpu_id) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << (cpu_id & 63));
#else
    (void)cpu_id;
#endif
}

template <typename T>
T* alloc_aligned(std::size_t n) {
    void* p = ::operator new[](n * sizeof(T), std::align_val_t{64});
    return static_cast<T*>(p);
}
template <typename T>
void free_aligned(T* p) { ::operator delete[](p, std::align_val_t{64}); }

// ── Kernels over [lo, hi) ─────────────────────────────────────────────────────

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

void kernel_copy(const float* a, float* b, std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) b[i] = a[i];
}
void kernel_triad(float* a, const float* b, const float* c, float s,
                  std::size_t lo, std::size_t hi) {
    for (std::size_t i = lo; i < hi; ++i) a[i] = b[i] + s * c[i];
}

// Q8 byte stream: read every weight byte with ~no math (sum all 32 quants/block), so
// the full Q8 buffer genuinely streams from DRAM — the load-bandwidth floor of the
// kernel's input, before any int8 dot work is added.
int64_t kernel_q8_stream(const BlockQ8_0* w, std::size_t lo, std::size_t hi) {
    int64_t acc = 0;
    for (std::size_t i = lo; i < hi; ++i)
        for (int j = 0; j < QK; ++j) acc += w[i].qs[j];
    return acc;
}

// Q8 GEMV: the REAL kernel. y[r] = dot_q8_0_q8_0(W + r*kblk, xq, kblk) for r in [lo,hi);
// the small activation xq is reused across rows — exactly the decode hot path.
void kernel_q8_gemv(const BlockQ8_0* w, const BlockQ8_0* xq, float* y,
                    std::size_t kblk, std::size_t lo, std::size_t hi) {
    for (std::size_t r = lo; r < hi; ++r)
        y[r] = cpu::dot_q8_0_q8_0(w + r * kblk, xq, static_cast<int64_t>(kblk));
}

enum class Kernel { Read, Copy, Triad, Q8Stream, Q8Gemv };

struct Bufs {
    float* A; float* B; float* C; std::size_t n;     // f32 STREAM arrays
    const BlockQ8_0* W; const BlockQ8_0* xq;          // Q8 weight buffer + activation
    std::size_t nblocks; std::size_t kblk;            // total blocks; blocks per GEMV row
};

// Run one kernel across `nthreads` pinned threads for `iters` timed reps; returns GB/s
// counting the bytes that actually stream from DRAM for that kernel.
double measure(Kernel k, int nthreads, int iters, const Bufs& b,
               float* gemv_y, volatile float* sink) {
    std::barrier sync(nthreads);
    std::atomic<double> elapsed_s{0.0};
    const std::size_t M = b.nblocks / b.kblk;   // q8_gemv rows

    const auto worker = [&](int tid) {
        pin_thread_to_cpu(tid);
        const auto nt = static_cast<std::size_t>(nthreads);
        const auto t  = static_cast<std::size_t>(tid);
        // f32 / q8_stream split the flat element/block range; q8_gemv splits rows.
        const std::size_t elo = (b.n       * t) / nt, ehi = (b.n       * (t + 1)) / nt;
        const std::size_t blo = (b.nblocks * t) / nt, bhi = (b.nblocks * (t + 1)) / nt;
        const std::size_t rlo = (M         * t) / nt, rhi = (M         * (t + 1)) / nt;

        float local = 0.0f;
        const auto run_once = [&] {
            switch (k) {
                case Kernel::Read:     local += kernel_read(b.A, elo, ehi); break;
                case Kernel::Copy:     kernel_copy(b.A, b.B, elo, ehi); break;
                case Kernel::Triad:    kernel_triad(b.A, b.B, b.C, 3.0f, elo, ehi); break;
                case Kernel::Q8Stream: local += static_cast<float>(kernel_q8_stream(b.W, blo, bhi)); break;
                case Kernel::Q8Gemv:   kernel_q8_gemv(b.W, b.xq, gemv_y, b.kblk, rlo, rhi); break;
            }
        };

        for (int w = 0; w < 3; ++w) run_once();   // warmup + page-fault settle
        sync.arrive_and_wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int it = 0; it < iters; ++it) run_once();
        sync.arrive_and_wait();
        if (tid == 0)
            elapsed_s.store(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
        *sink += local;
    };

    std::vector<std::thread> ts;
    ts.reserve(static_cast<std::size_t>(nthreads) - 1);
    for (int t = 1; t < nthreads; ++t) ts.emplace_back(worker, t);
    worker(0);
    for (auto& t : ts) t.join();

    const double secs = elapsed_s.load(std::memory_order_relaxed);
    double bytes = 0.0;
    switch (k) {
        case Kernel::Read:     bytes = 1.0 * static_cast<double>(b.n) * sizeof(float); break;
        case Kernel::Copy:     bytes = 2.0 * static_cast<double>(b.n) * sizeof(float); break;
        case Kernel::Triad:    bytes = 3.0 * static_cast<double>(b.n) * sizeof(float); break;
        case Kernel::Q8Stream: bytes = static_cast<double>(b.nblocks) * sizeof(BlockQ8_0); break;
        case Kernel::Q8Gemv:   bytes = static_cast<double>(M * b.kblk) * sizeof(BlockQ8_0); break;
    }
    bytes *= static_cast<double>(iters);
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
    int k_dim = 3840;          // Gemma D
    double peak = 0.0;

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        auto next = [&]() -> std::string_view { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--mib")              per_array_mib = static_cast<std::size_t>(parse_int(next(), 256));
        else if (a == "--iters")       iters = parse_int(next(), 30);
        else if (a == "--max-threads") max_threads = parse_int(next(), max_threads);
        else if (a == "--k")           k_dim = parse_int(next(), 3840);
        else if (a == "--peak")        peak = parse_double(next(), 0.0);
        else if (a == "--help" || a == "-h") {
            std::cout << "usage: bench_membw [--mib N] [--iters N] [--max-threads N] "
                         "[--k N] [--peak GBPS]\n";
            return 0;
        }
    }

    // f32 STREAM arrays.
    const std::size_t n = (per_array_mib * 1024 * 1024) / sizeof(float);
    float* A = alloc_aligned<float>(n);
    float* B = alloc_aligned<float>(n);
    float* C = alloc_aligned<float>(n);
    for (std::size_t i = 0; i < n; ++i) { A[i] = 1.0f; B[i] = 2.0f; C[i] = 3.0f; }

    // Q8 weight buffer (same MiB budget) + a single reused activation row of K weights.
    const std::size_t kblk    = static_cast<std::size_t>(k_dim) / static_cast<std::size_t>(QK);
    const std::size_t nblocks = ((per_array_mib * 1024 * 1024) / sizeof(BlockQ8_0) / kblk) * kblk;
    const std::size_t M       = nblocks / kblk;
    BlockQ8_0* W  = alloc_aligned<BlockQ8_0>(nblocks);
    BlockQ8_0* Xq = alloc_aligned<BlockQ8_0>(kblk);
    for (std::size_t i = 0; i < nblocks; ++i) {
        W[i].d = cpu::f32_to_f16(0.05f);
        for (int j = 0; j < QK; ++j) W[i].qs[j] = static_cast<int8_t>((static_cast<int>(i % 17) + j) % 17 - 8);
    }
    for (std::size_t i = 0; i < kblk; ++i) {
        Xq[i].d = cpu::f32_to_f16(0.05f);
        for (int j = 0; j < QK; ++j) Xq[i].qs[j] = static_cast<int8_t>((static_cast<int>(i % 13) * 3 + j) % 13 - 6);
    }
    float* gemv_y = alloc_aligned<float>(M);
    volatile float sink = 0.0f;

    Bufs bufs{A, B, C, n, W, Xq, nblocks, kblk};

    std::cout << std::format(
        "Memory-bandwidth + Q8 decode-ladder probe — {} MiB/array, {} reps, up to {} threads\n"
        "Q8 buffer: {} blocks ({:.0f} MiB), GEMV K={} ({} rows). The ladder walks from a\n"
        "pure read to the real int8 kernel; watch which rung falls off the read ceiling.\n\n",
        per_array_mib, iters, max_threads, nblocks,
        static_cast<double>(nblocks) * sizeof(BlockQ8_0) / (1024.0 * 1024.0),
        k_dim, M);

    std::vector<int> sweep;
    for (int t = 1; t <= max_threads; t = (t < 8 ? t * 2 : t + 4)) sweep.push_back(t);
    if (sweep.empty() || sweep.back() != max_threads) sweep.push_back(max_threads);

    std::cout << std::format("  {:>7s} {:>10s} {:>10s} {:>10s} {:>11s} {:>11s}   {:>12s}\n",
                             "threads", "read", "copy", "triad", "q8_stream", "q8_gemv",
                             "gemv/read");
    std::cout << std::format("  {:>7s} {:>10s} {:>10s} {:>10s} {:>11s} {:>11s}   {:>12s}\n",
                             "", "GB/s", "GB/s", "GB/s", "GB/s", "GB/s", "");

    double best_read = 0.0, best_gemv = 0.0;
    int best_read_t = 1, best_gemv_t = 1;
    double read_1 = 0.0, gemv_1 = 0.0;
    double read_at4 = 0.0, gemv_at4 = 0.0;

    // Brief cooldown between rungs/rows: this chip throttles under sustained load, so
    // spacing measurements keeps the comparison closer to a steady (non-drifting) state.
    const auto cooldown = [] { std::this_thread::sleep_for(std::chrono::milliseconds(120)); };
    // Bandwidth is a "peak capability" metric — take the best of 2 (noise only slows a
    // run, it can't make memory faster than it is), with a cooldown between attempts.
    const auto best2 = [&](Kernel k, int t) {
        cooldown();
        const double a = measure(k, t, iters, bufs, gemv_y, &sink);
        cooldown();
        const double b = measure(k, t, iters, bufs, gemv_y, &sink);
        return std::max(a, b);
    };

    for (int t : sweep) {
        const double rd = best2(Kernel::Read,     t);
        const double cp = best2(Kernel::Copy,     t);
        const double tr = best2(Kernel::Triad,    t);
        const double qs = best2(Kernel::Q8Stream, t);
        const double qg = best2(Kernel::Q8Gemv,   t);
        if (rd > best_read) { best_read = rd; best_read_t = t; }
        if (qg > best_gemv) { best_gemv = qg; best_gemv_t = t; }
        if (t == 1) { read_1 = rd; gemv_1 = qg; }
        if (t == 4) { read_at4 = rd; gemv_at4 = qg; }
        std::cout << std::format("  {:7d} {:10.1f} {:10.1f} {:10.1f} {:11.1f} {:11.1f}   {:11.0f}%\n",
                                 t, rd, cp, tr, qs, qg, 100.0 * qg / rd);
    }

    // Drift recheck: re-measure read at the best thread count; a big drop = thermal bias.
    const double read_recheck = measure(Kernel::Read, best_read_t, iters, bufs, gemv_y, &sink);

    std::cout << "\n";
    std::cout << std::format("EXERCISED MAX (read):  {:.1f} GB/s @ {}t\n", best_read, best_read_t);
    std::cout << std::format("BEST Q8 GEMV (kernel): {:.1f} GB/s @ {}t  = {:.0f}% of read ceiling\n",
                             best_gemv, best_gemv_t, 100.0 * best_gemv / best_read);
    if (peak > 0.0)
        std::cout << std::format("THEORETICAL (JEDEC):   {:.1f} GB/s  (read = {:.0f}% of paper)\n",
                                 peak, 100.0 * best_read / peak);
    if (read_1 > 0.0) {
        std::cout << std::format("\nSmall-N scaling (cleaner target than 20+ cores, below the bus wall):\n");
        std::cout << std::format("  read  1t {:.1f} -> 4t {:.1f}  ({:.2f}x, {:.0f}% per-core eff)\n",
                                 read_1, read_at4, read_at4 / read_1, 100.0 * (read_at4 / read_1) / 4.0);
        if (gemv_1 > 0.0)
            std::cout << std::format("  gemv  1t {:.1f} -> 4t {:.1f}  ({:.2f}x, {:.0f}% per-core eff)\n",
                                     gemv_1, gemv_at4, gemv_at4 / gemv_1, 100.0 * (gemv_at4 / gemv_1) / 4.0);
    }
    std::cout << std::format("\nThermal drift: read recheck {:.1f} GB/s vs {:.1f} best ({:+.0f}%){}\n",
                             read_recheck, best_read, 100.0 * (read_recheck - best_read) / best_read,
                             read_recheck < 0.95 * best_read ? "  <-- THROTTLING, interleave runs!" : "");
    std::cout << "\nDiagnosis: if q8_gemv << q8_stream we're compute/load-issue bound (ILP lever:\n"
                 "register-blocking, VNNI); if q8_stream << read it's the Q8 layout/stride; if both\n"
                 "track read we're bus-bound and only fewer bytes (lower-bit quant) helps.\n";
    std::cout << "LLM REAL: compare best q8_gemv GB/s against sub0llm-gemma's tok/s x model GB/token.\n";

    (void)sink;
    free_aligned(A); free_aligned(B); free_aligned(C);
    free_aligned(W); free_aligned(Xq); free_aligned(gemv_y);
    return 0;
}
