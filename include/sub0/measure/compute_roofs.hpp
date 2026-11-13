// sub0/measure/compute_roofs.hpp -- measured compute roofs for `sub0llm machine --roofs compute` (issue #78,
// increment 3).
//
// WHY THIS EXISTS. Prefill is arithmetic-bound: a prompt's tokens go through the same matrix products together, so
// the peak multiply-accumulate rate of the cores sets the prefill ceiling (see ceilings.hpp). This header measures
// that peak for three kernels, each keeping its data in registers so no memory traffic is involved:
//   fma_f32   _mm512_fmadd_ps (the AVX2 _mm256_fmadd_ps when AVX-512 is not compiled in);
//   dot_i8    _mm512_dpbusd_epi32 (AVX-512 VNNI), u8 x s8 products summed in 32-bit lanes;
//   dot_bf16  _mm512_dpbf16_ps (AVX-512 BF16), bf16 pairs summed into float lanes.
// Each has 12 independent accumulators: the FMA latency of 4 cycles times 2 pipes needs at least 8 chains.
//
// HOW IT IS MEASURED.
//   * N threads, each pinned to its CPU, placed exactly as the memory roofs place them (pack, spread, smt; see
//     memory_roofs.hpp), started together at the same barrier. Each thread times only its own loop; the slowest
//     thread sets the time. Rate = all threads' operations / slowest time.
//   * Each pass is calibrated from a short probe so that one pass lasts about `passSeconds` (50 ms by default).
//   * The effective clock is measured separately, on the same placement and thread count: a chain of 8 dependent
//     integer adds per iteration (1-cycle latency each), so iterations x 8 cycles = elapsed x frequency. It shows the
//     frequency actually delivered while all those cores are busy (it drops as more cores are busy).
//   * Every pass is CHECKED. The accumulators start from known values and the loop has a known result per lane:
//       fma_f32   acc = fma(acc, 1, 0) from start values k*L+j+1 (L = lanes): every lane must be unchanged;
//       dot_i8    u8 all 1 and s8 all 1 add exactly 4 per lane per instruction: every lane must equal 4 x iterations;
//       dot_bf16  bf16 1.0 pairs add exactly 2 per lane per instruction: every lane must equal 2 x iterations.
//     The iteration caps keep those results exact: i8 caps iterations at 2^28 (4 x iterations < 2^31 in int32); bf16
//     caps them at 2^23 (2 x iterations <= 2^24, every integer up to 2^24 is exact in float). A failed check makes
//     that point invalid (`verified` false, the reason set). The check proves the instructions ran as written;
//     the measured-to-theoretical ratio is the speed check.
//   * The compiler is kept from folding the loops away: the constants and accumulators pass through an empty asm
//     with a register constraint, and the results are used after the timed region.
//
// THEORETICAL ROOF. roof = physical cores x effectiveGHz x opsPerCycleCore. The per-core rates in the table below are
// ASSUMPTIONS about Zen 4 (the target host is a Ryzen 9 8945HX). The measured-to-theoretical ratio is the check: a
// ratio well above 1 means the assumption is too low, a ratio far below 1 means the kernel or the placement is not
// reaching the pipes. The roof counts physical cores, not threads: an FMA pipe belongs to a core, so the two SMT
// threads of one core share one roof (for thread counts up to the core count the two are the same).
//
// WHAT IT IS NOT. Descriptive ceilings for planning prefill, not an engine benchmark. Nothing here changes an
// engine default.
//
// PLATFORMS. Linux measures. Elsewhere `supported` is false and no timing is done; the header still compiles. The
// kernels run only where their ISA is compiled in (-march=native for the tool and its tests). A kernel whose ISA is
// not compiled in reports compiled=false with the reason; it never falls back to a slower kernel and reports that
// as the roof.

#pragma once

#include "sub0/measure/ceilings.hpp"
#include "sub0/measure/machine_facts.hpp"
#include "sub0/measure/memory_roofs.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sub0::measure {

/// The three compute kernels.
enum class ComputeKernel { fmaF32, dotI8, dotBf16 };

/// The kernel's name as it appears in the output.
[[nodiscard]] inline std::string_view computeKernelName(ComputeKernel k) {
    switch (k) {
        case ComputeKernel::fmaF32: return "fma_f32";
        case ComputeKernel::dotI8: return "dot_i8";
        case ComputeKernel::dotBf16: return "dot_bf16";
    }
    return "?";
}

/// The unit of the kernel's rate: GFLOP/s for the two float kernels, GOP/s for the integer one.
[[nodiscard]] inline std::string_view computeKernelUnit(ComputeKernel k) {
    return k == ComputeKernel::dotI8 ? "GOP/s" : "GFLOP/s";
}

/// Lanes of 32-bit elements in the fma kernel: 16 with AVX-512F, 8 with AVX2 and FMA, 0 when neither is compiled in.
[[nodiscard]] constexpr int fmaLanes() {
#if !defined(_MSC_VER) && defined(__AVX512F__)
    return 16;
#elif !defined(_MSC_VER) && defined(__AVX2__) && defined(__FMA__)
    return 8;
#else
    return 0;
#endif
}

/// Lanes of the integer and bf16 kernels: always 16 (they exist only with AVX-512).
inline constexpr int kVnniLanes = 16;

/// Whether this build compiled the kernel's ISA in.
[[nodiscard]] constexpr bool computeKernelCompiled(ComputeKernel k) {
    switch (k) {
        case ComputeKernel::fmaF32: return fmaLanes() > 0;
        case ComputeKernel::dotI8:
#if !defined(_MSC_VER) && defined(__AVX512F__) && defined(__AVX512VNNI__)
            return true;
#else
            return false;
#endif
        case ComputeKernel::dotBf16:
#if !defined(_MSC_VER) && defined(__AVX512F__) && defined(__AVX512BF16__)
            return true;
#else
            return false;
#endif
    }
    return false;
}

/// The ISA the kernel uses in this build, or "none".
[[nodiscard]] constexpr std::string_view computeKernelIsa(ComputeKernel k) {
    if (!computeKernelCompiled(k)) return "none";
    switch (k) {
        case ComputeKernel::fmaF32:
#if !defined(_MSC_VER) && defined(__AVX512F__)
            return "avx512f";
#else
            return "avx2+fma";
#endif
        case ComputeKernel::dotI8: return "avx512vnni";
        case ComputeKernel::dotBf16: return "avx512bf16";
    }
    return "none";
}

/// Why a kernel is not compiled in, or empty when it is.
[[nodiscard]] inline std::string computeKernelReason(ComputeKernel k) {
    if (computeKernelCompiled(k)) return {};
    switch (k) {
        case ComputeKernel::fmaF32:
            return "not compiled in: needs -mavx2 -mfma or -mavx512f (build with -march=native)";
        case ComputeKernel::dotI8:
            return "not compiled in: needs -mavx512f -mavx512vnni (build with -march=native)";
        case ComputeKernel::dotBf16:
            return "not compiled in: needs -mavx512f -mavx512bf16 (build with -march=native)";
    }
    return "not compiled in";
}

/// Counted operations per instruction in one lane group: 2 x lanes for the FMA (multiply and add), 2 x 4 x lanes for
/// the VNNI dot (four byte products per lane), 2 x 2 x lanes for the BF16 dot (two bf16 products per lane).
[[nodiscard]] constexpr int computeOpsPerInstruction(ComputeKernel k) {
    switch (k) {
        case ComputeKernel::fmaF32: return 2 * fmaLanes();
        case ComputeKernel::dotI8: return 2 * 4 * kVnniLanes;
        case ComputeKernel::dotBf16: return 2 * 2 * kVnniLanes;
    }
    return 0;
}

/// ASSUMED peak operations per cycle per core for each kernel (the roof's per-core factor). Zen 4 (Ryzen 9 8945HX):
///   fma_f32 32: two 256-bit FMA pipes x 8 lanes x 2 flops. AVX-512 executes double-pumped, so one zmm FMA per cycle
///               gives the same 32 flops (16 lanes x 2). The AVX2 path also gives 32 (two ymm FMAs per cycle).
///   dot_i8 128: one zmm VNNI instruction per cycle, 16 lanes x 4 byte products x 2 ops.
///   dot_bf16 64: one zmm BF16 instruction per cycle, 16 lanes x 2 bf16 products x 2 flops.
/// These are assumptions about Zen 4, not measurements. The measured/theoretical ratio is the check on them.
[[nodiscard]] constexpr double computeOpsPerCycleCore(ComputeKernel k) {
    switch (k) {
        case ComputeKernel::fmaF32: return 32.0;
        case ComputeKernel::dotI8: return 128.0;
        case ComputeKernel::dotBf16: return 64.0;
    }
    return 0.0;
}

/// One row of the ISA table: what the kernel is, whether it is compiled in, and the roof assumptions.
struct ComputeIsa {
    std::string kernel;                 // "fma_f32", "dot_i8", "dot_bf16"
    std::string unit;                   // "GFLOP/s" or "GOP/s"
    std::string isa;                    // "avx512f", "avx2+fma", "avx512vnni", "avx512bf16" or "none"
    bool compiled = false;
    std::string reason;                 // why not compiled; empty when compiled
    int lanes = 0;                      // lanes of the 32-bit accumulators
    int opsPerInstruction = 0;          // counted ops per instruction, all lanes
    double opsPerCycleCore = 0.0;       // assumed roof factor, see computeOpsPerCycleCore
};

/// The three kernels' ISA rows, in kernel order.
[[nodiscard]] inline std::vector<ComputeIsa> computeIsaTable() {
    std::vector<ComputeIsa> table;
    for (const ComputeKernel k : {ComputeKernel::fmaF32, ComputeKernel::dotI8, ComputeKernel::dotBf16}) {
        ComputeIsa row;
        row.kernel = std::string(computeKernelName(k));
        row.unit = std::string(computeKernelUnit(k));
        row.isa = std::string(computeKernelIsa(k));
        row.compiled = computeKernelCompiled(k);
        row.reason = computeKernelReason(k);
        row.lanes = k == ComputeKernel::fmaF32 ? fmaLanes() : kVnniLanes;
        row.opsPerInstruction = computeOpsPerInstruction(k);
        row.opsPerCycleCore = computeOpsPerCycleCore(k);
        table.push_back(row);
    }
    return table;
}

/// Distinct physical cores under the given logical CPUs (SMT siblings count once). A CPU the facts do not place is
/// counted as its own core.
[[nodiscard]] inline int physicalCoresOf(const std::vector<int>& cpus, const MachineFacts& facts) {
    std::map<int, int> groupOf;
    if (facts.smtGroups.value)
        for (std::size_t g = 0; g < facts.smtGroups.value->size(); ++g)
            for (const int cpu : (*facts.smtGroups.value)[g]) groupOf.emplace(cpu, static_cast<int>(g));
    std::set<long long> cores;
    for (const int cpu : cpus) {
        const auto it = groupOf.find(cpu);
        cores.insert(it != groupOf.end() ? static_cast<long long>(it->second) : -static_cast<long long>(cpu) - 1);
    }
    return static_cast<int>(cores.size());
}

/// Run settings. Defaults are the production run; tests shorten the passes and the budget.
struct ComputeConfig {
    double budgetSeconds = 30.0;                 // the whole compute run, clocks included
    int repeats = 5;                             // timed repeats per point, before the planner reduces them
    std::vector<int> threadCounts = {1, 2, 4, 8, 12, 16};
    double passSeconds = 0.05;                   // target duration of one timed pass (calibrated per point)
};

/// Keep each calibration pass finite and bounded, including direct API requests.
[[nodiscard]] inline bool validComputePass(double seconds) noexcept {
    return std::isfinite(seconds) && seconds > 0.0 && seconds <= 60.0;
}

/// The plan for the compute run: how many repeats, and which thread counts per placement, fit the budget.
struct ComputePlan {
    int repeats = 1;
    std::vector<std::vector<int>> threadCounts;  // per placement, after clipping and thinning
    double estimateSeconds = 0.0;
    std::string note;
};

/// One kernel or clock request: which CPUs, how many iterations, and how many checked repeats.
struct ComputeRequest {
    ComputeKernel kernel = ComputeKernel::fmaF32;
    std::vector<int> cpus;                       // one thread per entry, pinned to that CPU
    std::uint64_t iterations = 0;                // loop iterations per pass; 0 = calibrate to passSeconds
    double passSeconds = 0.05;                   // target pass duration when calibrating
    int repeats = 5;                             // timed repeats, each checked
    bool corruptOutput = false;                  // TEST HOOK: damages one accumulator so the check must fail
    std::chrono::steady_clock::time_point stopAt = std::chrono::steady_clock::time_point::max();
};

/// Result of one kernel point (one kernel, placement and thread count, all its repeats).
struct ComputeMeasurement {
    std::uint64_t iterations = 0;                // per pass, after calibration
    int repeats = 0;                             // repeats that completed AND passed their check
    bool verified = false;                       // true only when every completed repeat passed its check
    std::string verifyFailure;                   // why the point is invalid or was cut short; empty when clean
    Fact<double> medianRate;                     // GFLOP/s or GOP/s; empty unless verified
    Fact<double> minRate;
    Fact<double> maxRate;
    double seconds = 0.0;                        // wall time of the point, calibration included
};

/// One kernel at one placement and thread count.
struct ComputePoint {
    std::string kernel;                          // "fma_f32", "dot_i8", "dot_bf16"
    std::string unit;                            // "GFLOP/s" or "GOP/s"
    std::string placement;
    int threads = 0;
    int physicalCores = 0;                       // distinct cores under the threads
    std::string skipReason;                      // why it was not attempted; empty when it was
    ComputeMeasurement measurement;
    Fact<double> theoreticalRate;                // physicalCores x effectiveGHz x opsPerCycleCore
    Fact<double> ratioToRoof;                    // measured median / theoretical
};

/// The effective clock of one placement and thread count: GHz from a chain of dependent adds.
struct ClockPoint {
    std::string placement;
    int threads = 0;
    std::uint64_t iterations = 0;                // per pass, after calibration (8 dependent adds each)
    int repeats = 0;
    bool verified = false;                       // every repeat's chain ended at its exact expected value
    std::string verifyFailure;
    Fact<double> medianGHz;                      // mean over the threads, median over repeats
    Fact<double> minGHz;
    Fact<double> maxGHz;
    double seconds = 0.0;
};

/// Everything `measureComputeRoofs` produced.
struct ComputeRoofs {
    bool supported = false;
    std::string unsupportedReason;               // set when supported is false
    double budgetSeconds = 0.0;
    double passSeconds = 0.0;                    // target pass duration used for calibration
    double totalSeconds = 0.0;                   // wall time of the whole run
    int plannedRepeats = 0;                      // repeats the planner chose (<= ComputeConfig::repeats)
    std::string planNote;                        // what the planner reduced, empty when nothing
    std::vector<ComputeIsa> isas;                // one row per kernel: ISA, compile state, roof assumption
    std::vector<ClockPoint> clocks;              // one per placement and thread count
    std::vector<ComputePoint> points;            // kernel x placement x thread count
};

namespace detail::compute {

using Clock = roofs::Clock;

inline constexpr int kAccumulators = 12;            // independent chains per thread (see the header comment)
inline constexpr std::uint64_t kI8MaxIters = std::uint64_t{1} << 28;    // 4 x iterations < 2^31 (int32 lanes)
inline constexpr std::uint64_t kBf16MaxIters = std::uint64_t{1} << 23;  // 2 x iterations <= 2^24 (exact in float)
inline constexpr std::uint64_t kClockMaxIters = std::uint64_t{1} << 40;
inline constexpr std::uint64_t kCalibrateStart = std::uint64_t{1} << 12;
inline constexpr double kCalibrateMinSeconds = 0.002;   // probes grow until a pass lasts at least this long
inline constexpr double kCalibrationSeconds = 0.01;     // planner's allowance for one point's calibration
inline constexpr int kChainAddsPerIteration = 8;

[[nodiscard]] inline std::uint64_t iterationCap(ComputeKernel k) {
    switch (k) {
        case ComputeKernel::fmaF32: return std::uint64_t{1} << 40;
        case ComputeKernel::dotI8: return kI8MaxIters;
        case ComputeKernel::dotBf16: return kBf16MaxIters;
    }
    return 0;
}

[[nodiscard]] inline int lanesOf(ComputeKernel k) { return k == ComputeKernel::fmaF32 ? fmaLanes() : kVnniLanes; }

/// The fma kernel: 12 chains of fmadd(acc, 1, 0). Writes lane values to out[k * L + j] and returns the timed seconds.
inline double fmaLoop(std::uint64_t iters, double* out) {
#if !defined(_MSC_VER) && defined(__AVX512F__)
    constexpr int L = 16;
    alignas(64) float init[kAccumulators * L];
    for (int i = 0; i < kAccumulators * L; ++i) init[i] = static_cast<float>(i + 1);
    __m512 acc[kAccumulators];
    for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm512_load_ps(init + L * k);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 zero = _mm512_setzero_ps();
    asm volatile("" : "+v"(one), "+v"(zero));   // opaque: the compiler cannot fold fma(x, 1, 0) away
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+v"(acc[k]));
    const Clock::time_point t0 = Clock::now();
    for (std::uint64_t i = 0; i < iters; ++i)
        for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm512_fmadd_ps(acc[k], one, zero);
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+v"(acc[k]));
    const Clock::time_point t1 = Clock::now();
    alignas(64) float tmp[L];
    for (int k = 0; k < kAccumulators; ++k) {
        _mm512_store_ps(tmp, acc[k]);
        for (int j = 0; j < L; ++j) out[k * L + j] = static_cast<double>(tmp[j]);
    }
    return roofs::secondsBetween(t0, t1);
#elif !defined(_MSC_VER) && defined(__AVX2__) && defined(__FMA__)
    constexpr int L = 8;
    alignas(32) float init[kAccumulators * L];
    for (int i = 0; i < kAccumulators * L; ++i) init[i] = static_cast<float>(i + 1);
    __m256 acc[kAccumulators];
    for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm256_load_ps(init + L * k);
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 zero = _mm256_setzero_ps();
    asm volatile("" : "+x"(one), "+x"(zero));
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+x"(acc[k]));
    const Clock::time_point t0 = Clock::now();
    for (std::uint64_t i = 0; i < iters; ++i)
        for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm256_fmadd_ps(acc[k], one, zero);
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+x"(acc[k]));
    const Clock::time_point t1 = Clock::now();
    alignas(32) float tmp[L];
    for (int k = 0; k < kAccumulators; ++k) {
        _mm256_store_ps(tmp, acc[k]);
        for (int j = 0; j < L; ++j) out[k * L + j] = static_cast<double>(tmp[j]);
    }
    return roofs::secondsBetween(t0, t1);
#else
    (void)iters;
    (void)out;
    return 0.0;
#endif
}

/// The VNNI kernel: 12 chains of dpbusd(acc, 1 unsigned, 1 signed). Each lane gains 4 per instruction.
inline double dotI8Loop(std::uint64_t iters, double* out) {
#if !defined(_MSC_VER) && defined(__AVX512F__) && defined(__AVX512VNNI__)
    constexpr int L = kVnniLanes;
    __m512i acc[kAccumulators];
    for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm512_setzero_si512();
    __m512i unsignedOnes = _mm512_set1_epi8(1);
    __m512i signedOnes = _mm512_set1_epi8(1);
    asm volatile("" : "+v"(unsignedOnes), "+v"(signedOnes));
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+v"(acc[k]));
    const Clock::time_point t0 = Clock::now();
    for (std::uint64_t i = 0; i < iters; ++i)
        for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm512_dpbusd_epi32(acc[k], unsignedOnes, signedOnes);
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+v"(acc[k]));
    const Clock::time_point t1 = Clock::now();
    alignas(64) std::int32_t tmp[L];
    for (int k = 0; k < kAccumulators; ++k) {
        _mm512_store_si512(tmp, acc[k]);
        for (int j = 0; j < L; ++j) out[k * L + j] = static_cast<double>(tmp[j]);
    }
    return roofs::secondsBetween(t0, t1);
#else
    (void)iters;
    (void)out;
    return 0.0;
#endif
}

/// The BF16 kernel: 12 chains of dpbf16(acc, ones, ones), where each 32-bit lane holds two bf16 1.0 values. Each lane
/// gains 2 per instruction.
inline double dotBf16Loop(std::uint64_t iters, double* out) {
#if !defined(_MSC_VER) && defined(__AVX512F__) && defined(__AVX512BF16__)
    constexpr int L = kVnniLanes;
    __m512i pairs = _mm512_set1_epi32(0x3F803F80);   // bf16 1.0 in both halves of each 32-bit lane
    asm volatile("" : "+v"(pairs));
    __m512bh ones;
    std::memcpy(&ones, &pairs, sizeof ones);
    __m512 acc[kAccumulators];
    for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm512_setzero_ps();
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+v"(acc[k]));
    const Clock::time_point t0 = Clock::now();
    for (std::uint64_t i = 0; i < iters; ++i)
        for (int k = 0; k < kAccumulators; ++k) acc[k] = _mm512_dpbf16_ps(acc[k], ones, ones);
    for (int k = 0; k < kAccumulators; ++k) asm volatile("" : "+v"(acc[k]));
    const Clock::time_point t1 = Clock::now();
    alignas(64) float tmp[L];
    for (int k = 0; k < kAccumulators; ++k) {
        _mm512_store_ps(tmp, acc[k]);
        for (int j = 0; j < L; ++j) out[k * L + j] = static_cast<double>(tmp[j]);
    }
    return roofs::secondsBetween(t0, t1);
#else
    (void)iters;
    (void)out;
    return 0.0;
#endif
}

/// Runs the kernel's loop once and writes its lanes to `out`. Returns the timed seconds (0 when not compiled in,
/// which callers never reach: the compiled check comes first).
inline double runKernelLoop(ComputeKernel k, std::uint64_t iters, double* out) {
    switch (k) {
        case ComputeKernel::fmaF32: return fmaLoop(iters, out);
        case ComputeKernel::dotI8: return dotI8Loop(iters, out);
        case ComputeKernel::dotBf16: return dotBf16Loop(iters, out);
    }
    return 0.0;
}

/// Exact expected lane values after `iters` iterations. Empty when every lane matches.
inline std::string checkLanes(ComputeKernel k, const double* out, std::uint64_t iters) {
    const int L = lanesOf(k);
    for (int a = 0; a < kAccumulators; ++a) {
        for (int j = 0; j < L; ++j) {
            double expected = 0.0;
            switch (k) {
                case ComputeKernel::fmaF32: expected = static_cast<double>(a * L + j + 1); break;
                case ComputeKernel::dotI8: expected = 4.0 * static_cast<double>(iters); break;
                case ComputeKernel::dotBf16: expected = 2.0 * static_cast<double>(iters); break;
            }
            const double got = out[a * L + j];
            if (got != expected)
                return "accumulator " + std::to_string(a) + " lane " + std::to_string(j) + " is " +
                       std::to_string(got) + ", expected " + std::to_string(expected);
        }
    }
    return {};
}

/// One pinned multi-thread pass: per-thread seconds and per-thread check results (empty string when correct).
struct PinnedPass {
    bool pinned = false;
    std::vector<double> seconds;
    std::vector<std::string> failures;
};

/// One pass of a kernel on `cpus` with `iters` iterations per thread. `corrupt` damages thread 0's first lane after
/// the timed region, so its check must fail (the test hook).
inline PinnedPass runKernelPass(ComputeKernel k, const std::vector<int>& cpus, std::uint64_t iters, bool corrupt) {
    PinnedPass p;
    p.seconds.assign(cpus.size(), 0.0);
    p.failures.assign(cpus.size(), std::string());
    p.pinned = roofs::runPinnedThreads(cpus, [](int) {}, [&](int t) {
        const auto ti = static_cast<std::size_t>(t);
        double out[kAccumulators * 16] = {};
        p.seconds[ti] = runKernelLoop(k, iters, out);
        if (corrupt && t == 0) out[0] += 1.0;
        p.failures[ti] = checkLanes(k, out, iters);
    });
    return p;
}

/// The effective-clock chain: 8 dependent adds per iteration, each through an empty asm so the compiler keeps them
/// one after another. Writes the final value to `finalX`.
inline double clockLoop(std::uint64_t iters, std::uint64_t& finalX) {
#if defined(_MSC_VER)
    (void)iters;
    finalX = 0;
    return 0.0;
#else
    std::uint64_t x = 0;
    asm volatile("" : "+r"(x));
    const Clock::time_point t0 = Clock::now();
    for (std::uint64_t i = 0; i < iters; ++i) {
        for (int j = 0; j < kChainAddsPerIteration; ++j) {
            x += 1;
            asm volatile("" : "+r"(x));
        }
    }
    const Clock::time_point t1 = Clock::now();
    finalX = x;
    return roofs::secondsBetween(t0, t1);
#endif
}

/// One pinned clock pass on `cpus`. Each thread's chain must end at exactly 8 x iters.
inline PinnedPass runClockPass(const std::vector<int>& cpus, std::uint64_t iters) {
    PinnedPass p;
    p.seconds.assign(cpus.size(), 0.0);
    p.failures.assign(cpus.size(), std::string());
    p.pinned = roofs::runPinnedThreads(cpus, [](int) {}, [&](int t) {
        const auto ti = static_cast<std::size_t>(t);
        std::uint64_t x = 0;
        p.seconds[ti] = clockLoop(iters, x);
        const std::uint64_t expected = static_cast<std::uint64_t>(kChainAddsPerIteration) * iters;
        if (x != expected)
            p.failures[ti] = "chain ended at " + std::to_string(x) + ", expected " + std::to_string(expected);
    });
    return p;
}

/// Picks the iterations per pass so one pass lasts about `passSeconds`, from timed probes of `pass`. Returns 0 and
/// sets `why` when a probe cannot pin or fails its check.
template <typename PassFn>
inline std::uint64_t calibrateIters(std::uint64_t cap, double passSeconds, PassFn&& pass, std::string& why) {
    std::uint64_t iters = std::min(kCalibrateStart, cap);
    double t = 0.0;
    const auto timeOf = [&](std::uint64_t n) {
        const PinnedPass p = pass(n);
        if (!p.pinned) {
            why = "could not pin a thread to its CPU (sched_setaffinity refused)";
            return false;
        }
        for (const std::string& f : p.failures)
            if (!f.empty()) {
                why = "calibration pass failed its check: " + f;
                return false;
            }
        t = *std::max_element(p.seconds.begin(), p.seconds.end());
        return true;
    };
    if (!timeOf(iters)) return 0;
    for (int round = 0; round < 8 && t < kCalibrateMinSeconds && iters < cap / 8; ++round) {
        iters *= 8;
        if (!timeOf(iters)) return 0;
    }
    const double est = static_cast<double>(iters) * passSeconds / std::max(t, 1e-6);
    const double clamped = std::min(std::max(1.0, est), static_cast<double>(cap));
    return static_cast<std::uint64_t>(clamped);
}

/// Min, median and max of per-repeat values into three facts; all three are empty with `why` when not verified.
inline void setRateFacts(Fact<double>& median, Fact<double>& lo, Fact<double>& hi, const std::vector<double>& values,
                         bool verified, const std::string& why) {
    if (!verified || values.empty()) {
        median = Fact<double>{std::nullopt, why};
        lo = Fact<double>{std::nullopt, why};
        hi = Fact<double>{std::nullopt, why};
        return;
    }
    median = Fact<double>{roofs::medianOf(values), {}};
    lo = Fact<double>{*std::min_element(values.begin(), values.end()), {}};
    hi = Fact<double>{*std::max_element(values.begin(), values.end()), {}};
}

/// Planner allowance for one point of `reps` repeats.
inline double pointCostSeconds(int reps, double passSeconds) {
    return static_cast<double>(reps) * passSeconds + kCalibrationSeconds;
}

}  // namespace detail::compute

/// Measures one kernel at one placement. Every repeat is checked; the first failing repeat makes the whole point
/// invalid and stops it. Repeats stop early at `req.stopAt` (a cut-short point keeps the repeats it completed).
[[nodiscard]] inline ComputeMeasurement measureComputeKernel(const ComputeRequest& req) {
    using namespace detail::compute;
    ComputeMeasurement m;
    const Clock::time_point start = Clock::now();
    const int threads = static_cast<int>(req.cpus.size());
    const std::uint64_t cap = iterationCap(req.kernel);
    std::string why;
    if (!validComputePass(req.passSeconds) ||
        (req.iterations == 0 && req.stopAt != Clock::time_point::max() &&
         (req.stopAt <= start || std::chrono::duration<double>(req.stopAt - start).count() < req.passSeconds))) {
        why = "invalid pass duration or insufficient remaining budget";
    } else if (!kRoofsSupported) {
        why = "not supported on this platform";
    } else if (!computeKernelCompiled(req.kernel)) {
        why = std::string(computeKernelReason(req.kernel));
    } else if (threads < 1 || req.repeats < 1) {
        why = "invalid request (threads or repeats out of range)";
    } else {
        std::uint64_t iters = req.iterations;
        if (iters == 0)
            iters = calibrateIters(cap, req.passSeconds,
                                   [&](std::uint64_t n) { return runKernelPass(req.kernel, req.cpus, n, false); }, why);
        if (iters == 0 && why.empty()) why = "calibration gave no iterations";
        if (iters > cap) why = "iterations exceed the exact range of this kernel (" + std::to_string(cap) + ")";
        if (why.empty()) {
            m.iterations = iters;
            const double opsPerPass = static_cast<double>(threads) * kAccumulators * static_cast<double>(iters) *
                                      static_cast<double>(computeOpsPerInstruction(req.kernel));
            std::vector<double> rates;
            for (int r = 0; r < req.repeats; ++r) {
                if (Clock::now() >= req.stopAt) {
                    if (r == 0) why = "time budget exhausted before the first repeat";
                    break;
                }
                const PinnedPass pass = runKernelPass(req.kernel, req.cpus, iters, req.corruptOutput && r == 0);
                if (!pass.pinned) {
                    why = "could not pin a thread to its CPU (sched_setaffinity refused)";
                    break;
                }
                bool failed = false;
                for (std::size_t t = 0; t < pass.failures.size() && !failed; ++t)
                    if (!pass.failures[t].empty()) {
                        why = "repeat " + std::to_string(r) + " thread " + std::to_string(t) +
                              " failed its check: " + pass.failures[t];
                        failed = true;
                    }
                if (failed) break;
                const double slowest = std::max(*std::max_element(pass.seconds.begin(), pass.seconds.end()), 1e-9);
                rates.push_back(opsPerPass / slowest / 1e9);
                ++m.repeats;
            }
            m.verified = why.empty() && m.repeats > 0;
            if (!m.verified && why.empty()) why = "no repeat completed";
            setRateFacts(m.medianRate, m.minRate, m.maxRate, rates, m.verified, why);
        }
    }
    if (!why.empty() && !m.verified) {
        m.verifyFailure = why;
        setRateFacts(m.medianRate, m.minRate, m.maxRate, {}, false, why);
    }
    m.seconds = detail::roofs::secondsBetween(start, Clock::now());
    return m;
}

/// The effective clock on `cpus`: GHz from the dependent-add chain, mean over the threads, median over `repeats`.
[[nodiscard]] inline ClockPoint measureEffectiveClock(const std::vector<int>& cpus, double passSeconds, int repeats,
                                                      std::chrono::steady_clock::time_point stopAt) {
    using namespace detail::compute;
    ClockPoint c;
    c.threads = static_cast<int>(cpus.size());
    const Clock::time_point start = Clock::now();
    std::string why;
    if (!validComputePass(passSeconds) ||
        (stopAt != Clock::time_point::max() &&
         (stopAt <= start || std::chrono::duration<double>(stopAt - start).count() < passSeconds))) {
        why = "invalid pass duration or insufficient remaining budget";
    } else if (!kRoofsSupported) {
        why = "not supported on this platform";
    } else if (cpus.empty() || repeats < 1) {
        why = "invalid request (threads or repeats out of range)";
    } else {
        const std::uint64_t iters = calibrateIters(
            kClockMaxIters, passSeconds, [&](std::uint64_t n) { return runClockPass(cpus, n); }, why);
        if (iters == 0 && why.empty()) why = "calibration gave no iterations";
        if (why.empty()) {
            c.iterations = iters;
            const double addsPerThread = static_cast<double>(kChainAddsPerIteration) * static_cast<double>(iters);
            std::vector<double> ghz;
            for (int r = 0; r < repeats; ++r) {
                if (Clock::now() >= stopAt) {
                    if (r == 0) why = "time budget exhausted before the first repeat";
                    break;
                }
                const PinnedPass pass = runClockPass(cpus, iters);
                if (!pass.pinned) {
                    why = "could not pin a thread to its CPU (sched_setaffinity refused)";
                    break;
                }
                bool failed = false;
                for (std::size_t t = 0; t < pass.failures.size() && !failed; ++t)
                    if (!pass.failures[t].empty()) {
                        why = "repeat " + std::to_string(r) + " thread " + std::to_string(t) +
                              " failed its check: " + pass.failures[t];
                        failed = true;
                    }
                if (failed) break;
                double sum = 0.0;
                for (const double s : pass.seconds) sum += addsPerThread / std::max(s, 1e-9) / 1e9;
                ghz.push_back(sum / static_cast<double>(pass.seconds.size()));
                ++c.repeats;
            }
            c.verified = why.empty() && c.repeats > 0;
            if (!c.verified && why.empty()) why = "no repeat completed";
            setRateFacts(c.medianGHz, c.minGHz, c.maxGHz, ghz, c.verified, why);
        }
    }
    if (!c.verified) {
        c.verifyFailure = why;
        setRateFacts(c.medianGHz, c.minGHz, c.maxGHz, {}, false, why);
    }
    c.seconds = detail::roofs::secondsBetween(start, Clock::now());
    return c;
}

/// Plans the compute run for `cfg.budgetSeconds`: first the repeats shrink in proportion, then the thread-count lists
/// are thinned (the same rules as the memory planner). Each thread count costs one clock point and three kernel points.
[[nodiscard]] inline ComputePlan planComputeRun(const std::vector<Placement>& placements, const ComputeConfig& cfg) {
    using namespace detail::compute;
    ComputePlan plan;
    if (!validComputePass(cfg.passSeconds) || !std::isfinite(cfg.budgetSeconds) || cfg.budgetSeconds < 0.0 ||
        cfg.repeats < 1) {
        plan.note = "invalid pass duration, budget or repeats";
        return plan;
    }
    plan.threadCounts.resize(placements.size());
    for (std::size_t i = 0; i < placements.size(); ++i)
        plan.threadCounts[i] = threadCountsFor(placements[i], cfg.threadCounts);
    const auto estimate = [&](int reps) {
        double total = 0.0;
        for (const auto& counts : plan.threadCounts)
            total += 4.0 * static_cast<double>(counts.size()) * pointCostSeconds(reps, cfg.passSeconds);
        return total;
    };
    plan.repeats = std::max(1, cfg.repeats);
    double est = estimate(plan.repeats);
    if (est > cfg.budgetSeconds) {
        const double shrunk = std::floor(plan.repeats * (cfg.budgetSeconds / est));
        const int reps = boundedRoofRepeats(shrunk, plan.repeats);
        plan.note += "repeats " + std::to_string(plan.repeats) + " -> " + std::to_string(reps) + "; ";
        plan.repeats = reps;
        est = estimate(plan.repeats);
    }
    while (est > cfg.budgetSeconds) {
        bool thinned = false;
        for (std::vector<int>& counts : plan.threadCounts) {
            if (counts.size() <= 2) continue;
            counts = detail::roofs::thinCounts(counts);
            thinned = true;
        }
        if (!thinned) break;
        plan.note += "thread-count lists thinned; ";
        est = estimate(plan.repeats);
    }
    plan.estimateSeconds = est;
    return plan;
}

/// Runs the whole compute-roof measurement: for each placement and thread count, the effective clock and then the
/// three kernels, all inside the budget. Never throws for expected conditions; the reasons are in the result.
[[nodiscard]] inline ComputeRoofs measureComputeRoofs(const MachineFacts& facts, const ComputeConfig& config = {}) {
    using namespace detail::compute;
    ComputeRoofs out;
    if (!validRoofBudget(config.budgetSeconds)) {
        out.unsupportedReason = "budget must be finite and between 0 and 86400 seconds";
        return out;
    }
    if (!validComputePass(config.passSeconds) || config.passSeconds > config.budgetSeconds || config.repeats < 1) {
        out.unsupportedReason = "invalid pass duration or repeats";
        return out;
    }
    out.budgetSeconds = config.budgetSeconds;
    out.passSeconds = config.passSeconds;
    const Clock::time_point start = Clock::now();
    const Clock::time_point deadline =
        start + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(config.budgetSeconds));
    out.isas = computeIsaTable();

    if (!kRoofsSupported) {
        out.unsupportedReason = "compute roofs need Linux (sched_setaffinity); this platform cannot pin threads";
        return out;
    }
    const std::vector<Placement> placements = planPlacements(facts);
    if (placements.empty()) {
        out.unsupportedReason = "topology unknown: " +
                                (facts.l3Groups.value ? facts.onePerCore.reason : facts.l3Groups.reason);
        return out;
    }
    out.supported = true;

    const ComputePlan plan = planComputeRun(placements, config);
    out.plannedRepeats = plan.repeats;
    out.planNote = plan.note;

    const std::array<ComputeKernel, 3> kernels{ComputeKernel::fmaF32, ComputeKernel::dotI8, ComputeKernel::dotBf16};
    for (std::size_t p = 0; p < placements.size(); ++p) {
        const Placement& pl = placements[p];
        for (const int threads : plan.threadCounts[p]) {
            const std::vector<int> cpus = cpusFor(pl, threads);
            const int cores = physicalCoresOf(cpus, facts);

            ClockPoint clock = measureEffectiveClock(cpus, config.passSeconds, plan.repeats, deadline);
            clock.placement = pl.name;
            const bool haveClock = clock.verified && clock.medianGHz.value.has_value();
            const double ghz = haveClock ? *clock.medianGHz.value : 0.0;
            out.clocks.push_back(clock);

            for (const ComputeKernel k : kernels) {
                ComputePoint pt;
                pt.kernel = std::string(computeKernelName(k));
                pt.unit = std::string(computeKernelUnit(k));
                pt.placement = pl.name;
                pt.threads = threads;
                pt.physicalCores = cores;
                if (!computeKernelCompiled(k)) {
                    pt.skipReason = computeKernelReason(k);
                } else if (Clock::now() >= deadline) {
                    pt.skipReason = "time budget exhausted";
                } else {
                    ComputeRequest req;
                    req.kernel = k;
                    req.cpus = cpus;
                    req.passSeconds = config.passSeconds;
                    req.repeats = plan.repeats;
                    req.stopAt = deadline;
                    pt.measurement = measureComputeKernel(req);
                    if (!haveClock) {
                        pt.theoreticalRate = Fact<double>{std::nullopt, "no verified effective clock for this point"};
                    } else {
                        const double roof = static_cast<double>(cores) * ghz * computeOpsPerCycleCore(k);
                        pt.theoreticalRate = Fact<double>{roof, {}};
                        if (pt.measurement.medianRate.value)
                            pt.ratioToRoof = Fact<double>{ratioToCeiling(*pt.measurement.medianRate.value, roof), {}};
                        else
                            pt.ratioToRoof = Fact<double>{std::nullopt, "the measured point is not verified"};
                    }
                }
                out.points.push_back(std::move(pt));
            }
        }
    }
    out.totalSeconds = detail::roofs::secondsBetween(start, Clock::now());
    return out;
}

namespace detail::compute {

inline void writeIsa(detail::JsonObject& o, const ComputeIsa& row) {
    o.value("kernel", std::string_view(row.kernel));
    o.value("unit", std::string_view(row.unit));
    o.value("isa", std::string_view(row.isa));
    o.value("compiled", row.compiled);
    o.value("reason", std::string_view(row.reason));
    o.value("lanes", row.lanes);
    o.value("opsPerInstruction", row.opsPerInstruction);
    o.value("opsPerCycleCore", row.opsPerCycleCore);
}

inline void writeClock(detail::JsonObject& o, const ClockPoint& c) {
    o.value("placement", std::string_view(c.placement));
    o.value("threads", c.threads);
    o.value("iterations", c.iterations);
    o.value("repeats", c.repeats);
    o.value("verified", c.verified);
    o.value("verifyFailure", std::string_view(c.verifyFailure));
    o.fact("medianGHz", c.medianGHz);
    o.fact("minGHz", c.minGHz);
    o.fact("maxGHz", c.maxGHz);
    o.value("seconds", c.seconds);
}

inline void writePoint(detail::JsonObject& o, const ComputePoint& p) {
    o.value("kernel", std::string_view(p.kernel));
    o.value("unit", std::string_view(p.unit));
    o.value("placement", std::string_view(p.placement));
    o.value("threads", p.threads);
    o.value("physicalCores", p.physicalCores);
    o.value("skipReason", std::string_view(p.skipReason));
    o.value("iterations", p.measurement.iterations);
    o.value("repeats", p.measurement.repeats);
    o.value("verified", p.measurement.verified);
    o.value("verifyFailure", std::string_view(p.measurement.verifyFailure));
    o.fact("medianRate", p.measurement.medianRate);
    o.fact("minRate", p.measurement.minRate);
    o.fact("maxRate", p.measurement.maxRate);
    o.value("seconds", p.measurement.seconds);
    o.fact("theoreticalRate", p.theoreticalRate);
    o.fact("ratioToRoof", p.ratioToRoof);
}

/// The fields of the `compute` object, written into `comp` (which must be open inside `out`).
inline void writeComputeFields(std::string& out, detail::JsonObject& comp, const ComputeRoofs& r) {
    comp.value("supported", r.supported);
    comp.value("unsupportedReason", std::string_view(r.unsupportedReason));
    comp.value("budgetSeconds", r.budgetSeconds);
    comp.value("passSeconds", r.passSeconds);
    comp.value("totalSeconds", r.totalSeconds);
    comp.value("plannedRepeats", r.plannedRepeats);
    comp.value("planNote", std::string_view(r.planNote));
    comp.key("isa");
    detail::jsonArray(out, r.isas.size(), [&](std::size_t i) {
        detail::JsonObject o(out);
        writeIsa(o, r.isas[i]);
    });
    comp.key("clocks");
    detail::jsonArray(out, r.clocks.size(), [&](std::size_t i) {
        detail::JsonObject o(out);
        writeClock(o, r.clocks[i]);
    });
    comp.key("points");
    detail::jsonArray(out, r.points.size(), [&](std::size_t i) {
        detail::JsonObject o(out);
        writePoint(o, r.points[i]);
    });
}

inline std::string ghzText(double v) {
    std::string s;
    detail::appendFixed(s, v, 2);
    return s;
}

}  // namespace detail::compute

/// Deterministic JSON of the compute roofs alone: `{"schema":"sub0llm-machine-1","roofs":{"compute":{...}}}`.
/// Each measured number is `{"value": <number or null>, "reason": "<why absent, empty when known>"}`.
[[nodiscard]] inline std::string toJson(const ComputeRoofs& r) {
    std::string out;
    {
        detail::JsonObject root(out);
        root.value("schema", std::string_view("sub0llm-machine-1"));
        detail::JsonObject roofs(root, "roofs");
        detail::JsonObject comp(roofs, "compute");
        detail::compute::writeComputeFields(out, comp, r);
    }
    return out;
}

/// The `sub0llm-machine --roofs` document for any selection of roof sets:
/// `{"schema":"sub0llm-machine-1","facts":{...},"roofs":{"memory":{...},"compute":{...}}}`. A null set is left out.
/// The facts object is exactly the one `toJson(facts)` writes, and the memory object is exactly the one
/// `toJson(facts, memory)` writes.
[[nodiscard]] inline std::string toRoofsJson(const MachineFacts& facts, const MemoryRoofs* memory,
                                             const ComputeRoofs* compute) {
    std::string out = "{\"schema\":\"sub0llm-machine-1\",\"facts\":";
    out += detail::roofs::factsObject(facts) + ",\"roofs\":";
    {
        detail::JsonObject roofsObj(out);
        if (memory != nullptr) {
            detail::JsonObject mem(roofsObj, "memory");
            detail::roofs::writeMemoryFields(out, mem, *memory);
        }
        if (compute != nullptr) {
            detail::JsonObject comp(roofsObj, "compute");
            detail::compute::writeComputeFields(out, comp, *compute);
        }
    }
    out += '}';
    return out;
}

/// A human-readable summary: rate by kernel and placement (rows) and thread count (columns), the effective clock by
/// placement, and the best verified point of each kernel with its roof and ratio. Invalid points print INVALID.
[[nodiscard]] inline std::string toSummary(const ComputeRoofs& r) {
    using namespace detail::roofs;
    std::string o = "sub0llm machine roofs (compute)\n";
    if (!r.supported) {
        o += "  unsupported: " + r.unsupportedReason + "\n";
        return o;
    }
    o += "  pass " + fixedText(r.passSeconds * 1000.0, 0) + " ms, budget " + fixedText(r.budgetSeconds, 0) +
         " s, used " + fixedText(r.totalSeconds, 1) + " s\n";
    o += "  repeats planned " + std::to_string(r.plannedRepeats);
    if (!r.planNote.empty()) o += " (" + r.planNote + ")";
    o += "\n";
    for (const ComputeIsa& row : r.isas) {
        o += "  " + padRight(row.kernel, 9) + padRight(row.isa, 12) + " ";
        o += row.compiled ? "compiled, " + std::to_string(row.lanes) + " lanes, roof " +
                                fixedText(row.opsPerCycleCore, 0) + " ops/cycle/core (assumed)\n"
                          : row.reason + "\n";
    }

    // Table: rows are kernel x placement in first-seen order, columns are the thread counts measured.
    std::vector<std::pair<std::string, std::string>> rows;   // (kernel, placement)
    std::vector<int> cols;
    for (const ComputePoint& p : r.points) {
        const std::pair<std::string, std::string> key{p.kernel, p.placement};
        if (std::find(rows.begin(), rows.end(), key) == rows.end()) rows.push_back(key);
        if (std::find(cols.begin(), cols.end(), p.threads) == cols.end()) cols.push_back(p.threads);
    }
    std::sort(cols.begin(), cols.end());
    o += "  rate (median) by kernel, placement and threads\n";
    o += "    " + padRight("kernel placement", 20);
    for (const int c : cols) o += padLeft(std::to_string(c) + " thr", 10);
    o += "\n";
    for (const auto& row : rows) {
        o += "    " + padRight(row.first + " " + row.second, 20);
        for (const int c : cols) {
            std::string cell = "-";
            for (const ComputePoint& p : r.points) {
                if (p.kernel != row.first || p.placement != row.second || p.threads != c) continue;
                if (p.measurement.medianRate.value) cell = fixedText(*p.measurement.medianRate.value, 1);
                else if (p.skipReason.empty()) cell = "INVALID";
            }
            o += padLeft(cell, 10);
        }
        o += "\n";
    }

    // Effective clock: one line, placements in first-seen order, each as threads:GHz.
    o += "  effective GHz (median, per thread):";
    std::vector<std::string> placementsSeen;
    for (const ClockPoint& c : r.clocks)
        if (std::find(placementsSeen.begin(), placementsSeen.end(), c.placement) == placementsSeen.end())
            placementsSeen.push_back(c.placement);
    for (const std::string& name : placementsSeen) {
        o += " " + name + " ";
        bool first = true;
        for (const ClockPoint& c : r.clocks) {
            if (c.placement != name) continue;
            if (!first) o += ",";
            first = false;
            o += std::to_string(c.threads) + ":";
            o += c.medianGHz.value ? detail::compute::ghzText(*c.medianGHz.value) : std::string("INVALID");
        }
    }
    o += "\n";

    // Best verified point per kernel, with its roof and its ratio.
    for (const ComputeKernel k : {ComputeKernel::fmaF32, ComputeKernel::dotI8, ComputeKernel::dotBf16}) {
        const std::string name(computeKernelName(k));
        const ComputePoint* best = nullptr;
        for (const ComputePoint& p : r.points) {
            if (p.kernel != name || !p.measurement.medianRate.value) continue;
            if (!best || *p.measurement.medianRate.value > *best->measurement.medianRate.value) best = &p;
        }
        o += "  best " + padRight(name, 9);
        if (!best) {
            o += " none verified\n";
            continue;
        }
        o += " " + fixedText(*best->measurement.medianRate.value, 1) + " " + best->unit + " (" + best->placement +
             ", " + std::to_string(best->threads) + " threads, " + std::to_string(best->physicalCores) + " cores)";
        if (best->theoreticalRate.value)
            o += ", roof " + fixedText(*best->theoreticalRate.value, 1) + ", measured/roof " +
                 (best->ratioToRoof.value ? fixedText(*best->ratioToRoof.value, 2) : std::string("-"));
        o += "\n";
    }
    return o;
}

}  // namespace sub0::measure
