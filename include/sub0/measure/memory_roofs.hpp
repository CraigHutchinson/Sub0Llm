// sub0/measure/memory_roofs.hpp -- measured memory roofs for `sub0llm machine --roofs mem` (issue #78,
// increment 2).
//
// WHY THIS EXISTS. Decode speed is memory bandwidth divided by the bytes read per token, so the memory roof
// is the number that turns a model's size into a predicted tokens-per-second. This header measures it:
//   * read, write, copy and triad bandwidth (STREAM conventions) for three placements of N threads: `pack`
//     (fill one L3 group first, one thread per core), `spread` (alternate between L3 groups, one thread per
//     core) and `smt` (every one-per-core CPU, then the SMT siblings);
//   * a working-set sweep, single thread and one thread per core, for the L1/L2/L3/DRAM plateaus;
//   * a dependent-load latency sweep over a random single-cycle permutation.
//
// HOW THEY ARE MEASURED.
//   * Every thread is pinned (sched_setaffinity on the calling thread) before it touches memory, and it
//     first-touches its own slice of each array after pinning, so the pages sit where that thread runs.
//     A barrier starts all threads together; the timed region is per thread and the slowest thread sets the time.
//   * Each array is at least max(1 GiB, 8 x the total L3), so a pass streams from DRAM and not from cache.
//   * Bandwidth = bytes moved / slowest thread's time, STREAM convention: read moves 1 array, write 1, copy 2
//     (each array counted once), triad 3. Stores are plain stores (no non-temporal stores), so a write also pays
//     the read-for-ownership that the STREAM count leaves out. GB means 1e9 bytes.
//   * Every repeat is CHECKED. Read: each thread's sum must equal its element count (the fill is 1.0f and every
//     block sum is exact in float). Write, copy and triad: sampled elements must hold the value the kernel
//     should have stored. A failed check makes that point's numbers absent (`value` empty, reason set).
//   * Time budget: the planner first reduces the repeats, then thins the thread-count list, until the estimated
//     run fits; the run also stops at its budget. Each point reports its own wall time.
//
// WHAT IT IS NOT. Descriptive ceilings for planning a decode, not an engine benchmark. Nothing here changes an
// engine default.
//
// PLATFORMS. Linux measures. Elsewhere `supported` is false and no timing is done; the header still compiles.
// The AVX-512 and AVX2 read paths are selected by the compiler's macros, so the measuring target must be built
// for the host (the tool and its tests use -march=native in CMake).

#pragma once

#include "sub0/measure/machine_facts.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace sub0::measure {

/// A bounded duration keeps clock conversion and deadline arithmetic representable.
[[nodiscard]] inline bool validRoofBudget(double seconds) noexcept {
    return std::isfinite(seconds) && seconds > 0.0 && seconds <= 86400.0;
}

[[nodiscard]] inline bool validRoofRate(double gbPerSec) noexcept {
    return std::isfinite(gbPerSec) && gbPerSec > 0.0 && gbPerSec <= 1e9;
}

/// Clamp in floating point before converting to avoid out-of-range conversions.
[[nodiscard]] inline int boundedRoofRepeats(double estimate, int maximum) noexcept {
    if (!(estimate >= 1.0) || maximum <= 1) return 1;
    if (estimate >= static_cast<double>(maximum)) return maximum;
    return static_cast<int>(estimate);
}

/// Whether this build can pin threads and time memory roofs. False off Linux: the functions still compile.
#if defined(__linux__)
inline constexpr bool kRoofsSupported = true;
#else
inline constexpr bool kRoofsSupported = false;
#endif

/// One placement of threads: a name and the CPUs in the order threads are added. `cpusFor` takes a prefix.
struct Placement {
    std::string name;          // "pack", "spread" or "smt"
    std::vector<int> cpus;     // logical CPUs; the first N of them run N threads
};

/// The placements for this machine, derived from the facts only (l3Groups, onePerCore, smtGroups):
///   pack   the one-per-core CPUs of the first L3 group, then the next group's, and so on;
///   spread one-per-core CPUs taken round-robin across the L3 groups (2 threads = one core on each group);
///   smt    every one-per-core CPU in ascending order, then the SMT siblings core by core.
/// Empty when the facts do not give the topology (the reason is in the facts).
[[nodiscard]] inline std::vector<Placement> planPlacements(const MachineFacts& facts) {
    std::vector<Placement> out;
    if (!facts.l3Groups.value || !facts.onePerCore.value || !facts.smtGroups.value) return out;
    const std::vector<std::vector<int>>& groups = *facts.l3Groups.value;
    const std::vector<int>& firstThreads = *facts.onePerCore.value;

    std::map<int, std::size_t> groupOf;
    for (std::size_t g = 0; g < groups.size(); ++g)
        for (const int cpu : groups[g]) groupOf.emplace(cpu, g);

    std::vector<std::vector<int>> perGroup(groups.size());
    for (const int cpu : firstThreads) {
        const auto it = groupOf.find(cpu);
        if (it != groupOf.end()) perGroup[it->second].push_back(cpu);
    }

    Placement pack{"pack", {}};
    for (const auto& g : perGroup) pack.cpus.insert(pack.cpus.end(), g.begin(), g.end());

    Placement spread{"spread", {}};
    std::size_t longest = 0;
    for (const auto& g : perGroup) longest = std::max(longest, g.size());
    for (std::size_t k = 0; k < longest; ++k)
        for (const auto& g : perGroup)
            if (k < g.size()) spread.cpus.push_back(g[k]);

    Placement smt{"smt", firstThreads};
    for (const auto& core : *facts.smtGroups.value)
        for (std::size_t i = 1; i < core.size(); ++i) smt.cpus.push_back(core[i]);

    out.push_back(std::move(pack));
    out.push_back(std::move(spread));
    out.push_back(std::move(smt));
    return out;
}

/// The thread counts worth measuring for `p`: each wanted count that the placement has CPUs for, in the
/// order given, and for the `smt` placement also its full CPU count.
[[nodiscard]] inline std::vector<int> threadCountsFor(const Placement& p,
                                                      const std::vector<int>& wanted = {1, 2, 4, 8, 12, 16}) {
    std::vector<int> out;
    const int total = static_cast<int>(p.cpus.size());
    for (const int w : wanted)
        if (w >= 1 && w <= total && std::find(out.begin(), out.end(), w) == out.end()) out.push_back(w);
    if (p.name == "smt" && total > 0 && std::find(out.begin(), out.end(), total) == out.end()) out.push_back(total);
    return out;
}

/// The first `threadCount` CPUs of the placement's ordering. Empty when the count is outside 1..size.
[[nodiscard]] inline std::vector<int> cpusFor(const Placement& p, int threadCount) {
    if (threadCount < 1 || static_cast<std::size_t>(threadCount) > p.cpus.size()) return {};
    return std::vector<int>(p.cpus.begin(), p.cpus.begin() + threadCount);
}

/// Run settings. Defaults are the production run; tests override the buffer size and the sweeps.
struct RoofConfig {
    double budgetSeconds = 30.0;          // the whole run, sweeps and latency included
    int repeats = 5;                      // timed repeats per point, before the planner reduces them
    std::vector<int> threadCounts = {1, 2, 4, 8, 12, 16};
    double assumedGbPerSec = 20.0;        // starting estimate of the fill-plus-pass rate; the planner re-learns it
    bool runSweep = true;                 // working-set read sweep
    bool runLatency = true;               // pointer-chase latency sweep
    std::size_t arrayBytesOverride = 0;   // 0 = max(1 GiB, 8 x total L3); tests set a small size
    std::size_t sweepMaxBytes = 1ull << 30;     // largest working set in the bandwidth sweep
    std::size_t latencyMaxBytes = 1ull << 30;   // largest working set in the latency sweep
};

/// The four kernels.
enum class RoofKernel { read, write, copy, triad };

/// The kernel's name as it appears in the output.
[[nodiscard]] inline std::string_view kernelName(RoofKernel k) {
    switch (k) {
        case RoofKernel::read: return "read";
        case RoofKernel::write: return "write";
        case RoofKernel::copy: return "copy";
        case RoofKernel::triad: return "triad";
    }
    return "?";
}

/// Arrays moved per element by one pass (STREAM convention): read 1, write 1, copy 2, triad 3.
[[nodiscard]] inline int arraysMoved(RoofKernel k) {
    switch (k) {
        case RoofKernel::read: return 1;
        case RoofKernel::write: return 1;
        case RoofKernel::copy: return 2;
        case RoofKernel::triad: return 3;
    }
    return 0;
}

/// Arrays the untimed fill writes once per point (the arrays the kernel uses).
[[nodiscard]] inline int arraysFilled(RoofKernel k) {
    switch (k) {
        case RoofKernel::read: return 1;
        case RoofKernel::write: return 1;
        case RoofKernel::copy: return 2;
        case RoofKernel::triad: return 3;
    }
    return 0;
}

/// Bytes a point streams for `reps` repeats: one fill, then `reps` timed passes. The fill happens once because the
/// sources do not change between repeats and every timed pass rewrites the output in full.
[[nodiscard]] inline double pointStreamBytes(RoofKernel k, int reps, double arrayBytes) {
    return (static_cast<double>(arraysFilled(k)) + static_cast<double>(reps) * arraysMoved(k)) * arrayBytes;
}

/// Three 64-byte-aligned float arrays. Allocated, NOT initialised: the pinned worker threads first-touch
/// their own slices, which is what places the pages on the right node.
class RoofArrays {
public:
    /// Allocates three arrays of at least `bytesPerArray` (rounded up to 4 KiB). `ok()` is false on failure.
    explicit RoofArrays(std::size_t bytesPerArray) {
        if (bytesPerArray > std::numeric_limits<std::size_t>::max() - 4095) return;
        const std::size_t rounded = std::max<std::size_t>(4096, (bytesPerArray + 4095) / 4096 * 4096);
        elements_ = rounded / sizeof(float);
        a_ = allocate(rounded);
        b_ = allocate(rounded);
        c_ = allocate(rounded);
    }
    /// True when all three arrays were allocated.
    [[nodiscard]] bool ok() const noexcept { return a_ && b_ && c_; }
    /// Floats per array.
    [[nodiscard]] std::size_t elements() const noexcept { return elements_; }
    /// Bytes per array.
    [[nodiscard]] std::size_t bytes() const noexcept { return elements_ * sizeof(float); }
    [[nodiscard]] float* a() const noexcept { return a_.get(); }
    [[nodiscard]] float* b() const noexcept { return b_.get(); }
    [[nodiscard]] float* c() const noexcept { return c_.get(); }

private:
    struct FreeFn {
        void operator()(float* p) const noexcept {
#if defined(_WIN32)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }
    };
    using Buffer = std::unique_ptr<float, FreeFn>;
    static Buffer allocate(std::size_t bytes) {
#if defined(_WIN32)
        return Buffer(static_cast<float*>(_aligned_malloc(bytes, 64)));
#else
        return Buffer(static_cast<float*>(std::aligned_alloc(64, bytes)));
#endif
    }

    std::size_t elements_ = 0;
    Buffer a_, b_, c_;
};

/// Result of one measured point (one kernel or sweep configuration, all its repeats).
struct Measurement {
    int repeats = 0;                     // repeats that completed AND passed their check
    bool verified = false;               // true only when every completed repeat passed its check
    std::string verifyFailure;           // why the point is invalid or was cut short; empty when clean
    Fact<double> medianGbPerSec;         // empty unless verified
    Fact<double> minGbPerSec;
    Fact<double> maxGbPerSec;
    double seconds = 0.0;                // wall time of the point, fills included
};

/// One kernel at one placement and thread count.
struct KernelPoint {
    std::string kernel;                  // "read", "write", "copy", "triad"
    std::string placement;
    int threads = 0;
    std::string skipReason;              // why it was not attempted; empty when it was
    Measurement measurement;
};

/// One point of the working-set bandwidth sweep: read only.
struct SweepPoint {
    std::string mode;                    // "single" (one thread) or "one-per-core"
    std::size_t bytes = 0;               // total working set
    int threads = 0;
    Measurement measurement;
};

/// One point of the latency sweep: ns per dependent load over a single-cycle permutation.
struct LatencyPoint {
    std::size_t bytes = 0;
    int repeats = 0;
    bool verified = false;
    std::string verifyFailure;
    Fact<double> medianNsPerLoad;
    Fact<double> minNsPerLoad;
    Fact<double> maxNsPerLoad;
    double seconds = 0.0;
};

/// Everything `measureMemoryRoofs` produced.
struct MemoryRoofs {
    bool supported = false;
    std::string unsupportedReason;       // set when supported is false
    std::size_t arrayBytes = 0;          // bytes per array in the placement kernels
    std::size_t l3TotalBytes = 0;        // summed over L3 instances; 0 when unknown
    double budgetSeconds = 0.0;
    double totalSeconds = 0.0;           // wall time of the whole run
    int plannedRepeats = 0;              // repeats the planner chose (<= RoofConfig::repeats)
    std::string planNote;                // what the planner reduced, empty when nothing
    std::vector<KernelPoint> kernels;
    std::vector<SweepPoint> sweep;
    std::vector<LatencyPoint> latency;
};

/// Kernel-only result of one timed region across the threads.
struct PassResult {
    bool pinned = true;                  // every thread was pinned
    std::vector<double> seconds;         // per thread, timed region
    std::vector<double> sums;            // per thread, the read sum of the last pass (read kernel only)
};

/// What `measureBandwidth` runs.
struct BandwidthRequest {
    RoofKernel kernel = RoofKernel::read;
    std::vector<int> cpus;               // one thread per entry, pinned to that CPU
    std::size_t elements = 0;            // floats per array: the working set of each array
    int passes = 1;                      // kernel passes inside one timed region
    int repeats = 5;                     // timed repeats, each checked
    bool fillEachRepeat = true;          // false: fill once, before the first repeat (the first touch)
    bool corruptOutput = false;          // TEST HOOK: damage one element so the check must fail
    std::chrono::steady_clock::time_point stopAt = std::chrono::steady_clock::time_point::max();
};

/// Working-set sizes 16 KiB, 64 KiB, ... up to `maxBytes`, in factors of 4.
[[nodiscard]] inline std::vector<std::size_t> sweepSizes(std::size_t maxBytes) {
    std::vector<std::size_t> out;
    for (std::size_t s = 16u << 10; s <= maxBytes;) {
        out.push_back(s);
        if (s > maxBytes / 4) break;
        s *= 4;
    }
    return out;
}

/// Builds a single-cycle permutation of 0..n-1 with Sattolo's algorithm and a fixed-seed generator. Following
/// p[i] from any start visits every index once before returning, so it is a valid dependent-load chain.
[[nodiscard]] inline std::vector<std::uint64_t> buildSingleCyclePermutation(std::size_t n,
                                                                             std::uint64_t seed = 0x5ab1e5eed1234567ull) {
    std::vector<std::uint64_t> p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    std::uint64_t state = seed;
    // Sattolo: swap p[i] with a strictly earlier element. The result is one cycle of length n.
    for (std::size_t i = n; i-- > 1;) {
        state += 0x9E3779B97F4A7C15ull;   // splitmix64
        std::uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        z ^= z >> 31;
        const auto j = static_cast<std::size_t>(z % i);
        std::swap(p[i], p[j]);
    }
    return p;
}

/// True when `p` is a bijection of 0..size-1.
[[nodiscard]] inline bool isPermutation(const std::vector<std::uint64_t>& p) {
    std::vector<bool> seen(p.size(), false);
    for (const std::uint64_t v : p) {
        if (v >= p.size() || seen[static_cast<std::size_t>(v)]) return false;
        seen[static_cast<std::size_t>(v)] = true;
    }
    return true;
}

/// True when the walk from index 0 returns to 0 only after exactly size steps, i.e. `p` is one cycle.
[[nodiscard]] inline bool isSingleCycle(const std::vector<std::uint64_t>& p) {
    if (p.empty()) return false;
    std::uint64_t idx = 0;
    for (std::size_t step = 1; step <= p.size(); ++step) {
        if (idx >= p.size()) return false;
        idx = p[static_cast<std::size_t>(idx)];
        if (idx == 0) return step == p.size();
    }
    return false;
}

namespace detail::roofs {

using Clock = std::chrono::steady_clock;

/// Patterns that make copy and triad results checkable element by element.
inline float patternB(std::size_t i) { return static_cast<float>((i + 1) & 0x3FF); }
inline float patternC(std::size_t i) { return static_cast<float>((i * 7 + 3) & 0x3FF); }
inline constexpr float kWriteValue = 2.0f;
inline constexpr float kTriadScalar = 3.0f;
inline constexpr std::size_t kSampleStride = 4099;   // prime, so samples do not line up with pages
inline constexpr std::size_t kReadBlock = 8192;      // floats per read block: its float partials stay exact

/// Pins the calling thread to `cpu`. False when the CPU does not exist or the kernel refuses.
inline bool pinCurrentThread(int cpu) {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;
#endif
}

inline double secondsBetween(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

/// Starts one thread per CPU in `cpus`. Each thread pins itself, runs `setup(t)` when pinned, waits at a barrier
/// that every thread reaches (pinned or not, so the barrier cannot hang), then runs `body(t)` when pinned. Returns
/// true when every thread was pinned. Shared by the memory and compute roofs.
template <typename Setup, typename Body, typename Launch>
inline bool runPinnedThreadsWithLauncher(const std::vector<int>& cpus, Setup&& setup, Body&& body, Launch&& launch) {
    if (cpus.empty() || cpus.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
    const int threads = static_cast<int>(cpus.size());
    std::vector<char> pinned(cpus.size(), 0);
    std::barrier<> sync(threads);
    std::mutex gateMutex;
    std::condition_variable gate;
    bool started = false;
    bool cancelled = false;
    std::atomic<bool> failed{false};
    const auto work = [&](int t) {
        {
            std::unique_lock lock(gateMutex);
            gate.wait(lock, [&] { return started; });
            if (cancelled) return;
        }
        const bool ok = pinCurrentThread(cpus[static_cast<std::size_t>(t)]);
        pinned[static_cast<std::size_t>(t)] = ok ? 1 : 0;
        try { if (ok) setup(t); } catch (...) { failed = true; }
        sync.arrive_and_wait();
        try { if (ok && !failed) body(t); } catch (...) { failed = true; }
    };
    std::vector<std::thread> workers;
    workers.reserve(cpus.size());
    try {
        for (int t = 0; t < threads; ++t) workers.push_back(launch(work, t));
    } catch (...) {
        { std::lock_guard lock(gateMutex); cancelled = true; started = true; }
        gate.notify_all();
        for (std::thread& w : workers) w.join();
        return false;
    }
    { std::lock_guard lock(gateMutex); started = true; }
    gate.notify_all();
    for (std::thread& w : workers) w.join();
    return !failed && std::all_of(pinned.begin(), pinned.end(), [](char v) { return v != 0; });
}

template <typename Setup, typename Body>
inline bool runPinnedThreads(const std::vector<int>& cpus, Setup&& setup, Body&& body) {
    return runPinnedThreadsWithLauncher(cpus, setup, body,
                                       [](const auto& work, int t) { return std::thread(work, t); });
}

/// Start of thread t's slice of n elements. Boundaries are multiples of 16 floats (64 bytes).
inline std::size_t sliceBegin(std::size_t n, int threads, int t) {
    const std::size_t v = n * static_cast<std::size_t>(t) / static_cast<std::size_t>(threads);
    return v & ~std::size_t{15};
}

/// Sum of one block of at most kReadBlock floats. The partial sums are exact (every lane holds at most
/// kReadBlock/16 ones), so the result is exact for the 1.0f fill.
inline float readBlock(const float* p, std::size_t m) {
    std::size_t i = 0;
#if defined(__AVX512F__)
    __m512 acc[8];
    for (__m512& v : acc) v = _mm512_setzero_ps();
    for (; i + 128 <= m; i += 128)
        for (int k = 0; k < 8; ++k) acc[k] = _mm512_add_ps(acc[k], _mm512_loadu_ps(p + i + 16 * k));
    __m512 s = acc[0];
    for (int k = 1; k < 8; ++k) s = _mm512_add_ps(s, acc[k]);
    float r = _mm512_reduce_add_ps(s);
#elif defined(__AVX2__)
    __m256 acc[8];
    for (__m256& v : acc) v = _mm256_setzero_ps();
    for (; i + 64 <= m; i += 64)
        for (int k = 0; k < 8; ++k) acc[k] = _mm256_add_ps(acc[k], _mm256_loadu_ps(p + i + 8 * k));
    __m256 s = acc[0];
    for (int k = 1; k < 8; ++k) s = _mm256_add_ps(s, acc[k]);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, s);
    float r = 0.0f;
    for (const float v : lanes) r += v;
#else
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (; i + 8 <= m; i += 8)
        for (int k = 0; k < 8; ++k) acc[k] += p[i + k];
    float r = 0.0f;
    for (const float v : acc) r += v;
#endif
    for (; i < m; ++i) r += p[i];
    return r;
}

/// Sum of n floats, exact for the 1.0f fill, accumulated in double across blocks.
inline double readRange(const float* p, std::size_t n) {
    double total = 0.0;
    for (std::size_t i = 0; i < n; i += kReadBlock) total += readBlock(p + i, std::min(kReadBlock, n - i));
    return total;
}

inline void writeRange(float* __restrict a, float v, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) a[i] = v;
}

inline void copyRange(float* __restrict a, const float* __restrict b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) a[i] = b[i];
}

inline void triadRange(float* __restrict a, const float* __restrict b, const float* __restrict c, float s,
                       std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
}

/// Untimed first touch and fill of one thread's slice, done after pinning.
inline void fillSlice(RoofKernel k, float* a, float* b, float* c, std::size_t lo, std::size_t hi) {
    switch (k) {
        case RoofKernel::read:
            for (std::size_t i = lo; i < hi; ++i) a[i] = 1.0f;
            break;
        case RoofKernel::write:
            for (std::size_t i = lo; i < hi; ++i) a[i] = 0.0f;
            break;
        case RoofKernel::copy:
            for (std::size_t i = lo; i < hi; ++i) {
                b[i] = patternB(i);
                a[i] = 0.0f;
            }
            break;
        case RoofKernel::triad:
            for (std::size_t i = lo; i < hi; ++i) {
                b[i] = patternB(i);
                c[i] = patternC(i);
                a[i] = 0.0f;
            }
            break;
    }
}

/// The timed work of one thread's slice, once. Returns the read sum (0 for the other kernels).
inline double runSlice(RoofKernel k, float* a, const float* b, const float* c, std::size_t lo, std::size_t hi) {
    const std::size_t len = hi - lo;
    switch (k) {
        case RoofKernel::read: return readRange(a + lo, len);
        case RoofKernel::write: writeRange(a + lo, kWriteValue, len); break;
        case RoofKernel::copy: copyRange(a + lo, b + lo, len); break;
        case RoofKernel::triad: triadRange(a + lo, b + lo, c + lo, kTriadScalar, len); break;
    }
    return 0.0;
}

/// Runs `passes` passes of `k` over `elements` floats with one thread per CPU in `cpus`. Each thread is pinned,
/// fills its slice (when `fill`), waits at a barrier, then times its own passes.
inline PassResult runPass(RoofKernel k, RoofArrays& arr, std::size_t n, const std::vector<int>& cpus, int passes,
                          bool fill, bool corrupt) {
    const int threads = static_cast<int>(cpus.size());
    PassResult res;
    res.seconds.assign(static_cast<std::size_t>(threads), 0.0);
    res.sums.assign(static_cast<std::size_t>(threads), 0.0);
    float* a = arr.a();
    float* b = arr.b();
    float* c = arr.c();

    const auto setup = [&](int t) {
        if (!fill) return;
        fillSlice(k, a, b, c, sliceBegin(n, threads, t), sliceBegin(n, threads, t + 1));
        if (corrupt && t == 0 && k == RoofKernel::read) a[0] = 2.0f;   // the read sum must now disagree
    };
    const auto body = [&](int t) {
        const std::size_t lo = sliceBegin(n, threads, t);
        const std::size_t hi = sliceBegin(n, threads, t + 1);
        const Clock::time_point t0 = Clock::now();
        double sum = 0.0;
        for (int p = 0; p < passes; ++p) sum = runSlice(k, a, b, c, lo, hi);
        const Clock::time_point t1 = Clock::now();
        res.seconds[static_cast<std::size_t>(t)] = secondsBetween(t0, t1);
        res.sums[static_cast<std::size_t>(t)] = sum;
        if (corrupt && t == 0 && k != RoofKernel::read) a[0] = -1.0f;   // damage the output
    };
    res.pinned = runPinnedThreads(cpus, setup, body);
    return res;
}

/// Checks one element against what the kernel should have stored. Empty when correct.
inline std::string checkElement(RoofKernel k, const float* a, std::size_t i) {
    float expected = 0.0f;
    switch (k) {
        case RoofKernel::read: return {};
        case RoofKernel::write: expected = kWriteValue; break;
        case RoofKernel::copy: expected = patternB(i); break;
        case RoofKernel::triad: expected = patternB(i) + kTriadScalar * patternC(i); break;
    }
    if (a[i] == expected) return {};
    return "element " + std::to_string(i) + " is " + std::to_string(a[i]) + ", expected " + std::to_string(expected);
}

/// Checks one pass: read sums per thread, and sampled elements (every kSampleStride plus each slice's last
/// element) for the other kernels. Empty when the pass is correct.
inline std::string verifyPass(RoofKernel k, const RoofArrays& arr, std::size_t n, int threads, const PassResult& pass) {
    for (int t = 0; t < threads; ++t) {
        const std::size_t lo = sliceBegin(n, threads, t);
        const std::size_t hi = sliceBegin(n, threads, t + 1);
        const auto ti = static_cast<std::size_t>(t);
        if (k == RoofKernel::read) {
            const double expected = static_cast<double>(hi - lo);
            if (pass.sums[ti] != expected)
                return "thread " + std::to_string(t) + " read sum " + std::to_string(pass.sums[ti]) + ", expected " +
                       std::to_string(expected);
            continue;
        }
        for (std::size_t i = lo; i < hi; i += kSampleStride)
            if (const std::string why = checkElement(k, arr.a(), i); !why.empty()) return why;
        if (hi > lo)
            if (const std::string why = checkElement(k, arr.a(), hi - 1); !why.empty()) return why;
    }
    return {};
}

inline double medianOf(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t m = v.size();
    if (m == 0) return 0.0;
    return m % 2 == 1 ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}

/// Fills a Measurement's statistics from the per-repeat values (empty when none passed).
inline void setStats(Measurement& m, const std::vector<double>& values) {
    if (!m.verified || values.empty()) return;
    m.medianGbPerSec = Fact<double>{medianOf(values), {}};
    m.minGbPerSec = Fact<double>{*std::min_element(values.begin(), values.end()), {}};
    m.maxGbPerSec = Fact<double>{*std::max_element(values.begin(), values.end()), {}};
}

}  // namespace detail::roofs

/// Measures one kernel at one thread placement. Every repeat is checked; the first failing repeat makes the whole
/// point invalid and stops it. Repeats stop early at `req.stopAt` (reported as a cut-short point).
[[nodiscard]] inline Measurement measureBandwidth(RoofArrays& arr, const BandwidthRequest& req) {
    using namespace detail::roofs;
    Measurement m;
    const Clock::time_point start = Clock::now();
    const int threads = static_cast<int>(req.cpus.size());
    if (!kRoofsSupported) {
        m.verifyFailure = "not supported on this platform";
    } else if (!arr.ok() || threads < 1 || req.passes < 1 || req.repeats < 1 || req.elements == 0 ||
               req.elements > arr.elements() || req.elements % 16 != 0) {
        m.verifyFailure = "invalid request (threads, passes, repeats or elements out of range)";
    } else {
        const std::size_t n = req.elements;
        const double bytesPerRepeat =
            static_cast<double>(req.passes) * arraysMoved(req.kernel) * static_cast<double>(n) * sizeof(float);
        std::vector<double> gbps;
        for (int r = 0; r < req.repeats; ++r) {
            if (Clock::now() >= req.stopAt) {
                if (m.verifyFailure.empty() && r == 0) m.verifyFailure = "time budget exhausted before the first repeat";
                break;
            }
            const bool fill = req.fillEachRepeat || r == 0;
            const bool corrupt = req.corruptOutput && r == 0;
            const PassResult pass = runPass(req.kernel, arr, n, req.cpus, req.passes, fill, corrupt);
            if (!pass.pinned) {
                m.verifyFailure = "could not pin a thread to its CPU (sched_setaffinity refused)";
                break;
            }
            if (const std::string why = verifyPass(req.kernel, arr, n, threads, pass); !why.empty()) {
                m.verifyFailure = "repeat " + std::to_string(r) + " failed its check: " + why;
                break;
            }
            const double slowest = std::max(*std::max_element(pass.seconds.begin(), pass.seconds.end()), 1e-9);
            gbps.push_back(bytesPerRepeat / slowest / 1e9);
            ++m.repeats;
        }
        m.verified = m.verifyFailure.empty() && m.repeats > 0;
        setStats(m, gbps);
        if (!m.verified && m.verifyFailure.empty()) m.verifyFailure = "no repeat completed";
    }
    m.seconds = secondsBetween(start, Clock::now());
    if (!m.verified) {
        const std::string why = m.verifyFailure;
        m.medianGbPerSec = Fact<double>{std::nullopt, why};
        m.minGbPerSec = Fact<double>{std::nullopt, why};
        m.maxGbPerSec = Fact<double>{std::nullopt, why};
    }
    return m;
}

/// Dependent-load latency at one working-set size, timed on one thread pinned to `cpu`. The chain is a
/// single-cycle permutation: a bijection always, and a full one-cycle walk up to 4 M entries (above that the
/// cycle property comes from the Sattolo construction, which the unit tests check at small sizes).
[[nodiscard]] inline LatencyPoint measureLatency(std::size_t bytes, int cpu, int repeats,
                                                 std::chrono::steady_clock::time_point stopAt) {
    using namespace detail::roofs;
    LatencyPoint lp;
    lp.bytes = bytes;
    const Clock::time_point start = Clock::now();
    const std::size_t n = bytes / sizeof(std::uint64_t);
    if (!kRoofsSupported) {
        lp.verifyFailure = "not supported on this platform";
    } else if (n < 2) {
        lp.verifyFailure = "working set too small for a chain";
    } else if (Clock::now() >= stopAt) {
        lp.verifyFailure = "time budget exhausted before this size";
    } else {
        constexpr std::uint64_t kLoads = 1u << 20;
        const std::vector<std::uint64_t> next = buildSingleCyclePermutation(n);
        bool ok = isPermutation(next);
        if (!ok) lp.verifyFailure = "the permutation is not a bijection";
        if (ok && n <= (std::size_t{1} << 22) && !isSingleCycle(next)) {
            ok = false;
            lp.verifyFailure = "the permutation is not one cycle";
        }
        if (ok) {
            std::vector<double> nsPerLoad;
            bool pinnedOk = false;
            std::uint64_t finalIdx = 0;
            std::thread worker([&] {
                pinnedOk = pinCurrentThread(cpu);
                if (!pinnedOk) return;
                for (int r = 0; r < repeats && Clock::now() < stopAt; ++r) {
                    std::uint64_t idx = 0;
                    const Clock::time_point t0 = Clock::now();
                    for (std::uint64_t k = 0; k < kLoads; ++k) idx = next[static_cast<std::size_t>(idx)];
                    const Clock::time_point t1 = Clock::now();
                    finalIdx = idx;
                    nsPerLoad.push_back(secondsBetween(t0, t1) * 1e9 / static_cast<double>(kLoads));
                }
            });
            worker.join();
            if (!pinnedOk) {
                lp.verifyFailure = "could not pin the chase thread to its CPU";
            } else if (finalIdx >= n) {
                lp.verifyFailure = "chase left the permutation";
            } else if (nsPerLoad.empty()) {
                lp.verifyFailure = "time budget exhausted before the first repeat";
            } else {
                lp.verified = true;
                lp.repeats = static_cast<int>(nsPerLoad.size());
                lp.medianNsPerLoad = Fact<double>{medianOf(nsPerLoad), {}};
                lp.minNsPerLoad = Fact<double>{*std::min_element(nsPerLoad.begin(), nsPerLoad.end()), {}};
                lp.maxNsPerLoad = Fact<double>{*std::max_element(nsPerLoad.begin(), nsPerLoad.end()), {}};
            }
        }
    }
    lp.seconds = secondsBetween(start, Clock::now());
    return lp;
}

/// The plan for the placement kernels: how many repeats, and which thread counts, fit the time left.
struct RoofPlan {
    int repeats = 1;
    std::vector<std::vector<int>> threadCounts;   // per placement, after clipping and thinning
    double estimateSeconds = 0.0;
    std::string note;
};

namespace detail::roofs {

/// Drops every second interior thread count, keeping the first and the last. Each call on a list of three or more
/// shortens it, so repeated thinning always ends at two entries.
inline std::vector<int> thinCounts(const std::vector<int>& c) {
    std::vector<int> out;
    for (std::size_t i = 0; i < c.size(); ++i)
        if (i % 2 == 0 || i + 1 == c.size()) out.push_back(c[i]);
    return out;
}

}  // namespace detail::roofs

/// Plans the placement kernels for `kernelBudgetSeconds`, using `assumedGbPerSec` until measured. First the repeats
/// shrink in proportion to the budget; if one repeat still does not fit, the thread-count lists are thinned.
[[nodiscard]] inline RoofPlan planRoofRun(const std::vector<Placement>& placements, const RoofConfig& cfg,
                                          std::size_t arrayBytes, double kernelBudgetSeconds) {
    using namespace detail::roofs;
    RoofPlan plan;
    if (!validRoofRate(cfg.assumedGbPerSec) || !std::isfinite(kernelBudgetSeconds) || kernelBudgetSeconds < 0.0) {
        plan.note = "invalid rate or budget";
        return plan;
    }
    plan.threadCounts.resize(placements.size());
    for (std::size_t i = 0; i < placements.size(); ++i)
        plan.threadCounts[i] = threadCountsFor(placements[i], cfg.threadCounts);
    const double secPerByte = 1.0 / (std::max(cfg.assumedGbPerSec, 1e-3) * 1e9);
    const auto estimate = [&](int reps) {
        double total = 0.0;
        for (const auto& counts : plan.threadCounts)
            for (const RoofKernel k : {RoofKernel::read, RoofKernel::write, RoofKernel::copy, RoofKernel::triad})
                total += static_cast<double>(counts.size()) * pointStreamBytes(k, reps, static_cast<double>(arrayBytes)) *
                         secPerByte;
        return total;
    };
    plan.repeats = std::max(1, cfg.repeats);
    double est = estimate(plan.repeats);
    if (est > kernelBudgetSeconds) {
        const int reps = boundedRoofRepeats(plan.repeats * (kernelBudgetSeconds / est), plan.repeats);
        plan.note += "repeats " + std::to_string(plan.repeats) + " -> " + std::to_string(reps) + "; ";
        plan.repeats = reps;
        est = estimate(plan.repeats);
    }
    while (est > kernelBudgetSeconds) {
        bool thinned = false;
        for (std::vector<int>& counts : plan.threadCounts) {
            if (counts.size() <= 2) continue;
            counts = thinCounts(counts);
            thinned = true;
        }
        if (!thinned) break;
        plan.note += "thread-count lists thinned; ";
        est = estimate(plan.repeats);
    }
    plan.estimateSeconds = est;
    return plan;
}

/// Runs the whole memory-roof measurement: the working-set sweep and latency first (capped at 30% of the budget),
/// then the placement kernels with the time that remains. Never throws for expected conditions; the reasons are
/// in the result.
[[nodiscard]] inline MemoryRoofs measureMemoryRoofs(const MachineFacts& facts, const RoofConfig& config = {}) {
    using namespace detail::roofs;
    MemoryRoofs out;
    out.budgetSeconds = config.budgetSeconds;
    if (!validRoofBudget(config.budgetSeconds)) {
        out.unsupportedReason = "budget must be finite and between 0 and 86400 seconds";
        return out;
    }
    if (!validRoofRate(config.assumedGbPerSec) || config.repeats < 1) {
        out.unsupportedReason = "invalid rate or repeats";
        return out;
    }
    const Clock::time_point start = Clock::now();
    const auto deadline = start + std::chrono::duration_cast<Clock::duration>(
                                      std::chrono::duration<double>(config.budgetSeconds));
    const auto sideEnd = start + std::chrono::duration_cast<Clock::duration>(
                                     std::chrono::duration<double>(0.3 * config.budgetSeconds));

    if (!kRoofsSupported) {
        out.unsupportedReason = "memory roofs need Linux (sched_setaffinity); this platform cannot pin threads";
        return out;
    }
    const std::vector<Placement> placements = planPlacements(facts);
    if (placements.empty()) {
        out.unsupportedReason = "topology unknown: " +
                                (facts.l3Groups.value ? facts.onePerCore.reason : facts.l3Groups.reason);
        return out;
    }
    const Placement* pack = nullptr;
    for (const Placement& p : placements)
        if (p.name == "pack") pack = &p;
    if (pack == nullptr || pack->cpus.empty()) {
        out.unsupportedReason = "no one-per-core CPUs to place threads on";
        return out;
    }

    if (facts.caches.size() == 4 && facts.caches[3].sizeBytes.value && facts.caches[3].instances.value) {
        const auto size = static_cast<std::size_t>(*facts.caches[3].sizeBytes.value);
        const auto count = static_cast<std::size_t>(*facts.caches[3].instances.value);
        if (count != 0 && size > std::numeric_limits<std::size_t>::max() / count / 8) {
            out.unsupportedReason = "reported cache size is too large";
            return out;
        }
        out.l3TotalBytes = size * count;
    }
    const std::size_t wanted = config.arrayBytesOverride != 0
                                   ? config.arrayBytesOverride
                                   : std::max<std::size_t>(std::size_t{1} << 30, 8 * out.l3TotalBytes);
    RoofArrays arrays(wanted);
    if (!arrays.ok()) {
        out.unsupportedReason = "could not allocate three arrays of " + std::to_string(wanted) + " bytes";
        return out;
    }
    out.supported = true;
    out.arrayBytes = arrays.bytes();

    // ---- working-set sweep and latency: first, inside 30% of the budget ----
    if (config.runSweep) {
        const std::size_t maxBytes = std::min(config.sweepMaxBytes, arrays.bytes());
        const std::vector<int> single{pack->cpus.front()};
        for (const std::size_t bytes : sweepSizes(maxBytes)) {
            const int passes = static_cast<int>(std::max<std::size_t>(1, (std::size_t{256} << 20) / bytes));
            for (const bool allCores : {false, true}) {
                SweepPoint sp;
                sp.mode = allCores ? "one-per-core" : "single";
                sp.bytes = bytes;
                sp.threads = allCores ? static_cast<int>(pack->cpus.size()) : 1;
                BandwidthRequest req;
                req.kernel = RoofKernel::read;
                req.cpus = allCores ? pack->cpus : single;
                req.elements = bytes / sizeof(float);
                req.passes = passes;
                req.repeats = config.repeats;
                req.fillEachRepeat = false;
                req.stopAt = sideEnd;
                sp.measurement = measureBandwidth(arrays, req);
                out.sweep.push_back(std::move(sp));
            }
        }
    }
    if (config.runLatency) {
        const int latencyRepeats = std::min(config.repeats, 3);
        for (const std::size_t bytes : sweepSizes(config.latencyMaxBytes))
            out.latency.push_back(measureLatency(bytes, pack->cpus.front(), latencyRepeats, sideEnd));
    }

    // ---- placement kernels: with the time that remains, re-planned at each placement ----
    // The rate starts from the config's guess. After every point it is replaced by what was observed (wall time
    // per byte streamed, fill included), so the plan for the placements still to come uses the real machine.
    double secondsPerByte = 1.0 / (std::max(config.assumedGbPerSec, 1e-3) * 1e9);
    for (std::size_t p = 0; p < placements.size(); ++p) {
        const Placement& pl = placements[p];
        const double remainingBudget = std::max(0.0, config.budgetSeconds - secondsBetween(start, Clock::now()));
        const std::vector<Placement> rest(placements.begin() + static_cast<std::ptrdiff_t>(p), placements.end());
        RoofConfig replan = config;
        const double observedRate = 1.0 / (secondsPerByte * 1e9);
        if (validRoofRate(observedRate)) replan.assumedGbPerSec = observedRate;
        const RoofPlan plan = planRoofRun(rest, replan, arrays.bytes(), remainingBudget);
        if (p == 0) {
            out.plannedRepeats = plan.repeats;
            out.planNote = plan.note;
        } else if (!plan.note.empty()) {
            out.planNote += "[" + pl.name + "] " + plan.note;
        }

        // Points still to come, counted in the plan, for the per-point repeat share.
        double pointsLeft = 0.0;
        for (std::size_t q = 0; q < plan.threadCounts.size(); ++q)
            pointsLeft += 4.0 * static_cast<double>(plan.threadCounts[q].size());

        for (const int threads : plan.threadCounts[0]) {
            for (const RoofKernel k : {RoofKernel::read, RoofKernel::write, RoofKernel::copy, RoofKernel::triad}) {
                KernelPoint kp;
                kp.kernel = std::string(kernelName(k));
                kp.placement = pl.name;
                kp.threads = threads;
                const double arrayBytes = static_cast<double>(arrays.bytes());
                const double perRep = arraysMoved(k) * arrayBytes * secondsPerByte;
                const double perRun = pointStreamBytes(k, 1, arrayBytes) * secondsPerByte;
                const double remaining = std::chrono::duration<double>(deadline - Clock::now()).count();
                if (remaining < perRun) {
                    kp.skipReason = "time budget exhausted";
                    out.kernels.push_back(std::move(kp));
                    pointsLeft -= 1.0;
                    continue;
                }
                const int reps = boundedRoofRepeats(remaining / (pointsLeft * perRep), plan.repeats);
                BandwidthRequest req;
                req.kernel = k;
                req.cpus = cpusFor(pl, threads);
                req.elements = arrays.elements();
                req.passes = 1;
                req.repeats = reps;
                req.fillEachRepeat = false;   // one fill per point: the first repeat's fill is the first touch
                req.stopAt = deadline;
                const Clock::time_point t0 = Clock::now();
                kp.measurement = measureBandwidth(arrays, req);
                if (kp.measurement.repeats > 0) {
                    const double observed = secondsBetween(t0, Clock::now()) /
                                            pointStreamBytes(k, kp.measurement.repeats, arrayBytes);
                    if (observed > 0.0 && std::isfinite(observed)) secondsPerByte = observed;
                }
                out.kernels.push_back(std::move(kp));
                pointsLeft -= 1.0;
            }
        }
    }
    out.totalSeconds = secondsBetween(start, Clock::now());
    return out;
}

namespace detail::roofs {

inline void writeMeasurement(detail::JsonObject& o, const Measurement& m) {
    o.value("repeats", m.repeats);
    o.value("verified", m.verified);
    o.value("verifyFailure", std::string_view(m.verifyFailure));
    o.fact("medianGbPerSec", m.medianGbPerSec);
    o.fact("minGbPerSec", m.minGbPerSec);
    o.fact("maxGbPerSec", m.maxGbPerSec);
    o.value("seconds", m.seconds);
}

/// The fields of the `memory` object, written into `mem` (which must be open inside `out`).
inline void writeMemoryFields(std::string& out, detail::JsonObject& mem, const MemoryRoofs& r) {
    mem.value("supported", r.supported);
    mem.value("unsupportedReason", std::string_view(r.unsupportedReason));
    mem.value("arrayBytes", static_cast<std::uint64_t>(r.arrayBytes));
    mem.value("l3TotalBytes", static_cast<std::uint64_t>(r.l3TotalBytes));
    mem.value("budgetSeconds", r.budgetSeconds);
    mem.value("totalSeconds", r.totalSeconds);
    mem.value("plannedRepeats", r.plannedRepeats);
    mem.value("planNote", std::string_view(r.planNote));
    mem.key("kernels");
    detail::jsonArray(out, r.kernels.size(), [&](std::size_t i) {
        const KernelPoint& k = r.kernels[i];
        detail::JsonObject o(out);
        o.value("kernel", std::string_view(k.kernel));
        o.value("placement", std::string_view(k.placement));
        o.value("threads", k.threads);
        o.value("skipReason", std::string_view(k.skipReason));
        writeMeasurement(o, k.measurement);
    });
    mem.key("sweep");
    detail::jsonArray(out, r.sweep.size(), [&](std::size_t i) {
        const SweepPoint& s = r.sweep[i];
        detail::JsonObject o(out);
        o.value("mode", std::string_view(s.mode));
        o.value("bytes", static_cast<std::uint64_t>(s.bytes));
        o.value("threads", s.threads);
        writeMeasurement(o, s.measurement);
    });
    mem.key("latency");
    detail::jsonArray(out, r.latency.size(), [&](std::size_t i) {
        const LatencyPoint& l = r.latency[i];
        detail::JsonObject o(out);
        o.value("bytes", static_cast<std::uint64_t>(l.bytes));
        o.value("repeats", l.repeats);
        o.value("verified", l.verified);
        o.value("verifyFailure", std::string_view(l.verifyFailure));
        o.fact("medianNsPerLoad", l.medianNsPerLoad);
        o.fact("minNsPerLoad", l.minNsPerLoad);
        o.fact("maxNsPerLoad", l.maxNsPerLoad);
        o.value("seconds", l.seconds);
    });
}

inline std::string fixedText(double v, int digits) {
    std::string s;
    detail::appendFixed(s, v, digits);
    return s;
}

inline std::string padLeft(std::string s, std::size_t w) {
    if (s.size() < w) s.insert(0, w - s.size(), ' ');
    return s;
}

inline std::string padRight(std::string s, std::size_t w) {
    if (s.size() < w) s.append(w - s.size(), ' ');
    return s;
}

inline std::string bytesText(std::uint64_t b) {
    std::string s;
    detail::sBytes(s, b);
    return s;
}

/// The facts object exactly as `toJson(facts)` writes it (braces included), so a combined document agrees with the
/// facts document byte for byte. `null` when that document has an unexpected shape.
inline std::string factsObject(const MachineFacts& facts) {
    const std::string prefix = "{\"schema\":\"sub0llm-machine-1\",\"facts\":";
    const std::string factsDoc = toJson(facts);
    if (factsDoc.size() > prefix.size() + 1 && factsDoc.rfind(prefix, 0) == 0 && factsDoc.back() == '}')
        return factsDoc.substr(prefix.size(), factsDoc.size() - prefix.size() - 1);
    return "null";
}

}  // namespace detail::roofs

/// Deterministic JSON of the memory roofs alone: `{"schema":"sub0llm-machine-1","roofs":{"memory":{...}}}`.
/// Each measured number is `{"value": <number or null>, "reason": "<why absent, empty when known>"}`.
[[nodiscard]] inline std::string toJson(const MemoryRoofs& r) {
    std::string out;
    {
        detail::JsonObject root(out);
        root.value("schema", std::string_view("sub0llm-machine-1"));
        detail::JsonObject roofs(root, "roofs");
        detail::JsonObject mem(roofs, "memory");
        detail::roofs::writeMemoryFields(out, mem, r);
    }
    return out;
}

/// The full `sub0llm-machine --roofs mem` document:
/// `{"schema":"sub0llm-machine-1","facts":{...},"roofs":{"memory":{...}}}`. The facts object is exactly the one
/// `toJson(facts)` writes, so the two documents agree byte for byte on the facts.
[[nodiscard]] inline std::string toJson(const MachineFacts& facts, const MemoryRoofs& roofs) {
    const std::string prefix = "{\"schema\":\"sub0llm-machine-1\",\"facts\":";
    const std::string body = detail::roofs::factsObject(facts);
    std::string out = prefix + body + ",\"roofs\":";
    {
        detail::JsonObject roofsObj(out);
        detail::JsonObject mem(roofsObj, "memory");
        detail::roofs::writeMemoryFields(out, mem, roofs);
    }
    out += '}';
    return out;
}

/// A human-readable table: read bandwidth by placement (rows) and thread count (columns), then the best write,
/// copy and triad, the working-set sweeps and the latency plateaus. Invalid points print INVALID.
[[nodiscard]] inline std::string toSummary(const MemoryRoofs& r) {
    using namespace detail::roofs;
    std::string o = "sub0llm machine roofs (memory)\n";
    if (!r.supported) {
        o += "  unsupported: " + r.unsupportedReason + "\n";
        return o;
    }
    o += "  arrays " + bytesText(r.arrayBytes) + " each";
    o += ", L3 total " + (r.l3TotalBytes ? bytesText(r.l3TotalBytes) : std::string("unknown"));
    o += ", budget " + fixedText(r.budgetSeconds, 0) + " s, used " + fixedText(r.totalSeconds, 1) + " s\n";
    o += "  repeats planned " + std::to_string(r.plannedRepeats);
    if (!r.planNote.empty()) o += " (" + r.planNote + ")";
    o += "\n";

    // Table: rows are placements in first-seen order, columns are the thread counts measured for read.
    std::vector<std::string> rows;
    std::vector<int> cols;
    for (const KernelPoint& k : r.kernels) {
        if (k.kernel != "read") continue;
        if (std::find(rows.begin(), rows.end(), k.placement) == rows.end()) rows.push_back(k.placement);
        if (std::find(cols.begin(), cols.end(), k.threads) == cols.end()) cols.push_back(k.threads);
    }
    std::sort(cols.begin(), cols.end());
    o += "  read GB/s (median) by placement and threads\n";
    o += "    " + padRight("placement", 11);
    for (const int c : cols) o += padLeft(std::to_string(c) + " thr", 9);
    o += "\n";
    for (const std::string& row : rows) {
        o += "    " + padRight(row, 11);
        for (const int c : cols) {
            std::string cell = "-";
            for (const KernelPoint& k : r.kernels) {
                if (k.kernel != "read" || k.placement != row || k.threads != c) continue;
                if (k.measurement.medianGbPerSec.value) cell = fixedText(*k.measurement.medianGbPerSec.value, 1);
                else if (k.skipReason.empty()) cell = "INVALID";
            }
            o += padLeft(cell, 9);
        }
        o += "\n";
    }

    for (const RoofKernel k : {RoofKernel::write, RoofKernel::copy, RoofKernel::triad}) {
        const KernelPoint* best = nullptr;
        for (const KernelPoint& p : r.kernels) {
            if (p.kernel != kernelName(k) || !p.measurement.medianGbPerSec.value) continue;
            if (!best || *p.measurement.medianGbPerSec.value > *best->measurement.medianGbPerSec.value) best = &p;
        }
        o += "  best " + padRight(std::string(kernelName(k)), 6);
        if (best) {
            o += " " + fixedText(*best->measurement.medianGbPerSec.value, 1) + " GB/s  (" + best->placement + ", " +
                 std::to_string(best->threads) + " threads)\n";
        } else {
            o += " none verified\n";
        }
    }

    for (const std::string_view mode : {std::string_view("single"), std::string_view("one-per-core")}) {
        o += "  read sweep " + padRight(std::string(mode), 12) + " GB/s:";
        for (const SweepPoint& s : r.sweep) {
            if (s.mode != mode) continue;
            o += " " + bytesText(s.bytes) + " ";
            o += s.measurement.medianGbPerSec.value ? fixedText(*s.measurement.medianGbPerSec.value, 1) : "INVALID";
        }
        o += "\n";
    }

    o += "  latency ns per dependent load:";
    for (const LatencyPoint& l : r.latency) {
        o += " " + bytesText(l.bytes) + " ";
        o += l.medianNsPerLoad.value ? fixedText(*l.medianNsPerLoad.value, 1) : "INVALID";
    }
    o += "\n";
    return o;
}

}  // namespace sub0::measure
