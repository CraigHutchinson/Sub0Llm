#pragma once

// cpu_topology.hpp — the shared CPU threading model: P/E core topology + pinning.
//
// One detection, one pinning primitive, used by every threaded component (the
// gemma engine's GEMV pool, the diffusion ParallelTrainer, future kernels) so
// thread-placement policy is consistent across the project:
//   • compute-bound work → P-cores first (highest EfficiencyClass)
//   • bandwidth-bound work → E-cores can win by freeing P-cores (see Ch27 notes)
//
// Constexpr compatibility: this is the RUNTIME side. The specialized-build path
// (sub0diff-specialize → HostSpec) bakes perf/eff core COUNTS in as constexpr;
// these APIs take/return plain ints and vectors so a monomorphized binary can
// fold `HostSpec::perf_cores` into its pool size at compile time and only call
// detect_cpu_topology() in its `-verify` mode to assert the host still matches.

#include <string_view>
#include <thread>
#include <vector>

namespace sub0llm {

struct CpuTopology {
    int              n_logical = 0;   // logical CPU count (hardware_concurrency)
    std::vector<int> perf;            // ALL logical CPU ids of P-cores (top efficiency class)
    std::vector<int> efficiency;      // ALL logical CPU ids of E-cores (lower classes)

    // One logical CPU per PHYSICAL core (the lowest-id of each core's SMT group),
    // P-cores then E-cores. This is what compute-bound pinning must use: pinning two
    // workers to SMT siblings of the same physical core halves throughput. `perf`/
    // `efficiency` keep all logical ids for bandwidth-bound spreading (gemma decode).
    std::vector<int> perf_primary;
    std::vector<int> eff_primary;

    // Homogeneous CPUs report everything as "perf" via these helpers.
    [[nodiscard]] bool hybrid() const noexcept { return !efficiency.empty(); }
    [[nodiscard]] int  n_perf() const noexcept {
        return hybrid() ? static_cast<int>(perf.size()) : n_logical;
    }
    // Physical P-cores (the right count for compute-bound pool sizing).
    [[nodiscard]] int n_perf_cores() const noexcept {
        return static_cast<int>(perf_primary.size());
    }
};

// Detect the P/E split (Windows: GetLogicalProcessorInformationEx EfficiencyClass;
// other platforms: homogeneous fallback). Cheap, but cache it — topology can't change.
[[nodiscard]] CpuTopology detect_cpu_topology();

// Pin the calling thread to one logical CPU; cpu < 0 = no-op (OS default placement).
// Returns true iff the thread is now hard-affinitized to `cpu` (false for cpu < 0 or
// if the OS rejected the affinity request — callers can self-test pinning success).
bool pin_current_thread(int cpu) noexcept;

// The logical CPU the calling thread is executing on RIGHT NOW (Windows:
// GetCurrentProcessorNumber; Linux: sched_getcpu; -1 if unavailable). Sample it in a
// hot loop to verify a pinned worker really stays put — a worker observed on many
// different CPUs is migrating, i.e. its affinity didn't take.
[[nodiscard]] int current_cpu() noexcept;

// Worker→CPU assignment for compute-bound pools: P-cores first, then E-cores,
// then unpinned (-1) if the pool is larger than the machine.
[[nodiscard]] std::vector<int> compute_pin_set(const CpuTopology& topo, std::size_t n_workers);

// Parse an explicit logical-CPU list: "lo-hi" inclusive range, or "a,b,c" comma list.
[[nodiscard]] std::vector<int> parse_cpu_list(std::string_view s);

// Resolve a pin POLICY string to a worker→CPU pin set (one entry per worker, size
// n_workers) for a compute-bound pool. This is what lets concurrent jobs be placed on
// DISJOINT cores so they don't oversubscribe the same P-cores (the dominant cause of the
// ~30× slowdown when several ParallelTrainers run at once):
//   "auto"            → compute_pin_set (P-primaries first, then E, then unpinned) [default]
//   "all"             → no pinning (all -1; OS default placement)
//   "P" / "E"         → P-core / E-core primaries, cycled if fewer than n_workers
//   "lo-hi" / "a,b,c" → those explicit logical CPUs, cycled to n_workers
// e.g. run A `--pin 0,1,10,11` and run B `--pin 12,13,22,23` → two 4-worker jobs on
// disjoint P-cores. A typo (non-policy, non-number) throws via std::stoi (fails loudly).
[[nodiscard]] std::vector<int> resolve_pin_set(std::string_view policy,
                                               const CpuTopology& topo, std::size_t n_workers);

} // namespace sub0llm
