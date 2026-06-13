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

#include <thread>
#include <vector>

namespace sub0llm {

struct CpuTopology {
    int              n_logical = 0;   // logical CPU count (hardware_concurrency)
    std::vector<int> perf;            // logical CPU ids of P-cores (top efficiency class)
    std::vector<int> efficiency;      // logical CPU ids of E-cores (lower classes)

    // Homogeneous CPUs report everything as "perf" via this helper.
    [[nodiscard]] bool hybrid() const noexcept { return !efficiency.empty(); }
    [[nodiscard]] int  n_perf() const noexcept {
        return hybrid() ? static_cast<int>(perf.size()) : n_logical;
    }
};

// Detect the P/E split (Windows: GetLogicalProcessorInformationEx EfficiencyClass;
// other platforms: homogeneous fallback). Cheap, but cache it — topology can't change.
[[nodiscard]] CpuTopology detect_cpu_topology();

// Pin the calling thread to one logical CPU; cpu < 0 = no-op (OS default placement).
void pin_current_thread(int cpu) noexcept;

// Worker→CPU assignment for compute-bound pools: P-cores first, then E-cores,
// then unpinned (-1) if the pool is larger than the machine.
[[nodiscard]] std::vector<int> compute_pin_set(const CpuTopology& topo, std::size_t n_workers);

} // namespace sub0llm
