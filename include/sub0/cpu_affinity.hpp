// sub0/cpu_affinity.hpp -- P-core-first thread pinning for hybrid Intel CPUs (P+E cores).
//
// VTune's uarch-exploration data (project memory cpu-profiling-tooling-backlog, 2026-07-16) found this
// codebase's OpenMP-parallel regions (train_batch / evaluate / AdamW::step, all bare `#pragma omp
// parallel num_threads(DEFAULT_THREADS)`) have NO core-type awareness: E-cores did ~9x the total
// clockwork of P-cores despite being individually less efficient (CPI 1.099 vs P-core's 0.661), and
// effective CPU utilization averaged only 34.7% (8.3 of 24 logical cores). Windows' own hybrid-core
// scheduling does not guarantee OpenMP's worker threads land on P-cores first -- this module makes that
// explicit and deterministic instead of leaving it to the scheduler.
//
// Windows-only (this project's whole toolchain already is -- MSVC ABI, PowerShell workflow scripts).
// Single-processor-group only (correct for any machine with <=64 logical processors, true here at 24 --
// a >64-logical-processor machine needs the multi-group GROUP_AFFINITY-aware SetThreadGroupAffinity path,
// which this deliberately does not implement; detection degrades to invalid/no-op rather than guessing).
//
// This is a HINT, never a correctness requirement: every caller must tolerate detection or pinning
// failing (a VM, a non-hybrid CPU, an old Windows build without EfficiencyClass support) by just running
// with whatever placement the OS already gave the thread.

#pragma once

#define NOMINMAX                 // keep std::min/std::max, not the windows.h macros
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace sub0 {

// One entry per logical processor, ordered P-cores first (by EfficiencyClass descending, ties broken by
// detection order) -- rank i here is "the i-th most-performant logical processor to hand a thread to".
struct CoreOrder {
    std::vector<int> logical_ids;   // logical processor index (bit position in a single-group affinity mask)
    int p_core_count = 0;           // logical_ids[0 .. p_core_count) are the highest EfficiencyClass tier
    bool valid = false;             // false => detection failed or the machine has >1 processor group
};

// Detected once (thread-safe: C++11 static-local init), reused for the process lifetime -- topology
// doesn't change at runtime.
inline const CoreOrder& detect_core_order() {
    static const CoreOrder order = [] {
        CoreOrder o;
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len == 0) return o;   // API unsupported on this OS -> stay invalid, callers no-op
        std::vector<std::uint8_t> buf(len);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len))
            return o;

        struct Core { int logical_id; int efficiency; };
        std::vector<Core> cores;
        for (std::uint8_t* p = buf.data(); p < buf.data() + len; ) {
            auto* e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
            if (e->Relationship == RelationProcessorCore) {
                if (e->Processor.GroupCount != 1) return o;   // multi-group -> not handled, o is still invalid
                const GROUP_AFFINITY& ga = e->Processor.GroupMask[0];
                if (ga.Group != 0) return o;                  // only Group 0 supported
                for (int bit = 0; bit < 64; ++bit)
                    if (ga.Mask & (static_cast<KAFFINITY>(1) << bit))
                        cores.push_back({bit, e->Processor.EfficiencyClass});
            }
            p += e->Size;
        }
        if (cores.empty()) return o;
        std::stable_sort(cores.begin(), cores.end(),
            [](const Core& a, const Core& b) { return a.efficiency > b.efficiency; });
        for (const Core& c : cores) o.logical_ids.push_back(c.logical_id);
        const int top_eff = cores.front().efficiency;
        o.p_core_count = static_cast<int>(std::count_if(cores.begin(), cores.end(),
            [&](const Core& c) { return c.efficiency == top_eff; }));
        o.valid = true;
        return o;
    }();
    return order;
}

// Pin the CALLING thread to the `rank`-th most-performant logical processor (rank 0 = a P-core, if any
// exist), wrapping around if `rank` exceeds the logical processor count (more OpenMP threads than cores
// is legal, just shares a core). Returns false (no-op, thread keeps its current placement) if detection
// failed or the machine has multiple processor groups.
inline bool pin_current_thread_p_first(int rank) {
    const CoreOrder& o = detect_core_order();
    if (!o.valid || o.logical_ids.empty() || rank < 0) return false;
    const int logical_id = o.logical_ids[static_cast<std::size_t>(rank) % o.logical_ids.size()];
    const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << logical_id;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

}  // namespace sub0
