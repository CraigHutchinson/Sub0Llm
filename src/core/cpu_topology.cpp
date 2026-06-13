#include "sub0llm/core/cpu_topology.hpp"

#include <algorithm>
#include <map>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif

namespace sub0llm {

CpuTopology detect_cpu_topology() {
    CpuTopology topo;
    topo.n_logical = static_cast<int>(std::thread::hardware_concurrency());
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return topo;
    std::vector<char> buf(len);
    auto* first = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, first, &len)) return topo;

    // One record per physical core; EfficiencyClass is higher for faster cores.
    // GroupMask names the core's logical CPUs (SMT siblings share the class).
    std::map<int, std::vector<int>> by_class;
    for (char* p = buf.data(); p < buf.data() + len; ) {
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        if (info->Relationship == RelationProcessorCore) {
            const int ec = info->Processor.EfficiencyClass;
            const auto& gm = info->Processor.GroupMask[0];
            const int gbase = gm.Group * 64;
            for (int b = 0; b < 64; ++b)
                if (gm.Mask & (static_cast<KAFFINITY>(1) << b)) by_class[ec].push_back(gbase + b);
        }
        p += info->Size;
    }
    if (by_class.size() >= 2) {            // hybrid: top class = perf, the rest = efficiency
        const int top = by_class.rbegin()->first;
        for (auto& [cls, cpus] : by_class)
            for (int c : cpus) (cls == top ? topo.perf : topo.efficiency).push_back(c);
        std::ranges::sort(topo.perf);
        std::ranges::sort(topo.efficiency);
    } else if (!by_class.empty()) {        // homogeneous: expose the ids as perf
        topo.perf = by_class.begin()->second;
        std::ranges::sort(topo.perf);
    }
#endif
    return topo;
}

void pin_current_thread(int cpu) noexcept {
    if (cpu < 0) return;
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << (cpu & 63));
#else
    (void)cpu;
#endif
}

std::vector<int> compute_pin_set(const CpuTopology& topo, std::size_t n_workers) {
    std::vector<int> pins;
    pins.reserve(n_workers);
    for (int c : topo.perf)       if (pins.size() < n_workers) pins.push_back(c);
    for (int c : topo.efficiency) if (pins.size() < n_workers) pins.push_back(c);
    while (pins.size() < n_workers) pins.push_back(-1);   // oversubscribed: unpinned
    return pins;
}

} // namespace sub0llm
