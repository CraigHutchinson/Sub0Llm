#include "sub0llm/core/cpu_topology.hpp"

#include <algorithm>
#include <map>
#include <string>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sched.h>
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

    // One record per PHYSICAL core; EfficiencyClass is higher for faster cores.
    // GroupMask names the core's logical CPUs (SMT siblings share the class).
    // Track per class: all logical ids, and the primary (lowest id) of each core.
    std::map<int, std::vector<int>> by_class;        // class → all logical cpus
    std::map<int, std::vector<int>> primary_by_class; // class → one cpu per physical core
    for (char* p = buf.data(); p < buf.data() + len; ) {
        auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
        if (info->Relationship == RelationProcessorCore) {
            const int ec = info->Processor.EfficiencyClass;
            const auto& gm = info->Processor.GroupMask[0];
            const int gbase = gm.Group * 64;
            int core_primary = -1;
            for (int b = 0; b < 64; ++b)
                if (gm.Mask & (static_cast<KAFFINITY>(1) << b)) {
                    const int cpu = gbase + b;
                    by_class[ec].push_back(cpu);
                    if (core_primary < 0) core_primary = cpu;   // lowest-id SMT thread
                }
            if (core_primary >= 0) primary_by_class[ec].push_back(core_primary);
        }
        p += info->Size;
    }
    auto sort_into = [](std::map<int, std::vector<int>>& src, int top,
                        std::vector<int>& hi, std::vector<int>& lo) {
        for (auto& [cls, cpus] : src)
            for (int c : cpus) (cls == top ? hi : lo).push_back(c);
        std::ranges::sort(hi);
        std::ranges::sort(lo);
    };
    if (by_class.size() >= 2) {            // hybrid: top class = perf, the rest = efficiency
        const int top = by_class.rbegin()->first;
        sort_into(by_class, top, topo.perf, topo.efficiency);
        sort_into(primary_by_class, top, topo.perf_primary, topo.eff_primary);
    } else if (!by_class.empty()) {        // homogeneous: expose ids as perf
        topo.perf         = by_class.begin()->second;
        topo.perf_primary = primary_by_class.begin()->second;
        std::ranges::sort(topo.perf);
        std::ranges::sort(topo.perf_primary);
    }
#endif
    return topo;
}

bool pin_current_thread(int cpu) noexcept {
    if (cpu < 0) return false;
#ifdef _WIN32
    // SetThreadAffinityMask returns the PREVIOUS mask (nonzero) on success, 0 on
    // failure. Single processor group assumed (this host has 24 logical CPUs < 64).
    return SetThreadAffinityMask(GetCurrentThread(),
                                 static_cast<DWORD_PTR>(1) << (cpu & 63)) != 0;
#else
    (void)cpu;
    return false;
#endif
}

int current_cpu() noexcept {
#ifdef _WIN32
    return static_cast<int>(GetCurrentProcessorNumber());
#elif defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

std::vector<int> compute_pin_set(const CpuTopology& topo, std::size_t n_workers) {
    std::vector<int> pins;
    pins.reserve(n_workers);
    // ONE worker per physical core (P first, then E) — never two on SMT siblings of
    // the same core. Only once every physical core has one worker do we allow SMT
    // siblings (the remaining `perf`/`efficiency` logical ids) and finally unpinned.
    for (int c : topo.perf_primary) if (pins.size() < n_workers) pins.push_back(c);
    for (int c : topo.eff_primary)  if (pins.size() < n_workers) pins.push_back(c);
    auto add_remaining = [&](const std::vector<int>& all) {
        for (int c : all)
            if (pins.size() < n_workers &&
                std::ranges::find(pins, c) == pins.end()) pins.push_back(c);
    };
    add_remaining(topo.perf);
    add_remaining(topo.efficiency);
    while (pins.size() < n_workers) pins.push_back(-1);   // oversubscribed: unpinned
    return pins;
}

std::vector<int> parse_cpu_list(std::string_view s) {
    std::vector<int> out;
    const std::string str(s);
    if (auto dash = str.find('-');
        dash != std::string::npos && str.find(',') == std::string::npos) {
        const int lo = std::stoi(str.substr(0, dash)), hi = std::stoi(str.substr(dash + 1));
        for (int c = lo; c <= hi; ++c) out.push_back(c);
    } else {
        std::size_t i = 0;
        while (i < str.size()) {
            std::size_t j = str.find(',', i);
            if (j == std::string::npos) j = str.size();
            if (j > i) out.push_back(std::stoi(str.substr(i, j - i)));
            i = j + 1;
        }
    }
    return out;
}

std::vector<int> resolve_pin_set(std::string_view policy, const CpuTopology& topo,
                                 std::size_t n_workers) {
    if (policy == "auto") return compute_pin_set(topo, n_workers);
    if (policy == "all")  return std::vector<int>(n_workers, -1);
    std::vector<int> base;
    if (policy == "P")      base = topo.perf_primary.empty() ? topo.perf : topo.perf_primary;
    else if (policy == "E") base = topo.eff_primary.empty()  ? topo.efficiency : topo.eff_primary;
    else                    base = parse_cpu_list(policy);   // explicit list/range
    if (base.empty()) return std::vector<int>(n_workers, -1);
    std::vector<int> pins;
    pins.reserve(n_workers);
    for (std::size_t i = 0; i < n_workers; ++i) pins.push_back(base[i % base.size()]);
    return pins;
}

} // namespace sub0llm
