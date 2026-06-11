// host_detect.cpp (Ch28) — live host hardware detection for sub0diff::detect_host().
//
// Platform-specific and best-effort: any field we cannot determine is left 0 so the
// caller (and verify_host) can treat it as "unknown / skip" rather than crash. This is
// the runtime counterpart to the constexpr HostSpec emitted by sub0diff-specialize.

#include "sub0diff/spec/host_spec.hpp"

#include <thread>

// ── CPUID shim (x86 only) ─────────────────────────────────────────────────────────
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define SUB0DIFF_X86 1
  #if defined(_WIN32)
    #include <intrin.h>
    static inline void sub0diff_cpuidex(int out[4], int leaf, int sub) {
        __cpuidex(out, leaf, sub);
    }
  #elif __has_include(<cpuid.h>)
    #include <cpuid.h>
    static inline void sub0diff_cpuidex(int out[4], int leaf, int sub) {
        unsigned a = 0, b = 0, c = 0, d = 0;
        __get_cpuid_count(static_cast<unsigned>(leaf), static_cast<unsigned>(sub), &a, &b, &c, &d);
        out[0] = static_cast<int>(a); out[1] = static_cast<int>(b);
        out[2] = static_cast<int>(c); out[3] = static_cast<int>(d);
    }
  #else
    static inline void sub0diff_cpuidex(int out[4], int, int) { out[0] = out[1] = out[2] = out[3] = 0; }
  #endif
#else
  static inline void sub0diff_cpuidex(int out[4], int, int) { out[0] = out[1] = out[2] = out[3] = 0; }
#endif

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <vector>
#elif defined(__APPLE__)
  #include <sys/sysctl.h>
  #include <sys/types.h>
#else
  #include <unistd.h>
#endif

namespace sub0diff {

namespace {

unsigned detect_simd_width_bits() noexcept {
#if SUB0DIFF_X86
    int info[4] = {0, 0, 0, 0};
    sub0diff_cpuidex(info, 0, 0);
    const int max_leaf = info[0];
    if (max_leaf >= 7) {
        int leaf7[4] = {0, 0, 0, 0};
        sub0diff_cpuidex(leaf7, 7, 0);
        const unsigned ebx = static_cast<unsigned>(leaf7[1]);
        if (ebx & (1u << 16)) return 512;  // AVX-512F
        if (ebx & (1u << 5))  return 256;   // AVX2
    }
    return 128;  // SSE2 baseline on any x86-64
#else
    return 0;    // unknown on non-x86 (e.g. NEON would be 128, but don't assume)
#endif
}

std::string detect_cpu_brand() {
#if SUB0DIFF_X86
    int info[4] = {0, 0, 0, 0};
    sub0diff_cpuidex(info, static_cast<int>(0x80000000u), 0);
    if (static_cast<unsigned>(info[0]) < 0x80000004u) return {};
    char brand[49] = {};
    for (int i = 0; i < 3; ++i) {
        int regs[4] = {0, 0, 0, 0};
        sub0diff_cpuidex(regs, static_cast<int>(0x80000002u + static_cast<unsigned>(i)), 0);
        __builtin_memcpy(brand + i * 16, regs, 16);
    }
    std::string s(brand);
    // trim leading spaces some vendors pad with
    const auto first = s.find_first_not_of(' ');
    return first == std::string::npos ? std::string{} : s.substr(first);
#else
    return {};
#endif
}

} // namespace

HostFacts detect_host() noexcept {
    HostFacts f;
    f.logical_cores   = std::thread::hardware_concurrency();
    f.simd_width_bits = detect_simd_width_bits();
    try { f.cpu_brand = detect_cpu_brand(); } catch (...) { /* best-effort */ }

#if defined(_WIN32)
    // Physical cores + P/E split via the processor-core relation (EfficiencyClass).
    {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
        if (len > 0) {
            std::vector<unsigned char> buf(len);
            auto* base = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
            if (GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) {
                unsigned char max_class = 0;
                // First pass: count physical cores and find the top efficiency class.
                std::vector<unsigned char> classes;
                auto* p = buf.data();
                auto* end = buf.data() + len;
                while (p < end) {
                    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
                    if (info->Relationship == RelationProcessorCore) {
                        ++f.physical_cores;
                        const unsigned char ec = info->Processor.EfficiencyClass;
                        classes.push_back(ec);
                        if (ec > max_class) max_class = ec;
                    }
                    p += info->Size;
                }
                for (unsigned char ec : classes) {
                    if (ec == max_class) ++f.perf_cores; else ++f.eff_cores;
                }
            }
        }
    }
    // L1 data-cache line size via the cache relation.
    {
        DWORD len = 0;
        GetLogicalProcessorInformationEx(RelationCache, nullptr, &len);
        if (len > 0) {
            std::vector<unsigned char> buf(len);
            if (GetLogicalProcessorInformationEx(
                    RelationCache,
                    reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data()), &len)) {
                auto* p = buf.data();
                auto* end = buf.data() + len;
                while (p < end) {
                    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(p);
                    if (info->Relationship == RelationCache && info->Cache.Level == 1) {
                        f.cacheline_bytes = info->Cache.LineSize;
                        break;
                    }
                    p += info->Size;
                }
            }
        }
    }
    // Total physical RAM.
    {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) f.ram_bytes = ms.ullTotalPhys;
    }
#elif defined(__APPLE__)
    {
        int    nphys = 0;
        size_t sz = sizeof(nphys);
        if (sysctlbyname("hw.physicalcpu", &nphys, &sz, nullptr, 0) == 0 && nphys > 0)
            f.physical_cores = static_cast<unsigned>(nphys);
        std::int64_t mem = 0; sz = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &sz, nullptr, 0) == 0 && mem > 0)
            f.ram_bytes = static_cast<std::uint64_t>(mem);
        std::int64_t line = 0; sz = sizeof(line);
        if (sysctlbyname("hw.cachelinesize", &line, &sz, nullptr, 0) == 0 && line > 0)
            f.cacheline_bytes = static_cast<unsigned>(line);
        f.perf_cores = f.physical_cores;  // P/E split not split out here
    }
#else  // Linux / other POSIX
    {
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online > 0) f.physical_cores = static_cast<unsigned>(online);  // best-effort (SMT not divided)
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long psize = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && psize > 0)
            f.ram_bytes = static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(psize);
      #ifdef _SC_LEVEL1_DCACHE_LINESIZE
        const long line = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (line > 0) f.cacheline_bytes = static_cast<unsigned>(line);
      #endif
        f.perf_cores = f.physical_cores;
    }
#endif

    if (f.physical_cores == 0) f.physical_cores = f.logical_cores;
    if (f.perf_cores == 0)     f.perf_cores     = f.physical_cores;
    return f;
}

} // namespace sub0diff
