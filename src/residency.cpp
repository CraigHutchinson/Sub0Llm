#include "sub0/residency.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifndef PSAPI_VERSION
#define PSAPI_VERSION 2 // QueryWorkingSetEx resolves to kernel32's K32QueryWorkingSetEx: no psapi.lib
#endif
#include <psapi.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#if defined(__linux__)
#include <fstream>
#include <string>
#endif
#endif

namespace sub0::residency {

namespace {
constexpr std::size_t kChunk = 4096; // pages per query: bounds the stack buffer, allocates nothing
} // namespace

bool pin(void* data, std::size_t bytes) noexcept {
    if (data == nullptr || bytes == 0) return false;
#if defined(_WIN32)
    SIZE_T min_ws = 0, max_ws = 0;
    const HANDLE self = ::GetCurrentProcess();
    return ::GetProcessWorkingSetSize(self, &min_ws, &max_ws) &&
           ::SetProcessWorkingSetSizeEx(self, min_ws + bytes, max_ws + bytes,
                                        QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE) &&
           ::VirtualLock(data, bytes);
#else
    // mlock is bounded by RLIMIT_MEMLOCK (often 8-64 MiB on Linux, unlimited on macOS). A hard limit
    // below the request needs `ulimit -l` or CAP_IPC_LOCK.
    if (rlimit lim{}; ::getrlimit(RLIMIT_MEMLOCK, &lim) == 0 && lim.rlim_cur < lim.rlim_max) {
        lim.rlim_cur = lim.rlim_max;
        (void)::setrlimit(RLIMIT_MEMLOCK, &lim);
    }
    return ::mlock(data, bytes) == 0;
#endif
}

void release_quota([[maybe_unused]] std::size_t bytes) noexcept {
#if defined(_WIN32)
    SIZE_T min_ws = 0, max_ws = 0;
    const HANDLE self = ::GetCurrentProcess();
    if (::GetProcessWorkingSetSize(self, &min_ws, &max_ws) && min_ws > bytes && max_ws > bytes)
        (void)::SetProcessWorkingSetSizeEx(self, min_ws - bytes, max_ws - bytes,
                                           QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
#endif
}

Report query(const void* data, std::size_t bytes, [[maybe_unused]] bool locked_hint) noexcept {
    Report r;
    if (data == nullptr || bytes == 0) return r;
    const auto* base = static_cast<const std::byte*>(data);
#if defined(_WIN32)
    constexpr std::size_t kPage = 4096;
    std::array<PSAPI_WORKING_SET_EX_INFORMATION, kChunk> info{};
    r.pages = (bytes + kPage - 1) / kPage;
    for (std::uint64_t first = 0; first < r.pages; first += kChunk) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, r.pages - first));
        for (std::size_t i = 0; i < count; ++i)
            info[i].VirtualAddress = const_cast<std::byte*>(base + (first + i) * kPage);
        if (!::QueryWorkingSetEx(::GetCurrentProcess(), info.data(),
                                 static_cast<DWORD>(count * sizeof(PSAPI_WORKING_SET_EX_INFORMATION))))
            return Report{};
        for (std::size_t i = 0; i < count; ++i) {
            r.resident += info[i].VirtualAttributes.Valid;
            r.locked += info[i].VirtualAttributes.Valid && info[i].VirtualAttributes.Locked;
        }
    }
#else
    const auto page = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    // mincore needs a page-aligned start; widen the region to whole pages.
    const auto start = reinterpret_cast<std::uintptr_t>(base) & ~(static_cast<std::uintptr_t>(page) - 1);
    const auto end = reinterpret_cast<std::uintptr_t>(base) + bytes;
#if defined(__APPLE__)
    std::array<char, kChunk> vec{};
#else
    std::array<unsigned char, kChunk> vec{};
#endif
    r.pages = (end - start + page - 1) / page;
    for (std::uint64_t first = 0; first < r.pages; first += kChunk) {
        const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, r.pages - first));
        if (::mincore(reinterpret_cast<void*>(start + first * page), count * page, vec.data()) != 0) return Report{};
        for (std::size_t i = 0; i < count; ++i) r.resident += (vec[i] & 1) != 0;
    }
#if defined(__linux__)
    // No per-page lock bit on Linux: VmLck covers every lock this process holds, so it is a lower-bound
    // check -- below the region's size, the region cannot be fully locked.
    std::uint64_t vm_lck_kib = 0;
    std::ifstream status("/proc/self/status");
    for (std::string line; std::getline(status, line);)
        if (line.rfind("VmLck:", 0) == 0) vm_lck_kib = std::strtoull(line.c_str() + 6, nullptr, 10);
    r.locked = locked_hint && vm_lck_kib * 1024 >= bytes ? r.resident : 0;
#else
    // macOS exposes no per-page lock state; the caller's successful mlock is the evidence.
    r.locked = locked_hint ? r.resident : 0;
#endif
#endif
    return r;
}

} // namespace sub0::residency
