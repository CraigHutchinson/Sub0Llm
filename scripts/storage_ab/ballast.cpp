// ballast: hold N GiB of RAM resident so the OS cannot give it to the page cache, simulating a host
// with less memory. Locks the pages when the working-set quota allows, else re-touches them every
// second so they stay hot. Runs until killed. usage: ballast <gib>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: ballast <gib>\n"); return 1; }
    const SIZE_T bytes = static_cast<SIZE_T>(std::strtoull(argv[1], nullptr, 10)) << 30;
    const SIZE_T slack = SIZE_T{256} << 20;
    const BOOL quota = ::SetProcessWorkingSetSize(::GetCurrentProcess(), bytes + slack, bytes + 2 * slack);
    auto* p = static_cast<volatile char*>(::VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!p) { std::fprintf(stderr, "VirtualAlloc failed %lu\n", ::GetLastError()); return 2; }
    for (SIZE_T i = 0; i < bytes; i += 4096) p[i] = 1;
    const BOOL locked = quota && ::VirtualLock(const_cast<char*>(p), bytes);
    std::printf("ballast %zu GiB held: quota=%d locked=%d (err %lu)\n", bytes >> 30, quota, locked,
                locked ? 0ul : ::GetLastError());
    std::fflush(stdout);
    for (;;) {
        if (!locked)
            for (SIZE_T i = 0; i < bytes; i += 4096) p[i] = static_cast<char>(p[i] + 1);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
