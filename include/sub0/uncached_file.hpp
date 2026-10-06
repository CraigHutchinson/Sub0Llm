// sub0/uncached_file.hpp -- read the start of a file without going through the OS page cache.
//
// WHY THIS EXISTS. The owned expert cache can fill its slots with non-cached reads (--moe-cache-io
// uncached), and on Windows those reads stop overlapping -- each waits for the previous one -- while the
// same file has, or recently had, a cached reader (Sub0MemPage docs/investigations/
// unbuffered-read-ceiling.md: 7 x 256 KiB took ~2.4 ms beside a held buffered reader, ~0.6 ms alone, and
// the state outlives the reader by seconds). So in that mode nothing may read the sidecar through the
// cache, the header and descriptor table included; this is the non-cached read for those.
//
// SCOPE (AGENTS.md S8): one administrative read of a file's first bytes. Not a stream, not a hot path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <cstdlib>
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace sub0 {

/// Reads the first `bytes` of `path` into `out` with a non-cached open (FILE_FLAG_NO_BUFFERING /
/// O_DIRECT / F_NOCACHE), leaving nothing of the file in the OS page cache. Returns false and fills
/// `err` if the file cannot be opened that way or is shorter than `bytes`; there is no cached fallback,
/// because a silent one would defeat the reason to call this. Administrative: allocates.
[[nodiscard]] inline bool read_prefix_uncached(const std::string& path, std::uint64_t bytes,
                                               std::vector<std::uint8_t>& out, std::string& err) {
    constexpr std::size_t kBlock = 4096; // offset, length and buffer alignment for non-cached I/O
    const std::size_t aligned = static_cast<std::size_t>((bytes + kBlock - 1) / kBlock * kBlock);
    out.clear();
    if (bytes == 0) return true;
#if defined(_WIN32)
    const HANDLE file = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                      FILE_FLAG_NO_BUFFERING, nullptr);
    if (file == INVALID_HANDLE_VALUE) { err = "cannot open " + path + " non-cached"; return false; }
    void* const block = ::VirtualAlloc(nullptr, aligned, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    std::size_t total = 0;
    while (block != nullptr && total < aligned) {
        DWORD got = 0;
        const auto want = static_cast<DWORD>(aligned - total < (std::size_t{1} << 30) ? aligned - total : std::size_t{1} << 30);
        if (!::ReadFile(file, static_cast<std::uint8_t*>(block) + total, want, &got, nullptr) || got == 0) break;
        total += got;
        if (got < want) break; // end of file: a retry would start mid-block
    }
    ::CloseHandle(file);
    const bool ok = block != nullptr && total >= bytes;
    if (ok) out.assign(static_cast<const std::uint8_t*>(block), static_cast<const std::uint8_t*>(block) + bytes);
    if (block != nullptr) ::VirtualFree(block, 0, MEM_RELEASE);
#else
    int flags = O_RDONLY | O_CLOEXEC;
  #ifdef O_DIRECT
    flags |= O_DIRECT;
  #endif
    const int file = ::open(path.c_str(), flags);
    if (file < 0) { err = "cannot open " + path + " non-cached"; return false; }
  #ifdef F_NOCACHE
    if (::fcntl(file, F_NOCACHE, 1) != 0) { ::close(file); err = "cannot open " + path + " non-cached"; return false; }
  #endif
    void* block = nullptr;
    if (::posix_memalign(&block, kBlock, aligned) != 0) block = nullptr;
    std::size_t total = 0;
    while (block != nullptr && total < aligned) {
        const ssize_t got = ::pread(file, static_cast<std::uint8_t*>(block) + total, aligned - total, static_cast<off_t>(total));
        if (got <= 0) break;
        total += static_cast<std::size_t>(got);
        if (total % kBlock != 0) break; // end of file: a retry would start mid-block
    }
    ::close(file);
    const bool ok = block != nullptr && total >= bytes;
    if (ok) out.assign(static_cast<const std::uint8_t*>(block), static_cast<const std::uint8_t*>(block) + bytes);
    std::free(block);
#endif
    if (!ok) err = path + ": non-cached read failed or file too short";
    return ok;
}

} // namespace sub0
