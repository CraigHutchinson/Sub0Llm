// sub0/file_map.hpp -- a read-only, whole-file memory mapping with RAII lifetime.
//
// WHY THIS EXISTS (docs/WP4_SCOPE.md WP5b). `moeq::Store` read its entire payload into an owned
// buffer. At the 4-layer sub-stack that payload is 3.17 GiB and the choice was invisible; at the real
// 48 layers it is ~38 GiB of encoded routed experts, and an eager read of that alongside an 18.3 GiB
// f32 backbone and a 14 GiB activation arena does not fit in this machine's ~48 GiB of free RAM. A
// mapping fixes that without changing a single line above the accessor, because a forward pass touches
// only the experts it actually routes to: the pages it never reads are never faulted in, and the pages
// it does read are file-backed and evictable rather than committed private bytes.
//
// WHY A HEADER OF ITS OWN, rather than a Win32 call site inside moe_quant.hpp. There was already one
// hand-rolled mapping in this repo -- `TokMap` (include/sub0/tokmap.hpp), which inlines
// CreateFileMapping/MapViewOfFile and their POSIX counterparts into its own private section. Adding a
// SECOND copy of that platform code is exactly the duplication AGENTS.md S10 says to remove the class
// of rather than fix the instance of, so the mapping mechanics live here once, with RAII lifetime
// (AGENTS.md's `[[raii-cleanup-preferred-over-manual]]`) and a single error channel. `TokMap` predates
// this header and is deliberately left alone in this pass: it is on the training path, its own
// open/close is already exercised by the corpus tests, and converting it is a behaviour-neutral
// refactor of a hot, unrelated file -- worth doing, but not inside a work package whose gate is a
// bitwise-identity claim about something else.
//
// SCOPE, deliberately narrow (AGENTS.md S8): read-only, whole file, no partial views, no writes, no
// flush, no madvise/prefetch hints. Those are all real things a mapping CAN do and none of them has a
// consumer today. `size()` is the file's own length, so a caller still has to bounds-check its own
// structures against it -- the mapping guarantees the bytes are addressable, not that they mean
// anything.
//
// ONE PLATFORM NOTE WORTH STATING RATHER THAN DISCOVERING: a zero-length file cannot be mapped on
// either platform (CreateFileMapping refuses a 0-byte maximum size; mmap refuses a 0 length). That is
// reported as a normal failure with a message, not as a special empty-mapping success, because every
// consumer here wants a header out of the first bytes and an empty file cannot supply one.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#if defined(_WIN32)
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace sub0 {

// Move-only: a mapping is a unique resource, and copying one would either double-unmap or silently
// share a handle whose lifetime nothing owns.
class FileMap {
public:
    FileMap() = default;
    ~FileMap() { close(); }
    FileMap(const FileMap&) = delete;
    FileMap& operator=(const FileMap&) = delete;
    FileMap(FileMap&& o) noexcept { swap(o); }
    FileMap& operator=(FileMap&& o) noexcept {
        if (this != &o) { close(); swap(o); }
        return *this;
    }

    // Maps `path` read-only in its entirety. Returns false and fills `err` on any failure, leaving the
    // object empty -- never partially initialized, the same contract moeq::Store::open has.
    bool open(const std::string& path, std::string& err) {
        close();
#if defined(_WIN32)
        file_ = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) { err = "cannot open " + path; close(); return false; }
        LARGE_INTEGER sz{};
        if (!::GetFileSizeEx(file_, &sz) || sz.QuadPart <= 0) {
            err = path + ": cannot size the file, or it is empty";
            close();
            return false;
        }
        map_ = ::CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (map_ == nullptr) { err = path + ": CreateFileMapping failed"; close(); return false; }
        base_ = ::MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
        if (base_ == nullptr) { err = path + ": MapViewOfFile failed"; close(); return false; }
        size_ = static_cast<std::size_t>(sz.QuadPart);
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { err = "cannot open " + path; close(); return false; }
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
            err = path + ": cannot size the file, or it is empty";
            close();
            return false;
        }
        size_ = static_cast<std::size_t>(st.st_size);
        void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (p == MAP_FAILED) { err = path + ": mmap failed"; close(); return false; }
        base_ = p;
#endif
        return true;
    }

    void close() {
#if defined(_WIN32)
        if (base_ != nullptr) ::UnmapViewOfFile(base_);
        if (map_ != nullptr) ::CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) ::CloseHandle(file_);
        base_ = nullptr;
        map_ = nullptr;
        file_ = INVALID_HANDLE_VALUE;
#else
        if (base_ != nullptr) ::munmap(base_, size_);
        if (fd_ >= 0) ::close(fd_);
        base_ = nullptr;
        fd_ = -1;
#endif
        size_ = 0;
    }

    bool mapped() const { return base_ != nullptr; }
    const std::uint8_t* data() const { return static_cast<const std::uint8_t*>(base_); }
    std::size_t size() const { return size_; }

private:
    void swap(FileMap& o) noexcept {
        std::swap(base_, o.base_);
        std::swap(size_, o.size_);
#if defined(_WIN32)
        std::swap(file_, o.file_);
        std::swap(map_, o.map_);
#else
        std::swap(fd_, o.fd_);
#endif
    }

    void*       base_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE map_  = nullptr;
#else
    int fd_ = -1;
#endif
};

}  // namespace sub0
