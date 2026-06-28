// tokmap.hpp -- read-only memory map over a corpus.tok ("S0TK") token stream.
//
// Training reads the tokenized corpus by RANDOM windows, so the natural representation
// is a read-only memory map rather than a heap vector: the OS pages the stream in on
// demand and reclaims it under pressure, so the corpus may far exceed RAM without ever
// being fully resident -- and without a giant allocation (the mapping is file-backed
// virtual memory, which is exactly the out-of-core story for a FineWeb-scale corpus).
//
// File layout (written by sub0-configure), little-endian throughout:
//   "S0TK" (legacy):  u32 magic 'S0TK', u32 vocab, u32 ntok, then ntok int32 token ids.
//   "S0TD" (doc-aware): u32 magic 'S0TD', u32 vocab, u32 ntok, u32 ndoc, then ntok int32 token
//                       ids, then ndoc u32 document-start token indices (document 0 starts at 0).
// The id array begins 4-byte aligned and the mapped view base is page-aligned, so reading it as
// `const int*` (and the trailing index as `const u32*`) is sound. The doc index lets training
// keep each window inside one document; a legacy 'S0TK' file simply exposes an empty index.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

#if defined(_WIN32)
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <fcntl.h>
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace sub0 {

class TokMap {
public:
    enum class Err { Ok, Missing, BadMagic, Truncated };

    explicit TokMap(const std::string& path) { open(path); }
    ~TokMap() { close(); }
    TokMap(const TokMap&) = delete;
    TokMap& operator=(const TokMap&) = delete;

    Err  error() const { return err_; }
    bool ok()    const { return err_ == Err::Ok; }
    int  vocab() const { return vocab_; }
    std::span<const int> tokens() const { return {data_, count_}; }
    // Ascending document-start token indices (document 0 at index 0). Empty for a legacy 'S0TK'
    // file or when no boundary table was written; callers then sample windows without boundary
    // awareness.
    std::span<const std::uint32_t> doc_starts() const { return {docs_, ndoc_}; }

private:
    static constexpr std::uint32_t MAGIC     = 0x4B543053u;  // "S0TK" (legacy, no doc index)
    static constexpr std::uint32_t MAGIC_DOC = 0x44543053u;  // "S0TD" (doc-aware)

    const int*           data_  = nullptr;
    std::size_t          count_ = 0;
    const std::uint32_t* docs_  = nullptr;
    std::size_t          ndoc_  = 0;
    int                  vocab_ = 0;
    Err                  err_   = Err::Missing;

    // Validate the header against the mapped bytes and point data_ (+ docs_ for 'S0TD') at the
    // id array (and trailing document index).
    void parse(std::uint64_t filesize) {
        const auto* u = reinterpret_cast<const std::uint32_t*>(base_);
        if (filesize < 12)  { err_ = Err::Truncated; return; }
        const std::uint32_t magic = u[0];
        if (magic != MAGIC && magic != MAGIC_DOC) { err_ = Err::BadMagic; return; }
        vocab_ = static_cast<int>(u[1]);
        const std::uint32_t ntok = u[2];
        const bool          doc  = (magic == MAGIC_DOC);
        const std::uint64_t header = doc ? 16 : 12;
        const std::uint32_t ndoc = doc ? u[3] : 0u;
        const std::uint64_t tok_bytes = static_cast<std::uint64_t>(ntok) * sizeof(int);
        const std::uint64_t doc_bytes = static_cast<std::uint64_t>(ndoc) * sizeof(std::uint32_t);
        if (filesize < header + tok_bytes + doc_bytes) { err_ = Err::Truncated; return; }
        const char* p = static_cast<const char*>(base_) + header;
        data_  = reinterpret_cast<const int*>(p);
        count_ = ntok;
        if (doc) {
            docs_ = reinterpret_cast<const std::uint32_t*>(p + tok_bytes);
            ndoc_ = ndoc;
        }
        err_ = Err::Ok;
    }

#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE map_  = nullptr;
    void*  base_ = nullptr;

    void open(const std::string& path) {
        file_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) { err_ = Err::Missing; return; }
        LARGE_INTEGER sz{};
        if (!GetFileSizeEx(file_, &sz)) { err_ = Err::Truncated; return; }
        map_ = CreateFileMappingA(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_) { err_ = Err::Truncated; return; }
        base_ = MapViewOfFile(map_, FILE_MAP_READ, 0, 0, 0);
        if (!base_) { err_ = Err::Truncated; return; }
        parse(static_cast<std::uint64_t>(sz.QuadPart));
    }
    void close() {
        if (base_) UnmapViewOfFile(base_);
        if (map_)  CloseHandle(map_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        base_ = nullptr; map_ = nullptr; file_ = INVALID_HANDLE_VALUE;
    }
#else
    int         fd_     = -1;
    void*       base_   = MAP_FAILED;
    std::size_t maplen_ = 0;

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) { err_ = Err::Missing; return; }
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) { err_ = Err::Truncated; return; }
        maplen_ = static_cast<std::size_t>(st.st_size);
        base_ = ::mmap(nullptr, maplen_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (base_ == MAP_FAILED) { err_ = Err::Truncated; return; }
        parse(static_cast<std::uint64_t>(st.st_size));
    }
    void close() {
        if (base_ != MAP_FAILED) ::munmap(base_, maplen_);
        if (fd_ >= 0) ::close(fd_);
        base_ = MAP_FAILED; fd_ = -1; maplen_ = 0;
    }
#endif
};

}  // namespace sub0
