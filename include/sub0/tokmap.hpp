// tokmap.hpp -- read/write module for the tokenized corpus (corpus.tok).
//
// Training reads the tokenized corpus by RANDOM windows, so the natural representation is a read-only
// memory map rather than a heap vector: the OS pages the stream in on demand and reclaims it under
// pressure, so the corpus may far exceed RAM without ever being fully resident -- the out-of-core
// story for a FineWeb-scale corpus.
//
// File layout, little-endian throughout.
//   v2 "S0T2" (current):
//     u32 magic 'S0T2', u32 vocab, u32 bits_per_token, u32 flags,
//     u64 ntok, u64 ndoc,                                   (32-byte header)
//     ntok tokens, each (bits_per_token/8) bytes little-endian,   [byte-aligned: 2/3/4 bytes]
//     ndoc u64 document-start token indices (document 0 starts at 0).
//   Legacy "S0TK"/"S0TD": u32 magic, u32 vocab, u32 ntok[, u32 ndoc], then ntok INT32 ids
//     [, ndoc u32 doc-starts]. Read-only support so old builds' .tok still loads.
//
// Why v2: (1) u64 ntok/ndoc -- a 46 GB corpus is ~14B tokens, which OVERFLOWS the legacy u32 count
// (~4.29B) and silently truncates. (2) bits_per_token -- a 32K/64K vocab needs only 16 bits, so
// int32 wastes half the bytes; the width is chosen from the vocab (16/24/32 byte-aligned). The
// `flags` word reserves room for a future SUB-BYTE bit-packed layout (e.g. 20 bits/token) -- callers
// go through TokView, so only this file changes when that lands.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ostream>
#include <span>
#include <string>
#include <vector>

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

// Minimal byte-aligned token width for a vocabulary (ids 0..vocab-1). 2 bytes covers up to 65536,
// 3 up to ~16.7M, 4 beyond -- so the common 16K-64K vocab lands at 2 bytes (half of int32).
inline int tok_bytes_for_vocab(long long vocab) {
    return vocab <= 0x10000 ? 2 : (vocab <= 0x1000000 ? 3 : 4);
}

// A width-agnostic READ view over a token array: `bpt` bytes per token, little-endian. operator[]
// assembles the id, so callers are indifferent to the on-disk width (and to a future bit-packed
// layout, which would specialize this without touching them). Supports the span-like slicing the
// training data path uses (first / subspan).
struct TokView {
    const std::uint8_t* p = nullptr;
    std::size_t         n = 0;
    int                 bpt = 4;

    std::size_t size()  const { return n; }
    bool        empty() const { return n == 0; }
    int operator[](std::size_t i) const {
        const std::uint8_t* q = p + i * static_cast<std::size_t>(bpt);
        switch (bpt) {
            case 2:  return q[0] | (q[1] << 8);
            case 3:  return q[0] | (q[1] << 8) | (q[2] << 16);
            case 4:  return static_cast<int>(static_cast<std::uint32_t>(q[0]) | (static_cast<std::uint32_t>(q[1]) << 8)
                                             | (static_cast<std::uint32_t>(q[2]) << 16) | (static_cast<std::uint32_t>(q[3]) << 24));
            default: return q[0];
        }
    }
    TokView first(std::size_t m) const { return {p, std::min(m, n), bpt}; }
    TokView subspan(std::size_t off) const {
        const std::size_t o = std::min(off, n);
        return {p + o * static_cast<std::size_t>(bpt), n - o, bpt};
    }
    TokView subspan(std::size_t off, std::size_t len) const {
        const std::size_t o = std::min(off, n);
        return {p + o * static_cast<std::size_t>(bpt), std::min(len, n - o), bpt};
    }
    // A TokView over an int32 heap buffer (the on-demand path shares the training data interface).
    static TokView over_int32(const int* data, std::size_t count) {
        return {reinterpret_cast<const std::uint8_t*>(data), count, 4};
    }
    // Materialize n tokens from `off` into a contiguous int32 buffer -- for feeding the CPU engine's
    // forward(const int*, T), which needs a raw id array regardless of the on-disk width.
    void copy_to(std::size_t off, std::size_t n, int* dst) const {
        for (std::size_t i = 0; i < n; ++i) dst[i] = (*this)[off + i];
    }
};

// Pack `count` int ids from `src` into `dst` at `bpt` bytes each (little-endian, byte-aligned).
inline void tok_pack(std::uint8_t* dst, const std::int32_t* src, std::size_t count, int bpt) {
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint32_t v = static_cast<std::uint32_t>(src[i]);
        std::uint8_t* q = dst + i * static_cast<std::size_t>(bpt);
        for (int b = 0; b < bpt; ++b) q[b] = static_cast<std::uint8_t>(v >> (8 * b));
    }
}

// Streaming writer for a v2 corpus.tok: header (with a placeholder count), append packed token
// batches, then finish() writes the u64 doc index and back-patches ntok/ndoc. Keeps corpus.tok
// out-of-core on the write side too (nothing but the current batch is held).
class TokWriter {
public:
    static constexpr std::uint32_t MAGIC = 0x32543053u;   // "S0T2"

    TokWriter(std::ostream& os, int vocab) : os_(os), bpt_(tok_bytes_for_vocab(vocab)) {
        put32(MAGIC); put32(static_cast<std::uint32_t>(vocab)); put32(static_cast<std::uint32_t>(bpt_ * 8)); put32(0u);
        put64(0); put64(0);                               // ntok, ndoc -- patched in finish()
    }
    int  bytes_per_token() const { return bpt_; }
    // Append a batch of ids (packs to bpt_ bytes each). `scratch` is reused across calls.
    void append(const std::int32_t* ids, std::size_t count, std::vector<std::uint8_t>& scratch) {
        scratch.resize(count * static_cast<std::size_t>(bpt_));
        tok_pack(scratch.data(), ids, count, bpt_);
        os_.write(reinterpret_cast<const char*>(scratch.data()), static_cast<std::streamsize>(scratch.size()));
        ntok_ += count;
    }
    // Write the doc-start index (u64) and back-patch the counts. Call once, last.
    void finish(const std::vector<std::uint64_t>& doc_starts) {
        os_.write(reinterpret_cast<const char*>(doc_starts.data()),
                  static_cast<std::streamsize>(doc_starts.size() * sizeof(std::uint64_t)));
        os_.seekp(16, std::ios::beg); put64(ntok_);
        os_.seekp(24, std::ios::beg); put64(static_cast<std::uint64_t>(doc_starts.size()));
    }
    std::uint64_t token_count() const { return ntok_; }

private:
    void put32(std::uint32_t v) { os_.write(reinterpret_cast<const char*>(&v), sizeof v); }
    void put64(std::uint64_t v) { os_.write(reinterpret_cast<const char*>(&v), sizeof v); }
    std::ostream& os_;
    int           bpt_;
    std::uint64_t ntok_ = 0;
};

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
    int  bytes_per_token() const { return bpt_; }
    TokView tokens() const { return {data_, count_, bpt_}; }
    // Ascending document-start token indices (document 0 at index 0), materialized to u64 (small:
    // one entry per document). Empty for a legacy file with no boundary table.
    std::span<const std::uint64_t> doc_starts() const { return {docs_.data(), docs_.size()}; }

private:
    static constexpr std::uint32_t MAGIC_V2  = 0x32543053u;  // "S0T2"
    static constexpr std::uint32_t MAGIC     = 0x4B543053u;  // "S0TK" (legacy, no doc index)
    static constexpr std::uint32_t MAGIC_DOC = 0x44543053u;  // "S0TD" (legacy doc-aware)

    const std::uint8_t*        data_  = nullptr;
    std::size_t                count_ = 0;
    int                        bpt_   = 4;
    std::vector<std::uint64_t> docs_;
    int                        vocab_ = 0;
    Err                        err_   = Err::Missing;

    void parse(std::uint64_t filesize) {
        const auto* u = reinterpret_cast<const std::uint32_t*>(base_);
        if (filesize < 12) { err_ = Err::Truncated; return; }
        const std::uint32_t magic = u[0];
        const auto* b8 = static_cast<const std::uint8_t*>(base_);
        if (magic == MAGIC_V2) {                              // ---- v2 (u64 counts, width) ----
            if (filesize < 32) { err_ = Err::Truncated; return; }
            vocab_ = static_cast<int>(u[1]);
            bpt_   = static_cast<int>(u[2] / 8);
            if (bpt_ < 1 || bpt_ > 4) { err_ = Err::BadMagic; return; }
            std::uint64_t ntok = 0, ndoc = 0;
            std::memcpy(&ntok, b8 + 16, 8);
            std::memcpy(&ndoc, b8 + 24, 8);
            const std::uint64_t tok_bytes = ntok * static_cast<std::uint64_t>(bpt_);
            const std::uint64_t doc_bytes = ndoc * sizeof(std::uint64_t);
            if (filesize < 32 + tok_bytes + doc_bytes) { err_ = Err::Truncated; return; }
            data_  = b8 + 32;
            count_ = static_cast<std::size_t>(ntok);
            if (ndoc) {
                const auto* d = reinterpret_cast<const std::uint64_t*>(b8 + 32 + tok_bytes);
                docs_.assign(d, d + ndoc);
            }
            err_ = Err::Ok;
            return;
        }
        // ---- legacy S0TK / S0TD: u32 counts, int32 ids ----
        if (magic != MAGIC && magic != MAGIC_DOC) { err_ = Err::BadMagic; return; }
        vocab_ = static_cast<int>(u[1]);
        bpt_   = 4;
        const std::uint32_t ntok = u[2];
        const bool          doc  = (magic == MAGIC_DOC);
        const std::uint64_t header = doc ? 16 : 12;
        const std::uint32_t ndoc = doc ? u[3] : 0u;
        const std::uint64_t tok_bytes = static_cast<std::uint64_t>(ntok) * sizeof(std::int32_t);
        const std::uint64_t doc_bytes = static_cast<std::uint64_t>(ndoc) * sizeof(std::uint32_t);
        if (filesize < header + tok_bytes + doc_bytes) { err_ = Err::Truncated; return; }
        data_  = b8 + header;
        count_ = ntok;
        if (doc) {
            const auto* d = reinterpret_cast<const std::uint32_t*>(b8 + header + tok_bytes);
            docs_.assign(d, d + ndoc);                       // widen u32 -> u64
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
