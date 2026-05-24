#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace sub0llm {

// Raw file mapping (POSIX mmap with ifstream fallback).
// After construction, data() returns a read-only view of the file bytes.
class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path);
    ~MappedFile();

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&&)                 = delete;
    MappedFile& operator=(MappedFile&&)      = delete;

    [[nodiscard]] std::string_view data() const noexcept { return data_ ? std::string_view{data_, size_} : std::string_view{}; }
    [[nodiscard]] std::size_t      size() const noexcept { return size_; }

private:
    const char* data_ = nullptr;
    std::size_t size_ = 0;
    int         fd_   = -1;
    bool        mmaped_ = false;
    std::vector<char> fallback_; // used when mmap is unavailable
};

// A sliding-window dataset over a pre-tokenized integer sequence.
//
// Given token ids [t0 t1 t2 ... tN] and a context_length L, each sample is a
// consecutive window of length L+1; the first L tokens become input_ids and
// the last L tokens become target_ids (i.e. targets are inputs shifted by 1).
//
// stride controls how many tokens the window advances between samples:
//   stride = 1  → maximum overlap (default, maximises training data)
//   stride = L  → non-overlapping windows
class TextDataset {
public:
    using TokenId = int32_t;

    struct Sample {
        std::vector<TokenId> input_ids;
        std::vector<TokenId> target_ids;
    };

    // Build dataset from a pre-computed token id sequence.
    TextDataset(std::vector<TokenId> tokens,
                std::size_t          context_length,
                std::size_t          stride = 1);

    [[nodiscard]] std::size_t size()           const noexcept { return num_samples_; }
    [[nodiscard]] std::size_t context_length() const noexcept { return context_length_; }
    [[nodiscard]] bool        empty()          const noexcept { return num_samples_ == 0; }

    [[nodiscard]] Sample     operator[](std::size_t idx) const;

    // Direct span access into the token buffer (no copy).
    [[nodiscard]] std::span<const TokenId> input_span(std::size_t idx)  const;
    [[nodiscard]] std::span<const TokenId> target_span(std::size_t idx) const;

private:
    std::vector<TokenId> tokens_;
    std::size_t          context_length_;
    std::size_t          stride_;
    std::size_t          num_samples_;
};

} // namespace sub0llm
