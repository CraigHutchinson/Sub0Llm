#include "sub0llm/data/dataset.hpp"

#include <format>
#include <stdexcept>

// POSIX mmap
#if defined(__unix__) || defined(__APPLE__)
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  define SUB0LLM_HAVE_MMAP 1
#else
#  define SUB0LLM_HAVE_MMAP 0
#endif

#include <fstream>

namespace sub0llm {

// ── MappedFile ────────────────────────────────────────────────────────────────

MappedFile::MappedFile(const std::filesystem::path& path) {
#if SUB0LLM_HAVE_MMAP
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ == -1)
        throw std::runtime_error(
            std::format("MappedFile: cannot open '{}'", path.string()));

    struct stat sb{};
    if (::fstat(fd_, &sb) == -1) {
        ::close(fd_);
        throw std::runtime_error(
            std::format("MappedFile: fstat failed for '{}'", path.string()));
    }
    size_ = static_cast<std::size_t>(sb.st_size);

    if (size_ == 0) {
        // Empty file — no mapping needed; data_ stays nullptr.
        mmaped_ = false;
        return;
    }

    void* ptr = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (ptr == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error(
            std::format("MappedFile: mmap failed for '{}'", path.string()));
    }
    data_   = static_cast<const char*>(ptr);
    mmaped_ = true;
#else
    // Fallback: read entire file into memory.
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        throw std::runtime_error(
            std::format("MappedFile: cannot open '{}'", path.string()));
    size_ = static_cast<std::size_t>(f.tellg());
    f.seekg(0);
    fallback_.resize(size_);
    f.read(fallback_.data(), static_cast<std::streamsize>(size_));
    data_ = fallback_.data();
#endif
}

MappedFile::~MappedFile() {
#if SUB0LLM_HAVE_MMAP
    if (mmaped_ && data_)
        ::munmap(const_cast<char*>(data_), size_);
    if (fd_ != -1)
        ::close(fd_);
#endif
}

// ── TextDataset ───────────────────────────────────────────────────────────────

TextDataset::TextDataset(std::vector<TokenId> tokens,
                         std::size_t          context_length,
                         std::size_t          stride)
    : tokens_(std::move(tokens))
    , context_length_(context_length)
    , stride_(stride == 0 ? 1 : stride)
{
    if (context_length_ == 0)
        throw std::runtime_error("TextDataset: context_length must be > 0");

    // Each window needs context_length + 1 tokens (input + one extra for target).
    const std::size_t window = context_length_ + 1;
    if (tokens_.size() < window) {
        num_samples_ = 0;
    } else {
        // Number of stride-separated windows that fit.
        num_samples_ = (tokens_.size() - window) / stride_ + 1;
    }
}

TextDataset::Sample TextDataset::operator[](std::size_t idx) const {
    if (idx >= num_samples_)
        throw std::out_of_range(
            std::format("TextDataset: index {} out of range [0,{})",
                idx, num_samples_));

    const std::size_t offset = idx * stride_;
    Sample s;
    s.input_ids.assign(tokens_.begin() + static_cast<std::ptrdiff_t>(offset),
                       tokens_.begin() + static_cast<std::ptrdiff_t>(offset + context_length_));
    s.target_ids.assign(tokens_.begin() + static_cast<std::ptrdiff_t>(offset + 1),
                        tokens_.begin() + static_cast<std::ptrdiff_t>(offset + context_length_ + 1));
    return s;
}

std::span<const TextDataset::TokenId>
TextDataset::input_span(std::size_t idx) const {
    if (idx >= num_samples_)
        throw std::out_of_range(
            std::format("TextDataset: index {} out of range [0,{})",
                idx, num_samples_));
    const std::size_t offset = idx * stride_;
    return {tokens_.data() + offset, context_length_};
}

std::span<const TextDataset::TokenId>
TextDataset::target_span(std::size_t idx) const {
    if (idx >= num_samples_)
        throw std::out_of_range(
            std::format("TextDataset: index {} out of range [0,{})",
                idx, num_samples_));
    const std::size_t offset = idx * stride_ + 1;
    return {tokens_.data() + offset, context_length_};
}

} // namespace sub0llm
