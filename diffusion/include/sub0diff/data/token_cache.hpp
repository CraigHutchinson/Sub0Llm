#pragma once

// token_cache.hpp — cache the BPE-tokenized, split flat token streams to disk.
//
// BPE-encoding a large corpus (e.g. 146K paragraphs) is the dominant STARTUP cost and is
// DETERMINISTIC given a fixed (corpus, paragraph-limit, tokenizer). Caching the post-split
// train/eval streams to `ckpt_dir/tokens.bin` makes resumes and eval-only runs start near-
// instantly. The header fingerprints (corpus byte-size, paragraph limit, vocab size); any
// mismatch re-encodes and rewrites. (A vocab sweep is safe: each V uses its own ckpt_dir.)
// Reusable across training chapters that tokenize a corpus into flat streams.

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace sub0diff::data::tokcache {

constexpr std::uint32_t kMagic   = 0x53304454u;  // "S0DT"
constexpr std::uint32_t kVersion = 1u;

struct Header {
    std::uint32_t magic = kMagic, version = kVersion;
    std::uint64_t corpus_size = 0;
    std::int64_t  paragraphs = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t n_train = 0, n_eval = 0;
};

[[nodiscard]] inline bool load(const std::filesystem::path& path, std::uint64_t corpus_size,
                               std::int64_t paragraphs, std::uint64_t vocab_size,
                               std::vector<std::int32_t>& train_ids,
                               std::vector<std::int32_t>& eval_ids) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    Header h;
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!f || h.magic != kMagic || h.version != kVersion || h.corpus_size != corpus_size ||
        h.paragraphs != paragraphs || h.vocab_size != vocab_size)
        return false;
    train_ids.resize(h.n_train);
    eval_ids.resize(h.n_eval);
    if (h.n_train) f.read(reinterpret_cast<char*>(train_ids.data()),
                          static_cast<std::streamsize>(h.n_train * sizeof(std::int32_t)));
    if (h.n_eval)  f.read(reinterpret_cast<char*>(eval_ids.data()),
                          static_cast<std::streamsize>(h.n_eval * sizeof(std::int32_t)));
    return static_cast<bool>(f);
}

inline void save(const std::filesystem::path& path, std::uint64_t corpus_size,
                 std::int64_t paragraphs, std::uint64_t vocab_size,
                 const std::vector<std::int32_t>& train_ids,
                 const std::vector<std::int32_t>& eval_ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return;
    Header h{kMagic, kVersion, corpus_size, paragraphs, vocab_size,
             train_ids.size(), eval_ids.size()};
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    if (!train_ids.empty()) f.write(reinterpret_cast<const char*>(train_ids.data()),
                                    static_cast<std::streamsize>(train_ids.size() * sizeof(std::int32_t)));
    if (!eval_ids.empty())  f.write(reinterpret_cast<const char*>(eval_ids.data()),
                                    static_cast<std::streamsize>(eval_ids.size() * sizeof(std::int32_t)));
}

}  // namespace sub0diff::data::tokcache
