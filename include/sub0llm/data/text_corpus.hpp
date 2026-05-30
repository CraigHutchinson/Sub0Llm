#pragma once
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/core/tensor.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace sub0llm {

// Streaming token corpus for pretraining.
//
// Tokenises all documents once, concatenates into a flat buffer (EOS between docs),
// then yields (ids, targets) Tensor pairs of shape (seq_len,) as sliding windows.
// Windows are served in shuffled order (shuffled at sample-index level).
class TextCorpus {
public:
    using TokenId = int32_t;

    // Build from in-memory texts.
    TextCorpus(std::vector<std::string> texts, const BPETokenizer& tok,
               int64_t seq_len, uint32_t seed = 42);

    // Load from file path.  Lines starting with '{' -> JSONL ("text" field extracted).
    // Otherwise each line is a document.
    [[nodiscard]] static TextCorpus from_file(const std::string& path,
                                               const BPETokenizer& tok,
                                               int64_t seq_len, uint32_t seed = 42);

    // Fill ids (shape seq_len) and tgt (shape seq_len) for the next sample.
    // Returns false when all samples have been served (call reset() to restart).
    bool next_sample(Tensor& ids, Tensor& tgt);

    // Restart, optionally re-shuffling with a new seed.
    void reset(uint32_t seed = 42);

    [[nodiscard]] int64_t     total_tokens()    const noexcept;
    [[nodiscard]] int64_t     tokens_consumed() const noexcept;
    [[nodiscard]] std::size_t n_samples()       const noexcept;

private:
    std::vector<TokenId>     tokens_;   // flat token buffer (newline-separated docs)
    std::vector<std::size_t> order_;    // shuffled sample indices
    int64_t                  seq_len_;
    std::size_t              cursor_;   // next index into order_
    int64_t                  tokens_consumed_{0};

    void build_order(uint32_t seed);
};

} // namespace sub0llm
