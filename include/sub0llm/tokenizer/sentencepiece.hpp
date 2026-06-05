#pragma once

// sentencepiece.hpp (Ch27) — SentencePiece (SPM) tokenizer for Gemma-family models.
//
// Unlike GPT-2 byte-level BPE (BPETokenizer), SPM has no merge list: it greedily
// merges adjacent symbols by the *score* of the resulting vocab piece (priority
// queue), with byte fallback for unknown pieces. Word boundaries use the ▁ marker
// (U+2581). Matches llama.cpp's `llm_tokenizer_spm` for the "gemma*" tokenizer.

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sub0llm {

class SPTokenizer {
public:
    // Build from a GGUF vocab: tokens + per-token scores + special-token ids.
    [[nodiscard]] static SPTokenizer from_vocab(
        const std::vector<std::string>& tokens,
        const std::vector<float>&       scores,
        int32_t bos_id, int32_t eos_id, bool add_bos);

    // Text → token ids (prepends BOS when the model requests it).
    [[nodiscard]] std::vector<int32_t> encode(std::string_view text) const;

    // Token ids → text (▁ → space, byte tokens → raw bytes).
    [[nodiscard]] std::string decode(const std::vector<int32_t>& ids) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return id_to_piece_.size(); }

private:
    std::vector<std::string>                  id_to_piece_;
    std::vector<float>                        scores_;
    std::unordered_map<std::string, int32_t>  piece_to_id_;
    int32_t                                   bos_id_  = -1;
    int32_t                                   eos_id_  = -1;
    bool                                      add_bos_ = false;
};

} // namespace sub0llm
