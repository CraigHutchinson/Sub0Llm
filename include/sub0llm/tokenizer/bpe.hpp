#pragma once
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sub0llm {

// ── BPETokenizer ──────────────────────────────────────────────────────────────
//
// Byte-level Byte-Pair Encoding tokenizer.
//
// Design mirrors GPT-2's tokenizer:
//   • Vocabulary starts from the 256 raw byte types.
//   • BPE merge rules extend the vocab with subword units.
//   • Special tokens (<|endoftext|> etc.) are added on top.
//   • Text is pre-tokenized by splitting on whitespace/punctuation before
//     applying BPE within each word.  GPT-2's full regex pre-tokenizer is
//     added in Ch08 when we load real GPT-2 weights.
//
// Two construction paths:
//   1. BPETokenizer::train()  — learn merges from a corpus (Ch03)
//   2. BPETokenizer::load()   — load from GPT-2 JSON + merges.txt (Ch08+)

class BPETokenizer {
public:
    using TokenId  = std::int32_t;
    using MergePair = std::pair<std::string, std::string>;

    // ── Factory ───────────────────────────────────────────────────────────────

    // Learn BPE on `corpus` until `vocab_size` merges have been performed.
    // The starting vocabulary is the set of unique bytes found in the corpus.
    [[nodiscard]] static BPETokenizer train(
        const std::vector<std::string>& corpus,
        std::size_t vocab_size);

    // Load from GPT-2 format: vocab.json (token → id) + merges.txt (pairs).
    [[nodiscard]] static BPETokenizer load(
        const std::filesystem::path& vocab_json,
        const std::filesystem::path& merges_txt);

    // Build from in-memory data (e.g. extracted from a GGUF file).
    //   id_to_token: index is token id, value is token string
    //   merges:      each entry is "A B" (space-separated pair), in priority order
    [[nodiscard]] static BPETokenizer from_vocab(
        const std::vector<std::string>& id_to_token,
        const std::vector<std::string>& merges);

    // ── Encode / decode ───────────────────────────────────────────────────────

    // Convert a string to a sequence of token IDs.
    [[nodiscard]] std::vector<TokenId> encode(std::string_view text) const;

    // Convert a sequence of token IDs back to a string.
    [[nodiscard]] std::string decode(std::span<const TokenId> ids) const;

    // Convenience overload.
    [[nodiscard]] std::string decode(const std::vector<TokenId>& ids) const {
        return decode(std::span<const TokenId>{ids});
    }

    // ── Vocabulary info ───────────────────────────────────────────────────────
    [[nodiscard]] std::size_t vocab_size()  const noexcept { return id_to_token_.size(); }
    [[nodiscard]] std::size_t num_merges()  const noexcept { return merges_.size(); }

    // Returns the token string for an id, or throws if out of range.
    [[nodiscard]] std::string_view token_str(TokenId id) const;

    // Returns the id for a token string, or -1 if not in vocabulary.
    [[nodiscard]] TokenId token_id(std::string_view tok) const;

    // Special token id, or -1 if not present.
    [[nodiscard]] TokenId eos_id() const noexcept { return eos_id_; }
    [[nodiscard]] TokenId bos_id() const noexcept { return bos_id_; }
    [[nodiscard]] TokenId pad_id() const noexcept { return pad_id_; }
    [[nodiscard]] TokenId unk_id() const noexcept { return unk_id_; }

    // Add a special token (does nothing if already present).
    TokenId add_special_token(std::string token);

    // ── Serialisation ─────────────────────────────────────────────────────────
    void save(const std::filesystem::path& dir) const;

private:
    std::vector<std::string>                id_to_token_;  // id → token string
    std::unordered_map<std::string, TokenId> vocab_;        // token string → id
    std::vector<MergePair>                  merges_;        // ordered merge rules

    TokenId eos_id_{-1};
    TokenId bos_id_{-1};
    TokenId pad_id_{-1};
    TokenId unk_id_{-1};

    // True for GPT-2/Qwen-style byte-level vocabularies (set by from_vocab):
    // every token char is a remapped byte (e.g. newline→'Ċ', space→'Ġ'), so
    // decode must reverse the byte↔unicode mapping rather than emit the raw
    // token text. Trained/loaded vocabularies leave this false.
    bool    byte_level_{false};

    // Apply one round of BPE to a sequence of tokens.
    // Returns true if any merge was performed.
    [[nodiscard]] static bool apply_merges(
        std::vector<std::string>& tokens,
        const std::unordered_map<std::string, int>& merge_rank);

    // Tokenize a single pre-tokenized word using BPE merge rules.
    [[nodiscard]] std::vector<TokenId> encode_word(std::string_view word) const;

    // Register a new token; returns its id.
    TokenId add_token(std::string token);
};

} // namespace sub0llm
