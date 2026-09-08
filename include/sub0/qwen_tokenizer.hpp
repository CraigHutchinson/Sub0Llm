// sub0/qwen_tokenizer.hpp -- a from-scratch reimplementation of the REAL Qwen3.8-Flash-Next
// tokenizer: byte-level BPE, reading the real model's own vocab.json / merges.txt /
// tokenizer_config.json at runtime.
//
// WHY A SECOND TOKENIZER. sub0/tokenizer.hpp is this project's OWN from-scratch BPE scheme, trained
// on this project's own corpora, with its own marker set and its own vocabulary. The Qwen weights
// WP4 transplanted into this engine have no relationship whatsoever to those token ids -- feeding a
// sub0 token id to the real model is meaningless. Interactive inference against the real model
// therefore needs the real model's own tokenizer, which is what this is. Nothing here touches
// tokenizer.hpp/casing.hpp, and no non-Qwen build compiles differently because of it.
//
// WHAT THE REFERENCE ACTUALLY IS (verified this pass, not recalled -- docs/QWEN_TOKENIZER.md sec 2):
//   text -> [split out the 33 added/special tokens, matched LITERALLY on the raw bytes]
//        -> NFC normalize each remaining span
//        -> split into chunks with the pre-tokenization regex quoted in pretokenize() below
//        -> map every byte of each chunk through the GPT-2 bytes_to_unicode alphabet
//        -> greedy BPE by merge rank, per chunk, no merging across chunks
//        -> vocab lookup
// There is no unknown token (`unk_token = None`), no BOS/EOS insertion (the post-processor is a
// plain ByteLevel), and no prefix space (`add_prefix_space = false`).
//
// SCOPE. The tokenizer primitive only: text in, real Qwen ids out, and back. NOT a chat-template /
// prompt-formatting layer, and deliberately not yet wired into gen_stage.cpp -- both are separate
// later stages (AGENTS.md sec 8: land the stage that is actually wired up).
//
// Engine-free: this header and its .cpp depend on std + simdjson + sub0/qwen_unicode.hpp only, no
// generated config and no engine, so it lives in sub0_frontend and is unit-testable without a
// compiled model -- the same seam gguf.hpp and transplant.hpp already use.

#pragma once

#include "sub0/qwen_unicode.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sub0::qwen_tok {

// --- the GPT-2 byte alphabet -------------------------------------------------------------------
// Byte-level BPE runs over TEXT, not bytes, so every one of the 256 byte values is given a distinct
// printable-ish codepoint. The 188 bytes that are already printable ASCII/Latin-1 map to themselves;
// the remaining 68 (C0 controls, space, DEL, C1, U+00AD) map to U+0100.. in increasing byte order.
// This is the classic `bytes_to_unicode()` from the GPT-2 reference, reproduced exactly -- it is why
// a space shows up as "G-with-dot" in the vocabulary and why nothing is ever out-of-vocabulary.
struct ByteAlphabet {
    std::array<std::uint32_t, 256> to_cp{};   // byte -> codepoint
    std::array<int, 0x180>         to_byte{}; // codepoint -> byte, or -1 (only 0..0x17F is populated)
};
inline const ByteAlphabet& byte_alphabet() {
    static const ByteAlphabet a = [] {
        ByteAlphabet t;
        t.to_byte.fill(-1);
        bool direct[256] = {};
        for (int b = '!'; b <= '~'; ++b)      direct[b] = true;
        for (int b = 0xA1; b <= 0xAC; ++b)    direct[b] = true;
        for (int b = 0xAE; b <= 0xFF; ++b)    direct[b] = true;
        std::uint32_t next = 256;
        for (int b = 0; b < 256; ++b) t.to_cp[static_cast<std::size_t>(b)] =
            direct[b] ? static_cast<std::uint32_t>(b) : next++;
        for (int b = 0; b < 256; ++b) t.to_byte[t.to_cp[static_cast<std::size_t>(b)]] = b;
        return t;
    }();
    return a;
}

// --- pre-tokenization --------------------------------------------------------------------------
// Split `text` exactly the way the reference's Split pre-tokenizer does, appending each chunk (a
// view INTO `text`) to `out`. `out` is cleared first.
//
// THE REGEX, verbatim from the real model's own tokenizer.json pre_tokenizer (and from
// tokenizer_config.json's `pretokenize_regex`), quoted here per AGENTS.md sec 5:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// It cannot be handed to std::regex: no standard C++ regex implementation supports the Unicode
// property escapes (\p{L}, \p{N}, \p{M}) or the negative lookahead `(?!\S)`. It is therefore
// transliterated by hand, one function per alternative, in the alternation's own order, with the
// reference engine's leftmost-first-with-backtracking semantics reproduced explicitly (each
// alternative returns the exact extent its own backtracking would settle on -- see the per-clause
// comments in the .cpp). The character classes come from sub0/qwen_unicode.hpp, whose tables were
// checked against the real engine for every non-surrogate codepoint.
//
// Two traps this deliberately does NOT fall into, both confirmed against the reference:
//   * `\s` is the full Unicode White_Space property, not the ASCII six -- U+00A0 and U+3000 split
//     like spaces, not like symbols.
//   * `(?i:'s)` uses Unicode case folding, so it also matches `'` + U+017F LATIN SMALL LETTER LONG
//     S. That single non-ASCII fold is real; there are no others among s/t/r/e/v/m/l/d.
void pretokenize(std::string_view text, std::vector<std::string_view>& out);

// --- the tokenizer ------------------------------------------------------------------------------
class Tokenizer {
public:
    // Load the real model's own three files. Returns false and fills `error` (when non-null) on any
    // problem; a Tokenizer that failed to load stays unusable (loaded() == false) rather than
    // half-initialised.
    //
    // Three files rather than the single unified tokenizer.json, deliberately (docs sec 4):
    // merges.txt needs no JSON at all and makes "rank == line number" explicit; vocab.json is the
    // simplest possible simdjson shape (a flat string->int object); and tokenizer_config.json is the
    // authoritative source for the added-token table. It also avoids depending on tokenizer.json's
    // schema, which is a moving target across `tokenizers` releases (merges have been both a list of
    // strings and a list of pairs).
    bool load(const std::string& vocab_json_path,
              const std::string& merges_txt_path,
              const std::string& tokenizer_config_json_path,
              std::string* error = nullptr);

    bool loaded() const { return base_n_ > 0; }

    // Fill-into overload: a generation loop re-encodes a prompt per turn, so the caller keeps the
    // buffer (AGENTS.md sec 1 -- not the compute hot path, but no reason to churn either).
    void             encode(std::string_view text, std::vector<int>& out) const;
    std::vector<int> encode(std::string_view text) const;

    // Ids with no vocabulary entry are SKIPPED, not rendered. The real model's VOCAB axis is 248,320
    // while the tokenizer defines 248,077 tokens: rows 248,077..248,319 are padding the checkpoint
    // carries and sampling can legitimately land on, so decode has to survive them.
    std::string decode(std::span<const int> ids) const;

    int base_vocab_size() const { return base_n_; }        // BPE vocabulary alone (248,044)
    int vocab_size() const { return base_n_ + added_n_; }  // plus the added tokens (248,077)
    int eos_id() const { return eos_; }                    // <|im_end|>     (tokenizer_config eos_token)
    int pad_id() const { return pad_; }                    // <|endoftext|>  (tokenizer_config pad_token)

private:
    // Token text (byte-alphabet form) and decoded raw bytes, both as one blob + offsets rather than
    // ~248k separate std::strings: same content, a fraction of the allocator traffic and footprint.
    std::string                tok_blob_, dec_blob_;
    std::vector<std::uint32_t> tok_off_, dec_off_;              // size == count + 1
    std::unordered_map<std::string_view, int> id_of_;           // views into tok_blob_

    // (left id << 32 | right id) -> (rank, merged id). Keying on IDS, not on strings, is what makes
    // the merge loop allocation- and hash-of-string-free; every merge rule's result is itself a
    // vocabulary entry, so every symbol always has an id.
    struct Merge { int rank, id; };
    std::unordered_map<std::uint64_t, Merge> merges_;

    std::array<int, 256> byte_id_{};        // byte -> id of its single-character alphabet token

    struct Added { std::string content; int id; };
    std::vector<Added>   added_;            // longest content first, so matching is leftmost-longest
    std::array<bool, 256> added_first_{};   // fast reject: can an added token start with this byte?

    int base_n_ = 0, added_n_ = 0, eos_ = -1, pad_ = -1;

    void bpe_chunk(std::string_view chunk, std::vector<int>& out) const;
    void encode_span(std::string_view text, std::vector<int>& out) const;
};

}  // namespace sub0::qwen_tok
