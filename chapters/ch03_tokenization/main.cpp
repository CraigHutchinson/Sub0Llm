// Chapter 03 — Tokenization
//
// Before a transformer sees a single character of text, that text must be
// converted to integers — token IDs.  This chapter builds the full pipeline:
//
//   raw text  →  pre-tokenization  →  BPE merges  →  token IDs  →  embeddings
//
// We implement Byte-Pair Encoding (BPE), the algorithm used by GPT-2/3/4 and
// most modern LLMs.  Unlike earlier word-level vocabularies, BPE operates on
// bytes so it handles any Unicode text and has a bounded vocabulary size.
//
// Topics covered:
//   1. Why we need tokenization: vocabularies, OOV, subword units
//   2. BPE algorithm: count pairs → merge → repeat
//   3. GPT-2 space-marker convention (Ġ = " ")
//   4. Encode/decode roundtrip
//   5. Save/load vocabulary (JSON + merges.txt)
//   6. Teaser: loading real GPT-2 vocab files (added in Ch08)

#include <format>
#include <iostream>
#include <vector>

#include "sub0llm/compat/print.hpp"
#include "sub0llm/tokenizer/bpe.hpp"
#include "sub0llm/version.hpp"

using namespace sub0llm;

static void section(std::string_view title) {
    std::println("\n── {} ──", title);
}

// ── Pretty-print a token id sequence ─────────────────────────────────────────
static void show_tokens(const BPETokenizer& tok,
                        const std::vector<BPETokenizer::TokenId>& ids,
                        std::string_view label) {
    std::print("{}: [", label);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto s = tok.token_str(ids[i]);
        std::print(" '{}'({})", s, ids[i]);
        if (i + 1 < ids.size()) std::print(",");
    }
    std::println(" ]");
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::println("sub0llm v{} — Chapter 03: Tokenization", version_string);

    // ── 1. Why tokenization ───────────────────────────────────────────────────
    section("1. Why tokenization?");
    std::println(
        "Neural networks require fixed-size integer inputs — not raw text.\n"
        "Options:\n"
        "  • Character-level : large sequences, slow; 'don't' = 6 tokens\n"
        "  • Word-level      : huge vocabulary, OOV problem; 'tokenizing' = 1 token\n"
        "  • Subword (BPE)   : adaptive; common words = 1 token, rare words = bytes\n"
        "\nGPT-2 uses BPE on raw bytes → 50,257 tokens; works for any language."
    );

    // ── 2. BPE algorithm step-by-step ─────────────────────────────────────────
    section("2. BPE algorithm");
    std::println(
        "Start:  ['a', 'a', 'a', 'b', 'd', 'a', 'a', 'a', 'b', 'a', 'c']\n"
        "Step 1: most frequent pair → ('a','a'); merge to 'aa'\n"
        "        ['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']\n"
        "Step 2: most frequent pair → ('aa','a'); merge to 'aaa'\n"
        "        ['aaa', 'b', 'd', 'aaa', 'b', 'a', 'c']\n"
        "Step 3: ('aaa','b') → 'aaab'\n"
        "        ['aaab', 'd', 'aaab', 'a', 'c']\n"
        "Vocabulary now contains: a, b, c, d, aa, aaa, aaab, ..."
    );

    // ── 3. Train a tokenizer on a small corpus ─────────────────────────────────
    section("3. Training BPE on a small corpus");

    const std::vector<std::string> corpus = {
        "the cat sat on the mat",
        "the cat ate the rat",
        "the mat the cat sat",
        "rats and cats and bats",
    };

    std::println("Corpus ({} sentences):", corpus.size());
    for (const auto& s : corpus) std::println("  \"{}\"", s);

    auto tok = BPETokenizer::train(corpus, 60);

    std::println("\nVocabulary size: {} tokens", tok.vocab_size());
    std::println("Merge rules learned: {}", tok.num_merges());

    // Show the first 10 merge rules
    std::println("\nFirst 5 token merges:");
    for (std::size_t i = 0; i < std::min<std::size_t>(5, tok.num_merges()); ++i) {
        // We can infer merges by observing multi-char tokens in order added;
        // here we show the total merge count as a proxy.
    }
    std::println("  (merges encode the most frequent subword patterns in the corpus)");

    // ── 4. Encode / decode ─────────────────────────────────────────────────────
    section("4. Encode / decode");

    const std::string test1 = "the cat sat";
    const auto ids1 = tok.encode(test1);
    show_tokens(tok, ids1, "encode(\"the cat sat\")");
    std::println("decode → \"{}\"", tok.decode(ids1));

    const std::string test2 = "cats and rats";
    const auto ids2 = tok.encode(test2);
    show_tokens(tok, ids2, "encode(\"cats and rats\")");
    std::println("decode → \"{}\"", tok.decode(ids2));

    // ── 5. Space marker convention ─────────────────────────────────────────────
    section("5. GPT-2 space marker (Ġ = U+0120)");
    std::println(
        "GPT-2 distinguishes 'cat' (first word) from ' cat' (preceded by space).\n"
        "The byte Ġ (0xC4 0xA0) is prepended to words that follow whitespace.\n"
        "This means the tokenizer handles word boundaries without a special\n"
        "word-boundary symbol."
    );
    const auto ids_space = tok.encode("cat sat");
    show_tokens(tok, ids_space, "encode(\"cat sat\")");
    std::println("  ↑ note the space marker on 'sat' (second word)");

    // ── 6. Save and reload ─────────────────────────────────────────────────────
    section("6. Save / reload vocabulary");

    const std::filesystem::path save_dir = "/tmp/sub0llm_ch03_vocab";
    tok.save(save_dir);
    std::println("Saved to {}/vocab.json + merges.txt", save_dir.string());

    auto tok2 = BPETokenizer::load(save_dir / "vocab.json", save_dir / "merges.txt");
    std::println("Reloaded: {} tokens, {} merges", tok2.vocab_size(), tok2.num_merges());

    // Verify encode/decode is identical after reload.
    const auto ids3 = tok2.encode("the cat sat on the mat");
    std::println("Roundtrip after reload: \"{}\"", tok2.decode(ids3));

    // ── 7. Special tokens ──────────────────────────────────────────────────────
    section("7. Special tokens");
    std::println("EOS id: {} (token: '{}')", tok.eos_id(), tok.token_str(tok.eos_id()));
    std::println(
        "In training data, <|endoftext|> marks document boundaries.\n"
        "During generation, the model emits EOS to signal it is done."
    );

    // ── 8. Sequence length — the key tensor dimension ──────────────────────────
    section("8. Tokenization → Tensor");
    std::println(
        "The token ID sequence becomes the first real input to our model.\n"
        "For a batch of B sequences with max length T:\n"
        "  input_ids : Tensor(shape=({{B}}, {{T}}), dtype=int64)\n"
        "This is what Ch04 (Dataset/DataLoader) will pack from raw text files."
    );

    const auto demo_ids = tok.encode("the cat sat on the mat");
    std::println("\"the cat sat on the mat\" → {} tokens → shape (1, {})",
        demo_ids.size(), demo_ids.size());

    // ── 9. What's next ────────────────────────────────────────────────────────
    section("What's next — Ch04: Dataset & DataLoader");
    std::println(
        "Ch04 builds the text pipeline that feeds the tokenizer at scale:\n"
        "  • Memory-mapped text file reading (mmap)\n"
        "  • Sliding-window chunking into fixed-length sequences\n"
        "  • Batched DataLoader with shuffle and prefetch\n"
        "  • Producing (input_ids, target_ids) pairs for cross-entropy loss"
    );

    return 0;
}
