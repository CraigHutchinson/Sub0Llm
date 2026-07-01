// sub0/unigram.hpp — Unigram Language Model vocabulariser (Kudo 2018; SentencePiece's default).
//
// The replacement for BPE's greedy bottom-up merging. A Unigram vocab is a set of sub-word tokens,
// each with a probability; a word is encoded by the Viterbi segmentation that maximises the product
// of token probabilities (= minimises Σ −log p). The vocab is learned TOP-DOWN: seed a large
// candidate set (frequent substrings), fit the probabilities by EM, then prune to the target size by
// dropping the least-used tokens (single bytes are mandatory, so every word stays encodable). Unlike
// BPE this is GLOBAL (it optimises the corpus encoding cost, not a greedy local merge), so it spends
// no slots on dead intermediate tokens and is the basis for an order-independent, occurrence-optimal
// vocabulary. See docs/TOKENIZER_VOCAB_ANALYSIS.md.
//
// This header is config-independent (like casing.hpp / tokenizer.hpp) so the configurator can use it.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sub0::tok {

// A learned Unigram vocabulary: parallel arrays token[id] (its bytes) and logp[id] (log probability),
// plus an index for Viterbi segmentation. ids 0..255-ish are the mandatory single bytes; the rest are
// learned multi-byte pieces. Pure data + a segmenter; the spacing/casing markers are layered on top
// by the tokenizer, exactly as the BPE merges are today.
struct Unigram {
    std::vector<std::string> token;    // id -> byte sequence (length 1 = a base byte)
    std::vector<double>      logp;     // id -> log probability (higher = cheaper)
    std::unordered_map<std::string, int> index;  // byte sequence -> id (Viterbi lookup)
    int max_len = 1;                   // longest token, bounds the Viterbi inner loop

    int size() const { return static_cast<int>(token.size()); }

    // Viterbi: segment `word` into the token-id sequence of minimum total -log p (the encoding the
    // model assigns). Single bytes are always present, so any word over the learned byte set segments.
    std::vector<int> segment(std::string_view word) const;
};

struct UnigramOptions {
    int    target     = 8192;  // final vocab size (single bytes always kept on top of the budget floor)
    int    seed_mult  = 8;     // seed candidate set ~= target * seed_mult frequent substrings
    int    max_piece  = 16;    // longest candidate substring considered
    int    em_iters   = 4;     // EM passes between prunes
    double drop_frac  = 0.2;   // fraction of prunable tokens removed each prune round
    long long min_count = 2;   // a candidate must occur at least this many times to seed
    // Learn-set reduction (a huge corpus has millions of rare/hapax unique words that barely move the
    // vocabulary but dominate the EM/prune iteration count; single bytes always keep every word
    // encodable, so dropping the rare tail is near-lossless). Both default to "no reduction".
    long long min_word_freq   = 1;   // drop learn-set words rarer than this before EM/prune
    long long max_learn_words = 0;   // 0 = all; else keep only the top-N most-frequent words
    int    threads    = 0;     // EM/prune parallelism; 0 = hardware_concurrency
    bool   verbose    = false; // print seed size + per-prune-round progress to stderr
};

// Learn a Unigram vocab from (word-bytes, frequency) pairs. Single bytes that appear are mandatory;
// the multi-byte pieces are EM-fit then pruned to `opt.target`. Deterministic.
Unigram learn_unigram(const std::vector<std::pair<std::string, long long>>& words, const UnigramOptions& opt = {});

// Total tokens to Viterbi-encode the corpus (Σ freq * segmentation length) -- the A/B metric against
// BPE at the same vocab size (fewer = better compression). `bytes` (optional) receives Σ freq*len.
long long corpus_tokens(const Unigram& u, const std::vector<std::pair<std::string, long long>>& words,
                        long long* bytes = nullptr);

}  // namespace sub0::tok
