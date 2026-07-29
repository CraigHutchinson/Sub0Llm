// sub0/tokenizer.hpp — the reusable BPE tokenizer: learn from a corpus, encode,
// detokenize, (de)serialize.
//
// This is the single definition of the truecasing + BPE vocabulary the project is
// built around, factored out of the build-time configurator so it can run on an
// IN-MEMORY corpus too. Both consumers share it:
//   - sub0-configure streams a (possibly out-of-core) corpus through Scan and learn()
//     to emit tokenizer.tok / corpus.tok;
//   - the engine loads a serialized Tokenizer and encode()s prompts at runtime;
//   - the unit tests learn a tiny in-memory corpus and assert the encode/detokenize
//     contracts deterministically, with no dependency on the baked production corpus.
//
// Like casing.hpp, this header is intentionally free of any dependency on the
// generated config (no VOCAB, no sub0_config.hpp), so the configurator — which must
// not depend on the engine it configures — can use it.

#pragma once

#include "sub0/casing.hpp"
#include "sub0/modality.hpp"

#include <array>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sub0::tok {

// Hash for an adjacent-symbol pair, used by the BPE merge tables (configurator and
// the runtime merge_rank). Defined once here so both sides agree.
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const noexcept {
        return (static_cast<std::size_t>(static_cast<std::uint32_t>(p.first)) << 32) ^
               static_cast<std::uint32_t>(p.second);
    }
};

// --- Corpus scan -----------------------------------------------------------
// The mergeable, in-memory result of the two corpus passes that precede BPE. It is
// fed text in arbitrary newline-aligned chunks (whole corpus, a streamed chunk, or
// one parallel segment) and folded together with merge_*; the merged result is
// byte-identical to scanning the whole corpus in one piece because a newline never
// splits a word unit or the truecaser/name-detector look-back.
//
// Two passes, because pass 2 needs the global `attested` set derived from all of
// pass 1: call add_names() over the whole corpus first, derive_attested(), then
// add_words() over the whole corpus.
struct Scan {
    // Pass 1 — name detection: per lowercase form, how often it appears all-lowercase
    // vs. capitalized mid-sentence (a "name" use). A form whose name uses dominate is
    // withheld from case collapse (see derive_attested).
    std::unordered_map<std::string, long long> lower_count, midcap_count;

    // Pass 2 — unique word table: each word unit keyed by its raw byte-symbol sequence.
    std::unordered_map<std::string, int> index;     // raw-key -> word id
    std::vector<std::vector<int>>        word_syms;  // word id -> symbol sequence (raw bytes;
                                                     //            remapped to base ids by learn())
    std::vector<long long>               word_freq;  // word id -> occurrence count
    std::array<int, 256>                 byte_used{};// 0/1 byte usage flags
    bool                                 used_cap = false, used_up = false;
    casing::TokStats                     st;

    // Aggregate counters (configurator reporting).
    std::size_t raw_bytes = 0, norm_bytes = 0;
    long long   quote_repl = 0;

    // Per-codepoint spacing modality, accumulated over pass 2 (rides the same mergeable/cacheable
    // passes). learn() derives the corpus-adaptive per-byte glue-default table (D2) from it.
    modality::ModalityStats modality;

    // Pass 1 over one in-memory chunk.
    void add_names(std::string_view chunk);
    // Fold another Scan's pass-1 state into this one, clearing the source's tables.
    void merge_names(Scan& other);

    // Pass 2 over one in-memory chunk, using the derived `attested` set.
    void add_words(std::string_view chunk, const std::unordered_set<std::string>& attested);
    // Fold another Scan's pass-2 state into this one, clearing the source's tables.
    void merge_words(Scan& other);

    // Rebuild `index` from `word_syms` (used after loading a cached scan).
    void rebuild_index();
};

// Derive the set of lowercase forms eligible for Capitalized/UPPER -> marker collapse:
// a form is attested unless its mid-sentence-capital ("name") uses dominate its
// lowercase uses. `withheld`, if non-null, receives the count of withheld names.
[[nodiscard]] std::unordered_set<std::string> derive_attested(const Scan& s, long long* withheld = nullptr);

// --- Learned tokenizer -----------------------------------------------------
// The in-memory tokenizer: a base alphabet (used byte values then the case markers) plus a
// Unigram LM word-piece vocabulary -- the ONLY runtime word-encoding method (see learn() below).
// Both the runtime engine and the configurator hold one of these.
struct Tokenizer {
    bool                 loaded = false;
    int                  vocab  = 0;   // n_base + word pieces
    int                  n_base = 0;   // base alphabet size (== casing::TOK_MARKER_COUNT, a scheme
                                       // constant -- not corpus-derived, see deserialize()'s check)
    std::vector<int>     base_symbol;              // base id -> symbol code (0..255, 256 cap, 257 up)
    std::vector<std::vector<int>>    expansion;    // id -> base symbol codes
    std::unordered_set<std::string>  attested;     // lowercase words eligible for case collapse

    // Unigram LM word encoding: the word encoder is Viterbi over `piece_index` (byte-string -> id)
    // scored by `piece_logp` (per id). Base bytes + learned pieces are both candidates; markers are
    // not. See sub0/unigram.hpp.
    std::vector<float>                     piece_logp;   // id -> log prob (Viterbi cost = -logp); -inf for non-pieces
    std::unordered_map<std::string, int>   piece_index;  // byte sequence -> id
    int                                    max_piece = 0;// longest piece

    // v2 (schemeV4, D2): per-byte DEFAULT spacing, corpus-derived from the scan's modality (with the
    // hardcoded casing::glue_default as the floor). encode + decode BOTH read this via glue_lead/
    // glue_trail -- one source, no drift -- so a code corpus makes `=`/`.` glue where prose spaces.
    std::array<casing::GlueDefault, 256>   glue{};
    bool glue_lead(int b)  const { return b >= 0 && b < 256 && glue[static_cast<std::size_t>(b)].lead; }
    bool glue_trail(int b) const { return b >= 0 && b < 256 && glue[static_cast<std::size_t>(b)].trail; }
};

struct LearnOptions {
    int    vocab_target = 2048;  // target vocabulary size (base symbols + markers + word pieces)
    int    min_merge    = 2;     // drop a candidate piece scoring below this occurrence count
                                 // (passed through to UnigramOptions::min_count)
    // Learn-set reduction + progress (for huge corpora: the rare-word tail is near-lossless to
    // drop and the EM/prune parallelises). Passed straight through to UnigramOptions.
    long long min_word_freq   = 1;   // drop learn-set words rarer than this
    long long max_learn_words = 0;   // 0 = all; else cap to the top-N most-frequent words
    bool   verbose      = false; // per-prune-round progress to stderr
};

// Learn the base alphabet + Unigram LM word-piece vocabulary from a completed scan and the
// derived attested set -- the runtime tokenizer's ONLY word-encoding method (encode()/serialize()/
// deserialize() never produce or consume anything else). REMAPS scan.word_syms in place from raw
// byte symbols to final token ids (so the caller can emit a tokenized corpus from scan.index +
// scan.word_syms afterwards).
[[nodiscard]] Tokenizer learn(Scan& scan, const std::unordered_set<std::string>& attested,
                const LearnOptions& opts = {});

// Convenience for tests / small corpora: scan an in-memory corpus end to end (names ->
// attested -> words) and learn. The corpus is processed whole, single-threaded.
[[nodiscard]] Tokenizer learn(std::string_view corpus, const LearnOptions& opts = {});

// --- Offline BPE vocabulariser (analysis-only, NOT the runtime tokenizer) --------------------
// Greedy-merge BPE, kept purely for `sub0llm-configure --dump-vocab`'s A/B comparison against the
// Unigram default and its bytes/token vocab-size curve (merge_count records, per merge, how many
// corpus tokens it removed at selection -- the whole curve from one learn). This type is never
// loadable/serializable and never reaches encode()/detokenize() -- it exists so the runtime
// `Tokenizer` above doesn't have to carry BPE-only state (merges/merge_rank) it never uses.
struct BpeAnalysisVocab {
    int vocab = 0, n_base = 0;
    std::vector<std::pair<int, int>> merges;       // ordered learned merges (left,right)
    std::vector<long long>           merge_count;  // per-merge occurrence count at selection
    std::vector<std::vector<int>>    expansion;    // id -> base symbol codes
    std::vector<int>                 base_symbol;  // base id -> symbol code (0..255, 256 cap, ...)
};

// REMAPS scan.word_syms in place to BPE ids, same contract as learn() above (dump_vocab_files
// reads the remapped word_syms directly, not through this return value).
[[nodiscard]] BpeAnalysisVocab learn_bpe_analysis(Scan& scan, int vocab_target = 2048, int min_merge = 2);

// --- Encode / detokenize (operate on a given tokenizer) --------------------
[[nodiscard]] std::vector<int> encode(const Tokenizer& t, const std::string& text);
[[nodiscard]] std::string      detokenize(const Tokenizer& t, std::span<const int> ids);

// --- Document-boundary scan (for the training-window sampler; see sub0/window.hpp) -------------
// Scans one chunk of already-encoded tokens for document boundaries and appends each newly found
// document's start index (corpus-wide) to `doc_starts`. `base_index` is the corpus-wide index of
// toks[0] (a running total, so this can be folded over a streamed/parallel-tokenized corpus one
// chunk at a time); `nl_run` carries the blank-line-run count across chunks. PREFERS the explicit
// eos_id marker (the literal `<|endoftext|>` the extraction scripts insert between documents --
// unambiguous, so the next token always starts a new document); FALLS BACK to a run of >=2 newline
// tokens ("\n\n") for corpora tokenized before the marker existed (or without it). The two signals
// never both fire on the same corpus (get_fineweb.py's marker format uses single newlines around
// it), so this is a clean dual-mode fallback, not a heuristic needing per-corpus tuning. Caller
// seeds `doc_starts` with {0} before the first chunk (document 0 starts at token 0); this function
// only ever appends.
// Document boundaries from the EXPLICIT `<|endoftext|>` marker (TOK_EOS) only. A blank-line ("\n\n")
// fallback used to be OR'd in for corpora extracted before the marker existed; it is gone. It was
// unsound as a boundary signal -- a blank line also occurs mid-document as an ordinary paragraph break,
// so on any corpus that HAS paragraphs it manufactured spurious documents -- and every corpus this
// project ships now carries the marker (verified 2026-07-28: fineweb_edu, fineweb_smoke, tinystories,
// gsm8k all do; fineweb_edu contains zero blank lines at all). A corpus without markers now yields ONE
// document, which the configurator refuses rather than training on -- see its doc_count check.
//
// `nl_run` is retained in the signature (unused) so the streaming caller's per-chunk state threading
// does not have to change; it costs nothing and keeps the call sites identical.
void scan_doc_boundaries(std::span<const std::int32_t> toks, std::uint64_t base_index,
                         const Tokenizer& t, int& nl_run, std::vector<std::uint64_t>& doc_starts);

// --- Serialization (tokenizer.tok: base alphabet + merges + attested words) -
void serialize(const Tokenizer& t, std::ostream& os);
[[nodiscard]] bool deserialize(Tokenizer& t, std::istream& is);

// A 64-bit identity fingerprint of a tokenizer: an FNV-1a hash over its serialized bytes, so two
// tokenizers fingerprint equal IFF they encode/detokenize identically (same base alphabet, merges/
// pieces and attested set). A trained model stamps this so a decoder that does not match the vocab it
// was trained against is caught loudly instead of silently emitting garble. Stable across a
// serialize->deserialize round-trip (the value a saved model carries equals the value a reloaded
// tokenizer.tok computes), and independent of attested-set iteration order (serialize sorts it).
[[nodiscard]] std::uint64_t fingerprint(const Tokenizer& t);

}  // namespace sub0::tok
