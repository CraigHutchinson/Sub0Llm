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

#include <array>
#include <cstdint>
#include <iosfwd>
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
std::unordered_set<std::string> derive_attested(const Scan& s, long long* withheld = nullptr);

// --- Learned tokenizer -----------------------------------------------------
// The in-memory tokenizer: a base alphabet (used byte values then the case markers),
// an ordered BPE merge list, and the attested set. Both the runtime engine and the
// configurator hold one of these.
struct Tokenizer {
    bool                 loaded = false;
    int                  vocab  = 0;   // n_base + merges.size()
    int                  n_base = 0;   // base alphabet size
    std::array<int, 256> byte_base{};  // byte value -> base id (-1 if unused)
    int                  cap_id = -1, up_id = -1;  // base ids of the case markers
    // JOIN (implicit-space) markers. The base alphabet is the complete 256 bytes + these markers, and
    // encode/detokenize use the spacing FSM. This is the only scheme.
    int                  join_id = -1, newline_id = -1, para_id = -1;
    int                  odquote_id = -1, cdquote_id = -1;        // directional double quotes (§3)
    int                  spell_start_id = -1, spell_end_id = -1;  // spaceless-group delimiters (§4, N>=3 words)
    int                  space2_id = -1, space4_id = -1;          // run-length whitespace tokens (§6)
    int                  tab2_id = -1, tab4_id = -1;
    std::vector<int>     base_symbol;              // base id -> symbol code (0..255, 256 cap, 257 up)
    std::vector<std::pair<int, int>> merges;       // ordered learned merges (left,right)
    std::vector<long long>           merge_count;  // per-merge occurrence count at selection = corpus
                                                   // tokens that merge removes (the vocab-curve benefit;
                                                   // learn-time only, not serialized)
    std::unordered_map<std::pair<int, int>, int, PairHash> merge_rank;  // (l,r) -> merge index
    std::vector<std::vector<int>>    expansion;    // id -> base symbol codes
    std::unordered_set<std::string>  attested;     // lowercase words eligible for case collapse

    // Unigram LM word encoding (the default; replaces the BPE merges above). When `max_piece > 0` the
    // word encoder is Viterbi over `piece_index` (byte-string -> id) scored by `piece_logp` (per id),
    // instead of the greedy merge replay. Base bytes + learned pieces are both candidates; markers are
    // not. See sub0/unigram.hpp.
    std::vector<float>                     piece_logp;   // id -> log prob (Viterbi cost = -logp); -inf for non-pieces
    std::unordered_map<std::string, int>   piece_index;  // byte sequence -> id
    int                                    max_piece = 0;// longest piece (0 = BPE/merge mode)
};

struct LearnOptions {
    // Word-vocabulary method: Unigram LM (default; global, occurrence-optimal, no dead tokens) or the
    // legacy greedy BPE merges (kept for the vocab A/B + curve analysis).
    enum class Method { Unigram, BPE };
    int    vocab_target = 2048;  // target vocabulary size (base symbols + markers + word pieces/merges)
    int    min_merge    = 2;     // BPE: stop merging once the best pair occurs fewer than this many times
    Method method       = Method::Unigram;
    // Unigram learn-set reduction + progress (for huge corpora: the rare-word tail is near-lossless to
    // drop and the EM/prune parallelises). Passed straight through to UnigramOptions.
    long long min_word_freq   = 1;   // drop learn-set words rarer than this
    long long max_learn_words = 0;   // 0 = all; else cap to the top-N most-frequent words
    bool   verbose      = false; // per-prune-round progress to stderr
};

// Learn the base alphabet + BPE merges from a completed scan and the derived attested
// set. REMAPS scan.word_syms in place from raw byte symbols to final token ids (so the
// caller can emit a tokenized corpus from scan.index + scan.word_syms afterwards).
Tokenizer learn(Scan& scan, const std::unordered_set<std::string>& attested,
                const LearnOptions& opts = {});

// Convenience for tests / small corpora: scan an in-memory corpus end to end (names ->
// attested -> words) and learn. The corpus is processed whole, single-threaded.
Tokenizer learn(std::string_view corpus, const LearnOptions& opts = {});

// --- Encode / detokenize (operate on a given tokenizer) --------------------
std::vector<int> encode(const Tokenizer& t, const std::string& text);
std::string      detokenize(const Tokenizer& t, const std::vector<int>& ids);

// --- Serialization (tokenizer.tok: base alphabet + merges + attested words) -
void serialize(const Tokenizer& t, std::ostream& os);
bool deserialize(Tokenizer& t, std::istream& is);

// A 64-bit identity fingerprint of a tokenizer: an FNV-1a hash over its serialized bytes, so two
// tokenizers fingerprint equal IFF they encode/detokenize identically (same base alphabet, merges/
// pieces and attested set). A trained model stamps this so a decoder that does not match the vocab it
// was trained against is caught loudly instead of silently emitting garble. Stable across a
// serialize->deserialize round-trip (the value a saved model carries equals the value a reloaded
// tokenizer.tok computes), and independent of attested-set iteration order (serialize sorts it).
std::uint64_t fingerprint(const Tokenizer& t);

}  // namespace sub0::tok
