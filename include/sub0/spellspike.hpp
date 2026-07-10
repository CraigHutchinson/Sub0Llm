// sub0/spellspike.hpp -- synthetic "combine / uncombine" curriculum for the token-granularity spike.
//
// The reframed question (see project memory spellspike-poc-results): a word-piece token is opaque to
// its own letters, and the first spike proved that MEMORIZING spellings in weights does not
// generalize. So instead of teaching the model the letters, teach it to REQUEST a deterministic
// tokenizer-layer operation and USE the result:
//   UNCOMBINE: emit TOK_UNCOMBINE right after a token -> the decode harness (sub0/decode.hpp) expands
//     that token into its constituent byte fragments (from the tokenizer's own `expansion` -- known,
//     exact, free) and injects them + TOK_UNCOMBINE_END, so the model can now READ the characters.
//   COMBINE: emit TOK_COMBINE <fragment bytes> TOK_COMBINE_END -> the harness re-tokenizes that span
//     back into its minimal vocab token and injects it (the inverse op). Round-trip fidelity
//     (combine(uncombine(t)) == t) is the success measure.
//
// This header builds the TRAINING DATA (pre-baked traces, with the would-be-injected spans present)
// and the EVAL prompts / answers, and is engine-free (depends only on sub0::tok::Tokenizer + std) so
// it is unit-testable with no model. The three task types are the "proving ground" for the real use:
//   - Nth-char : "which character is at index N of this word?"  (needs positional access to letters)
//   - count-X  : "how many of character X are in this word?"     (needs to enumerate+aggregate letters)
//   - roundtrip: uncombine a word then combine the fragments back (tests combine + copy fidelity)
// All three are things a pure token-level model cannot do WITHOUT reaching the characters -- exactly
// what the uncombine mechanism provides. The framing bytes are ASCII (base alphabet), so a trace
// depends on no particular learned word token beyond the one being queried.

#pragma once

#include "sub0/tokenizer.hpp"   // sub0::tok::Tokenizer (pulls in casing.hpp -> the TOK_* markers)

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sub0::spellspike {

// Framing bytes (all in [0,256) -- real byte tokens, no dependency on learned word tokens).
constexpr int Q_NTH = '#';   // "# N word"  -> the Nth character
constexpr int Q_CNT = '?';   // "? x word"  -> count of character x
constexpr int Q_RT  = '~';   // "~ word"    -> uncombine/combine round-trip
constexpr int SEP   = '=';   // separates the question from the answer

// A task word is a learned word piece short enough that its length, indices and character counts all
// stay single ASCII digits (<= 9), keeping every answer a single byte token.
inline bool is_task_word(const tok::Tokenizer& t, int id) {
    return id >= t.n_base && id < t.vocab &&
           t.expansion[static_cast<std::size_t>(id)].size() >= 2 &&
           t.expansion[static_cast<std::size_t>(id)].size() <= 9;
}

inline std::vector<int> expansion_bytes(const tok::Tokenizer& t, int word_id) {
    return t.expansion[static_cast<std::size_t>(word_id)];   // each entry is a byte code == byte-token id
}

inline std::string word_text(const tok::Tokenizer& t, int word_id) {
    std::string s;
    for (int c : t.expansion[static_cast<std::size_t>(word_id)]) s.push_back(static_cast<char>(c & 0xFF));
    return s;
}

// --- One task = an eval PROMPT (fed to the model), a full TRAINING trace, a parallel loss MASK, and
// the expected ANSWER for scoring. `mask[i]` (aligned to `trace`) is 1 where the model must PRODUCE
// trace[i] (graded) and 0 where the decode harness would INJECT it (the uncombine bytes/END, the
// combine result) or it is prompt context -- so training grades ONLY invoking the ops + the answer +
// the copy, never the harness-provided content (the memorization the first spike showed doesn't
// generalize). `answer_byte` is the answer token for nth/count (-1 for round-trip, scored on the
// recombined token). See sub0/core.hpp LOSS_IGNORE_INDEX + train_batch's loss_mask.
struct Task {
    std::vector<int>          prompt;
    std::vector<int>          trace;
    std::vector<std::uint8_t> mask;      // parallel to trace: 1 = trained, 0 = masked (injected/prompt)
    int                       answer_byte = -1;
    int                       word_id     = -1;
};

namespace detail {
inline void push(Task& k, int tok, std::uint8_t m) { k.trace.push_back(tok); k.mask.push_back(m); }
// TOK_UNCOMBINE is EMITTED by the model (graded); the bytes + TOK_UNCOMBINE_END are harness-INJECTED
// (masked). The preceding token is what the interceptor expands, so the model only emits the marker.
inline void append_uncombine(Task& k, const tok::Tokenizer& t, int word_id) {
    push(k, casing::TOK_UNCOMBINE, 1);
    for (int b : expansion_bytes(t, word_id)) push(k, b, 0);
    push(k, casing::TOK_UNCOMBINE_END, 0);
}
inline void append_prompt(Task& k) {                         // prompt tokens are context (masked)
    for (int p : k.prompt) push(k, p, 0);
}
}  // namespace detail

// "Which character is at index n?"  prompt: [#, '0'+n, word]
inline Task nth_char_task(const tok::Tokenizer& t, int word_id, int n) {
    const std::vector<int> bytes = expansion_bytes(t, word_id);
    Task k; k.word_id = word_id;
    k.prompt = {Q_NTH, '0' + n, word_id};
    k.answer_byte = (n >= 0 && n < static_cast<int>(bytes.size())) ? bytes[static_cast<std::size_t>(n)] : '?';
    detail::append_prompt(k);
    detail::append_uncombine(k, t, word_id);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);                       // the graded answer
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// "How many of character x?"  prompt: [?, x, word]
inline Task count_task(const tok::Tokenizer& t, int word_id, int x_byte) {
    const std::vector<int> bytes = expansion_bytes(t, word_id);
    int cnt = 0; for (int b : bytes) if (b == x_byte) ++cnt;
    Task k; k.word_id = word_id;
    k.prompt = {Q_CNT, x_byte, word_id};
    k.answer_byte = '0' + std::min(cnt, 9);
    detail::append_prompt(k);
    detail::append_uncombine(k, t, word_id);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// Round-trip: uncombine the word, then combine the fragments back.  prompt: [~, word]
// trace: [~, word, UNCOMBINE, <bytes>, UNCOMBINE_END, COMBINE, <copied bytes>, COMBINE_END, word, EOS]
// The copied bytes are GRADED (the model must copy the injected chars); the trailing `word` is what
// the harness's combine op injects (masked); success = it equals word_id.
inline Task roundtrip_task(const tok::Tokenizer& t, int word_id) {
    const std::vector<int> bytes = expansion_bytes(t, word_id);
    Task k; k.word_id = word_id;
    k.prompt = {Q_RT, word_id};
    detail::append_prompt(k);
    detail::append_uncombine(k, t, word_id);
    detail::push(k, casing::TOK_COMBINE, 1);
    for (int b : bytes) detail::push(k, b, 1);               // the model must COPY the injected chars
    detail::push(k, casing::TOK_COMBINE_END, 1);
    detail::push(k, word_id, 0);                             // harness-injected combine result (masked)
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// The eligible task words, split into a DRILLED set (their tasks go into training) and a HELD-OUT set
// (never trained on -- the clean no-memorization probe: answering a task for a held-out word is only
// possible by USING the harness-injected characters, since the model never saw that word's letters).
struct WordSplit { std::vector<int> drilled, held_out; };

inline WordSplit split_task_words(const tok::Tokenizer& t, double drilled_frac, std::uint64_t seed) {
    std::vector<int> words;
    for (int id = t.n_base; id < t.vocab; ++id) if (is_task_word(t, id)) words.push_back(id);
    std::mt19937_64 rng(seed);
    std::shuffle(words.begin(), words.end(), rng);
    const std::size_t nd = static_cast<std::size_t>(drilled_frac * static_cast<double>(words.size()));
    WordSplit s;
    s.drilled.assign(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(nd));
    s.held_out.assign(words.begin() + static_cast<std::ptrdiff_t>(nd), words.end());
    std::sort(s.drilled.begin(), s.drilled.end());
    std::sort(s.held_out.begin(), s.held_out.end());
    return s;
}

struct DatasetOptions {
    int           tasks_per_word = 12;   // how many random task instances to emit per drilled word
    std::uint64_t seed           = 4321;
};

// A flat training token stream + a parallel loss MASK (aligned to tokens: 1 = trained, 0 = masked;
// feed straight to train_batch's loss_mask) + the document-start index the window sampler
// (sub0/window.hpp) consumes -- one task trace per document, so a window never straddles two tasks.
struct Dataset {
    std::vector<int>           tokens;
    std::vector<std::uint8_t>  mask;
    std::vector<std::uint64_t> doc_starts;
};

inline Dataset build_dataset(const tok::Tokenizer& t, const WordSplit& split, const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    std::uniform_int_distribution<int> pick_type(0, 2);
    for (int word : split.drilled) {
        const std::vector<int> bytes = expansion_bytes(t, word);
        const int len = static_cast<int>(bytes.size());
        for (int r = 0; r < opt.tasks_per_word; ++r) {
            switch (pick_type(rng)) {
                case 0: {
                    std::uniform_int_distribution<int> pn(0, len - 1);
                    docs.push_back(nth_char_task(t, word, pn(rng)));
                    break;
                }
                case 1: {
                    // Count a character that actually occurs in the word half the time, a random
                    // letter the rest (so 0-counts are represented too).
                    std::uniform_int_distribution<int> pick(0, len - 1);
                    const int x = (rng() & 1) ? bytes[static_cast<std::size_t>(pick(rng))]
                                              : static_cast<int>('a' + (rng() % 26));
                    docs.push_back(count_task(t, word, x));
                    break;
                }
                default:
                    docs.push_back(roundtrip_task(t, word));
                    break;
            }
        }
    }
    std::shuffle(docs.begin(), docs.end(), rng);

    Dataset ds;
    ds.doc_starts.push_back(0);
    for (const Task& d : docs) {
        ds.tokens.insert(ds.tokens.end(), d.trace.begin(), d.trace.end());
        ds.mask.insert(ds.mask.end(), d.mask.begin(), d.mask.end());
        ds.doc_starts.push_back(static_cast<std::uint64_t>(ds.tokens.size()));
    }
    return ds;
}

}  // namespace sub0::spellspike
