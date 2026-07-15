// sub0/repeatspike.hpp -- SPIKE: does HARNESS-DRIVEN (not model-requested) live collapse of a REPEATED
// mention work, and is it better than raw fuzzy copying? Unifies two threads from this session:
//
//   * scratchspike.hpp already proves "harness pre-binds, model resolves/reasons" works well (nth_char,
//     select, content-select ALL pre-bind before the episode starts) -- and separately proves the
//     ALTERNATIVE, "model asks for the binding" (#1 define_task), is copy-bottlenecked/weak at this scale
//     ("neither is learnable model-side at this scale", scratchspike.hpp's build_dataset_scratch comment).
//     So harness-driven beats model-requested for INITIAL binding; this spike asks whether the same holds
//     for a binding that forms MID-STREAM from a naturally-occurring REPEAT, not a pre-arranged preamble.
//   * The "live retroactive collapse" idea (see docs/DETERMINISTIC_MECHANISMS.md's persistent compound-word
//     cache extension note): a multi-token span the model writes out normally gets silently swapped for its
//     scratch slot on RECURRENCE, with no request from the model. Building the real thing needs new engine
//     work (KV-cache splice mid-generation -- not built). This spike sidesteps that: it operates on baked
//     training TRACES/eval prompts (the same full-forward path every other spike here uses), where "the
//     harness already collapsed recurrence #2, #3" is just a fact about the fed token stream, not something
//     that has to happen live inside a running KV-cache. Proves the MECHANISM before the engineering.
//
// The task: an OOV word W (never a single vocab piece) appears three times in a short passage. Then a
// query asks for the Nth character of W.
//   FUZZY   (no mechanism): all three mentions are the full spelling. Tests raw in-context lookup/copy --
//            exactly the "localization/copy wall" this whole codebase keeps finding weak at small scale
//            (docs/DETERMINISTIC_MECHANISMS.md's repeated finding: fuzzy copying of exact content doesn't
//            generalise the way delegation does).
//   COLLAPSE (the mechanism under test): mention 1 is the full spelling (as if the model had just written
//            it normally -- no special marker, no request); mentions 2 and 3 are REPLACED by the bound
//            scratch slot token directly (masked -- a live harness substituted them, the model never had
//            to predict or ask for this). The query then references the slot; resolving it needs UNCOMBINE,
//            same mechanism nth_char_task already proves works.
// Both arms answer the SAME query shape over the SAME entity -- only whether recurrence #2/#3 are spelled
// out or collapsed differs. Held-out OOVs (never seen bound in training) isolate resolution/tracking from
// memorisation, same discipline as every other spike in this codebase.
//
// Engine-free (tokenizer + std only), reuses scratchspike.hpp's OOV generation + Task/Dataset shapes
// directly rather than re-deriving them.

#pragma once

#include "sub0/scratchspike.hpp"

#include <random>
#include <string>
#include <vector>

namespace sub0::repeatspike {

namespace ss = sub0::scratchspike;

// A short filler byte between mentions so the three occurrences aren't touching (keeps the trace
// passage-shaped without depending on any learned word token). Distinct from ss::SEP ('=', the
// query/answer separator) and every OOV byte (lowercase a-z only).
constexpr int FILL = ',';

// FUZZY arm: W spelled out in full all three times; the query asks for its Nth char. No scratch
// mechanism at all -- pure in-context lookup over repeated, fully-visible text.
// prompt: W FILL W FILL W Q_NTH n   trace: prompt [SEP answer EOS]
inline ss::Task repeat_task_fuzzy(const std::string& oov, int n) {
    const std::vector<int> bytes = ss::oov_bytes(oov);
    ss::Task k; k.oov = oov;   // binds left EMPTY: nothing is ever scratch-bound in this arm
    auto emit_word = [&] { for (int b : bytes) ss::detail::append_query(k, b); };
    emit_word();
    ss::detail::append_query(k, FILL);
    emit_word();
    ss::detail::append_query(k, FILL);
    emit_word();
    ss::detail::append_query(k, ss::Q_NTH);
    ss::detail::append_query(k, '0' + n);
    k.answer_byte = (n >= 0 && n < static_cast<int>(bytes.size())) ? bytes[static_cast<std::size_t>(n)] : '?';
    ss::detail::push(k, ss::SEP, 1);
    ss::detail::push(k, k.answer_byte, 1);
    ss::detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// COLLAPSE arm: mention 1 is the full spelling (as harness-observed, ordinary text -- this is what
// BINDS the slot, mirroring what a live collapse would do the moment the word completes); mentions 2
// and 3 are the bound slot token directly, exactly as scratchspike's pre-bound tasks already reference a
// slot -- the only difference from those is that the binding formed FROM this passage's own first
// mention, not from an upfront harness pre-bind before the episode started. Query references the slot;
// resolving it needs UNCOMBINE (same mechanism nth_char_task proves works).
// prompt: W FILL <slot> FILL <slot> Q_NTH n <slot>
// trace:  prompt [UNCOMBINE <W bytes> UNCOMBINE_END] [SEP answer EOS]
inline ss::Task repeat_task_collapse(const std::string& oov, int n, int slot = 0) {
    const std::vector<int> bytes = ss::oov_bytes(oov);
    ss::Task k; k.oov = oov; k.binds = {oov};   // pre-declares the binding scratchspike's harness owns
    for (int b : bytes) ss::detail::append_query(k, b);        // mention 1: full spelling (binds the slot)
    ss::detail::append_query(k, FILL);
    ss::detail::append_query(k, ss::scratch_slot(slot));        // mention 2: COLLAPSED (harness substituted)
    ss::detail::append_query(k, FILL);
    ss::detail::append_query(k, ss::scratch_slot(slot));        // mention 3: COLLAPSED
    ss::detail::append_query(k, ss::Q_NTH);
    ss::detail::append_query(k, '0' + n);
    ss::detail::append_query(k, ss::scratch_slot(slot));        // the query references the slot, not raw text
    ss::detail::append_resolve(k, bytes);                       // model must UNCOMBINE to answer
    k.answer_byte = (n >= 0 && n < static_cast<int>(bytes.size())) ? bytes[static_cast<std::size_t>(n)] : '?';
    ss::detail::push(k, ss::SEP, 1);
    ss::detail::push(k, k.answer_byte, 1);
    ss::detail::push(k, casing::TOK_EOS, 1);
    return k;
}

inline ss::Dataset build_dataset_fuzzy(const ss::OovSplit& split, const ss::DatasetOptions& opt) {
    std::vector<ss::Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (const std::string& oov : split.drilled) {
        std::uniform_int_distribution<int> pn(0, static_cast<int>(oov.size()) - 1);
        for (int r = 0; r < opt.tasks_per_oov; ++r) docs.push_back(repeat_task_fuzzy(oov, pn(rng)));
    }
    std::shuffle(docs.begin(), docs.end(), rng);
    ss::Dataset ds; ds.doc_starts.push_back(0);
    for (const ss::Task& d : docs) ss::append_doc(ds, d);
    return ds;
}

inline ss::Dataset build_dataset_collapse(const ss::OovSplit& split, const ss::DatasetOptions& opt) {
    std::vector<ss::Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (const std::string& oov : split.drilled) {
        std::uniform_int_distribution<int> pn(0, static_cast<int>(oov.size()) - 1);
        for (int r = 0; r < opt.tasks_per_oov; ++r) docs.push_back(repeat_task_collapse(oov, pn(rng)));
    }
    std::shuffle(docs.begin(), docs.end(), rng);
    ss::Dataset ds; ds.doc_starts.push_back(0);
    for (const ss::Task& d : docs) ss::append_doc(ds, d);
    return ds;
}

}  // namespace sub0::repeatspike
