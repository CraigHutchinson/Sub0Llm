// sub0/scratchspike.hpp -- synthetic "scratch token" (context translation layer) curriculum.
//
// Scratch tokens EXTEND combine/uncombine (see sub0/spellspike.hpp + casing.hpp) with a per-context
// BINDING TABLE. When the model combines a fragment sequence that is NOT a single vocab piece -- an OOV
// construction -- the decode interceptor assigns it a free reserved SCRATCH SLOT and remembers
// slot -> fragments for the rest of the context; a later uncombine of that slot expands it back via the
// table. So an OOV word that would cost N tokens on every mention collapses to ONE scratch token after
// being defined once, saving BOTH context length and KV-cache (see project memory
// scratch-tokens-context-translation-layer). The binding table is entirely interceptor-side: because
// decode.hpp's expand/combine are std::function callbacks, a STATEFUL lambda supplies it with no engine
// or decode.hpp change (the harness owns the table; see scratchspike_engine_tests.cpp).
//
// The spike question, parallel to the combine/uncombine spike (spellspike-poc-results): can a small
// model treat a FRESH, embedding-less scratch slot as a valid in-context symbol -- resolve a reference
// to it by attending to its earlier binding -- and does that GENERALIZE to HELD-OUT OOV sequences never
// bound in training? A correct char-level answer for an OOV the model has never seen bound is only
// possible by using the slot's in-context binding, not memorized weights. That isolates whether
// dynamic in-context binding works BEFORE (and independently of) the content-derived-embedding engine
// change -- the scratch slot has no trained embedding, so resolution here is purely by attention.
//
// Engine-free (tokenizer + std only), unit-testable with no model -- same shape as spellspike.hpp.

#pragma once

#include "sub0/tokenizer.hpp"   // sub0::tok::Tokenizer (pulls in casing.hpp -> the TOK_* markers)

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sub0::scratchspike {

// Framing bytes (real byte tokens in [0,256), no dependency on learned word tokens) -- shared shapes
// with spellspike so a model can be trained on both curricula together later.
constexpr int Q_NTH = '#';   // "# n <slot>" -> the Nth character of the slot's bound fragments
constexpr int Q_RT  = '~';   // "~ <slot>"   -> uncombine then combine back (binding round-trip)
constexpr int Q_DEF = '&';   // "& <oov>"    -> the model emits combine to BIND the oov (model-driven)
constexpr int Q_SEL = '?';   // "<slot tag>* ? <tag>" -> select the slot carrying the queried tag
constexpr int Q_FST = '!';   // "<slot>* ! <c>" -> select the slot whose OOV starts with char c (CONTENT)
constexpr int Q_HAS = '%';   // "<slot>* %% <c>" -> select the slot whose OOV CONTAINS char c (order-agnostic)
constexpr int SEP   = '=';   // separates the question from the answer

// Answer tokens for the CONTAINS regime's universal/negative cases (the ALL and NONE outcomes that break
// the "exactly one matches" elimination shortcut). Each is a single opaque byte token that carries no prior
// meaning -- like a role/control token, it acquires its meaning purely from CONSISTENT use in the
// curriculum. Chosen outside [a-z] so they never collide with an OOV fragment byte or a slot id.
constexpr int ANS_NONE = '_';   // "%%<c>" -> NO bound slot's OOV contains c (forces checking every word)
constexpr int ANS_ALL  = '@';   // "%%<c>" -> EVERY bound slot's OOV contains c

// Per-slot VERDICT tokens for the chain-of-thought CONTAINS variant: right after resolving a slot, the
// model emits whether THAT slot's spelling contains the queried char. Making the match a LOCAL, explicit
// step (vs a hidden global recall at the end) turns final localization into "copy the slot that got a '+'"
// -- a short-range induction. Opaque single tokens, meaning acquired from consistent use.
constexpr int VERD_YES = '+';   // this slot's OOV contains the queried char
constexpr int VERD_NO  = '-';   // it does not

// The reserved SCRATCH SLOT pool. A handful is enough to prove single- and multi-binding in the spike;
// a production context-translation layer would carve a much larger reserved range (a tokenizer-budget
// follow-up). Slots are assigned in BINDING ORDER within a context (first OOV bound -> slot 0), so the
// baked training traces and the live interceptor agree deterministically.
constexpr int SCRATCH_BASE = casing::TOK_RESERVED_4;   // slot i == SCRATCH_BASE + i
constexpr int SCRATCH_POOL = 6;                        // TOK_RESERVED_4..9 (the full reserved range -> max K=6)
constexpr int scratch_slot(int i) { return SCRATCH_BASE + i; }

// An OOV "word": a random lowercase byte string of length 3..6 that is NOT a single vocab piece (so it
// genuinely CANNOT be represented by one token and must use a scratch slot). Its fragments are just its
// byte codes (each ASCII letter is a base byte token < 256), so a trace depends on no learned token.
inline std::string gen_oov(std::mt19937_64& rng, const tok::Tokenizer& t) {
    std::uniform_int_distribution<int> len_d(3, 6), ch(0, 25);
    for (int tries = 0; tries < 64; ++tries) {
        std::string s;
        const int len = len_d(rng);
        for (int i = 0; i < len; ++i) s.push_back(static_cast<char>('a' + ch(rng)));
        if (t.piece_index.find(s) == t.piece_index.end()) return s;   // ensure genuinely OOV (not one piece)
    }
    return "qxzjvw";   // fallback: essentially never a single vocab piece
}

inline std::vector<int> oov_bytes(const std::string& oov) {
    std::vector<int> b;
    for (char c : oov) b.push_back(static_cast<unsigned char>(c));
    return b;
}

// One task = an eval PROMPT (fed to the model), a full TRAINING trace, a parallel loss MASK, and the
// expected ANSWER, exactly like spellspike::Task. `mask[i]` (aligned to `trace`) is 1 where the model
// must PRODUCE trace[i] (graded), 0 where the interceptor INJECTS it (the assigned slot, the resolved
// fragments) or it is prompt/binding context -- so training grades ONLY invoking the ops + the answer,
// never the harness-provided content (the memorization the first spike showed does not generalize).
struct Task {
    std::vector<int>          prompt;    // fed to the model at eval (the query context)
    std::vector<int>          trace;     // full baked sequence (training)
    std::vector<std::uint8_t> mask;      // parallel to trace: 1 = trained, 0 = masked (injected/context)
    int                       answer_byte = -1;
    std::string               oov;       // the QUERIED OOV (held-out bookkeeping)
    std::vector<std::string>  binds;     // slot i is pre-bound to binds[i] (harness); single task = {oov}
};

namespace detail {
inline void push(Task& k, int tok, std::uint8_t m) { k.trace.push_back(tok); k.mask.push_back(m); }

// A query token that is part of the fed prompt (context, masked).
inline void append_query(Task& k, int tok) { push(k, tok, 0); k.prompt.push_back(tok); }
// The RESOLVE region the model GENERATES: `TOK_UNCOMBINE <injected bytes> TOK_UNCOMBINE_END`. Only the
// UNCOMBINE marker is graded (the model must INVOKE resolution); the injected bytes + END are masked --
// so the ONLY path to the chars is resolving the slot via the interceptor's binding table, never
// reading them off the stream (the isolation the in-stream-define variant lacked; see the header note).
inline void append_resolve(Task& k, const std::vector<int>& bytes) {
    push(k, casing::TOK_UNCOMBINE, 1);
    for (int b : bytes) push(k, b, 0);           // interceptor-injected from the binding table
    push(k, casing::TOK_UNCOMBINE_END, 0);
}
}  // namespace detail

// "Which character is at index n of the OOV bound to this scratch slot?" The slot is bound BEFORE the
// task (the harness pre-binds it -- modelling a definition that happened earlier in the context) and its
// OOV bytes are NOT in the stream, so the only way to answer is to RESOLVE the slot via uncombine. This
// isolates binding-table resolution; a correct answer for a HELD-OUT OOV proves it, not memorization.
// prompt: [# n <slot>]   trace: [# n <slot>] [UNCOMBINE <oov> UNCOMBINE_END] [= answer EOS]
inline Task nth_char_task(const tok::Tokenizer& /*t*/, const std::string& oov, int n, int slot = 0) {
    const std::vector<int> bytes = oov_bytes(oov);
    Task k; k.oov = oov; k.binds = {oov};
    k.answer_byte = (n >= 0 && n < static_cast<int>(bytes.size())) ? bytes[static_cast<std::size_t>(n)] : '?';
    detail::append_query(k, Q_NTH);
    detail::append_query(k, '0' + n);
    detail::append_query(k, scratch_slot(slot));            // reference the OOV by its 1-token scratch slot
    detail::append_resolve(k, bytes);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);                      // the graded answer
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// Round-trip: uncombine the bound slot, then combine the fragments back -- which must recover the SAME
// slot (the interceptor returns the existing binding, not a fresh slot). Tests binding-table consistency
// + the combine-back leg. prompt: [~ <slot>]
// trace: [~ <slot>] [UNCOMBINE <oov> UNCOMBINE_END] [COMBINE <copied oov> COMBINE_END] <slot> EOS
inline Task roundtrip_task(const tok::Tokenizer& /*t*/, const std::string& oov, int slot = 0) {
    const std::vector<int> bytes = oov_bytes(oov);
    Task k; k.oov = oov; k.binds = {oov};
    detail::append_query(k, Q_RT);
    detail::append_query(k, scratch_slot(slot));
    detail::append_resolve(k, bytes);
    detail::push(k, casing::TOK_COMBINE, 1);
    for (int b : bytes) detail::push(k, b, 1);              // the model must COPY the resolved fragments
    detail::push(k, casing::TOK_COMBINE_END, 1);
    detail::push(k, scratch_slot(slot), 0);                // interceptor-injected: the recombined slot
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// MULTI-BINDING: K distinct OOVs occupy slots 0..K-1 (all pre-bound by the harness). The context opens
// with a symbol-table preamble listing all K slot tokens (so they compete in attention + the KV-cache),
// then queries the Nth char of ONE of them -- the model must resolve the RIGHT slot amid the others.
// This is the interference/disambiguation test single-binding cannot exercise. prompt:
// [<slot_0..slot_{K-1}>  # n <slot_q>]   trace: prompt [UNCOMBINE <oov_q> UNCOMBINE_END] [= answer EOS]
inline Task multi_nth_char_task(const tok::Tokenizer& /*t*/, const std::vector<std::string>& oovs,
                                int query_idx, int n) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    const std::vector<int> bytes = oov_bytes(k.oov);
    k.answer_byte = (n >= 0 && n < static_cast<int>(bytes.size())) ? bytes[static_cast<std::size_t>(n)] : '?';
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) detail::append_query(k, scratch_slot(i));  // preamble
    detail::append_query(k, Q_NTH);
    detail::append_query(k, '0' + n);
    detail::append_query(k, scratch_slot(query_idx));      // query the chosen slot
    detail::append_resolve(k, bytes);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// MODEL-DRIVEN BINDING (#1): the model must EMIT combine to define an OOV given in the prompt -- the
// interceptor's combine callback assigns a free slot and records slot->fragments. Success = the model
// COPIED the OOV correctly into the combine region, so the slot binds to the right bytes (measured by
// the harness reading the resulting binding). No pre-binding: this tests the model DRIVING the binding,
// not just resolving a harness-supplied one -- the "when/how to bind" half the pre-bound tasks sidestep.
// prompt: [& <oov bytes>]   trace: [& <oov>] [COMBINE <oov copy> COMBINE_END] <slot> EOS
inline Task define_task(const tok::Tokenizer& /*t*/, const std::string& oov, int slot = 0) {
    const std::vector<int> bytes = oov_bytes(oov);
    Task k; k.oov = oov;                                    // binds stays EMPTY -> harness does NOT pre-bind
    detail::append_query(k, Q_DEF);
    for (int b : bytes) detail::append_query(k, b);         // the OOV to define (given context)
    detail::push(k, casing::TOK_COMBINE, 1);
    for (int b : bytes) detail::push(k, b, 1);              // the model COPIES the OOV to bind it (graded)
    detail::push(k, casing::TOK_COMBINE_END, 1);
    detail::push(k, scratch_slot(slot), 0);                // interceptor-injected slot (masked)
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// REASON-OVER-SLOTS (#2): K slots (pre-bound) are each tagged with a distinct byte in a preamble
// [slot_0 tag_0 ... slot_{K-1} tag_{K-1}], then the query asks which slot carries tag_j. The model must
// output slot_j -- using the slots as distinct in-context ENTITIES it recalls by association, over
// generic embedding-less reserved ids. The answer is the SLOT (not a char), so success needs no
// resolution, only correct selection. prompt: [<slot_i tag_i>* ? tag_j]   trace: prompt [= <slot_j> EOS]
inline Task select_task(const tok::Tokenizer& /*t*/, const std::vector<std::string>& oovs,
                        const std::vector<int>& tags, int query_idx) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) {   // slot<->tag pairing preamble (context)
        detail::append_query(k, scratch_slot(i));
        detail::append_query(k, tags[static_cast<std::size_t>(i)]);
    }
    detail::append_query(k, Q_SEL);
    detail::append_query(k, tags[static_cast<std::size_t>(query_idx)]);
    k.answer_byte = scratch_slot(query_idx);               // the graded answer IS the selected slot
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// CONTENT-based reason-over-slots (#4): K slots bound to K OOVs with DISTINCT first letters; a preamble
// lists only the slots (NO tags), then the query gives a first-letter and asks for the slot whose OOV
// starts with it. Unlike select (associative: a tag paired with each slot in-context), the discriminator
// is the slot's CONTENT -- which a generic reserved-id embedding does NOT carry. So this tests whether
// the model can reason about slot content WITHOUT resolving each slot: the case that motivates
// content-derived slot embeddings. prompt: [<slot_i>* ! <c>]   trace: prompt [= <slot_q> EOS]
inline Task content_select_task(const tok::Tokenizer& /*t*/, const std::vector<std::string>& oovs, int query_idx) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) detail::append_query(k, scratch_slot(i));   // slots only
    detail::append_query(k, Q_FST);
    detail::append_query(k, static_cast<int>(static_cast<unsigned char>(oovs[static_cast<std::size_t>(query_idx)][0])));
    k.answer_byte = scratch_slot(query_idx);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// CONTENT-based, ORDER-AGNOSTIC (#4b): exactly one of the K bound OOVs CONTAINS the queried char; output
// its slot. Unlike first-letter (positional), this needs only which chars are PRESENT -- what a mean-pool
// content embedding carries -- so it's the clean test of whether content-derived embeddings deliver a
// usable signal at all. `has_char` is a char known to be in oovs[query_idx] and in NO other listed oov.
inline Task content_contains_task(const tok::Tokenizer& /*t*/, const std::vector<std::string>& oovs,
                                  int query_idx, int has_char) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) detail::append_query(k, scratch_slot(i));
    detail::append_query(k, Q_HAS);
    detail::append_query(k, has_char);
    k.answer_byte = scratch_slot(query_idx);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// #4c CONTAINS via THINKING-UNCOMBINE (Route A): the SAME query as content_contains, but instead of
// answering off the content-free slot embedding in ONE hop, the model RESOLVES first. For each slot it
// emits [<slot_i> UNCOMBINE] (both GRADED -- it must choose to resolve), the interceptor injects that
// slot's fragments (MASKED), so every OOV's spelling enters the stream; then it emits the ANSWER: the
// matching slot, or ANS_NONE / ANS_ALL. Nothing is memorized (the bytes come from the binding table at
// generation time), so a correct HELD-OUT answer proves the exact, learn-nothing resolution route.
//
// The answer is a caller-supplied token so the regime spans THREE outcomes over the same trace shape:
// exactly-one-matches (answer = that slot), NONE matches (ANS_NONE), ALL match (ANS_ALL). Mixing NONE/ALL
// in is what forces GENUINE per-word checking: with "exactly one" guaranteed, K=2 collapses to a single
// check + elimination ("not word 0 -> word 1") that does NOT compose to K=3; once NONE/ALL are possible,
// "not word 0" no longer determines the answer, so the model must actually test every word -- the
// composable primitive the K=2 -> K=3 progression needs. prompt: [<slot_i>* %% <c>]
// trace: prompt [<slot_0> UNCOMBINE <bytes_0> UNCOMBINE_END] ... [<slot_{K-1}> ...] [= <answer> EOS]
inline Task content_contains_reason_task(const tok::Tokenizer& /*t*/, const std::vector<std::string>& oovs,
                                         int answer_tok, int has_char) {
    const int K = static_cast<int>(oovs.size());
    Task k; k.binds = oovs; k.oov = oovs.empty() ? std::string{} : oovs[0];   // held-out bookkeeping only
    for (int i = 0; i < K; ++i) detail::append_query(k, scratch_slot(i));   // the slot list (context)
    detail::append_query(k, Q_HAS);
    detail::append_query(k, has_char);
    for (int i = 0; i < K; ++i) {                                           // THINKING: resolve every slot
        detail::push(k, scratch_slot(i), 1);                               // graded: emit the slot to inspect
        detail::append_resolve(k, oov_bytes(oovs[static_cast<std::size_t>(i)]));   // UNCOMBINE + injected bytes
    }
    k.answer_byte = answer_tok;                                            // a slot, ANS_NONE, or ANS_ALL
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// CHAIN-OF-THOUGHT variant of the CONTAINS-reason task: identical prompt/answer, but right after resolving
// each slot the model emits a GRADED per-slot VERDICT (VERD_YES/VERD_NO -- does THIS slot's spelling contain
// the queried char). This scaffolds LOCALIZATION -- the wall the plain-reason task can't clear: each verdict
// is a LOCAL content-match (the spelling was just injected, right there), and the final answer becomes "copy
// the slot that got a '+'", a short-range induction, instead of a hidden global recall of which spelling
// matched. Tests whether the model CAN localize when the reasoning is made explicit ("show your work").
// trace: prompt [<slot_i> UNCOMBINE <bytes_i> UNCOMBINE_END <verdict_i>]* [= <answer> EOS]
//
// `local_query`: the plain CoT still fails because emitting slot i's verdict needs to isolate THAT segment's
// bytes from the rest -- the same segment-localization the model can't do -- while ALSO recalling the query
// char from the far-off front. Local mode removes BOTH crutches: it RESTATES the queried char immediately
// before each verdict (so the check is over adjacent tokens: <bytes_i> <c> <verdict>) and drops the now-
// redundant front slot-list (which the resolve phase re-emits anyway) to fit the window. If verdicts go
// right here, the wall was non-local binding (fixable); if not, segment isolation itself is the wall.
// `restate_query` and `drop_front` are independent so the two effects can be separated: the LOCAL variant
// sets both (restate the char next to each segment + drop the redundant front slot-list); a CONTROL sets
// only `drop_front` (no restatement) to prove the restatement -- not the shorter context -- is what cracks
// localization.
inline Task content_contains_cot_task(const tok::Tokenizer& /*t*/, const std::vector<std::string>& oovs,
                                      int answer_tok, int has_char, bool restate_query = false, bool drop_front = false) {
    const int K = static_cast<int>(oovs.size());
    Task k; k.binds = oovs; k.oov = oovs.empty() ? std::string{} : oovs[0];
    if (!drop_front)
        for (int i = 0; i < K; ++i) detail::append_query(k, scratch_slot(i));   // front slot-list (redundant; droppable)
    detail::append_query(k, Q_HAS);
    detail::append_query(k, has_char);
    for (int i = 0; i < K; ++i) {
        detail::push(k, scratch_slot(i), 1);                               // graded: emit the slot to inspect
        detail::append_resolve(k, oov_bytes(oovs[static_cast<std::size_t>(i)]));   // UNCOMBINE + injected bytes
        if (restate_query) detail::push(k, has_char, 1);                   // graded: restate the query char LOCALLY
        const bool contains = oovs[static_cast<std::size_t>(i)].find(static_cast<char>(has_char)) != std::string::npos;
        detail::push(k, contains ? VERD_YES : VERD_NO, 1);                 // graded: the per-slot verdict
    }
    k.answer_byte = answer_tok;
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// A generated pool of OOVs, split into a DRILLED set (their tasks go into training) and a HELD-OUT set
// (never trained on -- the clean no-memorization probe: answering a task for a held-out OOV is only
// possible by resolving its in-context scratch binding, since the model never saw that OOV bound).
struct OovSplit { std::vector<std::string> drilled, held_out; };

inline OovSplit make_oov_split(const tok::Tokenizer& t, int n_total, double drilled_frac, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<std::string> oovs;
    std::vector<std::string> seen;
    while (static_cast<int>(oovs.size()) < n_total) {
        std::string o = gen_oov(rng, t);
        if (std::find(seen.begin(), seen.end(), o) == seen.end()) { seen.push_back(o); oovs.push_back(o); }
    }
    const std::size_t nd = static_cast<std::size_t>(drilled_frac * static_cast<double>(oovs.size()));
    OovSplit s;
    s.drilled.assign(oovs.begin(), oovs.begin() + static_cast<std::ptrdiff_t>(nd));
    s.held_out.assign(oovs.begin() + static_cast<std::ptrdiff_t>(nd), oovs.end());
    return s;
}

struct DatasetOptions {
    int           tasks_per_oov = 8;
    std::uint64_t seed          = 20260711;
};

// A flat training token stream + a parallel loss MASK + the document-start index the window sampler
// (sub0/window.hpp) consumes -- one task trace per document, so a window never straddles two bindings.
// `doc_bindings[d]` are document d's per-slot fragments (slot i -> byte ids of binds[i]) -- what the
// engine needs for content-derived slot embeddings (a window inherits its document's bindings).
struct Dataset {
    std::vector<int>                             tokens;
    std::vector<std::uint8_t>                    mask;
    std::vector<std::uint64_t>                   doc_starts;
    std::vector<std::vector<std::vector<int>>>   doc_bindings;   // per doc: slot -> fragment token ids
};

// Append one task trace as a document, carrying its per-slot bindings (empty for tasks with no pre-bind).
inline void append_doc(Dataset& ds, const Task& d) {
    ds.tokens.insert(ds.tokens.end(), d.trace.begin(), d.trace.end());
    ds.mask.insert(ds.mask.end(), d.mask.begin(), d.mask.end());
    ds.doc_starts.push_back(static_cast<std::uint64_t>(ds.tokens.size()));
    std::vector<std::vector<int>> b;
    for (const std::string& oov : d.binds) b.push_back(oov_bytes(oov));
    ds.doc_bindings.push_back(std::move(b));
}

inline Dataset build_dataset(const tok::Tokenizer& t, const OovSplit& split, const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    std::uniform_int_distribution<int> pick_type(0, 1);
    for (const std::string& oov : split.drilled) {
        const int len = static_cast<int>(oov.size());
        for (int r = 0; r < opt.tasks_per_oov; ++r) {
            if (pick_type(rng) == 0) {
                std::uniform_int_distribution<int> pn(0, len - 1);
                docs.push_back(nth_char_task(t, oov, pn(rng)));   // single-binding: slot 0
            } else {
                docs.push_back(roundtrip_task(t, oov));
            }
        }
    }
    std::shuffle(docs.begin(), docs.end(), rng);

    Dataset ds;
    ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// Pick K distinct OOVs from `pool` (the query target first, then K-1 random distractors), shuffled so
// the target lands in a random slot; returns the built multi-binding task. Shared by the dataset
// builder and the harness so training and eval use the identical construction.
inline Task pick_multi_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                            int target_idx, int K, std::mt19937_64& rng) {
    std::vector<std::string> oovs{ pool[static_cast<std::size_t>(target_idx)] };
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    while (static_cast<int>(oovs.size()) < K) {
        const std::string& cand = pool[pick(rng)];
        if (std::find(oovs.begin(), oovs.end(), cand) == oovs.end()) oovs.push_back(cand);
    }
    std::shuffle(oovs.begin(), oovs.end(), rng);
    const int qi = static_cast<int>(std::find(oovs.begin(), oovs.end(), pool[static_cast<std::size_t>(target_idx)]) - oovs.begin());
    const int len = static_cast<int>(pool[static_cast<std::size_t>(target_idx)].size());
    std::uniform_int_distribution<int> pn(0, len - 1);
    return multi_nth_char_task(t, oovs, qi, pn(rng));
}

// Multi-binding training set: for each drilled OOV as the query target, `tasks_per_oov` docs each with
// K-1 random drilled distractors. Same shape as build_dataset otherwise.
inline Dataset build_dataset_multi(const tok::Tokenizer& t, const OovSplit& split, int K, const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_multi_task(t, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);

    Dataset ds;
    ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// #1 model-driven binding training set: one define per drilled OOV (define_task is deterministic).
inline Dataset build_dataset_define(const tok::Tokenizer& t, const OovSplit& split, const DatasetOptions& opt) {
    std::vector<Task> docs;
    for (const std::string& oov : split.drilled) docs.push_back(define_task(t, oov));
    std::mt19937_64 rng(opt.seed);
    std::shuffle(docs.begin(), docs.end(), rng);
    Dataset ds; ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// Build a select task: K OOVs (target first, then K-1 distractors), K distinct byte tags, both shuffled
// so tag<->slot pairing is non-positional, query one tag. Shared by the builder and the harness.
inline Task pick_select_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                             int target_idx, int K, std::mt19937_64& rng) {
    std::vector<std::string> oovs{ pool[static_cast<std::size_t>(target_idx)] };
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    while (static_cast<int>(oovs.size()) < K) {
        const std::string& cand = pool[pick(rng)];
        if (std::find(oovs.begin(), oovs.end(), cand) == oovs.end()) oovs.push_back(cand);
    }
    std::shuffle(oovs.begin(), oovs.end(), rng);
    std::vector<int> tags;
    for (int c = 'A'; static_cast<int>(tags.size()) < K; ++c) tags.push_back(c);   // distinct byte tags
    std::shuffle(tags.begin(), tags.end(), rng);
    const int query_idx = std::uniform_int_distribution<int>(0, K - 1)(rng);
    return select_task(t, oovs, tags, query_idx);
}

// #2 reason-over-slots training set: for each drilled OOV as the target, tasks_per_oov select tasks.
inline Dataset build_dataset_select(const tok::Tokenizer& t, const OovSplit& split, int K, const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_select_task(t, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);
    Dataset ds; ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// #4 content-select: K OOVs with DISTINCT first letters (target + distractors that differ in first char),
// query the target's first letter. Shared by the builder and the harness.
inline Task pick_content_select_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                                     int target_idx, int K, std::mt19937_64& rng) {
    std::vector<std::string> oovs{ pool[static_cast<std::size_t>(target_idx)] };
    std::vector<char> firsts{ pool[static_cast<std::size_t>(target_idx)][0] };
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    for (int guard = 0; static_cast<int>(oovs.size()) < K && guard < 2000; ++guard) {
        const std::string& cand = pool[pick(rng)];
        if (std::find(firsts.begin(), firsts.end(), cand[0]) == firsts.end()) {
            oovs.push_back(cand); firsts.push_back(cand[0]);
        }
    }
    std::shuffle(oovs.begin(), oovs.end(), rng);
    const int qi = static_cast<int>(std::find(oovs.begin(), oovs.end(),
                       pool[static_cast<std::size_t>(target_idx)]) - oovs.begin());
    return content_select_task(t, oovs, qi);
}

inline Dataset build_dataset_content(const tok::Tokenizer& t, const OovSplit& split, int K, const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_content_select_task(t, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);
    Dataset ds; ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// #4b contains SELECTION (shared by the single-hop and thinking-uncombine builders + the harness): K OOVs
// (target + distractors) and a char that is in the TARGET but in NO distractor, so exactly one slot
// matches. Returns the shuffled oovs, the target's landed slot index, and that unique char.
struct ContainsPick { std::vector<std::string> oovs; int qi = 0; int has = 0; };
inline ContainsPick pick_contains(const std::vector<std::string>& pool, int target_idx, int K, std::mt19937_64& rng) {
    const std::string& target = pool[static_cast<std::size_t>(target_idx)];
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    for (int attempt = 0; attempt < 128; ++attempt) {
        std::vector<std::string> oovs{ target };
        while (static_cast<int>(oovs.size()) < K) {
            const std::string& cand = pool[pick(rng)];
            if (std::find(oovs.begin(), oovs.end(), cand) == oovs.end()) oovs.push_back(cand);
        }
        std::string distract;                                   // all chars in the non-target OOVs
        for (const std::string& o : oovs) if (o != target) distract += o;
        int has = -1;
        for (char ch : target) if (distract.find(ch) == std::string::npos) { has = static_cast<unsigned char>(ch); break; }
        if (has < 0) continue;                                  // no char uniquely identifies the target -> retry
        std::shuffle(oovs.begin(), oovs.end(), rng);
        const int qi = static_cast<int>(std::find(oovs.begin(), oovs.end(), target) - oovs.begin());
        return { std::move(oovs), qi, has };
    }
    return { std::vector<std::string>(static_cast<std::size_t>(K), target), 0,   // degenerate fallback (rare)
             static_cast<unsigned char>(target[0]) };
}
inline Task pick_content_contains_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                                       int target_idx, int K, std::mt19937_64& rng) {
    const ContainsPick p = pick_contains(pool, target_idx, K, rng);
    return content_contains_task(t, p.oovs, p.qi, p.has);
}
// The selected CONTAINS case: the K oovs, the queried char, and the answer token (a slot / ANS_NONE /
// ANS_ALL) -- shared by BOTH the plain-reason and the chain-of-thought task builders (and thus by train +
// eval), so the only difference between the two regimes is the trace shape, never the selection.
struct ContainsCase { std::vector<std::string> oovs; int has = 0; int answer = 0; };

// Mixed CONTAINS-case selection. ONE-HEAVY (~60% ONE, 20% NONE, 20% ALL): NONE/ALL are EASY (a global
// "present anywhere?" OR) and saturate fast while ONE is HARD (localize WHICH slot), so an equal split lets
// the model settle for a present->ALL / absent->NONE heuristic (~0.67 for free) and never learn to point --
// the same gradient-starvation the production curriculum's 3:1 weighting avoids. NONE/ALL keep the
// elimination shortcut dead. All oovs are drawn from `pool`, so held-out probes stay valid for every outcome;
// a NONE/ALL that cannot be constructed falls back to ONE.
inline ContainsCase pick_contains_case(const std::vector<std::string>& pool, int anchor_idx, int K,
                                       std::mt19937_64& rng) {
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    const int roll = std::uniform_int_distribution<int>(0, 9)(rng);
    const int kind = (roll < 6) ? 0 : (roll < 8) ? 1 : 2;             // 0=ONE 1=NONE 2=ALL

    if (kind == 1) {                                                  // NONE: a char present in no oov
        for (int attempt = 0; attempt < 256; ++attempt) {
            std::vector<std::string> oovs{ pool[static_cast<std::size_t>(anchor_idx)] };
            while (static_cast<int>(oovs.size()) < K) {
                const std::string& cand = pool[pick(rng)];
                if (std::find(oovs.begin(), oovs.end(), cand) == oovs.end()) oovs.push_back(cand);
            }
            std::string present; for (const std::string& o : oovs) present += o;
            int has = -1;
            for (int c = 'a'; c <= 'z'; ++c)
                if (present.find(static_cast<char>(c)) == std::string::npos) { has = c; break; }
            if (has < 0) continue;                                    // every letter appears -> retry (rare)
            std::shuffle(oovs.begin(), oovs.end(), rng);
            return { std::move(oovs), has, ANS_NONE };
        }
    } else if (kind == 2) {                                          // ALL: a char present in every oov
        for (int attempt = 0; attempt < 64; ++attempt) {
            const int c = 'a' + static_cast<int>(rng() % 26);
            std::vector<std::string> withc;                          // pool words containing c
            for (const std::string& w : pool)
                if (w.find(static_cast<char>(c)) != std::string::npos) withc.push_back(w);
            if (static_cast<int>(withc.size()) < K) continue;        // not enough sharers -> try another char
            std::vector<std::string> oovs;
            if (pool[static_cast<std::size_t>(anchor_idx)].find(static_cast<char>(c)) != std::string::npos)
                oovs.push_back(pool[static_cast<std::size_t>(anchor_idx)]);
            std::uniform_int_distribution<std::size_t> wpick(0, withc.size() - 1);
            for (int guard = 0; static_cast<int>(oovs.size()) < K && guard < 4000; ++guard) {
                const std::string& cand = withc[wpick(rng)];
                if (std::find(oovs.begin(), oovs.end(), cand) == oovs.end()) oovs.push_back(cand);
            }
            if (static_cast<int>(oovs.size()) < K) continue;
            std::shuffle(oovs.begin(), oovs.end(), rng);
            return { std::move(oovs), c, ANS_ALL };
        }
    }
    const ContainsPick p = pick_contains(pool, anchor_idx, K, rng);   // ONE (or NONE/ALL unconstructible)
    return { p.oovs, p.has, scratch_slot(p.qi) };
}
inline Task pick_content_contains_reason_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                                              int anchor_idx, int K, std::mt19937_64& rng) {
    const ContainsCase c = pick_contains_case(pool, anchor_idx, K, rng);
    return content_contains_reason_task(t, c.oovs, c.answer, c.has);
}
inline Task pick_content_contains_cot_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                                           int anchor_idx, int K, std::mt19937_64& rng) {
    const ContainsCase c = pick_contains_case(pool, anchor_idx, K, rng);
    return content_contains_cot_task(t, c.oovs, c.answer, c.has);
}
inline Task pick_content_contains_cot_local_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                                                 int anchor_idx, int K, std::mt19937_64& rng) {
    const ContainsCase c = pick_contains_case(pool, anchor_idx, K, rng);
    return content_contains_cot_task(t, c.oovs, c.answer, c.has, /*restate_query=*/true, /*drop_front=*/true);
}
// Confound control for the local variant: drops the front slot-list like local mode but does NOT restate the
// query -- isolates whether the win came from the restatement (expected) or just the shorter context.
inline Task pick_content_contains_cot_ctrl_task(const tok::Tokenizer& t, const std::vector<std::string>& pool,
                                                int anchor_idx, int K, std::mt19937_64& rng) {
    const ContainsCase c = pick_contains_case(pool, anchor_idx, K, rng);
    return content_contains_cot_task(t, c.oovs, c.answer, c.has, /*restate_query=*/false, /*drop_front=*/true);
}

inline Dataset build_dataset_contains(const tok::Tokenizer& t, const OovSplit& split, int K, const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_content_contains_task(t, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);
    Dataset ds; ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// A CONTAINS task picker: (tokenizer, pool, anchor, K, rng) -> Task. Lets one dataset builder serve every
// contains variant (plain-reason / CoT / CoT-local / CoT-control) -- train and eval pass the SAME picker.
// The Route A training set is build_dataset_contains_via(..., pick_content_contains_reason_task): each doc
// resolves every slot before answering; plain masked training teaches the procedure (no content embeddings).
using ContainsPicker = Task (*)(const tok::Tokenizer&, const std::vector<std::string>&, int, int, std::mt19937_64&);

// Generic contains-curriculum builder: one doc per (drilled oov x tasks_per_oov) via `picker`.
inline Dataset build_dataset_contains_via(const tok::Tokenizer& t, const OovSplit& split, int K,
                                          const DatasetOptions& opt, ContainsPicker picker) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(picker(t, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);
    Dataset ds; ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

// The PRODUCTION scratch curriculum: the two WORKING model-side capabilities -- multi-slot RESOLUTION
// (index a bound slot's chars, generalizes) + associative REASON-over-slots (select a slot by its
// in-context tag). Deliberately EXCLUDES content-select (#4, needs content-derived embeddings) and
// model-driven define (#1, copy-bottlenecked) -- neither is learnable model-side at this scale. The mix
// is 3:1 RESOLUTION-heavy on purpose: associative is EASY (saturates fast) and a 50/50 mix let its
// gradient starve the HARDER resolution task (resolution held-out dropped 0.31 solo -> 0.13); weighting
// toward resolution protects the core capability while still teaching the easy associative one.
// `contains_k >= 2` adds the third VALIDATED capability -- CONTENT reasoning over slots (which slot's OOV
// contains a queried char) via LOCAL-GROUNDED CoT (resolve each slot, restate the query beside it, verdict,
// answer). That trace is much LONGER (~5 + 11*contains_k) than resolution/associative, so the CALLER sizes
// contains_k to the model's SEQ_LEN (0 = omit -- e.g. a context too short to hold the trace).
//
// Mix = 45% content / 45% resolution / 10% associative. Difficulty ordering (measured): content-localization
// and resolution are BOTH HARD and COMPETE (notably on the shared UNCOMBINE pattern -- content continues to
// a verdict+slot, resolution to a char), while associative is EASY and saturates at any share. A minority
// share STARVES a hard task: 20% content -> content ~random (res/assoc fine); 30% resolution -> resolution
// ~random (content/assoc fine). So the two hard tasks split the bulk evenly and associative gets a sliver.
// (Both hard tasks still want more absolute steps than a short run gives; a real production run -- 10x+ the
// steps -- has ample budget for both.) contains_k=0 reproduces the original 75/25 resolution+associative
// stream exactly.
inline Dataset build_dataset_scratch(const tok::Tokenizer& t, const OovSplit& split, int K,
                                     const DatasetOptions& opt, int contains_k = 0) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    std::uniform_int_distribution<int> pick4(0, 3);       // 0..2 resolution (75%), 3 associative (25%)
    std::uniform_int_distribution<int> pick20(0, 19);     // 0..8 content / 9..17 resolve / 18..19 associative
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r) {
            if (contains_k >= 2) {
                const int roll = pick20(rng);
                docs.push_back(roll < 9  ? pick_content_contains_cot_local_task(t, split.drilled, qi, contains_k, rng)
                             : roll < 18 ? pick_multi_task(t, split.drilled, qi, K, rng)
                                         : pick_select_task(t, split.drilled, qi, K, rng));
            } else {
                docs.push_back(pick4(rng) < 3 ? pick_multi_task(t, split.drilled, qi, K, rng)
                                              : pick_select_task(t, split.drilled, qi, K, rng));
            }
        }
    std::shuffle(docs.begin(), docs.end(), rng);
    Dataset ds; ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

}  // namespace sub0::scratchspike
