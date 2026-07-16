// sub0/persistent_scratchspike.hpp -- SPIKE: does the resolve-a-named-reference skill
// (scratchspike.hpp's multi_nth_char_task) still generalize when bound to the PERSISTENT (id >= VOCAB,
// unbounded) slot range instead of the ephemeral 6-slot pool, at concurrent-slot counts K well past that
// pool's hard ceiling? See sub0/scratch_slots.hpp's PersistentBindings for the engine-level substrate
// this exercises, and tests/persistent_slots_engine_tests.cpp for the proof that an id >= VOCAB dispatches
// safely through encode_slot in forward/backward/forward_one.
//
// Deliberately a NARROWER curriculum than scratchspike's full task family: only ATTEND-ONLY tasks (the
// query already NAMES which slot to resolve; the model only ever produces ordinary in-vocab bytes as its
// graded answer) transfer to the persistent range. A persistent id has no logit column -- cross-entropy
// targets are hard-bounded to [0,VOCAB) -- so it can NEVER be a model's predicted/graded answer, unlike
// the ephemeral pool's SELECTION tasks (select_task/content_select_task/content_contains_task/
// content_contains_reason_task), which grade the model on literally emitting the chosen slot id. That
// "choose which entity to reference" capability does not transfer here; it is an open, unattempted
// research question -- see project memory persistent-slot-selection-problem-backlog for why, and do not
// try to solve or work around it in this file.
//
// Reuses scratchspike.hpp's OOV generation/split machinery (gen_oov/oov_bytes/OovSplit/make_oov_split are
// already slot-range-agnostic -- they never reference SCRATCH_BASE) -- only the TASK shape (which id
// range a slot reference resolves into) is new here, mirroring multi_nth_char_task's exact structure.
//
// Engine-free (tokenizer + std only, via scratchspike.hpp), unit-testable with no model -- same shape as
// scratchspike.hpp itself. Matching scratch_slots.hpp's own PersistentBindings::base design choice, `base`
// (the persistent range's start -- VOCAB in every real caller) is an explicit PARAMETER throughout this
// file, never a baked-in constant: this header deliberately has no core.hpp/generated-config dependency,
// so it doesn't know VOCAB -- callers that DO have it in scope (the engine test harness) pass it in.

#pragma once

#include "sub0/scratchspike.hpp"   // gen_oov/oov_bytes/OovSplit/make_oov_split (range-agnostic, reused)

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sub0::persistent_scratchspike {

// Reuse scratchspike's own framing bytes -- same question SHAPE ("# n <slot-ref>" -> Nth char), just a
// different slot range, so a model could in principle be trained on both curricula together later without
// the question syntax itself needing to disambiguate which range a reference belongs to (the id value
// alone, ephemeral vs >= VOCAB, already does that).
constexpr int Q_NTH = scratchspike::Q_NTH;
constexpr int SEP   = scratchspike::SEP;

// Slot i in the persistent range == base + i (PersistentBindings' own `base` convention, scratch_slots.hpp
// -- `base` is VOCAB in every real caller, passed explicitly per this file's own header comment). No
// pool-size ceiling, unlike scratchspike::SCRATCH_POOL's hard cap of 6 -- that absence is the entire point
// of this spike.
inline int persistent_slot(int base, int i) { return base + i; }

// Mirrors scratchspike::Task exactly (same fields, same meaning). Kept as its own type rather than reusing
// scratchspike::Task so this file's slot-range choice (persistent_slot, not scratch_slot) lives entirely
// in the CONSTRUCTION functions below, never smuggled through a shared struct whose "slot" meaning would
// otherwise be ambiguous between the two ranges.
struct Task {
    std::vector<int>          prompt;      // fed to the model at eval (the query context)
    std::vector<int>          trace;       // full baked sequence (training)
    std::vector<std::uint8_t> mask;        // parallel to trace: 1 = trained, 0 = masked (injected/context)
    int                       answer_byte = -1;
    std::string               oov;         // the QUERIED OOV (held-out bookkeeping)
    std::vector<std::string>  binds;       // slot i is pre-bound to binds[i] (harness); binds.size() == K
};

namespace detail {
inline void push(Task& k, int tok, std::uint8_t m) { k.trace.push_back(tok); k.mask.push_back(m); }
inline void append_query(Task& k, int tok) { push(k, tok, 0); k.prompt.push_back(tok); }
// The RESOLVE region the model GENERATES: TOK_UNCOMBINE <injected bytes> TOK_UNCOMBINE_END -- identical
// shape to scratchspike::detail::append_resolve (only the UNCOMBINE invocation is graded; the interceptor
// supplies the bytes from the persistent binding table, never memorized weights).
inline void append_resolve(Task& k, const std::vector<int>& bytes) {
    push(k, casing::TOK_UNCOMBINE, 1);
    for (int b : bytes) push(k, b, 0);
    push(k, casing::TOK_UNCOMBINE_END, 0);
}
}  // namespace detail

// MULTI-BINDING, PERSISTENT range: K distinct OOVs pre-bound to persistent ids base..base+K-1, a preamble
// lists all K (so they compete in attention + the KV-cache, same interference test
// scratchspike::multi_nth_char_task exercises), then queries the Nth char of ONE -- direct analog of that
// function; only the id range differs, and K carries no pool-size ceiling.
// prompt: [<persistent_slot_0..K-1>  # n <persistent_slot_q>]
// trace:  prompt [UNCOMBINE <oov_q bytes> UNCOMBINE_END] [= answer EOS]
inline Task multi_nth_char_task(int base, const std::vector<std::string>& oovs, int query_idx, int n) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    const std::vector<int> bytes = scratchspike::oov_bytes(k.oov);
    k.answer_byte = (n >= 0 && n < static_cast<int>(bytes.size())) ? bytes[static_cast<std::size_t>(n)] : '?';
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) detail::append_query(k, persistent_slot(base, i));
    detail::append_query(k, Q_NTH);
    detail::append_query(k, '0' + n);
    detail::append_query(k, persistent_slot(base, query_idx));
    detail::append_resolve(k, bytes);
    detail::push(k, SEP, 1);
    detail::push(k, k.answer_byte, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// Pick K distinct OOVs from `pool` (the query target first, then K-1 random distractors), shuffled so the
// target lands in a random slot; returns the built multi-binding task. Direct analog of
// scratchspike::pick_multi_task, shared by the dataset builder and the harness so training and eval use
// the identical construction.
inline Task pick_multi_task(int base, const std::vector<std::string>& pool, int target_idx, int K,
                            std::mt19937_64& rng) {
    std::vector<std::string> oovs{ pool[static_cast<std::size_t>(target_idx)] };
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    while (static_cast<int>(oovs.size()) < K) {
        const std::string& cand = pool[pick(rng)];
        if (std::find(oovs.begin(), oovs.end(), cand) == oovs.end()) oovs.push_back(cand);
    }
    std::shuffle(oovs.begin(), oovs.end(), rng);
    const int qi = static_cast<int>(std::find(oovs.begin(), oovs.end(),
                       pool[static_cast<std::size_t>(target_idx)]) - oovs.begin());
    const int len = static_cast<int>(pool[static_cast<std::size_t>(target_idx)].size());
    std::uniform_int_distribution<int> pn(0, len - 1);
    return multi_nth_char_task(base, oovs, qi, pn(rng));
}

// A flat training token stream + a parallel loss MASK + the document-start index the window sampler
// (sub0/window.hpp) consumes -- one task trace per document, so a window never straddles two bindings.
// `doc_bindings[d][i]` = document d's persistent slot i's fragment token ids. UNLIKE scratchspike::Dataset
// (where doc_bindings is only consulted when --content-embed is separately enabled, since the ephemeral
// pool's slot ids are real vocab rows with their own learnable embedding as a fallback), this is NOT
// optional here: a persistent id has no embedding-table row at all, so doc_bindings is the ONLY source of
// any representation for it -- training MUST install a PersistentBindings view of this per document/window.
struct Dataset {
    std::vector<int>                             tokens;
    std::vector<std::uint8_t>                    mask;
    std::vector<std::uint64_t>                   doc_starts;
    std::vector<std::vector<std::vector<int>>>   doc_bindings;
};

inline void append_doc(Dataset& ds, const Task& d) {
    ds.tokens.insert(ds.tokens.end(), d.trace.begin(), d.trace.end());
    ds.mask.insert(ds.mask.end(), d.mask.begin(), d.mask.end());
    ds.doc_starts.push_back(static_cast<std::uint64_t>(ds.tokens.size()));
    std::vector<std::vector<int>> b;
    for (const std::string& oov : d.binds) b.push_back(scratchspike::oov_bytes(oov));
    ds.doc_bindings.push_back(std::move(b));
}

struct DatasetOptions {
    int           tasks_per_oov = 8;
    std::uint64_t seed          = 20260716;
};

// Multi-binding training set: for each drilled OOV as the query target, `tasks_per_oov` docs each with
// K-1 random drilled distractors. Direct analog of scratchspike::build_dataset_multi.
inline Dataset build_dataset_multi(int base, const scratchspike::OovSplit& split, int K,
                                   const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_multi_task(base, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);

    Dataset ds;
    ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

}  // namespace sub0::persistent_scratchspike
