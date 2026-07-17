// sub0/pairspike.hpp -- SPIKE (Phase 2 of docs/SCRATCH_TOKENS.md's word-level plan): the `<sigil> h`
// sentinel-PAIR addressing curriculum. Phase 1 ([.chaincapstone]'s HANDLE arm, 1.000 on every seed)
// proved a plain-byte handle pair fully replaces a reserved-id reference when the model only has to
// RE-EMIT a reference it was shown. This phase asks the harder, novel question the pair mechanism
// exists for: can the model CHOOSE which entity to reference -- content-select over K bound entities,
// answered by EMITTING the right pair -- when the pair's index position carries the entity's composed
// content vector (SentinelBindings' embedding override, scratch_slots.hpp)?
//
// Why this can't be asked of either existing range: the bounded pool's content_select_task caps at 6
// dedicated ids; the persistent range can never be a prediction target at all (no logit column). The
// pair has BOTH properties -- unbounded handle space AND ordinary next-token prediction -- so K here
// deliberately exceeds the pool's ceiling (the test's K=12 needs 12 concurrent addressable entities,
// impossible for either predecessor).
//
// The A/B this curriculum feeds: WITH the embedding override (pairs carry content) vs WITHOUT it (pairs
// embed as their plain rows). Without the override, the preamble contains NO information linking a
// handle to its entity's spelling -- first-letter queries are unanswerable in principle, pinning the
// chance floor -- so the gap directly measures what the override carries. (Contrast Phase 1, where the
// binding-site ASSOCIATION was in-stream and plain pairs sufficed.)
//
// Engine-free (scratchspike.hpp's OOV machinery + std only); the sigil id is caller-supplied (the spike
// commandeers casing::TOK_RESERVED_9 -- see SentinelBindings' own comment for the permanent-id plan).

#pragma once

#include "sub0/scratchspike.hpp"   // gen_oov/oov_bytes/OovSplit/make_oov_split (range-agnostic, reused)

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace sub0::pairspike {

constexpr int Q_FST = scratchspike::Q_FST;   // '!' -- "which entity starts with char c" (same shape as
                                             // scratchspike's content_select_task, different id space)
constexpr int SEP   = scratchspike::SEP;     // '='

// Handle for entity i: an ordinary lowercase-letter byte token. 'a'..'z' bounds K at 26 -- far past the
// bounded pool's 6, plenty for the spike; a production handle space could use any token subset.
constexpr int HANDLE_BASE = 'a';
inline int handle_token(int i) { return HANDLE_BASE + i; }

struct Task {
    std::vector<int>          prompt;      // fed to the model at eval (the query context)
    std::vector<int>          trace;       // full baked sequence (training)
    std::vector<std::uint8_t> mask;        // parallel to trace: 1 = trained, 0 = context
    int                       answer_handle = -1;   // the handle token of the queried entity
    std::string               oov;         // the QUERIED entity (held-out bookkeeping)
    std::vector<std::string>  binds;       // handle i is bound to binds[i]; binds.size() == K
};

namespace detail {
inline void push(Task& k, int tok, std::uint8_t m) { k.trace.push_back(tok); k.mask.push_back(m); }
inline void append_query(Task& k, int tok) { push(k, tok, 0); k.prompt.push_back(tok); }
}  // namespace detail

// CONTENT-select via pairs: K entities bound to handles a..; the preamble lists the K pairs (each pair's
// index position embeds the entity's content under the override -- the ONLY place its spelling can enter
// the model, the entities' bytes are never in the stream); the query names a first letter; the model
// answers by EMITTING the matching pair -- sigil AND handle both graded (the selection is an ordinary
// 2-token prediction). prompt: [<sigil> h_0 .. <sigil> h_{K-1} ! c]   trace: prompt [= <sigil> h_q EOS]
inline Task select_pair_task(int sigil, const std::vector<std::string>& oovs, int query_idx) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) {
        detail::append_query(k, sigil);
        detail::append_query(k, handle_token(i));
    }
    detail::append_query(k, Q_FST);
    detail::append_query(k, static_cast<int>(static_cast<unsigned char>(k.oov[0])));
    k.answer_handle = handle_token(query_idx);
    detail::push(k, SEP, 1);
    detail::push(k, sigil, 1);                    // graded: the model chooses to emit a pair...
    detail::push(k, k.answer_handle, 1);          // ...and WHICH handle (the actual selection)
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

// Pick K entities with DISTINCT first letters (target + distractors), shuffled so the target lands on a
// random handle -- the same construction scratchspike::pick_content_select_task uses, so the two tasks
// stay directly comparable. Shared by the dataset builder and the eval harness.
inline Task pick_select_pair_task(int sigil, const std::vector<std::string>& pool, int target_idx, int K,
                                  std::mt19937_64& rng) {
    std::vector<std::string> oovs{ pool[static_cast<std::size_t>(target_idx)] };
    std::vector<char> firsts{ pool[static_cast<std::size_t>(target_idx)][0] };
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    for (int guard = 0; static_cast<int>(oovs.size()) < K && guard < 4000; ++guard) {
        const std::string& cand = pool[pick(rng)];
        if (std::find(firsts.begin(), firsts.end(), cand[0]) == firsts.end()) {
            oovs.push_back(cand); firsts.push_back(cand[0]);
        }
    }
    std::shuffle(oovs.begin(), oovs.end(), rng);
    const int qi = static_cast<int>(std::find(oovs.begin(), oovs.end(),
                       pool[static_cast<std::size_t>(target_idx)]) - oovs.begin());
    return select_pair_task(sigil, oovs, qi);
}

// ROUTE-A variant (user-directed after the one-hop K=12 null): first-letter queries TRACK THROUGH THE
// UNCOMBINE instead of demanding one-hop extraction from the composed embedding -- the exact
// local-grounded CoT scaffold that cracked content localization for the bounded pool
// (scratchspike::content_contains_cot_task with restate_query, held-out 1.000 where one-hop sat at
// chance). Per candidate the model EMITS the pair to inspect + UNCOMBINE (graded -- it must choose to
// resolve), the interceptor injects the entity's bytes (masked -- exact, from the binding table, never
// memorized), the query char is restated LOCALLY beside them (graded), and a +/- verdict follows
// (graded); the final answer is then "copy the pair that got '+'" -- a copy-class step Phase 1 proved
// trivial for pairs. Content enters via INJECTED BYTES, so this route needs no embedding override for
// correctness -- the pair serves as a plain 2-token symbol; the override becomes an optional
// acceleration to A/B, not a load-bearing requirement. Front pair-list dropped (the resolve phase
// re-emits every pair anyway), matching the proven local-CoT trace shape.
// prompt: [! c]   trace: [! c] { <S> h_i UNCOMBINE <bytes_i> UNCOMBINE_END c <verdict_i> }*K [= <S> h_q EOS]
inline Task select_pair_cot_task(int sigil, const std::vector<std::string>& oovs, int query_idx) {
    Task k; k.binds = oovs; k.oov = oovs[static_cast<std::size_t>(query_idx)];
    const int qc = static_cast<int>(static_cast<unsigned char>(k.oov[0]));
    detail::append_query(k, Q_FST);
    detail::append_query(k, qc);
    for (int i = 0; i < static_cast<int>(oovs.size()); ++i) {
        detail::push(k, sigil, 1);                                 // graded: emit the pair to inspect...
        detail::push(k, handle_token(i), 1);                       // ...and WHICH one (fixed a.. order)
        detail::push(k, casing::TOK_UNCOMBINE, 1);                 // graded: choose to resolve it
        for (int b : scratchspike::oov_bytes(oovs[static_cast<std::size_t>(i)]))
            detail::push(k, b, 0);                                 // interceptor-injected spelling
        detail::push(k, casing::TOK_UNCOMBINE_END, 0);
        detail::push(k, qc, 1);                                    // graded: restate the query LOCALLY
        const bool match = oovs[static_cast<std::size_t>(i)][0] == static_cast<char>(qc);
        detail::push(k, match ? scratchspike::VERD_YES : scratchspike::VERD_NO, 1);   // graded verdict
    }
    k.answer_handle = handle_token(query_idx);
    detail::push(k, SEP, 1);
    detail::push(k, sigil, 1);
    detail::push(k, k.answer_handle, 1);
    detail::push(k, casing::TOK_EOS, 1);
    return k;
}

inline Task pick_select_pair_cot_task(int sigil, const std::vector<std::string>& pool, int target_idx,
                                      int K, std::mt19937_64& rng) {
    std::vector<std::string> oovs{ pool[static_cast<std::size_t>(target_idx)] };
    std::vector<char> firsts{ pool[static_cast<std::size_t>(target_idx)][0] };
    std::uniform_int_distribution<std::size_t> pick(0, pool.size() - 1);
    for (int guard = 0; static_cast<int>(oovs.size()) < K && guard < 4000; ++guard) {
        const std::string& cand = pool[pick(rng)];
        if (std::find(firsts.begin(), firsts.end(), cand[0]) == firsts.end()) {
            oovs.push_back(cand); firsts.push_back(cand[0]);
        }
    }
    std::shuffle(oovs.begin(), oovs.end(), rng);
    const int qi = static_cast<int>(std::find(oovs.begin(), oovs.end(),
                       pool[static_cast<std::size_t>(target_idx)]) - oovs.begin());
    return select_pair_cot_task(sigil, oovs, qi);
}

// Same Dataset shape as the sibling curricula: one task trace per document; doc_bindings[d][i] = the
// fragments handle i is bound to in document d (what the training loop installs via SentinelBindings).
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
    int           tasks_per_oov = 12;
    std::uint64_t seed          = 20260717;
};

inline Dataset build_dataset_select_pair(int sigil, const scratchspike::OovSplit& split, int K,
                                         const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_select_pair_task(sigil, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);

    Dataset ds;
    ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

inline Dataset build_dataset_select_pair_cot(int sigil, const scratchspike::OovSplit& split, int K,
                                             const DatasetOptions& opt) {
    std::vector<Task> docs;
    std::mt19937_64 rng(opt.seed);
    for (int qi = 0; qi < static_cast<int>(split.drilled.size()); ++qi)
        for (int r = 0; r < opt.tasks_per_oov; ++r)
            docs.push_back(pick_select_pair_cot_task(sigil, split.drilled, qi, K, rng));
    std::shuffle(docs.begin(), docs.end(), rng);

    Dataset ds;
    ds.doc_starts.push_back(0);
    for (const Task& d : docs) append_doc(ds, d);
    return ds;
}

}  // namespace sub0::pairspike
