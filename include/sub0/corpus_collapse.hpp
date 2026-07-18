// sub0/corpus_collapse.hpp -- scales wordspike's proven harness-driven compound-word collapse to a
// sampled subset of REAL base-corpus documents, instead of only a synthetic GSM8K-style template.
//
// wordspike.hpp proved (0.95 vs 0.13 held-out over repeatspike's synthetic passage, and op-delegation
// composing at 1.00 accuracy in its own natural-prose capstone) that a recurring multi-piece word's later
// mentions can collapse to a bound scratch slot with no cost to downstream reasoning, when the collapse is
// resolved by the harness rather than asked of the model. This header applies the SAME mechanism
// (ScratchTable::combine_recurrence, sub0::detail::word_span -- both already built in scratch.hpp, reused
// as-is, no new low-level logic) to real corpus text: a sampled subset of documents from the corpus.tok a
// training run is already using.
//
// Masking discipline is deliberately DIFFERENT from wordspike's own: wordspike masks essentially all prose
// immediately around an invented name (mask=0) because that content is synthetic and unlearnable either
// way. Real corpus prose is normal, predictable, learnable language, so only the exact substituted slot
// position is masked (mask=0) -- everything else, including a word's own first (spelled-out) mention,
// stays graded (mask=1), matching op_curriculum.hpp's own result-collapse precedent instead.
//
// Deliberately NOT a step toward the separate "persistent slot-range" substrate (unbounded ids >= VOCAB,
// cross-document) -- this reuses wordspike's bounded, per-document-reset 6-slot pool. See
// docs/CORPUS_COLLAPSE.md for the full design record, scope corrections, and known limitations (prefix
// truncation under-exercises recurrence that lives deep in long documents -- a named, accepted v1 gap, not
// silently glossed over).
//
// Engine-free (tokenizer + tokmap + scratch.hpp + std only), matching every other curriculum header here.

#pragma once

#include "sub0/scratch.hpp"
#include "sub0/tokmap.hpp"
#include "sub0/tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace sub0::corpus_collapse {

// Same Dataset shape as wordspike/op_curriculum/scratchspike -- what BlendSource consumes. `doc_starts`
// uses the "N+1 entries, trailing sentinel = total token count" convention every Dataset builder in this
// codebase follows (doc i spans [doc_starts[i], doc_starts[i+1])) -- NOT the raw corpus.tok on-disk
// convention (N entries, no trailing sentinel), which this header's INPUT `doc_starts` parameter uses
// instead; see build_dataset's own comments for exactly where that distinction matters.
struct Dataset {
    std::vector<int>                           tokens;
    std::vector<std::uint8_t>                  mask;
    std::vector<std::uint64_t>                 doc_starts;
    std::vector<std::vector<std::vector<int>>> doc_bindings;
};

struct Options {
    std::uint64_t seed           = 0;
    int           n_docs         = 3000;   // sampled subset size, matching wordspike's own default scale
    int           max_doc_tokens = 0;      // prefix-truncate cap; 0 = uncapped (caller derives from SEQ_LEN)
};

// Samples up to `opt.n_docs` documents (uniformly at random via `opt.seed`, reservoir sampling so a
// multi-million-document corpus never needs a fully materialized index list) from `base_tokens`/`doc_starts`
// -- the SAME already-tokenized, already-mmap'd corpus a training run's "base" source is already using
// (`doc_starts` here is the RAW corpus.tok convention: `doc_starts[d]` is document d's start, its end is
// `doc_starts[d+1]` or `train_tok` for the last document -- exactly `window.hpp`'s own convention, not this
// header's OUTPUT convention). For each sampled document: clamps its end to `train_tok` (mirrors
// `window.hpp:56`'s `if (de > train_tok) de = train_tok;` exactly -- a document straddling the train/val
// split must not leak held-out tokens into a training curriculum), prefix-truncates to `max_doc_tokens` if
// set (preserves yield from long documents rather than dropping them; see docs/CORPUS_COLLAPSE.md for why
// this is an accepted v1 limitation, not a bug), then walks it with a FRESH `ScratchTable` (reset per
// document -- recurrence must never leak across unrelated documents) via the same `detail::word_span` +
// `combine_recurrence` walk `prefill_collapse` already uses for prompt text. `mask=1` for everything except
// the exact position(s) of a substituted slot (`mask=0` there, harness-injected, matching op_curriculum's
// own result-collapse masking, not wordspike's more aggressive whole-mention masking -- real corpus prose
// is normal, learnable language, unlike a random invented name).
inline Dataset build_dataset(const tok::Tokenizer& tk, TokView base_tokens,
                              std::span<const std::uint64_t> doc_starts, std::size_t train_tok,
                              const Options& opt) {
    Dataset ds;
    ds.doc_starts.push_back(0);
    if (doc_starts.empty()) return ds;   // no document boundaries -- caller must pre-check and warn

    // Algorithm R reservoir sampling: a uniform random subset of `want` document indices out of
    // `doc_starts.size()`, in one pass, with no full index list ever materialized (a production corpus
    // can have millions of documents).
    const std::size_t n_total_docs = doc_starts.size();
    const std::size_t want = std::min(static_cast<std::size_t>(std::max(opt.n_docs, 0)), n_total_docs);
    std::mt19937_64 rng(opt.seed);
    std::vector<std::size_t> chosen;
    chosen.reserve(want);
    for (std::size_t d = 0; d < n_total_docs; ++d) {
        if (chosen.size() < want) {
            chosen.push_back(d);
        } else {
            std::uniform_int_distribution<std::size_t> pick(0, d);
            const std::size_t j = pick(rng);
            if (j < want) chosen[j] = d;
        }
    }
    std::sort(chosen.begin(), chosen.end());   // deterministic file-order processing (not required, just tidy)

    const std::size_t max_doc = opt.max_doc_tokens > 0 ? static_cast<std::size_t>(opt.max_doc_tokens)
                                                        : std::numeric_limits<std::size_t>::max();
    std::vector<int> doc_buf;
    for (std::size_t d : chosen) {
        const std::size_t ds_pos = static_cast<std::size_t>(doc_starts[d]);
        std::size_t de = (d + 1 < doc_starts.size()) ? static_cast<std::size_t>(doc_starts[d + 1]) : train_tok;
        if (de > train_tok) de = train_tok;                        // val-split clamp -- see window.hpp:56
        if (ds_pos >= de) continue;                                // doc entirely past the split, or empty
        std::size_t len = de - ds_pos;
        if (len > max_doc) len = max_doc;                          // prefix-truncate, preserve yield
        if (len < 2) continue;                                     // need >=1 (input,target) pair

        doc_buf.resize(len);
        base_tokens.copy_to(ds_pos, len, doc_buf.data());

        ScratchTable table;
        table.tk = &tk;   // required: without it, combine_recurrence can't recognize an ordinary vocab
                          // piece via piece_index and would wrongly treat every common word as bindable
        for (std::size_t i = 0; i < doc_buf.size(); ) {
            const auto [span_len, pieces] = detail::word_span(doc_buf, i);
            if (pieces.empty()) {   // unterminated SPELL span (truncation cut mid-word) -- pass through
                ds.tokens.push_back(doc_buf[i]); ds.mask.push_back(1);
                i += span_len; continue;
            }
            std::vector<int> bytes;
            for (int p : pieces) {
                const std::vector<int> e = table.expand(p);
                bytes.insert(bytes.end(), e.begin(), e.end());
            }
            const ScratchTable::Recurrence r = table.combine_recurrence(bytes);
            if (r.is_repeat) {
                for (int t : r.tokens) { ds.tokens.push_back(t); ds.mask.push_back(0); }   // harness-injected
            } else {
                for (std::size_t k = i; k < i + span_len; ++k) {
                    ds.tokens.push_back(doc_buf[k]); ds.mask.push_back(1);   // real, graded prose
                }
            }
            i += span_len;
        }
        ds.doc_starts.push_back(ds.tokens.size());
        ds.doc_bindings.push_back(std::move(table.bindings));
    }
    return ds;
}

}  // namespace sub0::corpus_collapse
