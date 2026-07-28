// sub0/window.hpp -- document-boundary-respecting training-window sampler.
//
// Training draws random windows from the token stream. Left to a flat uniform start, a window can
// straddle the `\n\n` boundary between two unrelated documents and teach the model to "continue"
// from one document into another -- exactly the broken context we want to avoid. Given the
// per-document start index the configurator records in corpus.tok, this samples a window that
// stays inside ONE document.
//
// A window also carries a `len`: the number of trained positions (input tokens whose next-token
// target is still inside the document). For a document long enough to host the step's full width T
// the window is full (len == T) and may start anywhere inside the document; for a SHORTER document
// the window is the whole document (len < T) and the caller pads the remainder, masking the loss
// on the padding -- so no document is dropped for being short, yet no window mixes two documents.
//
// Pure and std-only (no engine/config dependency) so it is unit-testable on a synthetic document
// index. With an EMPTY index (legacy corpus.tok / on-demand path) it degrades to a flat full-width
// window -- identical to the previous behaviour -- so callers need no special-casing.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

namespace sub0 {

// A sampled training window: `start` token index and `len` trained positions (1..T). The window
// covers input tokens [start, start+len) predicting targets [start+1, start+len]; positions
// [len, T) are padding the caller must loss-mask (only present when len < T, i.e. a short doc).
struct Window {
    std::size_t start = 0;
    int         len   = 0;
};

// --- corpus subsetting: train on a DISTRIBUTED fraction of the documents -----------------------
//
// Replaces the deleted data/fineweb_smoke.txt, a 1.07GB prefix file. A prefix was always the wrong
// shape: a second copy to keep in sync, and -- because fineweb_edu is not shuffled -- its first 25%
// is not 25% of the distribution, it is whatever the crawl happened to order first. This selects
// documents spread across the WHOLE corpus instead, so a subset run sees the real mixture.
//
// Membership is a hash of the document index, which buys several properties at once:
//   - distributed: selected documents are spread uniformly, not clustered at one end;
//   - document-aligned by construction, so a subset can never reintroduce the cross-document
//     window straddling this whole header exists to prevent;
//   - deterministic and seed-stable: (seed, fraction) reproduces a run exactly, with no subset
//     file to ship, store, or let drift out of sync with the corpus;
//   - complementary: the NOT-selected documents are a genuine held-out set from the same
//     distribution -- stronger than the current train/val tail split, which is positional;
//   - O(1) memory: no index, no prefix-sum table, nothing that scales with document count.
//
// splitmix64, the standard finalizer -- avalanches well enough that neighbouring document indices
// land independently, which a weaker mix (e.g. a plain multiply) would not give.
inline bool doc_in_subset(std::size_t doc_index, std::uint64_t subset_seed, double fraction) {
    if (fraction >= 1.0) return true;
    if (fraction <= 0.0) return false;
    std::uint64_t x = static_cast<std::uint64_t>(doc_index) + 0x9E3779B97F4A7C15ULL * (subset_seed | 1ULL);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^=  x >> 31;
    // Top 53 bits -> [0,1): exactly the mantissa width of a double, so the comparison is exact.
    return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0) < fraction;
}

// Sample a window of width up to T that stays within a single document. `docs` is the ascending
// list of document start token-indices (document 0 at index 0); `train_tok` is the number of
// trainable tokens (every index used stays in [0, train_tok)). With an empty index the result is a
// flat full-width window. Single-token documents are skipped (no input/target pair to learn from).
// `fraction` < 1 restricts sampling to the deterministic document subset defined by
// doc_in_subset(.., subset_seed, fraction); 1.0 (the default) is the whole corpus, bit-for-bit the
// previous behaviour -- the subset branch is not even reached.
inline Window sample_window(std::mt19937& rng, int T, std::size_t train_tok,
                            std::span<const std::uint64_t> docs,
                            double fraction = 1.0, std::uint64_t subset_seed = 0) {
    const std::size_t Tsz  = static_cast<std::size_t>(T);
    const std::size_t full = Tsz + 1;   // a full window needs T inputs + the last shifted target
    if (docs.empty()) {
        std::uniform_int_distribution<std::size_t> uni(0, train_tok - full);
        return { uni(rng), T };
    }
    // Rejection sampling, deliberately, rather than a precomputed table of selected documents.
    // Rejection preserves the sampler's existing TOKEN-uniform distribution exactly (drawing
    // uniformly over selected DOCUMENTS instead would bias toward short ones), and costs no memory
    // on a corpus that can hold tens of millions of documents. Expected draws per window is 1/f,
    // each an O(log ndocs) binary search -- negligible beside the step's GEMMs even at f = 0.05.
    const int budget = (fraction >= 1.0)
                     ? 8
                     : static_cast<int>(8.0 / (fraction > 0.001 ? fraction : 0.001)) + 8;
    std::uniform_int_distribution<std::size_t> uni(0, train_tok - 2);   // need >=1 (input,target) pair
    for (int tries = 0; tries < budget; ++tries) {
        const std::size_t pos = uni(rng);
        const std::size_t k = static_cast<std::size_t>(
            std::upper_bound(docs.begin(), docs.end(), static_cast<std::uint64_t>(pos)) - docs.begin()) - 1;
        if (!doc_in_subset(k, subset_seed, fraction)) continue;   // not in this run's subset
        const std::size_t ds = docs[k];
        std::size_t de = (k + 1 < docs.size()) ? static_cast<std::size_t>(docs[k + 1]) : train_tok;
        if (de > train_tok) de = train_tok;                 // a doc straddling the train/val split
        const std::size_t cap = de - ds - 1;                // max trained positions if we start at ds
        if (cap < 1) continue;                              // single-token doc: no pair to train on
        if (cap >= Tsz) {                                   // long doc: a full window, snapped in-doc
            std::size_t start = pos;
            if (start > de - full) start = de - full;       // keep [start, start+T] inside the doc
            return { start, T };
        }
        return { ds, static_cast<int>(cap) };               // short doc: whole document, caller pads
    }
    // Exhausted the budget. With a subset active the flat fallback below is NOT acceptable: it
    // ignores document boundaries, so it would reintroduce cross-document windows precisely in the
    // configuration where the subset was meant to keep sampling clean. Walk forward instead to the
    // next selected, trainable document -- bounded by the document count, and boundary-correct.
    if (fraction < 1.0) {
        const std::size_t nd = docs.size();
        std::uniform_int_distribution<std::size_t> pick(0, nd - 1);
        const std::size_t k0 = pick(rng);
        for (std::size_t i = 0; i < nd; ++i) {
            const std::size_t k = (k0 + i) % nd;
            if (!doc_in_subset(k, subset_seed, fraction)) continue;
            const std::size_t ds = docs[k];
            std::size_t de = (k + 1 < nd) ? static_cast<std::size_t>(docs[k + 1]) : train_tok;
            if (de > train_tok) de = train_tok;
            if (ds + 1 >= de) continue;                     // single-token doc
            const std::size_t cap = de - ds - 1;
            return { ds, static_cast<int>(cap >= Tsz ? Tsz : cap) };
        }
        return { 0, 1 };   // no selected document is trainable: caller's fraction is unusable
    }
    std::uniform_int_distribution<std::size_t> uni2(0, train_tok - full);
    return { uni2(rng), T };                                // pathological: fall back to a flat window
}

}  // namespace sub0
