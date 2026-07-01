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

// Sample a window of width up to T that stays within a single document. `docs` is the ascending
// list of document start token-indices (document 0 at index 0); `train_tok` is the number of
// trainable tokens (every index used stays in [0, train_tok)). With an empty index the result is a
// flat full-width window. Single-token documents are skipped (no input/target pair to learn from).
inline Window sample_window(std::mt19937& rng, int T, std::size_t train_tok,
                            std::span<const std::uint64_t> docs) {
    const std::size_t Tsz  = static_cast<std::size_t>(T);
    const std::size_t full = Tsz + 1;   // a full window needs T inputs + the last shifted target
    if (docs.empty()) {
        std::uniform_int_distribution<std::size_t> uni(0, train_tok - full);
        return { uni(rng), T };
    }
    std::uniform_int_distribution<std::size_t> uni(0, train_tok - 2);   // need >=1 (input,target) pair
    for (int tries = 0; tries < 8; ++tries) {
        const std::size_t pos = uni(rng);
        const std::size_t k = static_cast<std::size_t>(
            std::upper_bound(docs.begin(), docs.end(), static_cast<std::uint64_t>(pos)) - docs.begin()) - 1;
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
    std::uniform_int_distribution<std::size_t> uni2(0, train_tok - full);
    return { uni2(rng), T };                                // pathological: fall back to a flat window
}

}  // namespace sub0
