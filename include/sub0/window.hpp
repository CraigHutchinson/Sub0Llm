// sub0/window.hpp -- document-boundary-respecting training-window start sampler.
//
// Training draws random fixed-or-variable-length windows from the token stream. Left to a flat
// uniform start, a window can straddle the `\n\n` boundary between two unrelated documents and
// teach the model to "continue" from one document into another -- exactly the broken, unnatural
// context we want to avoid. Given the per-document start index the configurator records in
// corpus.tok, this samples a start so the whole window stays inside ONE document.
//
// Pure and std-only (no engine/config dependency) so it is unit-testable on a synthetic document
// index without standing up a corpus. With an EMPTY index (an old corpus.tok with no boundary
// table, or the on-demand tokenization path) it degrades to a plain uniform start -- identical to
// the previous behaviour -- so callers need no special-casing.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

namespace sub0 {

// Sample a window start in [0, train_tok) such that the window inputs [start, start+T) and the
// last shifted target at index start+T all fall within a single document. `docs` is the ascending
// list of document start token-indices (document 0 begins at index 0); `train_tok` is the number
// of trainable tokens (the window must satisfy start+T < train_tok). When `docs` is empty the
// result is a plain uniform start. Documents shorter than T+1 tokens cannot hold the window; after
// a few rejections the sampler falls back to a flat window rather than loop (rare for real corpora).
inline std::size_t sample_window_start(std::mt19937& rng, int T, std::size_t train_tok,
                                       std::span<const std::uint32_t> docs) {
    const std::size_t span = static_cast<std::size_t>(T) + 1;   // inputs [start,start+T) + target start+T
    std::uniform_int_distribution<std::size_t> uni(0, train_tok - span);
    if (docs.empty()) return uni(rng);
    for (int tries = 0; tries < 8; ++tries) {
        std::size_t pos = uni(rng);
        // Document containing pos = the last start <= pos.
        const std::size_t k = static_cast<std::size_t>(
            std::upper_bound(docs.begin(), docs.end(), static_cast<std::uint32_t>(pos)) - docs.begin()) - 1;
        const std::size_t ds = docs[k];
        std::size_t de = (k + 1 < docs.size()) ? static_cast<std::size_t>(docs[k + 1]) : train_tok;
        if (de > train_tok) de = train_tok;                     // a doc straddling the train/val split
        if (de < ds + span) continue;                           // document too short for this T
        if (pos + span > de) pos = de - span;                   // snap so the window ends inside the doc
        return pos;
    }
    return uni(rng);   // pathologically short documents only: accept a flat window
}

}  // namespace sub0
