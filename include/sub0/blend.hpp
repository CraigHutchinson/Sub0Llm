// sub0/blend.hpp -- weighted, on-the-fly, per-window corpus blending.
//
// A training run may want to draw its windows from more than one source: the base corpus, plus (per
// docs/ROADMAP.md) a conversational corpus, a reasoning corpus, and the synthetic combine/uncombine
// curriculum (sub0/spellspike.hpp). Rather than concatenate them into one .tok (which fixes the mix
// ratio at build time and reshuffles the whole file to retune it), each source keeps its own token
// view + document index + weight, and EVERY batch window independently picks a source in proportion
// to the weights and samples a window inside it. Retuning the mix is then just a weight change; a
// generated source (no file) drops in the same way a mmap corpus does.
//
// The key enabler is that both backends already materialize each step's windows host-side (the CPU
// path copies each window into a contiguous buffer; GpuTrainer::step builds host ids[]/targets[] and
// uploads them) -- so "which source" is a host-side per-window choice, and a source that carries a
// loss MASK simply writes LOSS_IGNORE_INDEX into the masked target positions, reusing the ignore-index
// cross-entropy foundation (sub0/core.hpp) with no new device-side mask stream.
//
// Engine-free (TokView + the window sampler + std only), so it is unit-testable with no model. A
// single source consumes NO rng draw (pick_source short-circuits), so a one-source blend leaves the
// window-sampling rng stream byte-identical to the pre-blend code -- resume determinism is preserved.

#pragma once

#include "sub0/tokmap.hpp"   // TokView (a width-agnostic read view over a token stream)
#include "sub0/window.hpp"   // Window + sample_window (document-boundary-respecting sampler)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace sub0 {

// One weighted training source. `view` + `docs` are exactly what sample_window consumes (a token
// stream and its ascending document-start index; an empty `docs` degrades to a flat full-width
// window). `mask`, when non-empty, is a per-token 0/1 array parallel to `view` (1 = the model must
// PRODUCE this token, graded; 0 = masked -> LOSS_IGNORE_INDEX, no loss/grad) -- the same interior
// mask train_batch's loss_mask consumes. The caller owns all backing storage (a mmap corpus, or an
// in-memory generated buffer); this struct only borrows the spans.
struct BlendSource {
    TokView                        view;
    std::span<const std::uint64_t> docs;
    std::span<const std::uint8_t>  mask;      // empty => every position trained
    double                         weight = 1.0;

    bool masked() const { return !mask.empty(); }
};

// The index of the document owning corpus position `pos`, given its ascending `doc_starts` (doc i spans
// [doc_starts[i], doc_starts[i+1])). Used to recover a training window's source document from a source's
// `docs` span + a sampled window start -- e.g. to look up per-document side data (like a scratch-mix
// window's slot->fragment bindings) that a plain token+mask stream doesn't itself carry. Precondition:
// `doc_starts` is non-empty and ascending (guaranteed by every Dataset builder in this codebase) and
// `pos < doc_starts.back()`.
inline std::size_t doc_of(std::span<const std::uint64_t> doc_starts, std::size_t pos) {
    return static_cast<std::size_t>(
        std::upper_bound(doc_starts.begin(), doc_starts.end(), pos) - doc_starts.begin() - 1);
}

// Pick a source index in proportion to the weights. A single source returns 0 WITHOUT drawing from
// `rng` -- so the common one-source run's window-sampling rng stream (hence its resume point) is
// unchanged by the blender's mere presence. Weights need not sum to 1 (normalized by their total).
inline int pick_source(std::mt19937& rng, const std::vector<BlendSource>& sources) {
    if (sources.size() <= 1) return 0;
    double total = 0.0;
    for (const BlendSource& s : sources) total += s.weight;
    std::uniform_real_distribution<double> u(0.0, total);
    double r = u(rng);
    for (int i = 0; i < static_cast<int>(sources.size()); ++i) {
        r -= sources[static_cast<std::size_t>(i)].weight;
        if (r < 0.0) return i;
    }
    return static_cast<int>(sources.size()) - 1;   // guard against fp drift on the last bucket
}

// A blended draw: which source, and a window sampled inside it (same Window contract as
// sample_window -- [start, start+len) inputs predicting [start+1, start+len]).
struct BlendDraw {
    int    src = 0;
    Window win;
};

// Pick a source by weight, then sample a document-respecting window inside it. For a single source
// this is exactly sample_window on that source (no extra rng draw) -- see pick_source.
inline BlendDraw sample_blend(std::mt19937& rng, int T, const std::vector<BlendSource>& sources) {
    const int s = pick_source(rng, sources);
    const BlendSource& src = sources[static_cast<std::size_t>(s)];
    return { s, sample_window(rng, T, src.view.size(), src.docs) };
}

}  // namespace sub0
