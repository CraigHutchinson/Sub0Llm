// sub0/blend.hpp -- BlendSource: the per-source data a corpus blend draws windows from.
//
// A training run may want to draw its windows from more than one source: the base corpus, plus a
// synthetic curriculum (sub0/scratchspike.hpp, sub0/op_curriculum.hpp, sub0/spellspike.hpp) or (per
// docs/ROADMAP.md) a conversational/reasoning corpus. Rather than concatenate them into one .tok (which
// fixes the mix ratio at build time and reshuffles the whole file to retune it), each source keeps its
// own token view + document index + weight, and each batch window picks a source and samples a window
// inside it. A generated source (no file) drops in the same way a mmap corpus does.
//
// The key enabler is that both backends already materialize each step's windows host-side (the CPU
// path copies each window into a contiguous buffer; GpuTrainer::step builds host ids[]/targets[] and
// uploads them) -- so "which source" is a host-side per-window choice, and a source that carries a
// loss MASK simply writes LOSS_IGNORE_INDEX into the masked target positions, reusing the ignore-index
// cross-entropy foundation (sub0/core.hpp) with no new device-side mask stream.
//
// Engine-free (TokView + the window sampler + std only), so it is unit-testable with no model.
//
// The actual source-selection scheduler lives in sub0/blend_schedule.hpp (a staged, epoch-fair deficit
// scheduler -- see that file's header comment for why: a flat per-draw weighted-random pick, which used
// to live here as pick_source/sample_blend, let a small source relative to a huge one get statistically
// re-covered dozens of times before the huge one finished a single epoch, a real production incident).

#pragma once

#include "sub0/tokmap.hpp"   // TokView (a width-agnostic read view over a token stream)
#include "sub0/window.hpp"   // Window + sample_window (document-boundary-respecting sampler)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace sub0 {

// One training source. `view` + `docs` are exactly what sample_window consumes (a token stream and its
// ascending document-start index; an empty `docs` degrades to a flat full-width window). `mask`, when
// non-empty, is a per-token 0/1 array parallel to `view` (1 = the model must PRODUCE this token, graded;
// 0 = masked -> LOSS_IGNORE_INDEX, no loss/grad) -- the same interior mask train_batch's loss_mask
// consumes. `doc_bindings`, when non-empty, is per-document slot->fragment content (scratchspike/
// op_curriculum sources only -- a plain corpus or spellspike source leaves it empty, meaning no
// content-embed binding applies to any of its windows). `name` matches a ScheduleStage::weights key
// (blend_schedule.hpp) so the scheduler can resolve this source's target rate per stage. The caller owns
// all backing storage (a mmap corpus, or an in-memory generated Dataset); this struct only borrows spans.
struct BlendSource {
    std::string                                    name;
    TokView                                        view;
    std::span<const std::uint64_t>                 docs;
    std::span<const std::uint8_t>                  mask;          // empty => every position trained
    std::span<const std::vector<std::vector<int>>> doc_bindings;  // empty => no content-embed binding

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

// A blended draw: which source, and a window sampled inside it (same Window contract as
// sample_window -- [start, start+len) inputs predicting [start+1, start+len]). Produced by
// sub0/blend_schedule.hpp's sample_blend_staged, the sole source-selection mechanism now (see that
// file's header comment for why the flat-weight picker that used to live here was replaced).
struct BlendDraw {
    int    src = 0;
    Window win;
};

}  // namespace sub0
