// sub0/blend_schedule.hpp -- staged, epoch-fair corpus blend scheduling.
//
// Replaces the flat per-draw-fraction blending in blend.hpp (pick_source/sample_blend, both deleted by
// this change -- confirmed exactly 2 call sites existed in the whole tree, train_stage.cpp and
// blend_tests.cpp, so there was no other caller to preserve). The flat scheme drew a source in proportion
// to a FIXED weight on every window, forever: `scratch_mix=0.3` meant "30% of every step's windows," with
// zero tracking of how much of a source had actually been covered. A small curriculum blended against a
// huge base corpus got statistically re-covered every few steps, hit a memorization phase-transition
// almost immediately, and its cratering loss contribution produced a visible cliff-then-plateau in the
// blended training loss while the base corpus was still a tiny fraction into its own first epoch -- a real
// incident this design fixes structurally, not by tuning a magic number (project memory
// hybrid-cpu-gpu-execution-design's sibling doc, gpu-large-batch-access-violation, and this file's own
// motivating diagnosis are recorded in project history; see docs/DETERMINISTIC_MECHANISMS.md).
//
// The mechanism: a deficit / weighted-fair scheduler (the same family as Linux CFS's argmin(vruntime)
// task scheduling and network WFQ packet scheduling). Track each source's own cumulative "epoch progress"
// (tokens drawn / its own size) and always draw from whichever ACTIVE source is furthest behind its
// target pace -- by default every source's target pace is EQUAL (equal epoch coverage regardless of
// size), matching a real, established precedent (GPT-3's own per-dataset epoch-count table, Brown et al.
// 2020 Table 2.2, deliberately let small high-quality corpora run multiple epochs while a huge corpus
// stayed under one). A STAGED schedule generalizes this to curriculum-learning-style mixtures that change
// over the course of training (e.g. "corpus A alone until epoch 0.5, then B ramps in to 25% until 0.9") --
// each stage's per-source weight is a target epoch-rate RATIO relative to its peers in that stage, not a
// sampling fraction.
//
// Engine-free (TokView/window-sampler/std only, mirroring blend.hpp's own testability property) so the
// scheduler math is unit-testable with no model and no JSON parser -- the simdjson-based parser that turns
// a schedule file into a ScheduleSpec lives in blend_schedule.cpp, out-of-line, for the same DLL-boundary
// reason registry.hpp documents for read_config_json (simdjson types never cross sub0_core.dll).

#pragma once

#include "sub0/blend.hpp"          // BlendSource, BlendDraw
#include "sub0/scratch_slots.hpp"  // ContentEmbedKind (engine-free: only depends on casing.hpp)
#include "sub0/window.hpp"         // sample_window

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace sub0 {

// Export macro for parse_blend_schedule_json's out-of-line definition (src/blend_schedule.cpp, compiled
// once into sub0_core) -- self-contained copy of core.hpp's own SUB0_API, matching registry.hpp's own
// precedent for read_config_json (this header is deliberately engine-free, no core.hpp dependency).
#ifndef SUB0_API
  #if defined(_WIN32)
    #define SUB0_API __declspec(dllexport)
  #else
    #define SUB0_API __attribute__((visibility("default")))
  #endif
#endif

// One declared blend source, as parsed from the schedule JSON -- not yet materialized into a BlendSource
// (that needs the actual corpus/tokenizer to invoke a generator against). Exactly one of `corpus` /
// `generator` is set (parse-time validated, not enforced by the type -- keeping this a plain data struct
// keeps the parser and the scheduler independently testable).
struct SourceSpec {
    std::string name;
    // non-empty => this is the file-backed base-corpus source (at most one per schedule, and exactly one
    // is REQUIRED -- see blend_schedule.cpp's validation: `epoch` is defined against the base corpus's
    // own token count, so a schedule with none would have no meaningful epoch clock). The VALUE is not
    // read back against the CLI's actual corpus path anywhere -- train_stage.cpp always trains whatever
    // corpus it was invoked with regardless of what string is written here; treat this field as the
    // schedule author's own label/reminder of which corpus the schedule was written for, not a path the
    // engine resolves or validates.
    std::string corpus;
    std::string generator;   // non-empty => "scratchspike" | "op_curriculum" | "spellspike" | "wordspike"
                              //              | "corpus_collapse"
    std::string gsm8k_path;  // op_curriculum only: a real GSM8K file instead of the synthetic generator
    // Generator knobs, defaulting to today's train_stage.cpp-hardcoded values when unset (-1 sentinel).
    // Safety clamps (SCRATCH_POOL, the SEQ_LEN-derived contains_k/max_digits ceilings) are enforced by the
    // caller that actually invokes the generator, not here -- this struct only carries the parsed request.
    int    n_oov         = -1;   // scratchspike: OOV pool size (default 400)
    int    tasks_per_oov = -1;   // scratchspike (default 12)
    int    contains_k    = -1;   // scratchspike: content-reasoning sub-curriculum size (default derived)
    int    n_examples    = -1;   // op_curriculum synthetic (default 6000); also corpus_collapse's
                                  // sampled-document count (n_docs, default 3000)
    double chain_frac    = -1.0; // op_curriculum synthetic (default 0.4)
    int    max_digits    = -1;   // op_curriculum synthetic (default derived)
    int    tasks_per_word = -1;  // spellspike (default 12)
};

// One schedule stage: active from the PREVIOUS stage's until_epoch up to this one's (exclusive), or from
// 0 for the first stage. `until_epoch = +infinity` is the "end" sentinel (JSON `"until_epoch": "end"`) --
// must be the LAST stage (parser-validated). `weights` is source-name -> target epoch-rate ratio, RELATIVE
// to its peers ALSO active in this stage (not a fraction of the whole run, and not comparable in
// magnitude across sources absent from this stage). A source with no entry (or weight <= 0) is INACTIVE
// for this stage's duration -- never drawn until a later stage reactivates it.
struct ScheduleStage {
    double until_epoch = std::numeric_limits<double>::infinity();
    std::vector<std::pair<std::string, double>> weights;   // insertion order preserved; small (<=~4 sources)
};

// The full parsed, validated schedule -- everything sub0_train_stage needs to build real BlendSources and
// drive the scheduler, plus the two whole-run settings (content_embed, expected_plateau_epoch) that apply
// across every stage rather than varying per-stage.
struct ScheduleSpec {
    std::vector<SourceSpec>   sources;
    std::vector<ScheduleStage> stages;    // ascending by until_epoch; last entry's until_epoch == +inf
    std::optional<ContentEmbedKind> content_embed;       // nullopt = off (plain reserved-id embeddings)
    std::optional<double>           expected_plateau_epoch;
};

// Parse + validate a schedule JSON file into `out`. Returns false on any structural problem (bad JSON,
// a source missing exactly-one-of corpus/generator, a stage referencing an undeclared source, non-
// ascending until_epoch, a missing/misplaced "end" stage, an all-zero-weight stage, an unrecognized
// content_embed value, more than one corpus-typed source -- see blend_schedule.cpp for the exact checks),
// with a human-readable reason in `error`. Non-fatal issues (a declared source never given a positive
// weight anywhere) are appended to `warnings` instead of failing the parse. Does NOT log anything itself
// (see this header's own top comment for why -- a sub0_core.dll/sub0_train.dll logger-instance mismatch)
// -- the caller decides how to surface `error`/`warnings`.
[[nodiscard]] SUB0_API bool parse_blend_schedule_json(const std::filesystem::path& path, ScheduleSpec& out,
                                                       std::string& error,
                                                       std::vector<std::string>& warnings);

// The scheduler's runtime state: each source's cumulative TRAINED tokens drawn (win.len, not window
// width), index-aligned with the caller's `sources` vector (NOT SourceSpec -- this is keyed to whatever
// order the caller actually built real BlendSources in). `last_stage` tracks the most recently resolved
// stage index so a stage TRANSITION (see resolve_stage/apply_transition_clamp below) is detected and
// handled exactly once, not re-applied every draw. Persisted across checkpoints (CKPT_VERSION 3) so a
// resume doesn't discard the fairness state and re-introduce a scaled-down version of the original bug.
struct BlendFairness {
    std::vector<double> drawn_tokens;
    int last_stage = -1;
    explicit BlendFairness(std::size_t n) : drawn_tokens(n, 0.0) {}
};

// Precomputed per-stage target rates, index-aligned with the caller's `sources` vector -- resolved ONCE
// (name -> index lookups happen here, not in the per-draw hot path) from a ScheduleSpec + the actual
// source names the caller built BlendSources in. rate[stage][i] == 0.0 means source i is inactive in that
// stage. `until_epoch` is index-aligned with `rate` (parallel arrays, not a struct-of-arrays for its own
// sake -- lower_bound over a flat double array is simpler than over a vector of structs).
struct ResolvedSchedule {
    std::vector<double>              until_epoch;   // ascending; last == +infinity
    std::vector<std::vector<double>> rate;           // rate[stage][source_index]
};

// Build a ResolvedSchedule from the parsed spec and the ACTUAL order `sources` (real BlendSources) were
// constructed in -- `source_names[i]` names the source at index i. A stage's weight entries not matching
// any name in `source_names` are a parse/validation bug (must have been caught earlier); this function
// assumes a spec already validated by the JSON parser (blend_schedule.cpp) or a hand-built test fixture
// following the same contract, and does not re-validate.
inline ResolvedSchedule resolve_schedule(const ScheduleSpec& spec, std::span<const std::string> source_names) {
    ResolvedSchedule out;
    out.until_epoch.reserve(spec.stages.size());
    out.rate.reserve(spec.stages.size());
    for (const ScheduleStage& stage : spec.stages) {
        out.until_epoch.push_back(stage.until_epoch);
        std::vector<double> rate_by_index(source_names.size(), 0.0);
        for (const auto& [name, w] : stage.weights) {
            for (std::size_t i = 0; i < source_names.size(); ++i)
                if (source_names[i] == name) { rate_by_index[i] = w; break; }
        }
        out.rate.push_back(std::move(rate_by_index));
    }
    return out;
}

// The stage index active at `frac_epoch` -- the first stage whose until_epoch has not yet been reached.
inline int resolve_stage(const ResolvedSchedule& sched, double frac_epoch) {
    const auto it = std::lower_bound(sched.until_epoch.begin(), sched.until_epoch.end(), frac_epoch,
                                     [](double until_epoch, double fe) { return until_epoch <= fe; });
    return static_cast<int>(it - sched.until_epoch.begin());
}

// Rebase each CONTINUING source's progress at a stage transition so a source whose RATE DECREASED can't
// carry disproportionate "credit" into the new stage and get silently starved for the rest of the run (a
// real bug found in design review: a taper from e.g. 50%->5% mid-run computed out to ~5.7 epochs before
// the source would be drawn again -- effectively "off," with no error). Fix: for each source i active in
// BOTH the old and new stage ("continuing" -- a brand-new source, inactive before, is never touched: its
// progress is already correctly 0 and starting there IS the intended catch-up burst, not something to
// clamp away), cap its progress at what it would be if its OWN normalized value (progress/rate) equalled
// the most-caught-up OTHER CONTINUING source's normalized value. Comparing only against continuing peers
// matters: a brand-new peer's normalized value is always 0 (nothing drawn yet), and using it as the "max
// peer" reference would incorrectly zero out a legitimately-accumulated continuing source's progress too
// (caught by blend_tests.cpp's own "newly activated mid-schedule source" test). This only ever LOWERS an
// over-credited progress value (never inflates true coverage, never touches an inactive, brand-new, or
// under-credited source) and bounds worst-case post-transition starvation to a handful of draws instead of
// an unbounded stretch. Computed via a temp snapshot so the clamp is order-independent.
inline void apply_transition_clamp(BlendFairness& fair, const std::vector<sub0::BlendSource>& sources,
                                   const std::vector<double>& old_rate, const std::vector<double>& new_rate) {
    const std::size_t n = sources.size();
    std::vector<double> normalized(n, 0.0);
    std::vector<bool> continuing(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        if (old_rate[i] <= 0.0 || new_rate[i] <= 0.0) continue;   // not active in both stages
        continuing[i] = true;
        const double size = static_cast<double>(sources[i].view.size());
        normalized[i] = size > 0.0 ? fair.drawn_tokens[i] / size / new_rate[i] : 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!continuing[i]) continue;
        double max_peer = 0.0;
        bool has_peer = false;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i && continuing[j]) { max_peer = std::max(max_peer, normalized[j]); has_peer = true; }
        if (!has_peer) continue;   // no continuing peer to compare against -- leave progress untouched
        const double size = static_cast<double>(sources[i].view.size());
        const double cap = new_rate[i] * max_peer * size;
        if (size > 0.0) fair.drawn_tokens[i] = std::min(fair.drawn_tokens[i], cap);
    }
}

// Pick the active source furthest behind its own target epoch pace: argmin_i(progress[i]/rate_i) among
// sources with rate_i > 0 in the CURRENT stage. Deterministic (no rng draw) -- a real, intentional change
// from blend.hpp's old pick_source, which drew a weighted-random sample for >=2 sources; this scheduler's
// choice is a pure function of accumulated state, not chance. Ties (all-equal normalized values, e.g. at
// the very start of a run) resolve to the lowest index -- stable and simple; the fairness dynamics make a
// sustained tie vanishingly unlikely to matter after the first few draws.
inline int pick_source_staged(BlendFairness& fair, const std::vector<sub0::BlendSource>& sources,
                              const ResolvedSchedule& sched, double frac_epoch) {
    const int stage = resolve_stage(sched, frac_epoch);
    if (stage != fair.last_stage) {
        if (fair.last_stage >= 0)   // no rebase before the very first stage -- nothing accumulated yet
            apply_transition_clamp(fair, sources, sched.rate[static_cast<std::size_t>(fair.last_stage)],
                                   sched.rate[static_cast<std::size_t>(stage)]);
        fair.last_stage = stage;
    }
    const std::vector<double>& rate = sched.rate[static_cast<std::size_t>(stage)];
    int best = -1;
    double best_normalized = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < sources.size(); ++i) {
        if (rate[i] <= 0.0) continue;
        const double size = static_cast<double>(sources[i].view.size());
        const double normalized = size > 0.0 ? fair.drawn_tokens[i] / size / rate[i] : 0.0;
        if (normalized < best_normalized) { best_normalized = normalized; best = static_cast<int>(i); }
    }
    return best;   // -1 only if every source is inactive in this stage -- a validation bug, not a runtime case
}

// Pick + sample a window + record the draw in one call, mirroring sample_blend's own shape so a caller
// can't forget the record-draw step. `len` (the window's TRAINED length, <= T for a short document) is
// what accumulates into drawn_tokens, not the requested width T -- matching what the source actually
// contributed to training, the same quantity a short-document window already reports via Window::len.
inline sub0::BlendDraw sample_blend_staged(std::mt19937& rng, BlendFairness& fair,
                                           const std::vector<sub0::BlendSource>& sources,
                                           const ResolvedSchedule& sched, int T, double frac_epoch) {
    const int s = pick_source_staged(fair, sources, sched, frac_epoch);
    // pick_source_staged's -1 sentinel means every source was inactive in the resolved stage -- a
    // validation bug (parse_blend_schedule_json requires every stage to have >=1 positive-weight source),
    // not a case a hand-built ResolvedSchedule in a test or future caller is guaranteed to avoid. Fail
    // loudly here rather than silently reading sources[SIZE_MAX] via the cast below.
    assert(s >= 0 && "sample_blend_staged: no active source in the current schedule stage -- every stage "
                     "must have at least one source with rate > 0 (see parse_blend_schedule_json)");
    const sub0::BlendSource& src = sources[static_cast<std::size_t>(s)];
    const sub0::Window win = sub0::sample_window(rng, T, src.view.size(), src.docs,
                                                 src.subset_fraction, src.subset_seed);
    fair.drawn_tokens[static_cast<std::size_t>(s)] += static_cast<double>(win.len);
    return { s, win };
}

// Reattribute per-source fairness progress from an OLD source ordering to a NEW one, matched BY NAME --
// not by index/position. This matters specifically for --blend-config-replace (train_stage.cpp): a
// deliberate schedule swap that happens to keep the same NUMBER of sources but changes what any given
// index MEANS (a source reordered, or one swapped out for a differently-named one at the same slot) must
// not silently hand a new, unrelated source the old one's accumulated progress -- exactly the class of
// silent fairness-state corruption this whole scheduler exists to prevent (see this header's own top
// comment). A name present in both keeps its progress; a name new to `new_names` starts at 0 (the correct
// "just activated" state, same as any other newly-active source); a name no longer in `new_names` is
// dropped. `old_names`/`old_tokens` must be the same length (index-aligned, as persisted); a name
// duplicated in either list matches its FIRST occurrence only (schedules are parser-validated to have
// unique source names, so this only matters for a malformed/hand-built input).
inline std::vector<double> carry_forward_by_name(std::span<const std::string> old_names,
                                                  std::span<const double> old_tokens,
                                                  std::span<const std::string> new_names) {
    std::vector<double> out(new_names.size(), 0.0);
    const std::size_t n_old = std::min(old_names.size(), old_tokens.size());
    for (std::size_t ni = 0; ni < new_names.size(); ++ni)
        for (std::size_t oi = 0; oi < n_old; ++oi)
            if (old_names[oi] == new_names[ni]) { out[ni] = old_tokens[oi]; break; }
    return out;
}

}  // namespace sub0
