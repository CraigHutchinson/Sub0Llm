---
name: blend-schedule-clean-room-redesign
description: "DONE + committed (2026-07-16): clean-room rebuild of corpus blending -- staged, epoch-fair (deficit/WFQ) scheduler replaces the old fixed-per-draw-fraction pick_source/sample_blend entirely; JSON blend-schedule config fully replaces --scratch-mix/--op-mix/--spell-mix/--content-embed/--gsm8k; RunConfig config_schema versioning guards old model dirs; checkpoint v4 persists per-source fairness state BY NAME (v3's positional version had a real --blend-config-replace misattribution bug, found+fixed in self-review). Verified on real GPU hardware."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

## The bug that started this

A real training run (d=256, batch=512, old `--scratch-mix 0.3 --content-embed`) showed loss descend
smoothly to 2.66 by step 483, cliff to 1.19 by step 522, then dead-flat for 450+ steps. Root cause:
`include/sub0/blend.hpp`'s old `pick_source` drew a blend source in proportion to a FIXED weight on
every window, forever -- no tracking of how much of a source had been covered. The 141K-token scratch
curriculum, blended against a 633M-token base corpus, got statistically re-covered every ~4 steps,
memorized within ~500 steps, and its cratering loss contribution produced the cliff while the base
corpus was still <20% into its own first epoch. See [[gpu-large-batch-access-violation]] (a separate,
unrelated bug found incidentally while chasing this one) and [[hybrid-cpu-gpu-execution-design]] (the
work that led to running the verification scale that surfaced this).

## What shipped (full replacement, not a patch -- explicit user direction: "clean-room... build from
## the ground up")

**Core mechanism** (`include/sub0/blend_schedule.hpp`): a deficit/weighted-fair scheduler -- same
family as Linux CFS's `argmin(vruntime)` task scheduling and network WFQ. Track `drawn_tokens[i]` per
source, `progress[i] = drawn_tokens[i]/size[i]`, always draw from whichever ACTIVE source is furthest
behind its target rate. Every source targets EQUAL epoch coverage by default regardless of size (a real,
established precedent -- GPT-3's own per-dataset epoch table, Brown et al. 2020 Table 2.2, deliberately
let small corpora run multiple epochs while a huge one stayed under one). Generalizes to STAGED
schedules (curriculum-learning-style: "corpus A alone until epoch 0.5, then B ramps to 25% until 0.9").

**A real correctness bug found in self-review** (not just noted, fixed): a source whose target rate
DECREASES at a stage boundary can accumulate disproportionate normalized "credit" and get silently
starved for the rest of the run (a 50%->5% taper computed out to ~5.7 epochs before redraw -- effectively
permanently off, no error). Fixed by rebasing continuing-active sources' progress at each transition,
capped against the most-caught-up CONTINUING peer (excluding brand-new sources from that comparison --
a second bug caught by the new unit test itself: the first fix version zeroed out a legitimate
continuing source's progress by comparing it against a brand-new source's correctly-zero progress).
Both bugs are pinned by regression tests in `tests/blend_tests.cpp`.

**JSON schedule format** (`src/blend_schedule.cpp`, new simdjson::ondemand parser -- first array/nested-
object JSON parsing in this codebase, `run_config.cpp` was flat-object-only) fully replaces
`--scratch-mix`/`--op-mix`/`--spell-mix`/`--content-embed`/`--gsm8k`. `sub0llm-train` gains
`--blend-config <path>` (required for anything beyond a single corpus; a bare corpus positional stays as
sugar for the trivial case) and `--blend-config-replace <path>` (the explicit, separately-named override
for deliberately switching schedules mid-run -- default resume behavior reads ONLY the pinned copy,
stricter than `bundle_into_model_dir`'s tokenizer.tok/corpus.tok precedent which re-syncs every
invocation, because a schedule identity change literally changes what data trains the model).

**RunConfig breaking change** (`include/sub0/registry.hpp`): removed `spell_mix`/`scratch_mix`/`op_mix`/
`content_embed`/`content_embed_kind` (5 fields); added `config_schema` (struct default 1 = pre-redesign,
new code writes 2). `train`/`gen`/`eval` refuse to operate on a `config_schema<2` train resume (loud
error, would otherwise let a content-embed model resume silently decoding without its interceptor);
`gen`/`eval` degrade gracefully instead (warn + generate without any interceptor -- read-only inference
on an old model dir shouldn't hard-refuse the way continuing to TRAIN incorrectly should).

**Checkpoint v3**: `RunState::drawn_tokens` (the scheduler's fairness state) persisted, `CKPT_VERSION`
2->3, following the exact v1->v2 append precedent. Without this, every resume would silently discard
fairness state and reintroduce a scaled-down version of the original bug on whatever cadence the process
restarts.

**Epoch-aware plateau confidence** (`coherence::trend_plateaued`'s caller in `train_stage.cpp`):
`plateaued()` gained a hard `PLATEAU_MIN_EPOCH=0.75` floor (never plateau-stop before this, regardless of
hint) plus an optional `expected_plateau_epoch` schedule hint that scales the effective `PLATEAU_MIN_REL`
threshold between `PLATEAU_MIN_REL_FAR` (stricter, far from the hint) and `PLATEAU_MIN_REL_AT_HINT`
(looser, at/past it). Self-review correction worth remembering: the ORIGINAL motivating narrative claimed
the cliff-then-plateau bug would false-trigger this detector -- WRONG, `plateaued()` only ever reads
`rs.evals` (base-corpus-only val NELBO, structurally disjoint from blend dynamics, already isolated
before this redesign). The real, verifiable justification: blending in any secondary source at all
dilutes the base corpus's own per-step gradient budget, which can slow ITS OWN val-NELBO improvement
early on -- that's the mechanism documented in the actual code comment, not the unverifiable one.

**BlendSource generalized** (`include/sub0/blend.hpp`): gained `name` + `doc_bindings` (a
`std::span<const std::vector<std::vector<int>>>`, generic over which generator produced it) --
collapsed the old `scratch_source_idx`/`op_source_idx`-hardcoded per-window dispatch in
`train_stage.cpp`'s `materialize_cpu_window` and the hybrid CPU/GPU partition's `needs_cpu` check into
one generic `!src.doc_bindings.empty()` check each, a real simplification over what existed before.

## Verified on real GPU hardware (RTX 5070, d=256/L=6/H=4, batch=512)

Full 292-test suite green (up from 269 before this session's work -- new blend/blend_schedule tests).
A fresh equal-epoch-rate run (`{"sources":[{"name":"base","corpus":...},{"name":"scratch",
"generator":"scratchspike"}], "schedule":[{"until_epoch":"end","weights":{"base":1.0,"scratch":1.0}}],
"content_embed":"hrr"}`) showed loss 5.28->3.67->3.14->1.56->1.54->1.53->1.48 over 1000 steps -- critically,
it KEEPS improving after the step-483-662 drop instead of going dead flat like the original bug (1.19->1.12
stall). A resume with ZERO `--blend-config` flag correctly re-read the pinned schedule and continued
seamlessly (1.4811->1.4640->1.4641, no discontinuity). `config.json` showed `config_schema: 2` (none of
the 5 removed fields); `blend_schedule.json` was pinned exactly as written.

## Production relaunch + the "step ~483 cliff" is corpus-intrinsic, not a bug (2026-07-16, post-commit)

Relaunched the exact original-incident recipe (d=256/L=6/H=4, tinystories, batch=512, scratchspike+HRR
content-embed) fresh under the new scheduler as `ce256_prod_fixed` (real production run, `--steps 0`,
not a capped verification run) after the redesign was committed. The user spotted the SAME-shaped fast
loss drop (~3.14->~1.56) around step 483-663 in its log and asked whether the eval/hybrid-CPU-GPU path
was still "stomping" something. Investigated with two controls rather than asserting:

1. A same-architecture plain run with NO blend/scratch/content-embed/hybrid (`ce256_plain_control`,
   pure GPU) showed the IDENTICAL drop shape at the identical step (483->584: 3.14->1.58).
2. Checked EVERY prior registered TinyStories model run in `models/` (d128/d192/d256/d320/d448/d512/d768,
   dated 2026-07-07 through 07-10, all pre-blend-schedule, pre-content-embed, plain single-corpus, two
   different batch sizes) -- ALL show the identical drop (~2.6-3.1 -> ~1.4-1.6) at the identical relative
   epoch position (~0.10-0.16 epoch, mid-LR-warmup), regardless of d_model or batch size. For contrast,
   `sub0llm_fineweb_edu_d448...` (real web text, not TinyStories) shows NO such cliff -- smooth monotonic
   decline instead.

Conclusion: the fast early drop is an intrinsic TinyStories-corpus + this-architecture-family training
dynamic (a real early "phase transition," plausible given TinyStories' narrow, highly templated
vocabulary), present in EVERY TinyStories run ever captured in this repo's history, totally independent
of blending/content-embed/hybrid-CPU-GPU/eval. It is NOT the bug. The bug's actual, distinguishing
signature was always the DEAD-FLAT PLATEAU immediately after (1.19->1.12 over 450 steps in the original
incident) -- and `ce256_prod_fixed` does NOT do that: val_nelbo kept improving past the equivalent point
(1.3687 @2898 -> 1.3004 @9177 -> 1.2927 best @11109, 2.3 epochs in and still descending). The fix holds
under a real, long production load, not just the earlier 1000-step verification run.

Also confirmed (useful cross-reference for future TinyStories work): dims-sweep best val_nelbo from the
2026-07-08/10 runs -- d128 1.5719, d192 1.4296, d256 1.3381, d320 1.2597, d448 1.1902, d512 1.1653-1.1864,
d768 1.1235 -- matches [[tinystories-sweep-saturation]]'s "d448=knee, d512/d768 diminishing returns"
finding exactly.

**Final result**: `ce256_prod_fixed` plateaued cleanly at step 13524 (2.8 epochs), best val_nelbo 1.2895 --
BETTER than the equivalent plain (no content-embed) d256 sweep's 1.3381, no dead-flat stall anywhere, no
errors in the log. Generation from the final checkpoint (`sub0llm-gen ... "Once upon a time"`) produced
coherent TinyStories-style prose and correctly showed `interceptor enabled (spell/scratch)` +
`content-embed hrr` -- the scratch/content-embed machinery this whole redesign protects is confirmed
live and working end to end, not just loss-curve-healthy.

**Operational gotcha hit while launching this (not a code bug, a CLI-usage footgun worth remembering)**:
gave `sub0llm-train` a BARE filename as the model arg (`ce256_prod_fixed`, no directory component)
instead of a directory-style path (`ce256_prod_fixed/model`). `meta_dir = parent_path(model_path)`
resolves to EMPTY for a bare filename, and every `meta_dir`-gated side effect (`config.json`/`meta.txt`/
`blend_schedule.json` writes, `train.log`'s own `set_file` tee) is silently skipped when `meta_dir` is
empty -- training itself is unaffected (the schedule still loads fine via the "no model dir to pin into --
read the live path directly" branch), but the resulting model has no config.json/meta.txt, so it's
invisible to the model registry and `gen`/`eval` can't auto-detect its capabilities. Had to manually
reconstruct a proper directory (copy model.bin + .ckpt in, hand-write config.json + blend_schedule.json)
to run gen against it afterward. **Lesson: always give sub0llm-train a directory-style path
(`<name>/model`), never a bare filename**, when the model needs to be resumable/generatable/registry-
visible later. FIXED same-day (commit `0e820a6`): `sub0_train_stage` now logs a loud `[warn]` the moment
`meta_dir` resolves empty, naming exactly what won't be written and suggesting the `<name>/model` fix --
verified it fires for a true bare name and stays silent for both a full-path-with-directory and an
explicit `<name>/model` path.

## Self-review pass (2026-07-16, before commit): 1 MUST bug found + fixed, plus smaller hardening

A `/cpp-review`-style self-review (prompted by the user's explicit "Commit and review, check test
coverage is good") caught a real correctness bug before it ever shipped:

- **MUST, fixed: `--blend-config-replace`'s "carries forward BY NAME" claim was false.** The CLI help
  text and this very memory documented name-based reattribution, but the actual code only checked
  `rs.drawn_tokens.size() == sources.size()` and did a blind POSITIONAL copy. A replace schedule that
  keeps the same source COUNT but reorders or renames sources (the exact `--blend-config-replace` use
  case) would have silently handed a new, unrelated source the old one's fairness progress -- the same
  class of silent corruption this whole redesign exists to eliminate. Fixed: `RunState` gained
  `drawn_names` (index-aligned with `drawn_tokens`), persisted in a new **CKPT_VERSION 4** (append-only,
  same precedent as v1->v2->v3), and a new pure, unit-tested `sub0::carry_forward_by_name()`
  (`blend_schedule.hpp`) does real name-keyed reattribution in `train_stage.cpp`'s resume-reconciliation
  block. **Verified on real hardware**: a smoke run (gsm8k corpus, base+scratchspike) trained 8 steps,
  resumed to 16 (plain resume, names unchanged -- no fallback warning fired), then
  `--blend-config-replace`d to a schedule with the SAME two sources **reordered** (scratch now index 0,
  base index 1 -- exactly the case the old positional code would have silently swapped) and resumed to
  24 steps cleanly, with the "SCHEDULE CHANGED" log firing and no "does not match" fallback -- proof the
  by-name match succeeded despite the reorder.
- Smaller fixes from the same pass: `pick_source_staged`'s -1 sentinel (all-sources-inactive) was cast
  unchecked to `std::size_t` in `sample_blend_staged`, a latent OOB read if a hand-built `ResolvedSchedule`
  ever violated the invariant -- added an `assert` (matches `tokenizer.cpp`'s existing `assert(cond &&
  "message")` idiom). The parser only required *at most* one corpus-typed source, not *at least* one --
  a schedule of pure generator sources would have run against a corpus whose epoch clock is meaningless;
  now rejected with a clear error. `gen`/`eval` silently swallowed a `blend_schedule.json` parse failure
  (fell back to "no interceptor" with zero diagnostic, unlike the loud `config_schema<2` case) -- now
  prints the parse error. `SourceSpec::corpus`'s doc comment overclaimed the field's value is validated
  against the build's actual corpus path; corrected to state it's never read back (presence-only check).
  Cosmetic: the `SUB0_API` export-macro block was sandwiched inside `ScheduleSpec`'s doc comment --
  moved to top-of-file.
- New tests: `carry_forward_by_name` (reorder+drop+new-source case, and a no-op-when-unchanged case) in
  `blend_tests.cpp`; a "zero corpus sources is an error" case in `blend_schedule_tests.cpp`. Full suite:
  `sub0_tests` 148 cases / 12.3M assertions green, `sub0_frontend_tests` 147 cases / 102K assertions
  green (up from 146 pre-review -- the 2 new `blend_tests.cpp` cases plus the 1 new
  `blend_schedule_tests.cpp` case, net +3 across both binaries after accounting for the pre-existing
  counts already reflected in "Verified on real GPU hardware" above).

## Not yet done (explicitly deferred, not forgotten)

- Docs cleanup is DONE, not outstanding (verified 2026-07-16 by re-grepping the old flag names post-
  commit): the only remaining `--scratch-mix`/`--op-mix`/`--content-embed` mentions in
  `docs/DETERMINISTIC_MECHANISMS.md` are intentional past-tense/historical narrative (the bug story, the
  "superseded" follow-up notes) describing what the OLD flags were, not stale how-to-use prose;
  `docs/SCRATCH_TOKENS.md` and `scripts/get_gsm8k.py` have zero matches left.
- No *unit* test exercises `--blend-config-replace`'s CLI plumbing end-to-end (the pure by-name
  reattribution logic now IS unit tested; the pin/replace file-handling + checkpoint round-trip was only
  verified via the real smoke run described above, not a pinned automated test).
- A genuine SECOND file-backed corpus source (not just generator-type sources) is explicitly out of
  scope -- the parser rejects more than one `"corpus"`-typed source. Real new plumbing (its own
  tokenize-against-shared-vocab pass, doc-index, on-demand handling, model-dir pinning) if ever wanted.
- Committed to git as of this memory's update (see git log for the commit).
