---
name: blended-scratch-op-capstone-validated
description: "DONE (f5a7f34, 072c2f7): validated a model trained on scratch+op BLENDED via the real epoch-fair scheduler retains both capabilities -- op-delegation was already perfect (1.0/1.0/1.0) solo AND blended; scratch held-out resolution was actually HIGHER blended (0.215) than solo (0.166). New tests/blended_capstone_engine_tests.cpp. Also found+CLOSED a real footgun: train_stage.cpp now REFUSES an explicit model path resolving to the shared models root, or to a directory another model already occupies, instead of silently sharing config.json/blend_schedule.json between unrelated models."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User asked to look at training with scratch and op support blended, with a test-generation step for
each and "blended function validation" -- also asked to clean up any obsolete old approaches first.

## Cleanup finding: nothing obsolete to remove

Surveyed `train_stage.cpp`'s per-generator dataset-building loop and `gen_stage.cpp`'s interceptor
setup before adding anything. Both were already properly generalized by the earlier
[[blend-schedule-clean-room-redesign]]: `train_stage.cpp` builds sources from an arbitrary
`schedule.sources` loop (no hardcoded `scratch_source_idx`/`op_source_idx` branches remain -- only a
comment referencing that history); `gen_stage.cpp` computes `sp_on`/`scratch_on`/`op_on` as
INDEPENDENT booleans via `generator_ever_active(schedule, ...)` and composes them unconditionally
(`sp_on || op_on` gates whether the interceptor is even built; both `expand`/`combine` (scratch) and
`compute` (op-collapse) can be wired in together with no extra flag). So blending scratch+op was
already architecturally supported end-to-end before this session touched anything -- confirmed, not
assumed, by reading both files in full. Nothing to delete.

## The one real new risk: shared scratch-slot pool

`sub0::ScratchTable::bindings` (scratch.hpp) is ONE vector shared by BOTH mint paths: scratchspike's
`combine()` (mints on an unrecognized OOV) and gen_stage.cpp's op_on branch (mints an op-collapse
result at `scratch.bindings.size()`). Blending means both draw from the same finite
`SCRATCH_SLOT_COUNT` (6) range within one live gen session. Added a fast, model-free test
(`[blended]`, no training) that mints 2 OOV bindings then an op-collapse result and confirms they land
on distinct, correctly-value()-resolving slots (0, 1, 2) with no collision.

## The capstone: matched-budget A/B, not eyeballed numbers

`tests/blended_capstone_engine_tests.cpp`'s `[.blendedcapstone]` (hidden) trains THREE models at a
matched per-mechanism step budget in one run (so "does blending hurt" is a real A/B against a solo
baseline trained the same session, not compared against a different test file's numbers from a
different run/seed):
- scratch-solo (2000 steps): DRILLED nth=0.167, HELD-OUT nth=0.166
- op-solo (2000 steps): in-dist delegated=1.000 exact=1.000, OOD exact=1.000
- BLENDED (4000 steps total, ~2000/mechanism under the equal-weight epoch-fair scheduler):
  scratch DRILLED nth=0.246, HELD-OUT nth=0.215 (HIGHER than solo); op delegated=1.000 exact=1.000
  OOD exact=1.000 (unchanged, already saturated)

Result: neither mechanism degrades when trained together. Op was already perfect at this tiny
d128/2000-op-example scale so there was no room to see degradation there; scratch resolution was
measurably BETTER blended than solo in this single-seed run (plausibly just step-to-step/seed
variance at this scale, not a claimed causal improvement -- would need multi-seed to trust the
direction, but it flatly rules out "blending silently breaks resolution" as a concern).

Uses the REAL production scheduler (`sub0::sample_blend_staged`/`ResolvedSchedule`/`BlendFairness`
from blend_schedule.hpp), not a hand-rolled analog -- a hand-built `ScheduleSpec` (bypassing the JSON
parser, matching the header's own engine-free testability property) with a single equal-weight
`{"scratch":1.0,"op":1.0}` stage. Op-side scoring mirrors `gen_stage.cpp`'s `sub0_eval_stage` collapse
callback exactly (own copy, matching this project's per-file self-contained-capstone convention --
`reset_opt_state`/`byte_encode`-style helpers are duplicated per file elsewhere too, not shared).

## Real production-path smoke train + gen (not just the in-process test)

Also ran a genuine `sub0llm-train --blend-config` combining base+scratch+op (400 steps, d128, gsm8k.txt
corpus, on the CUDA build) -- all three `blend: '...'` lines appeared (confirming the config actually
took effect, see the footgun below), hybrid GPU router correctly split content-embed windows, trained
clean to val_nelbo 1.80. Then ran `sub0llm-gen` against the trained model:
`gen: interceptor enabled (spell/scratch + op-dispatch, collapse)` + `gen: content-embed hrr` --
confirms the REAL CLI gen path composes both interceptors on a jointly-trained model, matching
gen_stage.cpp's designed (already-correct) composition.

## Footgun found AND CLOSED IN CODE (commit `072c2f7`, same session, user-directed follow-up)

Passing an explicit model path like `sub0llm-train models/foo ...` (rather than an auto-named path, or
`models/foo/foo`) did NOT create `models/foo/` as a subdirectory -- `foo` became a flat weights-file
basename and `foo`'s PARENT (`models/`, the top-level shared directory) became "the model directory"
for config.json/blend_schedule.json/corpus.tok/meta.txt/train.log purposes (matches `gen_stage.cpp`'s
own `dir = is_directory(mp) ? mp : mp.parent_path()` logic). Ran THREE separate "isolated" smoke tests
this way across this session and the prior one -- all three silently shared ONE config.json/
blend_schedule.json the whole time. This directly caused a real methodology bug: a blended (base+
scratch+op) smoke run silently trained against a STALE 2-source (base+scratch only) pinned schedule
left by an earlier unrelated smoke test -- the op source was NEVER ACTUALLY TRAINED in that first
attempt, caught only by noticing the training log had no `blend: 'op'...` line. Cleaned up the stray
top-level `models/config.json`/`blend_schedule.json`/`meta.txt`/`train.log`/scattered `.ckpt` files
(confirmed via `git status`/timestamps these were all this session's own debris, not pre-existing
production state -- every real model here lives in its own auto-named subdirectory, e.g.
`ce256_prod_fixed/`, matched by `file`/`ls -la` before deleting anything, and only after explicit user
confirmation since the classifier correctly blocked the first bulk-delete attempt).

User then asked to make the CODE bulletproof against this (and similar) user error, not just document
it. Fixed in `train_stage.cpp`'s `sub0_train_stage`, right after the existing bare-filename warning
(the `0e820a6` precedent for a RELATED but different footgun -- a path with NO directory component at
all, e.g. "myrun", which only WARNS since skipping the directory layout is a legitimate if unusual
choice). This new case has no legitimate use, so it REFUSES outright (return 1, matching
SingleInstanceGuard's own precedent), on two checks, both against the resolved `meta_dir`:
1. `meta_dir` canonicalizes to `SUB0_MODELS_ROOT` itself (a CMake-baked absolute path,
   `${CMAKE_SOURCE_DIR}/models` -- NOT the bare relative fallback the `#ifndef` in the same file
   suggests; that fallback is dead in the real build since CMakeLists.txt always defines it) -- the
   exact mistake hit here.
2. `meta_dir` already contains a checkpoint/weights file (`*.ckpt` substring match) whose basename
   does NOT match this run's own model-path basename -- generalizes to "pointed at a directory ANOTHER
   model already occupies" (e.g. accidentally nesting inside `ce256_prod_fixed/`), not just the bare
   root case. Skips the known shared-by-design filenames (config.json/meta.txt/blend_schedule.json/
   train.log/corpus.tok/tokenizer.tok) and anything matching our OWN basename (so resuming the SAME
   model's own directory is unaffected).

Verified via 4 manual CLI scenarios on a real (tiny, fast) build: bare-root path refused with a clear
message; pointing at `ce256_prod_fixed/different_name` refused (correctly names the foreign
`.ckpt`); a genuine fresh dedicated subdirectory trains normally; resuming that SAME model's directory
a second time proceeds normally (`RESUMING from step 3`, not flagged as foreign). Full CPU-only suite
stayed green on both configs after the change (118 cases, 9,192,152 assertions, no regressions --
this logic isn't unit-tested directly, consistent with how `SingleInstanceGuard`/`config_schema`
refusal are also CLI-integration-verified rather than unit-tested, since `sub0_train_stage` is a large
CLI entry point).

## Follow-up dry-check (2026-07-18, same day): the new check caught a REAL pre-existing instance

User asked for a dry check on the models/ data folders + a constexpr/defaults audit. Systematically
scanned every model directory for the exact dual-basename pattern the check above targets --
`models/ce256_prod_fixed` genuinely had it: `ce256_prod_fixed.step000013524.ckpt` (the run's own,
canonical basename -- its train.log's own final line reads "plateaued at step 13524 ... ->
ce256_prod_fixed") alongside a byte-identical (confirmed via md5sum) `model.bin` +
`model.bin.step000013524.ckpt` pair, almost certainly from a LATER bare-directory-path invocation
against this same directory (`is_directory(mp) ? mp : mp.parent_path()` rewrites a directory path to
`<dir>/model.bin` -- the exact mechanism the new check now guards, just triggered from the OTHER
direction: a real basename dir being resumed via its bare directory path rather than two distinct
model names sharing one directory). This is now precisely what the new foreign-checkpoint check would
refuse on a future resume attempt. User confirmed dedupe: removed `model.bin` +
`model.bin.step000013524.ckpt`, kept the canonical `ce256_prod_fixed.*` files. Verified the check no
longer fires: a resume attempt against `ce256_prod_fixed/ce256_prod_fixed` now proceeds past the
directory-sharing check to the (correct, unrelated) checkpoint-compat error, instead of the
foreign-file refusal it would have hit pre-dedupe. No other model directory had this pattern (a
`bin` false-positive from the crude first-pass scan was just the `bundle` command's runtime-copy
subdirectory, not a stray basename).

**Lesson for next time regardless**: still prefer an explicit model path that is ALREADY (or will
become) its own subdirectory for any one-off smoke/A-B test -- the fix makes the WRONG shape refuse
loudly instead of silently corrupting shared state, but doesn't auto-correct it to the right shape.

## Verification

Full CPU-only default suite green throughout (both before-fix crash-diagnosis and after-fix runs):
118 cases, 9,192,152 assertions (117/9,192,142 plus the new fast `[blended]` case's 10 assertions).
One bug found+fixed in the new test itself: `eval_op`'s `sub0::encode()` call needs the ENGINE's global
tokenizer loaded via `sub0::load_tokenizer()` (separate from the local `Tokenizer tk` the dataset
builders consume directly) -- omitting it segfaulted (`g_tok` default-constructed/empty). Fixed by
adding the load call to `load_tk()`.

Committed as `f5a7f34`.
