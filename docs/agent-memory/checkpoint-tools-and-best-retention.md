---
name: checkpoint-tools-and-best-retention
description: prune_ckpts now always keeps the best-val_nelbo checkpoint (CKPT_VERSION v1->v2); new sub0llm bundle + ckpt2model commands for cross-model-size comparison without rebuilding
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Real bug found and fixed (2026-07-07, `f6fb19d`).** `prune_ckpts(model_path, keep)` kept only the
newest `keep` (default 3) progress checkpoints by step -- no notion of "best". Caught live: the d448
GPU training run's val_nelbo hit its best (1.2933) at step 23229 (0.90 epoch), then ticked up slightly
for two evals (1.2958, 1.2961) before a real, larger improvement to 1.2663 at step 30972 (1.20 epoch) --
so the early uptick was noise, not a plateau. But the default keep=3 rotation pruned the step-23229
checkpoint before that clarified itself (confirmed: it was gone by the time step 30972 was checked).
Manually backed up the file under a name outside `prune_ckpts`' glob before that happened, but the
underlying gap was real and would silently lose the best-so-far checkpoint on any run whose val curve
dips temporarily before continuing to improve -- likely common, not an edge case.

**Fix:** `RunState` gained `best_step`; `prune_ckpts(model_path, keep, best_step)` now always keeps the
best-step checkpoint in ADDITION to the newest `keep`, even if it has aged out of that window.
Persisted in the checkpoint itself (`best_step` written/read in `save_checkpoint`/`load_checkpoint`,
`CKPT_VERSION` bumped 1->2 -- old checkpoints are rejected on load, matching this project's existing
breaking-version-bump convention, e.g. [[rope-convention-mismatch-forward-vs-forward-one]]'s v2 bump).

**New tools, same commit**, since keeping the best checkpoint is pointless without a way to actually
USE it:
- `sub0llm ckpt2model <ckpt> <model_out>` -- extracts just the weights from a `.ckpt` (which also
  carries optimizer moments/RNG/eval-history, a DIFFERENT binary layout from `model.bin`) into a plain
  `model.bin` gen/report can load. `model.bin` itself is overwritten with the LATEST weights on every
  eval tick, never the best, so this was the missing piece. Smoke-tested end to end: converted a fresh
  checkpoint, loaded the result via `gen`, confirmed it decodes.
- `sub0llm bundle <model>` -- copies the CURRENT build's `sub0llm.exe` + `sub0_core/gen/train.dll`
  (deliberately NOT the CUDA backend -- gen/report/eval always run on the CPU engine regardless of how
  a model was trained) into `<model_dir>/bin/`, so an old model stays runnable later even after this
  build's own dims change and it can no longer load that checkpoint (every model dir's dims are pinned
  to the exact build that produced it, see `sub0llm models`' loadable/incompatible split). Opt-in, not
  part of the normal train/gen flow -- built for an upcoming multi-model-size comparison sweep (see
  [[cuda-parity-backlog-deferred]]'s downstream-dependency note) where several incompatible builds need
  to stay runnable side by side without a rebuild each time. Smoke-tested live: bundled `ts_cuda`'s
  binaries (d448/L11/H7/vocab16517) into an OLDER CPU-trained checkpoint dir at the same dims and ran
  `report` from the bundled copy successfully.

**Also extended the same session:** `sub0llm report`'s sample-generation battery, 1 prompt x 2 temps ->
2 prompts x 3 temps (matching `gen`'s real 0.8 default), fixed seed, persisted to `report.txt` -- so a
later cross-model-size comparison can read saved `report.txt` files side by side instead of re-running
generation on every old model.

**Follow-up fix, same day (`eed2f8d`): `load_checkpoint` made version-tolerant (v1 accepted, not just
v2).** First attempt at "prove it against the real live run" failed: the live d448 GPU training process
was still running its own pre-bump (v1) binary, so its on-disk checkpoints were v1 -- and the freshly
rebuilt v2-expecting `ckpt2model` rejected them outright ("bad magic/version"). Distinguished this from
the RoPE v2 bump (a genuine semantic/math change, where reading old weights with new code would be
WRONG) -- here v1->v2 only ADDED a field (`best_step`); the weights/optimizer/RNG bytes are identical.
Fixed by making `load_checkpoint` accept EITHER version 1 (defaults `best_step=-1`, i.e. unknown) or the
current version, magic still required to match. This matters beyond just this one conversion: since
`load_checkpoint` is also the TRAINING RESUME path, rejecting v1 outright would have meant the next
`sub0llm train` resume of this live run, after any future rebuild, forced a fresh restart -- silently
discarding all the real GPU-hours already invested. Verified against the real thing: converted the live
run's actual step-30972 checkpoint (best-so-far val_nelbo 1.2663, ~1.2 epochs) and ran `gen` on it --
produced genuinely coherent TinyStories-style text ("Once upon a time, there was a little girl who
loved to play...").

**Open design idea, not yet implemented:** the user reasoned about eval frequency vs noise/cost --
checked the actual code (`evaluate()` in `train_stage.cpp`): the val-NELBO measurement itself is
DETERMINISTIC (same fixed, evenly-spaced windows every eval, no resampling) and CHEAP (capped at
`EVAL_WINDOWS_MAX=128` forward-only windows, a small fraction of the ~2581-step training interval
between evals) -- so evaluating more often would barely cost wall-clock and would NOT add measurement
noise. The real cost tied to eval cadence is the CHECKPOINT SAVE (writing the full param+2-Adam-moment
tensors, hundreds of MB, every eval tick). A clean future improvement: decouple eval cadence (cheap,
could run much finer, e.g. every 0.02-0.05 epoch) from the expensive checkpoint-save cadence, to
localize exactly where a val curve's true minimum sits without paying more disk I/O.
