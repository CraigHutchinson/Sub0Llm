---
name: train-diagnostics-backlog
description: Training-loop diagnostic gaps found investigating the workflow build -- silent resume-override, checkpoint-save retry, epoch-ETA jitter, on-demand token-count estimate bias. ALL DONE.
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Found while investigating why `models/fineweb_edu_workflow/train.log` kept showing
`lr: peak 2.83e-03 (batch 64)` even though the user configured/tuned this workflow run for
batch=196. Root-caused via `src/train_stage.cpp`. **All three items are now DONE** (as of
2026-07-03) -- this file is now a historical record, not an open backlog.

**1. Silent resume-override of batch/lr/seed (the "batch 64" mystery) -- DONE.**
`train_stage.cpp` (~line 985-1014): before `load_checkpoint()` overwrites the local
`batch`/`lr`/`seed`, the invocation's originally-requested values are captured
(`requested_batch`/`requested_lr`/`requested_seed`); if the checkpoint's values differ, an explicit
`sub0::log::info` line fires ("checkpoint overrides this invocation's settings: batch X -> Y, lr
... , seed ..."). No longer reads as "the binary must not have rebuilt" -- a resumed run's actual
knobs are now always visible in the log.

**2. Checkpoint-save retry/fallback -- DONE (commit 3ca4477, 2026-07-03).** `save_checkpoint()` and
`atomic_replace()` now return bool instead of silently giving up (`atomic_replace`'s failure path was
previously **completely unlogged** -- a checkpoint that wrote fine but couldn't be renamed into place
vanished with zero trace). Design refined by the user mid-implementation: a periodic save failing
DURING training must never block the loop -- the next checkpoint interval (eval cadence or the
`CKPT_SECONDS` crash-resistance tick) is the natural retry, so those two call sites just warn and
move on, no blocking retry. A real bounded retry-with-backoff (`retry_with_backoff`, 5 attempts / 3s
apart) is reserved for the single END-of-training save, which has no later interval to fall back on;
if that still fails, it falls back to `path + ".rescue-<unix-epoch>"`. The graceful-stop (Ctrl+C)
path deliberately does NOT get blocking retry -- it has a short OS grace period before a force-kill,
and blocking there risks a worse outcome (killed mid-retry) than a single quick attempt; it now at
least logs a failure instead of staying silent. `sub0::save_model` (engine_core.cpp) changed
void->bool too, for the same "detect the failure at all" reason. Verified: engine_tests green
(124.5M assertions) + a real train-to-completion smoke run.

**3. Epoch-ETA -- exists, and its JITTER BUG is now fixed (2026-07-03).** The progress lines (both
the per-eval line and the `TICK_SECONDS`=180s interim heartbeat) already print `[frac ep, next ep in
<eta>]`. The user flagged real jitter in a live production run (`workflow/fineweb_edu`): consecutive
heartbeats swung 59.5h -> 59.1h -> 59.3h with no clear trend. Root cause: the heartbeat branch computed
`eta_next_epoch` from the rate over ONLY the last `TICK_SECONDS` (180s) tick
(`last_log`/`last_log_step`, reset every tick) -- a single 3-minute sample is noisy (scheduling jitter,
window-length variance under `vary_seq`), so the derived ETA bounced instead of converging. **Fix**:
the heartbeat's ETA now uses `win_t0`/`win_steps0` (the SAME cumulative-since-last-eval accumulators
the eval branch already used) instead of the tick-local ones -- a much wider, steadier sample (up to
`eval_every` steps, not just one tick), while the displayed instantaneous `wps`/`tok/s` still uses the
tick-local rate (legitimately a "right now" figure, not the thing that was broken). Not yet
re-verified against a long live run (the fix is logically sound and compiles, but jitter reduction
over hours of wall-clock is not something a quick smoke test can confirm) -- worth a spot-check next
session if the workflow run is still going.

**4. On-demand token-count estimate was a single, potentially biased sample -- fixed (2026-07-03).**
Separately, the user cross-checked the on-demand path's `~N train tok est` against the SAME corpus's
actual pretokenized count and found a systematic ~4.4% undershoot (12.14B est vs 12.70B actual on
fineweb_edu). Root cause: `train_stage.cpp`'s tokens/byte probe sampled exactly ONE 64KB region at
byte offset 0 to extrapolate a tokens/byte ratio across the whole corpus -- a single sample is exactly
the kind of thing that goes wrong on a non-homogeneous corpus (FineWeb-style corpora cluster document
types, so the very start of the file is not necessarily representative). **Fix**: now samples
`OD_TPB_SAMPLES=8` regions evenly spread across `[0, train_byte_hi)` and averages the ratio -- still
trivial cost (a few hundred KB read total against a multi-GB corpus). This only affects the printed
estimate and the epoch-schedule math (both explicitly heuristic-only per the existing code comment),
not training correctness. Not yet re-verified against a fresh on-demand run to confirm the bias
shrank -- worth checking against a pretokenized count on the next on-demand corpus tried.
