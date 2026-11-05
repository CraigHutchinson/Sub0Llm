---
name: winrate-metric-suspect-2-5x
description: "OPEN BUG — the training log's win/s appears over-reported ~2.5x; tok/s and the epoch ETA agree with each other and disagree with it"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-28T16:52:25.949Z
---

# RETRACTED 2026-07-28 -- win/s was NOT wrong. Read this whole note; the finding moved.

**The claim in this file's own title and description is FALSE.** `win/s`, `tok/s` and the ETA are all
self-consistent. The error was mine: I computed "actual win/s" as `steps/s * 256`, assuming a constant
256 windows per step. The token-budget scheduler explicitly does NOT work that way -- it admits
`batch_t * seq_t <= batch * SEQ_LEN`, so shorter windows mean MORE of them per step for the same token
budget. `train_stage.cpp:2050` says exactly this, and I had quoted that comment earlier in the same
session before contradicting it.

Reconciled: `366 win/s / 0.6006 steps/s = 609 windows/step`; `131072 / 609 = 215` mean tokens per
window; `0.6006 * 131072 = 78,722 tok/s` vs ~79,000 reported. Everything agrees.

**The one REAL bug found here was the epoch clock** -- `epoch_steps` ignored `--corpus-fraction`, so a
0.8 run reported "1 epoch" after ~1.25 real passes. Fixed in `1d6772a`. The user spotted the cause
("is the calculation full corpus?") faster than I did.

## What is still genuinely open, and is probably MY regression

Mean window is **215 tokens** but Cosmopedia's median document is **820**, with only 7.3% under 512.
Those do not fit. The earlier "short-document padding is ruled out" conclusion rested on the same bad
constant-batch assumption and is NOT safe either.

Prime suspect: the corpus-subset sampler added in `26b826d`. `sample_window` returns `{ds, cap}` --
the WHOLE document -- whenever `cap < T`, and both the rejection loop and its forward-scan fallback
live in that path. If the fallback fires often at `fraction 0.80` it will return short windows
disproportionately, depressing mean `seq_t`, wasting compute on padding, and doing so ONLY when
subsetting is active. Check `fraction 1.0` vs `0.8` mean `seq_t` on the same corpus before blaming the
corpus.

## Method lessons (the durable part)

1. **Do not compute rates by averaging since launch on a shared machine.** An earlier attempt inferred
   an "18 h epoch" that way and raised a false alarm about tok/s; the window was polluted by this
   session's own concurrent verification runs and a CUDA rebuild. A clean 363 s window matched the log
   to 0.4%.
2. **Sample across at least one log line.** They appear every ~110 steps; a 90 s window sees zero
   progress and reads as a hang.
3. **When two instrumented numbers disagree, suspect the DERIVATION before the instrument** -- twice
   now the "disagreement" was my own arithmetic assumption, not the code.

See [[thermal-confounds-ab-wallclock-testing]].
