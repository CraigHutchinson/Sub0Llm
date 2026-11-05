---
name: arm-a-plateau-run-is-the-baseline
description: "Arm A resumed to ~5.6 epochs reaching val_nelbo 2.0739 (from 2.1332) -- this is THE BASELINE for the training-recipe work, and it never plateaued"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T18:50:56.859Z
---

Arm A (`models/ls_A_deep16_s1`, d192 16-exec deep, 9.66M params, batch 448, corpus_fraction 0.4,
subset_seed 1) was resumed 2026-07-30 past its earlier false stop at step 8586 and run to ~5.6 epochs.

```
step 8586  (1.80 ep)  val_nelbo 2.1332   <- the old "plateau", and the point the 4-arm
                                            token-matched comparison was taken at
step 26235 (5.50 ep)  val_nelbo 2.0739   <- best; cumulative -0.0593
train 1.9982 at the same point (crossed below 2.0)
throughput ~218-219 win/s, ~111,900 tok/s (CLEAN window -- one earlier segment is
contaminated by my own concurrent gen probes and must not be quoted)
```

**It never plateaued.** The detector reached 2-of-3 patience SIX times and was broken each time by a
step-down: -0.0063, -0.0048, -0.0048, -0.0022, -0.0049, -0.0009. The trend test had been trivially
satisfied for most of the run, so only the patience guard prevented six premature stops. Stopped
deliberately by the user at ~5.6 epochs to free the GPU for the training-recipe work — NOT because it
converged.

## Why this is the baseline

The plan (user decision) is to improve training effectiveness FIRST, then re-run all arms plus seeds under
the better recipe, on the reasoning that a faster/more effective recipe returns the whole set in less
wall-clock. Arm A's curve is what any recipe change is measured against, at matched tokens, three pillars.

## What it proves about the earlier comparison

The 4-arm result at step 8586 (A 2.1332 / B 2.2140 / C 2.2300 / D 2.2085) was taken while **every arm was
still setting a new best at every one of its final six evals** — verified from their train.logs. Arm A
alone then gained -0.0593 by continuing, roughly **10x the entire D-B margin** that comparison rested on.
Treat those rows as an early snapshot, not a second data point of equal standing. See
[[loopsplit-3arm-clean-result-looping-works]] and
[[plateau-detector-calibrated-for-tinystories-era]].

## Handoff state (all recorded, nothing in flight)

* `docs/PLATEAU_DESIGN.md` — stopping-criterion redesign, 10 open questions, NOT agreed. Includes the
  novelty/coverage reframe (§4a) and the TinyStories smoke experiment (§4b).
* `docs/LR_SCHEDULE.md` — inverse-sqrt is legacy; DeepSeek-V4 shape is cosine cooldown to 10% of peak.
* [[lr-schedule-wsd-cooldown-backlog]] — carries the **do-not-bump-CKPT_VERSION** constraint.
* Arms B/C/D build dirs are rebuilt and their models remain resumable.
