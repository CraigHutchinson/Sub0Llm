---
name: plateau-detector-fires-while-still-improving
description: "FIXED 2026-07-29 — the plateau stop fired while val_nelbo was still hitting a NEW BEST at every eval; a threshold-free patience guard now blocks it. The resume-inflation claim in here was MISDIAGNOSED"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T17:24:17.917Z
---

Found 2026-07-28 in the 3-arm LoopSplit sweep. **Both** arms stopped at the *identical* step 8586
(1.80 epochs) with `status=plateaued`, while `val_nelbo` was still setting a new best at every single
eval:

| eval | A_deep16 | B_loop10x6 |
|------|----------|------------|
| 16   | 2.1435   | 2.2191     |
| 17   | 2.1385   | 2.2179     |
| 18 (stop) | **2.1332** | **2.2140** |

Monotonically decreasing, every one a new best, then "plateaued". Nothing had plateaued.

**Likely cause, not yet confirmed**: the detector's minimum RELATIVE improvement threshold. A's last
two gains were 0.23% and 0.25% per eval; B's were 0.05% and 0.18%. A threshold demanding more than
~0.5% relative improvement reads genuine steady learning as a plateau. That it fired on two
different architectures at the *same eval boundary* points at a threshold/schedule interaction rather
than anything data-driven. See the `min_rel` interpolation around `train_stage.cpp:394` -- start
there, and check whether the fix from the `--corpus-fraction` epoch-clock change (`1d6772a`) shifted
the eval cadence relative to what the threshold was tuned against.

## What the affected results still mean

Both arms consumed EXACTLY 1,969,422,336 tokens, so seed-1 A-vs-B remains a valid **matched-budget**
comparison -- just not a convergence one. Label it that way:

> At a matched 1.97B-token budget: A 2.1332, B 2.2140. Looping did NOT fully recover the capacity of
> six real layers at this budget.

Weak additional signal: A was improving FASTER at the cut (-0.0050, -0.0053 per eval vs B's -0.0012,
-0.0039), so extending would more likely widen A's lead than close it.

## FIXED 2026-07-29 -- and the root cause was confirmed, not guessed

The `min_rel` suspicion above was right. With `expected_plateau_epoch` defaulting to 2.0 and the runs
sitting at 1.80 epochs, the hint interpolation loosened the threshold to
`0.002 + 0.8*(0.02-0.002) = 0.0164` -- a demanded 1.64% fitted drop per 6-eval window. Both arms were
really shedding ~1.4%. Genuine improvement, judged against a threshold that the proximity-to-hint
loosening had raised above it. `tests/coherence_tests.cpp` now asserts exactly that:
`trend_plateaued(arm_a, 5, 0.0164)` is TRUE on the real series.

**The fix is a second, THRESHOLD-FREE condition rather than a retune**: `coherence::improved_recently`
(a classic early-stopping patience check). A plateau now additionally requires no new best val_nelbo in
the last `PLATEAU_PATIENCE = 3` evals. Any threshold can be wrong for a run whose real improvement rate
is slower than assumed; "is this still producing the best model it ever has?" cannot be. Retuning
`min_rel` was deliberately NOT done -- it would trade this false stop for a different one elsewhere.

Pinned by three regression cases on the real observed series (both arms' last six evals).

## Correction: the token/epoch difference was NOT a resume bug

An earlier version of this note attributed arm C's larger reported tokens/epochs to resume accounting.
That was wrong -- C simply ran at batch 512 where A and B ran at 448. See
[[loopsplit-3arm-batch-confound]]. Resume accounting was nonetheless improved in passing (the count is
now carried from state.json instead of reconstructed as `step*batch*SEQ_LEN`, which was wrong under
varied sequence length), but it was not the cause of anything observed here.

## Fix options considered at the time (user decision 2026-07-28: keep the sweep as-is, fix later)

1. Raise/retune the relative-improvement threshold so runs converge.
2. Pin `--steps` explicitly for A/B work, removing the detector from the experiment entirely. For a
   controlled comparison this is arguably better regardless -- "matched budget" becomes a deliberate
   choice rather than an artifact of a heuristic.

**Checkpoints exist, so the truncated runs can be RESUMED from their stop point rather than rerun** --
`models/ls_*/model.bin.step*.ckpt`. That makes this cheap to correct after the fact: continue each
arm to real convergence and re-read the comparison, no restart needed.

Related: [[d512-fault-resume-optimizer-switch-bug]] (resume correctness),
[[tinystories-d448-plateau-result]] (an earlier plateau that WAS genuine, for contrast).
