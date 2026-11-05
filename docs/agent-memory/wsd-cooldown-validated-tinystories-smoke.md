---
name: wsd-cooldown-validated-tinystories-smoke
description: "MEASURED 2026-07-31 — the WSD cooldown extracted 0.0478 nelbo in ~0.2 epochs from a run that had been genuinely FLAT for 5 evals at constant LR. The under-finishing hypothesis is confirmed; the whole pipeline (schemeV5 + WSD + pre-anneal ckpt) validated end to end"
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T23:19:58.066Z
---

First WSD-era trajectory. TinyStories, GPU, d196 L11 H7 seq256, 8.72M params, batch 512, schemeV5
tokenizer (vocab 16824), plateau-stopped. `models/smoke_v5_tinystories/`, build `out/build/smoke_v5`.

## The result

```
stable phase (constant lr 8.00e-03), evals from step 4972:
  1.3905  1.3986  1.3936  1.3987  1.4017     <- flat/oscillating, no new best for 5 evals
plateau fires @ step 6780 (1.50 ep) -> pre-anneal ckpt written -> cosine 8e-3 -> 8e-4 over 678 steps
  1.3570  1.3427                             <- both new bests
stopped @ 7684, best 1.3427
```

**The anneal extracted 0.0478 nelbo in ~904 steps (~0.2 epochs)** from a run that was genuinely flat.
For scale: arm A gained 0.0593 over **3.8 extra epochs** of ordinary training.

## What this confirms

1. **The under-finishing hypothesis** — the central justification for replacing inverse-sqrt. Every prior
   model in this project stopped on the plateau detector and therefore **never ran a cooldown at all**,
   leaving roughly this much on the table. Confirmed, and the effect is large.
2. **A constant stable phase makes the detector well-posed.** The flat line was REAL — five evals, no new
   best — rather than a decaying LR being mistaken for saturation. This is the subtler half of the WSD
   argument and it held.
3. **The mechanics work end to end**: schemeV5 tokenizer loaded, GPU training ~141K tok/s,
   `model.bin.preanneal.step000006780.ckpt` written AND survived pruning alongside the ordinary
   `step000006780.ckpt` (the pre-anneal one carries `cooldown_start = -1`, so restoring it resumes IN the
   stable phase). Coherent samples throughout.

## Do NOT cross-compare these numbers

* **1.3427 is not comparable to the historical TinyStories best of 1.2388.** Different tokenizer
  (schemeV5 changes vocab and token count, so per-token entropy differs) AND different model size
  (d196 vs d448). See [[tinystories-d448-plateau-result]].
* **Plateau at 1.50 ep vs the ledger's ~1.9-2.0** is suggestive but confounded the same way. It is
  consistent with constant LR removing the slow decay-driven creep that used to keep the curve
  descending, but it is NOT yet evidence about `DEFAULT_EXPECTED_PLATEAU_EPOCH` — that still needs the
  unbounded control with the detector DISABLED, or it measures the detector rather than the corpus.

## Bugs this smoke run found (the reason to smoke at all)

A 200-step validation pass beforehand announced `cosine cooldown from step 150` against a **1129-step
warmup** — a cooldown that can never fire, because `lr_at()` gives warmup precedence. The math was right;
the announcement described a schedule the run would not follow. Fixed + pinned in `1e10c23`.

## Still open

* The **A/B has not been run**: inverse-sqrt vs WSD at matched tokens. This shows the cooldown pays;
  it does not show WSD beats the old schedule overall at equal budget.
* Single run, single seed, small model, one corpus.

Related: [[lr-schedule-wsd-cooldown-backlog]], [[plateau-detector-calibrated-for-tinystories-era]],
[[truecasing-collapse-unconditional]].
