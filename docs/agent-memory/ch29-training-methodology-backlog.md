---
name: ch29-training-methodology-backlog
description: Backlog — vocab/arch A/Bs must be training-BUDGET-MATCHED (early-stop confounds bigger models); plus sample-weighting refinement
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Backlog refinements surfaced 2026-06-14 from the Ch29 vocab sweep (see [[ch29-model-quality-founded-dims]]):

**1. A/Bs are NOT training-budget-matched — early-stop confounds bigger models.** The self-terminating early-stop (patience on held-out NELBO + 20-epoch min floor) is implicitly calibrated to the V=512 / D=128 regime. A bigger vocab (or wider/deeper model) has more embedding params + more output classes and plausibly needs MORE iterations to converge — but it often early-stops SOONER (its harder, sparser NELBO plateaus within patience faster). EVIDENCE: V=512 trained 24030 steps, V=1024 only 15515 steps (FEWER), V=2048 fewer still — so "bigger vocab = worse recall" is confounded with "bigger vocab trained less." This likely ALSO affected the earlier width sweep (D=192 "worse + fragile") and the whole-word run (stopped at the min-epochs floor). FIX: for capacity/vocab A/Bs, compare at a FIXED step (or epochs/tokens-seen) budget with early-stop OFF — `--steps N --patience <huge> --min-epochs <huge>` — or scale patience/min-epochs with param count. Re-run the vocab sweep budget-matched before trusting the V=512-wins conclusion (proxy is also only ~136K tokens → over-penalizes large vocab via sparsity; a full-corpus budget-matched run is the real test). The undertraining test (`--track-recall`) is the V=512 instance of this same question — so far FLAT (≈converged).

**2. Per-sample / per-token loss weighting may need refining.** Current objective: NELBO 1/t importance weight `n_masked/(t·T)` × a mean-over-masked CE (`weighted_cross_entropy`). Open questions: (a) the mean-over-masked normalization weights every masked subword equally — with different vocab sizes the subwords-per-word differs, so the effective per-WORD learning signal shifts across tokenizations; (b) rare tokens (common at larger vocab) get little gradient signal → consider frequency-aware class weighting or label smoothing; (c) whether to weight per-word vs per-token under whole-word masking. Refine + A/B these as a lever, especially when comparing vocab sizes. [[verify-correctness-against-reference-before-perf]]
