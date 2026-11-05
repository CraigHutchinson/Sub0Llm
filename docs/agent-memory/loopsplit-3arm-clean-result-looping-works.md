---
name: loopsplit-3arm-clean-result-looping-works
description: "RESULT 2026-07-29 — the CLEAN token-matched 3-arm sweep: A 2.1332 / B_loop 2.2140 / C_shallow 2.2300. Looping DOES recover capacity (beats the shallow control), but real depth is worth ~5x more"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T21:21:12.682Z
---

The definitive three-arm LoopSplit result, after arm C was re-run at arms A/B's pinned batch 448 /
lr 7.48e-3 for 8586 steps -- **1,969,422,336 tokens, byte-identical budget across all three arms**.

| arm | params | executions | best val_nelbo | ctx 64->512 gain | bits/byte |
|-----|--------|-----------|----------------|------------------|-----------|
| A_deep16    | 9.66M | 16 | **2.1332** | 14.2% (0.3517) | 0.781 |
| B_loop10x6  | 7.23M | 16 | **2.2140** | 13.6% (0.3472) | 0.811 |
| C_shallow10 | 7.23M | 10 | **2.2300** | 13.3% (0.3409) | 0.817 |

## CORRECTION to an over-claim made from the confounded data

On the confounded run I reported the context-length gain as A 14.2% / B 13.6% / C **11.8%** and
concluded that "context gain tracks EXECUTION COUNT, not parameter count" -- a striking pattern.
**Re-measured on the clean arm C it is 13.3%, and that claim is much weaker than I stated.** The
ordering still holds directionally (14.2 > 13.6 > 13.3, executions 16 / 16 / 10) but the magnitudes are
compressed: B-vs-C is 0.3 percentage points, not 1.8. The over-fed arm C had an artificially DEPRESSED
context gain, which flattered the hypothesis.

Treat B-vs-C on context gain as **within noise until multi-seed**. The mean-nelbo gap (0.0160) is the
sturdier of the two signals, and even that is single-seed.

`models/ls_C_shallow10_s1b/` is the clean arm C. The old `ls_C_shallow10_s1` ran at batch 512 and is
NOT comparable -- see [[loopsplit-3arm-batch-confound]].

## The answer: looping works, weakly

- **B beats C by 0.0160 nelbo at IDENTICAL parameters and IDENTICAL tokens**, purchased with 60% more
  executions. Looping really does convert compute into quality at a fixed parameter count. This is a
  POSITIVE result and it reverses what the confounded sweep appeared to show.
- **A beats B by 0.0808** -- real depth is worth about 5x what looped depth is.
- Looping recovers roughly **16.5%** of the gap between shallow-10 and deep-16
  (0.0160 / 0.0968).

## Why this matters for the mechanism

"Some benefit, but far less than distinct layers" is exactly what the fixed-point argument predicts:
identical passes applied repeatedly do converge, so later passes contribute progressively less. The
result is consistent with [[depth-attention-rescues-loopsplit-hypothesis]] and it means the premise is
sound rather than the whole idea being dead. Depth attention (or a writable thought context --
[[loop-as-thinking-loop-needs-writable-context]]) is aimed at exactly the residual 83.5%.

## Corrections this supersedes

- The earlier "C now BEATS B, looping is worse than the shallow control" reading is **RETRACTED**. It
  was an artifact of C training on 14.3% more tokens.
- [[loopsplit-ab-negative-result]] (the TinyStories A/B) remains separately caveated -- that one was
  confounded by corpus, not batch.

## Caveats that still stand

- **Single seed (1).** Seed noise on this project has previously been 1.73x the signal
  ([[gqa-ab-three-pillar-result]]). B-vs-C at 0.0160 is NOT safely outside that. Needs the multi-seed
  run before it is load-bearing.
- Both A and B were plateau-truncated at 8586 steps by the (now-fixed) false-plateau bug, so all three
  are matched-BUDGET, not converged. C's re-run used an explicit `--steps`, so no arm was
  plateau-stopped -- the budgets match exactly.
- C's context-length gain (11.8%) is from the CONFOUNDED run; re-report it on `ls_C_shallow10_s1b`.
