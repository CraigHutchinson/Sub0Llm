---
name: loopsplit-3arm-batch-confound
description: "CORRECTION — the 3-arm LoopSplit sweep is NOT token-matched: arm C ran at batch 512/lr 8.0e-3 while A and B ran at 448/7.48e-3, so C trained on 14.3% more tokens at the same step count"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T17:23:44.228Z
---

Found 2026-07-29 while trying to fix what I had recorded as a "resume token/epoch accounting bug".
**That diagnosis was wrong.** The numbers were never inflated by resume bookkeeping -- they are correct,
and they are reporting a real difference in what the arms actually trained on.

## The evidence

From the three `config.json` files and the `train.log` schedule lines:

| arm | layers | batch | lr | steps | tokens_seen | epochs | best val_nelbo |
|-----|--------|-------|----|-------|-------------|--------|----------------|
| A_deep16    | 16 | **448** | 7.48e-3 | 8586 | 1,969,422,336 | 1.80 | **2.1332** |
| B_loop10x6  | 10 | **448** | 7.48e-3 | 8586 | 1,969,422,336 | 1.80 | 2.2140 |
| C_shallow10 | 10 | **512** | 8.00e-3 | 8586 | 2,250,573,552 | 2.06 | 2.2034 |

2,250,573,552 / 1,969,422,336 = 1.1428 = **512/448 exactly**. The epoch figures differ by the same
ratio for the same reason (`epoch_steps = epoch_tokens / (batch * SEQ_LEN)`). Everything is
self-consistent; the sweep simply matched **steps**, and steps are not tokens when batch differs.

## Why the batches differ

The harness sized each arm's batch from available VRAM. C (10 real layers, no loop) has the smallest
activation footprint of the three, so it got a larger batch -- and, being derived from batch, a
different learning rate. A and B coincidentally matched because B's 16 EXECUTIONS cost about what A's
16 layers do, which is exactly the property the experiment was built around.

## What this does to the result

- **A vs B stays clean.** Identical batch, lr, steps and tokens. A 2.1332 vs B 2.2140 -- real depth
  beats looping at a matched budget.
- **B vs C is CONFOUNDED.** C beat B by 0.0106 nelbo while training on 14.3% more tokens. The
  conclusion I previously recorded -- "the shallow control now beats looping" -- is not supported by
  this data. It may still be true; this sweep cannot say.
- The A-vs-C gap is likewise not a clean capacity comparison.

## What to do about it

Pin **batch and lr explicitly and identically across arms**, not per-arm auto-tuning, and match on
TOKENS rather than steps. The batch must be the one the most expensive arm can fit (448 here), and
every arm runs at it even where more would fit -- the wasted headroom is the price of a controlled
comparison. Re-run B and C on that basis before drawing any B-vs-C conclusion.

Supersedes the resume-accounting claim in [[plateau-detector-fires-while-still-improving]] (the
plateau finding in that note is real and is now fixed; only the token-inflation attribution was wrong).
Related: [[three-pillar-shootout-policy]], [[thermal-confounds-ab-wallclock-testing]] --
this is the same class of error, a comparison whose arms differ in something other than the variable
under test.
