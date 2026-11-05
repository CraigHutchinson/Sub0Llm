---
name: loopsplit-ab-negative-result
description: "LoopSplit three-pillar A/B, 2026-07-27: a clear NEGATIVE result. 6 vs 8 vs 10 executions at identical parameter count costs 18-31% throughput and 21-43% activation memory for NO measurable quality gain (noise exceeds signal 1.89x, ranking flips between seeds). Do not enable LoopSplit at this scale on this evidence."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-27T15:01:13.129Z
---

> **SCOPE CAVEAT added 2026-07-28 -- read before citing this result.** This A/B ran on
> **TinyStories**, which is synthetically constrained to a ~1.5k-word vocabulary of concrete nouns and
> simple actions. Weight-shared looping is supposed to earn its cost by ITERATIVELY REFINING
> ABSTRACTIONS -- contracting complex/technical terms toward shared representations. A corpus with
> essentially no abstraction to contract cannot exercise the mechanism under test. So the honest
> statement is **"negative on a corpus that cannot express what LoopSplit is for"**, NOT "negative at
> this parameter scale". The throughput and memory costs below are solid and corpus-independent; the
> **quality** half of the verdict is confounded and must be re-measured on a denser corpus before it
> is used to reject looping -- or to reject its relatives (depth attention, mHC), which inherit the
> same weak prior. See [[roadmap-2026-07-21]] for the corpus plan.

Matched-budget, two seeds from the start (the GQA A/B had already established that single-seed quality
claims at this budget are unsupported -- see [[gqa-ab-three-pillar-result]]). `out/build/gqa_ab`, d384
L6 H8 seq256, TinyStories 671M tokens, Muon, 6000 steps, batch PINNED 256, seeds 42 + 1337.

**Split under test**: `LOOP_MIDDLE_LAYERS=2` of `N_LAYERS=6`, so `LOOP_MID_START=(6-2)/2=2` -- head
[0,2), repeated middle [2,4), tail [4,6). Execution orders: `0 1 2 3 4 5` / `0 1 2 3 2 3 4 5` /
`0 1 2 3 2 3 2 3 4 5`. **Parameter count identical across all three (16.97M)** -- that is the point:
LoopSplit's proposition is "more compute from the same weights", so holding params fixed and varying
executions is the comparison that tests it.

| variant | nelbo s42 | s1337 | mean | seed spread | median tok/s | acts MB |
|---|---|---|---|---|---|---|
| 6 exec (off)   | 1.2845 | 1.2793 | 1.2819 | 0.0052 | 276,096 (1.000x) | 300.2 |
| 8 exec (mid2x2)| 1.2817 | 1.2748 | 1.2782 | 0.0069 | 225,814 (0.818x) | 364.8 |
| 10 exec(mid2x3)| 1.2816 | 1.2767 | 1.2792 | 0.0049 | 189,175 (0.685x) | 429.4 |

## Verdict: negative, and decisively so per unit of compute

- **Quality NOT resolvable.** Max seed spread within a variant 0.0069 vs max between-variant spread
  0.0037 -- **noise exceeds signal 1.89x**. The ranking flips between seeds (s42: 10<8<6; s1337:
  8<10<6). The un-looped variant was worst in BOTH seeds, which is suggestive but with n=2 has ~1/3
  probability by chance -- NOT a result.
- **Throughput costs 18-31%.** Monotonic in execution count and well outside run-to-run spread.
- **Activation memory costs 21-43%.** Params are genuinely fixed; activations are NOT. They scale with
  EXECUTIONS, sub-linearly (1.43x at 1.667x executions -- only the per-layer term scales, not the final
  block). So LoopSplit trades activation memory for depth, and repeats are VRAM-capped at a given batch.
- Per unit of compute it is strictly behind: you pay ~1.45x the wall clock for a quality change that is
  inside the noise floor.

**Recommendation**: do not enable LoopSplit at this scale on this evidence.

## What this does NOT rule out

One point in a two-dimensional space. Only the REPEAT COUNT was varied; the BLOCK WIDTH (2 of 6 layers)
was fixed, and `LOOP_MID_START` centres it automatically. A wider repeated block, a different placement,
or a much larger model could all behave differently -- Nanbeige apply this at 3B+, and the mechanism's
premise is that a deep reasoning core benefits from re-application, which a 17M-param model on
TinyStories may simply not have. The negative is about THIS scale and THIS shape, not the idea.

## Cross-cutting: GPU training is not run-to-run reproducible

Two nominally IDENTICAL runs (same config, seed 42, same 6000 steps -- the GQA A/B's kv8 and this A/B's
r1) gave **1.2940 vs 1.2845**, a 0.0095 gap with byte-identical config banners. Most likely cause is
`atomicAdd` ordering in the backward (embeddings, dgamma) plus cuBLAS kernel selection; NOT confirmed by
a controlled same-binary double-run, so treat as strong suspicion rather than established. Either way it
means fixed-seed does NOT imply reproducible on GPU, and it further supports both A/Bs' "not
resolvable" conclusions.
