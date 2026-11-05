---
name: gqa-ab-three-pillar-result
description: "First real A/B of any Nanbeige feature (2026-07-27): GQA on CUDA at d384/L6/H8, kv-heads 8 vs 4 vs 2, TinyStories, matched budget, TWO seeds. Throughput +8.4%/+12.6%, KV cache -50%/-75%, and quality differences NOT resolvable -- seed noise exceeds the between-variant signal 1.73x and the ranking flips completely between seeds."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-27T04:25:32.900Z
---

The first of the three Nanbeige features to get actual numbers rather than only correctness proofs
([[nanbeige-features-progress]] had flagged that gap explicitly). Setup: `out/build/gqa_ab`, d384 L6 H8
seq256 vocab16535, gated-ffn + tied + qk-norm, TinyStories (671M tokens, a REAL corpus per `AGENTS.md`),
Muon, 6000 steps (~0.6 epoch), **batch PINNED at 256**, seeds 42 and 1337. RTX 5070.

Batch was pinned deliberately: letting GQA auto-pick a bigger batch off its lower memory would confound
the quality comparison with the memory benefit. The memory saving is reported separately as what WOULD
permit a larger batch.

## Performance -- real, monotonic, well outside run-to-run spread

| variant | median tok/s | stdev | vs MHA |
|---|---|---|---|
| kv=8 (MHA)   | 264,803 | 9,597 | 1.000x |
| kv=4 (GQA-2) | 286,993 | 7,335 | **1.084x** |
| kv=2 (GQA-4) | 298,281 | 8,029 | **1.126x** |

Gaps (+22K, +33K) are 2-3x the within-variant stdev, so not thermal drift
([[thermal-confounds-ab-wallclock-testing]]).

## Memory -- exact, not sampled

| variant | params | KV cache / sequence |
|---|---|---|
| kv=8 | 16.97M | 4.50 MB |
| kv=4 | 16.09M (-5.2%) | 2.25 MB (**-50%**) |
| kv=2 | 15.64M (-7.8%) | 1.125 MB (**-75%**) |

The KV-cache row is the number GQA exists for and is now measured-config-real rather than derived from
the layout. Param deltas match `2*D_MODEL*D_KV*L` exactly (0.885M then 0.442M).

## Quality -- NOT RESOLVABLE at this scale/budget. Do not quote a winner.

best val_nelbo:

| variant | seed 42 | seed 1337 | mean | seed spread |
|---|---|---|---|---|
| kv=8 | 1.2940 | 1.2699 | 1.2820 | 0.0241 |
| kv=4 | 1.2884 | 1.2773 | 1.2829 | 0.0111 |
| kv=2 | 1.2934 | 1.2984 | 1.2959 | 0.0050 |

- max seed-to-seed spread WITHIN a variant: **0.0241**
- max spread BETWEEN variant means: **0.0139**
- noise/signal ratio: **1.73x** -- the experiment cannot resolve the variants.
- **The ranking flips completely between seeds**: seed 42 gives kv4 < kv2 < kv8 (MHA WORST), seed 1337
  gives kv8 < kv4 < kv2 (MHA BEST). That flip is the decisive evidence, more than the ratio is.

**Had I stopped at one seed I would have reported "GQA-2 slightly best" -- a false claim.** Seed 42's
between-variant spread was 0.006 and non-monotonic (middle variant best), which is what noise looks like;
the second seed was run specifically to quantify the floor rather than to confirm a hunch. Any future
single-seed quality claim on this codebase at this budget should be treated as unsupported: the floor is
~0.024 nelbo.

## Verdict

GQA delivers its textbook benefit here -- **+8-13% throughput and 50-75% KV-cache reduction at no
measurable quality cost** -- with "no measurable cost" meaning exactly that, not "no cost". A real
quality difference smaller than ~0.024 nelbo would be invisible to this setup. Resolving kv=2's
apparent (mean-level, unconfirmed) deficit would need more seeds or a longer budget.

## Verification trap this run exposed

`sub0-configure`'s summary line does NOT echo `kv-heads`, so all three variants print an identical
config banner. Proof they actually differed came from the param counts (16.97M / 16.09M / 15.64M)
matching the predicted Wk/Wv narrowing. Without that independent check there was nothing distinguishing
"three variants" from "the same build three times". Check a DERIVED quantity, not the banner.

See also [[gpu-large-batch-access-violation]] -- the auto-tuned batch (704) segfaults on this config, which
is why batch is pinned here.
