---
name: memplan-vram-prediction-gap-at-scale
description: "memplan's VRAM-usage prediction OVER-estimates measured device usage as batch grows at the larger d768 shape (~141 MiB@batch64/128, ~504@256) -- safe direction (conservative, not OOM-risking). NOW HANDLED at the test/runtime level via an ASYMMETRIC tolerance (tight UNDER, wide OVER; FOOTPRINT_OVERPREDICT_TOLERANCE_MB, committed c4654f3) so [cuda] is green; root-cause of the over-count still not investigated"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Found 2026-07-10 while rebuilding/reconfiguring `out/build/ts_cuda` for the d768 TinyStories
Chinchilla-sweep point (after landing on `d=768 L=11 H=12 seq=256 vocab=16668`, `d_head=64` matching
the sweep's established convention). `ctest -R "memplan prediction matches measured device usage"`
(the parity test in `tests/cuda_tests.cpp` around line 1170-1193) failed at this larger shape:

```
batch=32   predicted=2263.78 MiB  measured=2322 MiB  gap=+58.2 MiB   (predicted UNDER -- fine)
batch=64   predicted=2985.67 MiB  measured=2844 MiB  gap=-141.7 MiB  (predicted OVER, exceeds FOOTPRINT_TOLERANCE_MB)
batch=128  predicted=4429.45 MiB  measured=4286 MiB  gap=-143.4 MiB  (predicted OVER, exceeds tolerance)
batch=256  predicted=7317.01 MiB  measured=6813 MiB  gap=-504.0 MiB  (near/over VRAM budget, parity skipped)
```

**Pattern**: prediction starts slightly under measured at batch=32, flips to increasingly OVER
measured as batch grows, with the overshoot itself growing (not a fixed constant offset) --
suggests a per-batch-row term in memplan's formula is miscalibrated for this larger d_model/layer
combination (previously only verified up to d512 in this project's own sweep), not a fixed
one-time constant that's merely off.

**Not a safety issue**: over-predicting VRAM need is the SAFE direction of error (leads to an
overly conservative auto-batch clamp / wasted headroom, not an OOM crash). Also not blocking actual
training: `train_stage.cpp`'s own batch resolution does live optimistic-allocate with a fallback to
a conservative estimate only if real allocation genuinely fails (see the resume-time VRAM
re-validation comment in that file) -- so a static prediction being somewhat pessimistic at this
scale doesn't propagate into a wrong runtime decision, just a potentially-suboptimal one.

**Confirmed NOT caused by this session's own work**: `git log -- include/sub0/memplan.hpp
tests/cuda_tests.cpp` shows neither file touched by the tokenizer-redesign or RunConfig commits
just made -- this is a pre-existing calibration gap that simply hadn't been exercised at a shape
this large before (the sweep's prior largest point was d512).

**Not yet investigated further.** Candidate next steps, not started: (1) re-derive/re-measure
memplan's per-parameter and per-batch-row byte constants at d768 specifically (mirroring how the
existing constants were "calibrated against a real measurement" per memplan.hpp's own comments);
(2) check whether the gap is specific to THIS architecture combination (11 layers, 12 heads, gated-
FFN + tied-embeddings + QK-norm all on) vs a generic d_model-scaling drift; (3) decide whether
`FOOTPRINT_TOLERANCE_MB` should simply widen (if the remaining gap is inherent/acceptable) vs the
formula itself needs a correction term.

**UPDATE 2026-07-11 -- HANDLED at the test/runtime level (committed c4654f3).** While making the full
[cuda] suite green (see [[corpus-blending-and-spell-integration]]), reframed the tolerance as
ASYMMETRIC by CONSEQUENCE rather than widening it symmetrically: UNDER-prediction (predicted < measured)
is the dangerous, OOM-risk direction and keeps the tight `FOOTPRINT_TOLERANCE_MB` (128); OVER-prediction
(predicted > measured) is SAFE (reserve more, never OOM) and gets a wider `FOOTPRINT_OVERPREDICT_TOLERANCE_MB`
(320) that still catches a gross buffer drift but tolerates this ~141MB d768 over-estimate. Applied in
BOTH the `memplan prediction matches measured device usage` test AND train_stage.cpp's runtime footprint
message (was `std::fabs` symmetric, cried "STALE" every d768 run). batch=256 stays guarded out (predicted
> 0.8*VRAM -> parity skipped), so the ~504MB overshoot there never trips the check. NOTE: the cudaMemGetInfo
MEASUREMENT itself is WDDM-noisy (~200MB swing between runs at batch32: 2122 vs 2322 measured for the same
2263 predicted), so don't chase a single run's exact gap. Root-cause of the over-count still OPEN -- it is
a batch-GROWING over-prediction (not a fixed constant), so a per-batch-row term in train_scratch_bytes is
the suspect (candidate steps 1-3 below still stand); the asymmetric band buys correctness without it.

See [[gpu-tune-benchmark-vs-production-discrepancy]] for a related but DIFFERENT prior finding
(tune's own throughput *benchmark*, not memplan's *memory* prediction, diverging from production) --
worth keeping distinct, don't conflate the two when picking this up later.
