---
name: ch30-iterative-refinement-precondition
description: Iterative diffusion decoding LOSES to one-step on weak models (error compounding); commit precision is the precondition
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

Ch30 measured (2026-06-13, 857K-param denoiser, shakespeare): iterative canvas refinement
loses to one-step recovery by 0.5-1.7pp recall at 25-90% noise, under every variant —
confidence-threshold commits, one-token-at-a-time, low-confidence remasking. With ~10-15%
top-1 accuracy, most committed tokens are wrong and self-generated context compounds errors
that one-step's clean masked context never sees.

**Why:** the reverse process only pays when the model's confident predictions are usually
RIGHT (the implicit assumption behind MaskGIT/LLaDA-style decoding). Commit precision, not
sampler cleverness, is the binding constraint.

**How to apply:** re-run ch30_iterative_sampling's §3 comparison whenever a stronger
checkpoint lands (knobs: --conf/--min-commit/--remask/--temperature); expect the sign to flip
as base accuracy grows. Don't burn time tuning sampler schedules on weak models. The sampler
(refine_canvas in diffusion/include/sub0diff/nn/sampler.hpp) supports remasking + entropy_bound
early stop; model loading via load_model_dir(). See [[ch28-curriculum-findings]],
[[conditioning-embedding-amplitude]].

**UPDATE (2026-06-16): the sign FLIPPED to break-even.** On the founded 6M model (~33% recall,
[[diffusion-nelbo-ceiling-reframe]]), iterative vs one-step is now ~TIED: 25% noise +0.3 to +1.0pp,
50% noise -0.1pp (small-budget, noisy). So iterative refinement NO LONGER COMPOUNDS errors — it has
crossed from net-negative to neutral, exactly as predicted as base accuracy grew. BUT not yet a clear
WIN, and GENERATION is still incoherent (mask-artifact fragments, not Shakespeare). Verdict on "is the
model sufficient for ch30 to work": the MECHANISM works now (break-even, no longer harmful) but the
model is not strong enough for iterative to clearly beat one-step or to generate coherent text — Ch30
will run/refine but won't impress until the model strengthens (data/proportions, not the sampler).
NOTE: ch30's §3 recall sweep over many windows × iterative passes is SLOW and timed out at 1500/300
windows while contending with a live training run — use --recall-windows ~120 for a quick read.
