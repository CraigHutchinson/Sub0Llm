---
name: conditioning-embedding-amplitude
description: Additive conditioning embeddings must match token-embedding init scale (1/sqrt(D)) or learning stalls ~10x
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

Unit-amplitude sinusoidal time/noise embeddings added to token embeddings (init N(0,1/sqrt(D)),
~0.09/dim at D=128) drown the content signal — token identity becomes <1% of input energy and
training crawls. Measured on Ch29 (2026-06-12): held-out NELBO stuck at 6.13≈uniform(6.24) for
16K steps; after scaling the sinusoid amplitude to 1/sqrt(D) in
diffusion/include/sub0diff/nn/time_embedding.hpp, the same run reached 5.17 (PPL 175 vs 514)
and recall 8.4%@10% noise on held-out Shakespeare.

**Why:** RMSNorm at block entry normalizes the SUM; whichever added embedding is loudest owns
the variance budget. Ch28's slow ramp (~200K steps on a trivial corpus) was the same pathology
masked by an easy task.

**How to apply:** any additive conditioning signal (time, noise level, future: position/segment)
should be initialized/scaled to the token-embedding std. Debug heuristic: if loss hovers at
ln(vocab) but a single-sample overfit test converges, check input-energy ratios first.
Ch28 numbers in [[ch28-curriculum-findings]] were measured PRE-fix; rankings likely hold but
absolute recall should improve if re-measured.
