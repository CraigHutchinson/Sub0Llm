---
name: three-pillar-shootout-policy
description: "STANDING POLICY (user-directed 2026-07-17): every shootout/A-B reports all THREE pillars -- correctness (accuracy), performance (wall-clock/throughput), memory (bytes added) -- never accuracy alone; a 'winner' can hide a hidden step-cost or memory tax."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User directive (2026-07-17, during the pairspike/encoder shootout work): "we should also weight
performance, correctness, and memory on the results as/if these shootouts are sensitive to any of these
aspects but the three core pillars to ensure we strive to optimize against."

**Why:** the encoder shootout made the risk concrete -- an accuracy-only table crowned ConvPool for
persistent-range resolve, but the arms differ materially on the other two pillars: HRR's compose is
O(C^2) circular convolution per fragment, ConvPool adds 2*C*C learned params PLUS grads and Adam moments
(and is single-thread-training-locked until per-thread enc_w reduction exists), while MeanPool/Hash are
O(C*len) and parameter-free. At d448 production scale those differences compound into real throughput
and VRAM/RAM costs the accuracy table never shows.

**How to apply:** every future shootout/A-B report includes, per arm: (1) accuracy (the existing
drilled/held-out numbers), (2) wall-clock -- train s/seed and eval s/seed at minimum, tokens/s where
cheap to compute, measured not estimated, with thermal-confound care per
[[thermal-confounds-ab-wallclock-testing]] when wall-clock differences are close, and (3) memory delta --
static per-arm facts (added params/state bytes) suffice when allocation is deterministic. First
implementation: pairspike_engine_tests.cpp's Route-A experiment (train/eval s/seed + encoder-mem note in
the MEAN summary line, 2026-07-17). Weighting between pillars stays a judgment call per decision -- the
policy is that all three are MEASURED and REPORTED so the trade is visible, not that a fixed formula
picks winners.
