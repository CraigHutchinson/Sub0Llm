---
name: loop-as-thinking-loop-needs-writable-context
description: "DESIGN FRAME (user, 2026-07-29) — a recursive loop is a THINKING loop, so what it needs is an accumulating thought context. Reframes arm D: depth attention is a READ channel; the project already has a WRITE substrate (persistent slots)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T18:49:38.623Z
---

User's framing: the recursive loop is effectively a thinking loop, so building a **thought context** --
accumulated knowledge that drives further iteration and the final outcome -- is what conceptually makes
it work.

## Why this is the right frame, and what it predicts

A LoopSplit pass hands the next pass ONE thing: the residual stream. That is a fixed-capacity,
overwrite-in-place channel with no addressability -- pass 3 cannot ask "what did pass 1 conclude about
position 7?", it can only receive a sum. A thinking loop whose only memory is its latest conclusion is
exactly the failure mode the measured result shows (B 2.2140 vs A 2.1332 at matched budget).

**This project has already proven this exact point elsewhere, twice**, which is why the frame should be
trusted rather than treated as an analogy:

- [[factspike-piece-transfer-validated]] -- injecting a representation ONCE and relying on ordinary
  computation to carry it forward loses fidelity versus real per-layer computation.
- [[repeatspike-harness-collapse-result]] / [[wordspike-natural-prose-collapse]] -- an explicit harness
  beats fuzzy re-derivation (0.95 vs 0.13 held-out).
- [[deterministic-mechanisms-thesis]] -- the standing thesis is exact addressable state over fuzzy
  re-computation.

The loop is the same "carry it in the residual and hope" bet, in a new place. So the fix should be the
same shape as the fixes that already worked here.

## The consequence: READ channel vs WRITE substrate

- **Depth attention is a READ channel.** Later passes softmax over earlier passes' (K,V) and mix their
  V. It gives addressable *access* to what earlier passes computed. Zero parameters, cheap
  ([[depth-attention-stage0-done]]).
- **It does NOT give the loop a place to deliberately WRITE a conclusion.** The mixing is soft and
  implicit; nothing lets pass 1 commit "this span resolved to X" in a form pass 3 can retrieve exactly.
- **This project already has that write substrate**: the persistent slot-range engine
  ([[persistent-slot-range-engine-substrate]], unbounded id >= VOCAB) plus the sentinel-pair addressing
  design ([[persistent-slot-selection-problem-backlog]]). A loop that writes intermediate conclusions
  into scratch slots and reads them back next pass is a far more direct realization of "build a thought
  context" than soft V-mixing is -- and it is the project's own thesis rather than a borrowed feature.

## Cheap diagnostic to run FIRST, before building either

If the loop is a thinking loop, the question is whether later passes do DIFFERENT work. That is
directly measurable on the arms that already exist:

1. **Per-pass residual delta norm.** If ||Δh|| collapses across passes, the loop is converging to a
   fixed point and the extra compute buys nothing -- the mechanism this whole line predicts.
2. **Depth-attention probability mass on SELF vs earlier passes** (once arm D exists). If it peaks on
   its own entry, the passes are not using each other and depth attention is inert.

Both are far cheaper than a training arm and either could falsify the premise before more is spent.

## Honest caveat

At d192 / ~7M params / 1.8 epochs, nothing here is "thinking" in any strong sense. The frame is a
DESIGN hypothesis about what the mechanism needs, and it is a good one; it is not an established
property of these models. Do not let the vocabulary smuggle in a claim the measurements do not support.

Related: [[depth-attention-rescues-loopsplit-hypothesis]] (the symmetry-breaking argument this extends),
[[loopsplit-3arm-batch-confound]] (why the current numbers must be re-read carefully).
