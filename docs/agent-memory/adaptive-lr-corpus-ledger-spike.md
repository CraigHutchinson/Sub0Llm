---
name: adaptive-lr-corpus-ledger-spike
description: "BACKLOG SPIKE 'Tutor' (docs/TUTOR.md) — per-entry adaptive LR from a corpus mastery surface. KEY: weight by learning VELOCITY not NELBO level, which separates unlearnable from not-yet-learnt with NO reference model. Velocity MUST be normalised by applied learning or the controller confounds its own actuator. Heat map leads"
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T21:12:00.027Z
---

Written up in `docs/TUTOR.md` (was ADAPTIVE_LR_LEDGER.md). **Sandbox spike, not mainline.** Open question
is not feasibility but **"does the feedback pay for its own cost and beat global WSD at matched tokens"**.

Working name **Tutor** — the schedule stops being a clock and becomes a teacher in a feedback loop
(checking progress, finding gaps, rehearsing). Per-entry state = the **mastery surface**. Provisional;
alternatives were *isoprogress* and plain *corpus ledger*. Spike would be `tutorspike`.

## THE KEY INSIGHT (user's, and it dissolves the blocker)

I had filed "high loss ≠ novel" as the killer risk, solvable only as RHO-LOSS does it — with a held-out
reference model. Not needed:

**The goal is not flat NELBO across the corpus. It is a flat/consistent LEARNING GRADIENT.** An
unlearnable entry has a high NELBO whose value **does not shift**. So weight by the RATE OF CHANGE:

| state | NELBO | ΔNELBO | weight |
|---|---|---|---|
| not yet learnt | high | falling | **high** |
| unlearnable (noise/foreign/corrupt) | high | ~0 | **low** |
| mastered | low | ~0 | **low** |
| regressing (forgotten) | rising | rising | **high** |

Level cannot separate rows 1–2; velocity does, with no reference model — and gets **forgetting** for free.

## THE NORMALIZATION TRAP (user's catch — a real feedback pathology)

Velocity must be per unit of **APPLIED LEARNING**, never per visit or per step. Otherwise the controller
confounds its actuator with its measurement: down-weight → less effective learning → NELBO stops moving
→ **looks mastered** → down-weight further. An entry can be driven to zero weight purely by having been
given a low weight, with nothing in the signal revealing it.

```
velocity_i = -Δ NELBO_i / Δ (Σ effective_lr × tokens)_i
```

So the ledger tracks accumulated applied learning per entry — the same field the heat map wants.
Unsolved second-order: the model's global state also moves between visits, so velocity is never a clean
partial derivative.

## Other load-bearing points

* **A per-entry LR IS a per-entry LOSS WEIGHT** (gradient contribution scales linearly), so the existing
  per-window loss mask (`tests/loss_mask_tests.cpp`) already carries it — no optimizer surgery.
* **Corpus: SPLICED, not synthetic-only.** TinyStories (simple, saturates fast) + cosmopedia (complex,
  retains weight) + a deliberate unlearnable slice. Two of three populations are real text; the ordering
  prediction is known in advance, so it is falsifiable. The garbage slice is the direct test: a
  level-based rule up-weights it, a velocity-based rule must not.
* **The HEAT MAP should lead.** Read-only, changes no training behaviour, and useful even if Tutor is
  never adopted: it instruments the coverage question [[plateau-detector-calibrated-for-tinystories-era]]
  §4a raises and nothing currently measures — letting a plateau be SEEN (uniform coverage vs a stubborn
  slice) instead of inferred from one scalar. Also de-risks the rest: a feedback controller whose state
  you cannot visualise is very hard to debug. Per doc/window: visits, accumulated applied learning,
  NELBO + trend, current weight/effective LR, velocity.
* It structurally removes ICLR 2026 arXiv 2511.18903's finding (LR decay wastes late curriculum data) —
  that failure is *caused by* the LR being global.

Related: [[lr-schedule-wsd-cooldown-backlog]], [[deterministic-mechanisms-thesis]],
`sub0/blend_schedule.hpp` (deficit/weighted-fair BETWEEN sources — natural host for a revisit policy).
