---
name: engine-tests-gradient-check-dims-dependent
description: "OBSOLETE / RESOLVED 2026-07-27. The old claim -- engine_tests' gradient check fails at production dims (d448 L11 H7) and passes at tiny dims -- is NO LONGER TRUE and is now inverted: d448 passes exactly, and the only config that misbehaves is a TOY-VOCAB spike build. Kept (not deleted) because several memories link here; read the correction before citing it."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-27T19:35:15.639Z
---

**Do not cite the old claim. It was re-measured on 2026-07-27 and is false at current HEAD.**

## The claim that was here (2026-07-03/04, still cited 2026-07-21)

`engine_tests.cpp`'s "analytic gradients match finite differences" was recorded as failing
DETERMINISTICALLY at production dims (d448 L11 H7) with a ~12% gap (`directional=9.566` vs
`gnorm=10.843`, byte-identical across outer seeds), while passing cleanly on a tiny d32 L2 H4 seq32
vocab756 build. That was read as genuine dims-dependence, left with "root cause not investigated", and
cited by `AGENTS.md` 7 as the motivating example for multi-scale testing.

## What is actually true now (measured, not inferred)

| config | vocab | ratio numeric/\|\|g\|\| | result |
|---|---|---|---|
| d448 L11 H7 (the config claimed to FAIL) | 16535 | ~1.0 | **PASSES** |
| d196 L11 H7 | 16535 | 0.999941 | PASSES |
| d96 L8 H2 | 16535 | 0.999968 | PASSES |
| d96 L8 H2 (`factspike96`) | **493** | **0.96199** | marginal; fails outright under other probe shapes |

d448 L11 H7 at a real vocabulary passes the gradient check, and passes the **full** suite in that
configuration (176,570,618 assertions / 127 cases) -- so it is not test-order or optimizer-state
contamination either. The direction of the old claim is exactly inverted: large dims are fine, and the
only config that misbehaves is the SMALL one, because of its 493-token toy vocabulary rather than its size.

## What changed

Not identified, and not worth bisecting. A great deal landed between 2026-07-04 and now that could
plausibly have fixed a real large-dims gradient error -- the RoPE convention unification
([[rope-convention-mismatch-forward-vs-forward-one]]), the FTZ/denormal fix
([[denormal-stalls-in-training-need-ftz]]), QK-norm, tied embeddings, gated FFN. The actionable fact is
that the failure no longer reproduces at the exact config it was recorded against.

## The lesson that replaces it

The original observation paired "d448, real vocab" against "d32, vocab 756" and blamed DIMS. Vocabulary
differed too, and vocabulary is what actually moves this check -- see [[nanbeige-review-findings]] for
the one-variable discriminator. Two variables changed at once, the wrong one got the blame, and the
claim then sat unexamined in `AGENTS.md` for three weeks while several memories cited it as settled.

**A toy vocabulary is not a valid fast scale for numerical gradient work.** Use d96 L8 H2 at vocab 16535
(d196check's corpus with dims overridden) -- same speed, no artefact. `factspike96` (vocab 493) should
not gate this test.

The multi-scale RULE itself stands; only this example was wrong. A genuine two-scale catch from the
Nanbeige work makes the point better: LoopSplit's head/tail symmetry `static_assert` rejected every
ODD-layer model even with the feature OFF, visible only because factspike96 has 8 layers and d196check
has 11.
