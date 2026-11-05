---
name: fixed-point-hypothesis-falsified
description: "RESULT 2026-07-29 — measured on arm B: pass 2 of the looped middle block contributes 31.5% MORE than pass 1. The passes are NOT converging to a fixed point, which removes depth attention's specific motivation"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T21:35:43.916Z
---

Measured before building arm D, for about two minutes of compute. `sub0llm report` now prints this for
any looped build (commit `1dd0b9e`).

## The claim under test

[[depth-attention-rescues-loopsplit-hypothesis]] argued: a looped block applies the SAME function
repeatedly, so repeated application of a contractive map converges to a fixed point -- later passes
perturb the residual stream less and less and the extra compute buys nothing. Depth attention was
scoped specifically to break that symmetry.

## The measurement

Arm B (d192, L10, middle 6 x2 = 16 executions), averaged over 8 held-out windows. Middle block is
layers 2-7; pass 1 is executions 2-7, pass 2 is executions 8-13.

```
    exec  layer        |dh|    |dh|/|h|
       2      2    1477.738      0.5544     <- pass 1
       7      7    1981.567      0.3212
       8      2    1826.766      0.2810     <- pass 2
      13      7    2529.422      0.2587
    middle-block contribution, pass 2 / pass 1: 1.315
```

**Pass 2 contributes 31.5% MORE in absolute norm than pass 1.** The repeats are doing real, comparable
work. The relative measure `|dh|/|h|` does decline (~0.31 -> ~0.27, a ratio of ~0.87), but that is
ordinary residual-stream norm growth, not convergence -- nowhere near inert.

## Consequence: arm D's motivation is materially weakened

Depth attention was not "another Nanbeige feature"; it was scoped as a targeted fix for THIS mechanism.
The mechanism does not appear to exist, so the specific argument for building it is gone. Reconsider
before spending Stage 1 + Stage 2 (the CUDA cross-execution dK/dV accumulation is days of work and the
most error-prone piece in the plan).

**The likelier explanation for B trailing A is mundane**: B has 7.23M parameters against A's 9.66M.
Extra executions add compute, not capacity. B beating C by 0.0160 at EQUAL parameters
([[loopsplit-3arm-clean-result-looping-works]]) says compute genuinely does help -- it just converts to
quality at a poor exchange rate. Weight sharing also forces one function to serve two roles at
different depths, which is a capacity constraint rather than a convergence one.

## What is NOT falsified, and the cheap next probe

That the passes do **redundant** work. `||dh||` is a magnitude only: two passes could contribute
equal-norm but near-PARALLEL updates, which would still waste the compute while looking healthy here.
**Cosine similarity between pass-1 and pass-2 contributions, per layer, is the direct test** -- same
hook, same cost, minutes not days. Run that before any decision on arm D.

## Method note worth keeping

The hook instruments the REAL forward rather than a reimplementation of the layer loop. A separate
diagnostic copy would drift from the code that actually trains and then measure the wrong thing
convincingly -- the same reasoning as [[verify-correctness-against-reference-before-perf]].

Also incidentally confirmed: arm B's model.bin loaded fine into a build reconfigured after
`DEPTH_ATTN_STRIDE` entered `ARCH_FINGERPRINT`, which exercises the bit-identity-at-stride-0 claim
([[depth-attention-stage0-done]]) on a real checkpoint rather than only in a unit test.
