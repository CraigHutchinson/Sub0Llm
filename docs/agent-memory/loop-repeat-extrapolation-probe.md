---
name: loop-repeat-extrapolation-probe
description: "RUN 2026-07-29 — RESULT: quality peaks EXACTLY at the trained repeat count and degrades both ways (1: 2.9234 | 2: 2.2140 | 3: 2.4202 | 4: 2.8634). The loop learned a depth-specific pipeline, not an iterative operator"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T21:42:34.194Z
---

## RESULT (run 2026-07-29, commit df91d02) -- READ THIS FIRST

Arm B, trained at `LOOP_REPEATS = 2`, weights held FIXED:

| repeats | executions | val nelbo | ctx gain |
|---------|-----------|-----------|----------|
| 1 | 10 | **2.9234** | 9.0% |
| 2 | 16 | **2.2140** | 13.6%  <- as trained |
| 3 | 22 | **2.4202** | 11.8% |
| 4 | 28 | **2.8634** | 8.9% |

**Quality peaks EXACTLY at the trained count and degrades in BOTH directions.**

1. **The loop is load-bearing.** B@1 (2.9234) is far WORSE than arm C's 2.2300 -- a model actually
   trained flat at the same 10 executions and same 7.23M params. So looped training does NOT produce
   weights that stand alone. **"Train looped, deploy flat" is dead** -- that was the most attractive
   of the four predicted outcomes and it is falsified.
2. **The middle block learned a DEPTH-SPECIFIC PIPELINE, not an iterative operator.** An iterative
   operator would hold or improve at 3-4. Instead the per-pass diagnostic reports pass3/pass1 = 1.375
   and pass4/pass1 = 1.398: the extra passes contribute LARGE-norm updates that make the model WORSE.
   The same function applied at the wrong depth is actively harmful.
3. Context gain simply tracks nelbo here (13.6% at peak, 9-12% off-peak), so it is not an independent
   signal in this probe.

**This closes the arm-D question from a second, independent direction.** Depth attention was scoped to
break a symmetry between near-identical passes ([[depth-attention-rescues-loopsplit-hypothesis]]). The
passes turn out to be so differentiated that they are not interchangeable at all. Building depth
attention on that rationale would be solving a problem that does not exist. See also
[[fixed-point-hypothesis-FALSIFIED]].

**Open question this raises**: if the two passes do genuinely different depth-specific work, LoopSplit
is functionally "two layers that share weights" -- so the interesting comparison is no longer against
depth attention but against other weight-sharing schemes, and against simply spending the parameters.

---

## Original design notes

User's proposal 2026-07-29: arm B trained at `LOOP_REPEATS = 2`; run the SAME weights at 3, 4, ...
and re-measure. This is the standard evaluation for recurrent-depth architectures (Universal
Transformers and successors) and it is the sharpest available probe of what the loop actually learned.

## Why it is mechanically free

**LoopSplit adds no parameters.** The middle block reuses the same weights, and `PARAM_LAYOUT` is keyed
by layer INDEX, not by execution. So `PARAM_FLOATS` is IDENTICAL at repeats 2, 3, 4 -- arm B's
`model.bin` is byte-compatible with a repeats=3 build. `LOOP_EXEC_COUNT = N_LAYERS +
MIDDLE*(REPEATS-1)`, so 10 + 6*2 = 22 executions at repeats=3, all served by the same 7.23M params.

`LOOP_REPEATS` is `constexpr` and `LAYER_EXEC_ORDER` is `consteval`, so each repeat count needs its own
build dir -- not a runtime flag. That is cheap (configure reuses corpus.tok via the reuse stamp).

## The one obstacle, and the right fix

`ARCH_FINGERPRINT` packs `LOOP_REPEATS` specifically to REFUSE this cross-load, because `nfloat` cannot
catch it (a looped and un-looped build have byte-identical blobs -- that guard exists because such a
mismatch once went unnoticed). It will reject the experiment.

**Do NOT add a blanket bypass.** `layout.hpp`'s own rule for compile-time axes already has the right
category: *"3. Deliberately variable between train and inference -> EXCLUDE, and record why"*, whose
only current members are ROPE_SCALING / ROPE_SCALE_FACTOR ("training short and running long WITH
scaling on is the intended workflow"). Train-at-2/run-at-3 is exactly that shape.

The targeted change: **training resume and inference already load through different paths.**
- `load_checkpoint` (train resume, train_stage.cpp) -- KEEP the hard refusal. A repeats mismatch there
  is always a bug.
- `load_model` (gen / report / eval, engine_core.cpp ~line 165) -- ALLOW a repeats-ONLY mismatch with a
  loud warning naming both counts. Every other fingerprint axis stays a hard refusal.

That preserves the guard where it protects correctness and opens it only where the mismatch is the
deliberate subject of the measurement.

## repeats = 1 is the LEFT ANCHOR, and the most informative single point (user, 2026-07-29)

Run B's weights with the loop REMOVED. At `LOOP_REPEATS = 1`, `LOOP_EXEC_COUNT = 10 + 6*0 = 10` and
`LAYER_EXEC_ORDER` is just 0..9 -- arm B's weights running as a plain 10-layer network, which is
**structurally identical to arm C**. Same 7.23M parameters, same 10 executions. The ONLY difference
between B@1 and C is that B's weights were TRAINED under a looped regime.

That gives a decomposition nothing else can:

| comparison | what it isolates |
|---|---|
| B@2 (2.2140) vs **B@1** | what the second pass buys AT INFERENCE, on FIXED weights -- a clean ablation of the mechanism, with no confound from different weights or training |
| **B@1** vs C (2.2300) | whether looped TRAINING produces better weights even when run un-looped, i.e. LoopSplit as a training scheme rather than an inference structure |

Outcomes and what each would mean:

- **B@1 ~= B@2** -> the second pass contributes norm (31.5% MORE than pass 1, per
  [[fixed-point-hypothesis-FALSIFIED]]) but no QUALITY. The work is real and useless -- a striking
  result, and it would kill looping at this scale outright.
- **B@1 much worse than B@2** -> the model genuinely depends on the second pass; the loop is load-bearing.
- **B@1 ~= C (2.2300)** -> looped training buys nothing for the weights themselves; all of B's
  advantage over C is the extra inference pass.
- **B@1 BETTER than C** -> LoopSplit is valuable as a TRAINING method independent of inference-time
  looping (a regularizer / implicit depth-augmentation). That would be a surprising and independently
  useful finding -- train looped, deploy flat, keep the gain at C's inference cost.

**Configure it as `--loop-middle-layers 6 --loop-repeats 1`, NOT as middle 0 / repeats 1.** Both give a
plain 10-layer execution order, but only the first leaves `middle_layers` unchanged in the fingerprint,
so the mismatch is repeats-ONLY and the narrow `load_model` exemption above covers it. Setting middle=0
too would differ in two fields and fall outside the exemption.

## What each outcome would MEAN across the full sweep

Run `report` at repeats 1, 2 (baseline, as trained), 3, 4 and read val_nelbo + the context-length curve
+ the per-pass diagnostic:

- **Nelbo improves or holds at 3-4** -> the middle block learned a genuinely ITERATIVE operator that
  converges. Strong support for the thinking-loop frame
  ([[loop-as-thinking-loop-needs-writable-context]]), and it would mean free quality at inference time
  by simply looping more -- a substantial practical win.
- **Nelbo degrades sharply at 3** -> the block learned a DEPTH-SPECIFIC pipeline: its two executions do
  different, position-in-the-stack-specific work rather than iterating toward a solution. Weight
  sharing is being used as "two different layers that happen to share weights".

**The second is my prediction**, because it is what [[fixed-point-hypothesis-FALSIFIED]] already shows:
pass 2 contributes 31.5% MORE than pass 1, which is the signature of differentiated work, not of
iteration toward a fixed point. If that is right it also explains why depth attention is unnecessary --
the passes are ALREADY differentiated, so there is no symmetry left to break.

Either answer is decisive for arm D, and it costs one configure + build + report per repeat count.

## Caveat to check, not predict

The stream norm grows with depth (`|dh|/|h|` data in [[fixed-point-hypothesis-FALSIFIED]]), so more
passes push a larger-norm residual into `ln_f`. RMSNorm renormalizes, so it may be fine -- but if nelbo
degrades, distinguish "the loop does not extrapolate" from "the final norm/head saw an out-of-
distribution scale" before concluding. The per-execution `|h|` column the diagnostic already prints
separates them.

## Follow-up if it DOES extrapolate

Then adaptive halting becomes the obvious next question (Universal Transformers' ACT): let the model
stop looping when it is done, rather than at a baked count. That would be a genuinely new capability
for this project, not a port.
