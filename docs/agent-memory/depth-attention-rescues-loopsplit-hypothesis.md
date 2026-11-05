---
name: depth-attention-rescues-loopsplit-hypothesis
description: "HYPOTHESIS — depth attention may be what makes LoopSplit work, by breaking the fixed-point symmetry between identical loop passes; build it only after the 3-arm sweep says which outcome we are in"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T08:16:40.628Z
---

Raised by the user 2026-07-28 while the 3-arm LoopSplit sweep was running, and it reframes depth
attention from "another Nanbeige feature, unclear benefit" into a targeted fix for a specific,
predicted failure mode.

## The argument

**A looped block applies the SAME function repeatedly, so its passes differ only by their input.**
Repeated application of a contractive map converges to a fixed point: later passes perturb the
residual stream less and less, and the extra compute buys nothing. That is the most plausible reason
looping underperforms, and it fits the shape of the earlier negative
([[loopsplit-ab-negative-result]]).

**Depth attention breaks that symmetry.** It caches each participating layer's (K,V) across DEPTH
(not sequence position) and lets later layers softmax-attend over the accumulated set. Applied to a
loop, pass 3 is then no longer the same function as pass 1 -- it has strictly more context. The loop
gains *iteration memory* rather than only a mutated residual stream.

**It targets a failure this project already measured.** Phase G/H found that injecting a
representation ONCE and relying on ordinary computation to carry it forward loses fidelity versus
real per-layer computation ([[factspike-piece-transfer-validated]]). A looped block depends on
exactly that -- the residual stream as the sole carrier between passes. Depth attention gives passes
a direct channel instead. Same insight as the per-layer gated re-injection result
([[nanbeige-architecture-reference]] §1), reached from a different direction.

**Corroboration**: Nanbeige ships the two TOGETHER -- the architecture reference has them in one
section, "LoopSplit + depth attention", not as independent features. Worth re-reading
`modeling_nanbeige.py`'s `_apply_depth_attention` / `_depth_attention_mix_value` against this
hypothesis specifically, to see whether their depth cache participates in the looped block or only
in the un-looped head/tail. **That detail decides whether this hypothesis is what they built or a
different thing that merely resembles it.**

## Do NOT build it yet -- the running sweep decides whether it is worth building

The 3-arm sweep asks "does looping alone recover capacity?" Its answer changes what you would do:

- **B ~= A** -- looping already works. Depth attention is optional polish, and adding it would
  confound an otherwise clean result.
- **B ~= C** -- looping does nothing, AND we would have a mechanistic reason why (indistinguishable
  passes) plus a specific remedy. Far stronger ground than "Nanbeige has it".
- **B between** -- depth attention becomes the obvious candidate to close the gap.

Two of the three outcomes change the decision, so build after, not during. Adding it now would also
make the loop arm test two mechanisms at once -- exactly the confound this experiment was
restructured to remove.

## Cost, unchanged

Hand-written forward AND backward on CPU and CUDA, plus a cross-layer K/V cache that grows activation
memory -- and at 16 executions that cache is not small. Same four-implementation burden as LoopSplit
itself. Spend it once the sweep says which outcome we are in.

Sweep state at capture: arm A plateaued at 1.80 epochs, best val_nelbo 2.1332; arm B running.
