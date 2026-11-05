---
name: independent-reimplementation-catches-identity-swap-bugs
description: "A real dt_bias/a_log argument-swap bug sat in already-merged, already-\"verified\" Gated DeltaNet CPU code (Stage 1+2) invisible to fixture tests, mutation tests, parity checks, AND a whole-model finite-difference gradient check -- only caught when an independent, from-scratch CUDA reimplementation (Stage 3) disagreed with the CPU engine"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-02T08:39:38.980Z
---

**What happened** (full detail: [[pivot-to-existing-model-support-qwen4-preview]]'s Stage 3 entry,
2026-09-02): `backend_cpu.cpp`'s `op_gdn` passed two structurally-similar, same-shaped parameter tensors
(`dt_bias`, `a_log`) in swapped positions relative to `gdn::forward()`'s real declared signature, while
`gdn::backward()`'s own call site already had them correct — a genuine forward/backward mismatch that
had already passed Stage 1's fixture-match review, Stage 1's mutation/presence tests, Stage 2's
real-PyTorch-autograd-oracle gradient check, AND my own independent re-verification of all of the above.
It was only found because Stage 3's CUDA implementation — built independently from the real reference
math, not derived from or sharing code with the buggy CPU glue — disagreed with the CPU engine's actual
output at a real mixed-layer model.

**Why every existing check missed it, specifically**:
- The real-weight fixture test and the mutation tests call `gdn::forward()`/`recurrence_step()` directly
  with their own correctly-ordered local variables, never through `op_gdn` — so they validate the core
  math in isolation but never validate that the engine's own glue code (the part that maps trained
  `Layer` parameters onto that math's argument list) wires it up correctly.
- The `forward_one`-vs-`forward` parity check stayed consistent because BOTH call sites had the identical
  swap — a differential check between two paths cannot catch an error common to both paths.
- The whole-model finite-difference gradient check compares a NAMED tensor's numeric and analytic
  gradients, which stays self-consistent under a symmetric relabeling of "which tensor plays which role"
  — and a whole-vector aggregate (directional cosine, norm ratio) is additionally just less sensitive to
  an error localized to 1-2 of many parameter tensors than a per-tensor isolated check would be.

**Why**: verification methods that are all downstream of, or self-referential within, the same
implementation share its blind spots. A fixture match, a real-oracle gradient check, and a parity check
are each individually strong (this project has good reason to trust all three, generally) — but none of
them independently re-derives "does the ENGINE's actual glue code hand the right buffer to the right
argument," because they were each designed to check a different, narrower thing.

**How to apply**: when a new op has been "verified" by fixture/gradient/parity checks alone, and a
genuinely independent reimplementation becomes available for any reason (a second backend, a second
scale, a rewrite) — treat a disagreement as a serious signal worth chasing to ground truth, not a bug
in the new code by default. More proactively: for any op with multiple structurally-similar,
same-shaped parameters (two gate vectors, two projection matrices of identical shape, etc.), add a check
that specifically distinguishes their IDENTITY, not just their gradient magnitude — e.g. perturb each
one with a distinguishable, asymmetric signal and confirm the measured effect matches which ROLE that
specific tensor is supposed to play, not just that some plausible-looking gradient exists. A per-tensor
isolated FD check is also strictly more diagnostic here than a whole-model aggregate one, for the same
reason. Related: [[depth-attention-stage0-done]]'s "a gradient check alone cannot validate this op"
finding — this is the same family of lesson, extended one level further: even a REAL-weight fixture
match and a REAL-oracle gradient check can both be blind to a bug in the glue code that feeds them.
