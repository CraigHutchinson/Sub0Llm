---
name: cuda-parity-backlog-deferred
description: CLOSED 2026-07-08 -- all 4 CPU-only features now have CUDA parity (tied-embeddings 2026-07-04, QK-norm 2026-07-06, gated-FFN + Muon both 2026-07-08), all via phased Fable dispatch + independent review
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Originally 4 CPU-only features guarded by a `static_assert` in `backend_cuda.cu` blocking GPU use.
**Tied embeddings DONE (2026-07-04)** -- see [[weight-tying-tied-embeddings]]. **QK-norm DONE
(2026-07-06)** -- see [[qk-norm]] for the full CUDA writeup (3 kernels, constexpr-verified per-layer
offsets, ncu-profiled); both `static_assert`s are lifted, GPU is a fully valid backend for both now,
verified at production scale, committed `c05c6ff`. This memory now tracks the REMAINING two: gated-FFN/
SwiGLU (`!USE_GATED_FFN`, new SwiGLU forward+backward kernel needed, has a clear CPU reference in
`op_swiglu`), Muon optimizer (runtime-gated, no CUDA `newton_schulz5`/Muon step kernel, most novel
numerically of the two). User said "ensure our cuda versions are all in place soon too" -- explicit ask
to close this gap, not a someday-maybe.

**How tied-embeddings actually got done, as a template for the remaining three:** a PHASED Fable
dispatch (Phase 1 audit+plan only -> review/greenlight -> Phase 2 implement -> Phase 3 test/verify),
resumed via `SendMessage` after an initial session-limit failure mid-Phase-1. See
[[fable-subagent-for-big-risky-tasks]] for the full delegation writeup, including a real gotcha:
`isolation: "worktree"` did NOT survive the `SendMessage` resume -- Phase 2/3 ended up editing the main
working tree directly, requiring active coordination with concurrent unrelated work happening in the
same session. Worth planning for that up front next time (either avoid touching the same files during
the dispatch, or warn the agent proactively).

**Scoping notes for whoever picks this up** (from auditing `backend_cuda.cu` this session):
- The CUDA backend assumes FIXED, hardcoded `lm_head`/`lm_bias` offsets in multiple call sites across
  forward, forward-with-training-graph, backward, AND the `forward_one`/KV-cache decode path (e.g.
  `const float* lm_head = base + L[fi + 1].off;`-style fixed-offset reads) -- this is a bigger audit
  than the math itself. cuBLAS already supports a transposed GEMM operand (`CUBLAS_OP_T`) so the tied
  head's math is NOT the hard part; finding and updating every fixed-offset site is.
- Recommended order (least to most novel-kernel-work): **tied-embeddings first** (reuses existing GEMM
  machinery via `CUBLAS_OP_T`, no new kernel math, "just" layout/offset wiring -- lowest risk, good
  first Fable task to establish a reviewable pattern), then **gated-FFN** (new SwiGLU forward+backward
  kernel, has a clear CPU reference in `backend_cpu.cpp`'s `op_swiglu`), then **Muon** (new
  Newton-Schulz-on-GPU kernel, most novel numerically).
- Each needs: CPU-parity tests in the `cuda_tests.cpp` pattern at MULTIPLE dims (per
  [[agents-md-established]] rule #7 -- the gradient-check bug already proved production dims can behave
  differently from toy dims), the `static_assert` guard lifted only once parity is proven, must not
  regress the existing (untied/non-gated/non-Muon) CUDA suite, and per the user's own reminder use
  ncu (see [[ncu-vbs-memory-backup-blocker]] for the profiling gotcha) once functional to confirm it's
  not just correct but fast.
- Given the depth revealed (see [[fable-subagent-for-big-risky-tasks]]), this is a strong Fable
  candidate -- but brief it thoroughly and review the diff, per that memory's own guidance.

**Downstream dependency (2026-07-07):** the planned multi-size TinyStories Chinchilla-ratio
investigation (train d128/d192/d256/d320/d448 side by side, see [[checkpoint-tools-and-best-retention]])
is explicitly ON HOLD until gated-FFN + Muon land here -- the user wants that comparison to run under
the SAME optimizer/architecture options across all sizes, not re-run once more after this gap closes.

**Gated-FFN CUDA parity IN PROGRESS (2026-07-08), via Fable, phased dispatch per the template above.**
Phase 1 (audit+plan) complete and reviewed -- thorough, file:line-grounded, correctly identified real
hazards rather than a naive "just add a third GEMM" story:
- **Real hazard found**: `b1`/`b2` bias pointers are initialized UNCONDITIONALLY at ~7 call sites in
  `backend_cuda.cu`; under `USE_GATED_FFN` the offset arithmetic (`kB2 = kW1+3`) reads PAST the last
  layer's `PARAM_LAYOUT` entry on a combined gated+tied build -- silent runtime UB the compile-time
  `check_layer_offsets()` cross-check cannot catch. Every site must move behind
  `if constexpr (!USE_GATED_FFN)`. Required test: `--gated-ffn 1 --qk-norm 1 --tie-embeddings 1`
  combined-axes build, specifically to reproduce this.
- **Backward is NOT a naive 3rd-GEMM add**: plain FFN's backward checkpoint-RECOMPUTES `ff1/gact`
  (never saved) via 2 intermediates; gated needs to recompute THREE (gate_pre, up_pre, hswi) and reuses
  the plain path's `ff1/gact/dff1/dgact` scratch buffers in a REMAPPED role sequence -- flagged as the
  single highest-risk piece of the whole change (subtle-wrong-gradient risk, not a crash risk).
- GEMM shapes for `dWgate`/`dWup`/`dWdown` all reuse `launch_linear_bwd_t`'s EXISTING shapes (same
  pattern as tied-embeddings' backward reusing `op_linear`'s shapes) -- one needed primitive extension:
  a `beta`/accumulate parameter on `gemm_t` (default off, mirrors the already-tested FP32 `gemm()` beta
  path) so a second dX contribution can accumulate rather than overwrite.
- Offset-constants mechanism (`kQNorm`/`kKNorm`/etc. pattern from QK-norm) generically supports the new
  `Wg` slot with a small, mechanical extension to the existing named-constant block -- not new plumbing.
- `memplan.hpp` needs a `gated` bool on `Dims` (mirrors `tied`/`qk_norm`) threaded through FOUR
  independent `Dims`-construction sites (configurator, backend_cuda.cu, train_stage.cpp x2, cuda_tests.cpp)
  -- the audit flagged this is exactly the class of miss that caused a real bug last time (see
  [[dq-dk-register-spill-fix]]-adjacent memplan drift history).

Greenlit Phase 2 (implementation) with explicit guidance: prefer the gemm beta-extension over an extra
scratch buffer; treat the buffer-role-reuse backward as the highest-risk piece (verify via
gradient-check before declaring done, fall back to a clearer dedicated buffer rather than fight subtle
bugs indefinitely); `forward_one_device` and its CUDA-graph-captured twin
`forward_one_device_graphed` MUST get identical branches, verified via the (normally `[.slow]`-hidden)
decode parity test run explicitly; the combined-axes test is REQUIRED not optional; standard bar is
tiny-dims full gradient-check + production d448/HD64 isolated `[cuda]`-tag run + DH=96 spot-check +
byte-identical default-build assertion counts before/after + an ncu pass once functionally green.

**DONE (2026-07-08, `ca4649c`).** Real hiccup mid-Phase-2: the resumed agent, after the greenlight
message, described "dispatching to a Fable subagent to implement" instead of implementing directly
itself -- it misread background context (explaining the project's general Fable-delegation practice)
as an instruction to delegate FURTHER, when it WAS already the delegate with no one else to hand the
work to. Caught immediately via `git status`/`git diff --stat` showing zero changes despite the "done"
narrative -- corrected with an explicit "there is no subagent, YOU are Fable, write the code yourself"
message; see [[fable-subagent-for-big-risky-tasks]] for the full writeup of this new failure mode. Also
hit (and recovered from, via a plain resume once past the reset time) a mid-Phase-2/3 API session-limit
cutoff.

**My own independent review, not just trusting the agent's report** (matching the tied-embeddings/
QK-norm precedent): hand-verified the offset-constant arithmetic (`kWg`/`kW1`/`kW2`/`kB1`/`kB2`/`kLPer`)
against `layout.hpp`'s real slot layout -- confirmed `kB1`/`kB2` DO alias into `kW2`/past-the-block on a
gated build exactly as flagged, and confirmed every read site (7 total) is properly `if constexpr`
-guarded, not just claimed. Hand-traced the backward buffer-role-reuse sequence step by step against
the plan -- correct GEMM shapes, correct overwrite-then-accumulate stream ordering, correct in-place
aliasing (read-before-write). Verified the new SwiGLU kernels' math character-for-character against
`op_swiglu`/`Op::SwiGLU` in `backend_cpu.cpp` (silu/dsilu forms, store-not-accumulate convention
confirmed to match the existing `gelu_backward_act_kernel`, not a divergence). Verified `gemm_t`'s new
`beta` param truly defaults to `0.0f` (zero behavior change for every pre-existing call site). Then
LIVE-reproduced two of the reported test results myself from scratch (not just re-reading the agent's
transcript): tiny d32/L2/H4 gated build -- exact match, 1,109,239 assertions/29 cases/0 fail (had to
first resolve an apparent discrepancy: a bare `[cuda]` filter also pulls in 2 pre-existing `.bench`
-tagged perf tests with a hardcoded T=128/64 that exceeds this tiny build's SEQ_LEN=32 -- confirmed
dims-only and unrelated to gated-FFN by excluding `~[.bench]`, which reproduced the agent's exact
number). Combined gated+qk_norm+tied build (the direct reproduction case for the b1/b2 hazard) --
exact match, 564,218 assertions/29 cases/0 fail. Did not independently rebuild the production-scale
result given CUDA compile time cost, but the math/offset logic verified is scale-independent (no
per-dims branches), and both independent live reproductions landed exactly on the agent's claimed
numbers -- high confidence the production number is genuine too.

Muon is now the ONLY remaining CUDA-parity gap.

**Muon CUDA parity DONE too (2026-07-08, `40f96a1`) -- backlog fully CLOSED.** See
[[muon-cuda-parity-in-progress]] for the full writeup: first-principles precision strategy (double
norm reduction only, `CUBLAS_COMPUTE_32F_PEDANTIC` GEMMs, no FP64), 2 real hazards found and designed
around (TF32 leaking via the shared cuBLAS handle; naive grad-zeroing would have corrupted Muon's
momentum), 1 real bug caught by the agent's own testing (a Muon-skip gate that was unconditional on
kind alone, silently breaking plain AdamW-only GPU runs), and a fully independent review including a
from-scratch hand-derivation of the no-transpose cuBLAS op-flag algebra (the riskiest, most novel piece
of the whole 4-feature backlog) cross-checked against two already-proven GEMM patterns in the same
file, plus a live from-scratch rebuild+test run reproducing the reported numbers exactly.

**All 4 items done: tied-embeddings, QK-norm, gated-FFN, Muon.** The multi-size TinyStories
Chinchilla-ratio investigation's blocker (see [[checkpoint-tools-and-best-retention]]) is now CLEARED.
