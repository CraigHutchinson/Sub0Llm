---
name: ncu-depth-vs-nsys-busy-time
description: User correction (2026-07-04) -- nsys "GPU busy %" is a timeline/occupancy-gap signal, not an efficiency signal; ncu's per-kernel depth metrics are what show whether the hardware is actually being used well
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's correction (2026-07-04): "remember ncu shows real cuda performance - 100% gpu usage does not
mean the gpu is being used effectively etc - always run a multi layer profiling on the performance to
ensure the data covers all the relevant aspects."

**What I got wrong:** cited [[training-step-nsys-occupancy-findings]]'s "93%+ GPU busy, near-zero idle
gaps" finding as if it settled whether the training step's kernels are running efficiently, and
concluded "little slack" for further optimization on that basis. That finding only answers "is the GPU
doing SOMETHING" (a timeline/occupancy-gap question, nsys's job) -- it says nothing about whether each
kernel is using the SM's compute/memory resources well (occupancy limiters, warp stall reasons, achieved
vs theoretical throughput, register/shared-memory pressure) once it IS running. A kernel can be
memory-latency-bound, register-spilled, or badly parallelism-shaped and still show up as "the GPU was
busy" in a timeline trace -- busy is not the same as effective. That earlier memory actually already
flagged this gap correctly ("STILL PENDING: no ncu deep-dive done yet on rmsnorm_backward/attn_bwd_dk/
the FP32 lm-head GEMM... that's the natural next step to turn 'these three are unusually expensive'
into 'here's the specific fix'") -- I just glossed over that caveat when I summarized it.

**How to apply, standing:** treat nsys (timeline: gaps, overlap, launch structure) and ncu (per-kernel
depth: occupancy, stall reasons, memory/compute throughput vs roofline) as two DIFFERENT, both-required
layers, not substitutes for each other. Before declaring a GPU code path "already checked" or
"optimized," confirm both layers were actually run -- a wall-clock throughput number or an nsys
busy-percentage alone is not sufficient evidence, no matter how good it looks. This applies to:
- Any future performance claim I make about CUDA code in this project (mine or a delegated agent's).
- Verification gates for CUDA work handed to Fable -- Phase 3-style verification asks should include an
  actual ncu profile of the new/changed kernels, not just "full test suite green + throughput held," per
  [[fable-subagent-for-big-risky-tasks]]'s existing ncu reminder (that one was framed as "for new
  kernels specifically"; this one is the broader "don't trust busy-time alone" principle behind it).

**Concrete open item this reframes -- DONE (2026-07-07, via Fable's hardware-efficiency Phase 1 audit):**
[[training-step-nsys-occupancy-findings]]'s own "STILL PENDING" ncu deep-dive on `rmsnorm_backward_act_kernel`,
`attn_bwd_dk_kernel`, and the FP32 lm-head GEMM is complete. Important correction found in the process:
that old 11.0%/10.4%/6.8% breakdown was captured on the SINCE-ABANDONED d768/L16/H8 architecture (see
[[production-line-pivot-medium-model]]), not the current d448/L11/H7 production line -- the percentages
don't transfer. Fresh numbers at the ACTUAL current production dims: lm-head GEMM fix (`force_tc`)
confirmed still landed, zero FP32 fallback kernels anywhere in the trace, 88-96.5% SM throughput --
nothing left there. `rmsnorm_backward_act_kernel`: 95.67% occupancy (not occupancy-limited), 92-94% of
stall cycles from L1TEX long-scoreboard = atomicAdd fan-in contention on `dgamma[1,D_MODEL]` (same root
cause as [[qk-norm]]'s already-documented qknorm_backward atomic issue -- a shared fix is now proposed,
see [[hardware-efficiency-iteration-standing-task]]). `attn_bwd_dk_kernel`: zero register spill confirmed
(closes the "NOT yet confirmed at production dims" flag in [[dq-dk-register-spill-fix]]), but 170
regs/thread caps occupancy at 8.33% theoretical -- register pressure is now attention's own limiting
factor, flagged as a separate higher-risk redesign candidate, not bundled into the atomic fix.
Attention (fwd+bwd combined) is now the single largest cost center at 37.1% of a training step, not
rmsnorm -- a materially different picture than the old d768 numbers suggested.
