---
name: overlap-memcpy-compute-standing-reminder
description: Standing reminder (2026-07-04) -- keep watching for memory-copy/compute overlap opportunities across the FULL pipeline, not just the CUDA training step's kernel sequence
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's explicit reminder (2026-07-04): "remember to keep eye out for optimization opportunities around
overlapping memory copy and computation for the full pipeline."

**How to apply:** an ongoing lens to hold while doing CUDA/performance work, not a one-shot task. Watch
for H2D/D2H transfers that could overlap with kernel execution (multi-stream, async copies,
double-buffering) ANYWHERE in the pipeline -- training step, generation/decode loop, checkpoint I/O,
data loading -- not just the training step's own kernel sequence (which prior investigation already
covered).

**What's already established (timeline-level only -- see the important caveat below):**
- [[training-step-nsys-occupancy-findings]]: a real nsys trace of the actual production training step
  showed the GPU is 93%+ busy overall, 99%+ in steady state once warmed up -- ZERO idle gaps >1ms after
  the first 1.15s of a captured window. The "host marshals ids/targets while GPU sits idle" hypothesis
  (a real structural gap in the code) was REFUTED in practice -- kernel launch queuing already hides it.
- [[cuda-wave6-orchestration-memory-findings]]: a separate architectural review explicitly concluded
  stream overlap / multi-stream training-step orchestration is "NOT worth another wave" -- bounded to
  ≤7% by the same nsys finding.
**CAVEAT (user correction, see [[ncu-depth-vs-nsys-busy-time]]):** the above is a TIMELINE finding only
-- "the GPU is busy" is not the same as "the GPU is being used efficiently." An nsys busy-percentage
says nothing about per-kernel occupancy, stall reasons, or achieved-vs-theoretical throughput -- that
earlier memory's own "STILL PENDING: no ncu deep-dive done yet" caveat is the part that actually matters
here and was never done. Don't treat "little slack for overlap" as settled without pairing it with an
actual ncu per-kernel efficiency pass.

**Open ground, resolved (2026-07-07, via Fable's hardware-efficiency Phase 1 audit):**
- Decode loop: my original framing ("overlap the D2H logits / H2D next-token-id round-trip") was WRONG
  -- there is no per-token H2D at all. `id`/`pos` are passed as plain kernel-launch scalar arguments into
  `embed_one_kernel`, never via `cudaMemcpy` (independently confirmed via grep, not just trusted). The
  REAL bottleneck is launch-count: ~147 kernel launches per decode token, `cudaLaunchKernel` alone is
  68.6% of decode wall-time, GPU only 31.16% busy during decode. Lever = CUDA-graph capture-once +
  per-node parameter patching for the id/pos-dependent nodes, reusing this file's existing
  `capture_graph`/`cudaGraphLaunch` pattern already used for batched forward. Greenlit for Phase 2
  implementation, see [[hardware-efficiency-iteration-standing-task]].
- Checkpoint save vs. GPU compute: confirmed BLOCKING, not overlapped (`train_stage.cpp`'s main loop has
  zero `std::thread`/`std::async`, `gpu.sync_to_host()` -> eval -> `save_checkpoint()` -> `save_model()`
  all serialize on one thread with no GPU work in between). Measured ~3.1s of pure disk I/O per boundary
  (real 497MB .ckpt + ~158MB model.bin at ~204MB/s on this disk) plus an unmeasured CPU-only eval cost.
  Real but small given `eval_every` fires only every 1000-8000 steps -- LOW priority, deferred.
- B1 chunk-to-chunk pipelining: DEAD END, confirmed via ncu -- `ce_backward_kernel` already runs at
  98.7% occupancy / 88.5% memory throughput, the associated head-GEMMs at 88-96.5% SM throughput. No
  idle bubble exists for a second stream to fill; multi-stream overlap would only contend for
  already-saturated SMs. Explicitly not worth pursuing -- don't revisit without new evidence.
