---
name: training-step-nsys-occupancy-findings
description: nsys trace of the real fineweb_edu training step (d768/L16/H8) shows GPU is ~93% busy overall, 99%+ in steady state -- the host/device ping-pong hypothesis was REFUTED; real lever is kernel-level (rmsnorm backward + attn dk + fp32 lm-head gemm)
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Question (2026-07-02, user):** during the real fineweb_edu (d768/L16/H8, batch 30, seq up to 512)
training run, is there a GPU idle window per step worth fixing via overlap/async/orchestration? Code
reading first (before any trace) found a real structural gap worth checking: `GpuTrainer::step()`
(train_stage.cpp ~587-610) builds the `ids`/`targets` host arrays in a single-threaded scalar loop
*then* calls `sub0_cuda_train_step`, which blocks on `cudaStreamSynchronize` (backend_cuda.cu:2065 and
again :2100) before returning -- no double-buffering, so in principle every step is
marshal-batch-on-host (GPU idle) -> block-on-GPU (host idle) -> repeat.

**Nsight tooling notes (Windows, this host):** both `nsys` (Nsight Systems, timeline tracer) and `ncu`
(Nsight Compute, per-kernel profiler) ARE installed but NOT on PATH -- `nsys.exe` is at
`C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.6.3\target-windows-x64\nsys.exe`, `ncu.bat` at
`...\Nsight Compute 2026.1.1\ncu.bat`. On this Windows build, `--trace=osrt` is NOT a valid value
(unlike the Linux docs) -- use `--trace=cuda` (add `,wddm` only if running elevated; wddm/CPU-sampling/
context-switch tracing all silently disable without admin rights, no hard failure). `nsys stats
--report cuda_gpu_trace --format csv` exports a proper per-event Start/Duration CSV, from which GPU
idle gaps can be computed directly (merge overlapping intervals, look at the gaps) -- there is no
built-in "gpu idle" report, but this is a five-line computation over the CSV.

**CRITICAL gotcha that cost real time:** launching `nsys profile ... > file 2>&1` via `run_in_background:
true` reliably left an ORPHANED `sub0llm-train.exe` running after the wrapping Bash tool call's own
timeout killed `nsys.exe` -- nsys itself died but its child kept running, undetected (repeated
`tasklist` checks showed nothing running moments later, presumably a process-listing/timing artifact,
compounding the confusion). This happened twice and both times raced a second orphaned instance against
whatever ran next, corrupting nothing only by luck (no eval/checkpoint boundary was reached in the
short window). **Fix: always run `nsys profile` in the FOREGROUND (not `run_in_background`), piped
through `| tee` (not `> file 2>&1`), with an explicit `--duration=N` so nsys itself terminates the
target cleanly** -- this is the pattern that worked reliably (confirmed twice). Also ALWAYS verify
`tasklist` shows zero matching processes immediately before relaunching anything against the same
output files -- this session's earlier near-miss ([[eos-token-and-fixed-marker-scheme]]) was the exact
same class of mistake (impatient retry -> concurrent duplicate processes).

**Finding: the host/device ping-pong hypothesis was REFUTED by the trace.** Captured a 25s window
(`nsys profile --trace=cuda --duration=25`) resuming the real checkpoint (step 3834). GPU activity
only starts ~17.4s into the 25s window (CPU-only startup: checkpoint load + the full-corpus.tok
validation scan -- see [[out-of-core-corpus-pipeline]]/train_stage.cpp's `for (i<data.size())` OOB
check, which also happens to pre-warm the whole file into the OS page cache). Within the ~7.6s of
actual GPU activity: **93.4% GPU busy, 6.6% idle** -- and of the idle time, ALL 5 gaps >10ms are
clustered in the FIRST 1.15 seconds (each immediately followed by a first-use kernel/GEMM-variant
lazy-load: `cuLibraryLoadData`/CUTLASS template resolution, or the initial param H2D copy). **After
t=1.15s, there are ZERO gaps >1ms for the remaining ~6.4 seconds** -- kernels run back-to-back. This
directly contradicts the code-reading hypothesis: the marshaling-loop-then-sync pattern exists
structurally, but in practice the GPU is never left waiting once warmed up -- either the marshaling
loop really is sub-millisecond (plausible: it's ~15-30K scalar int copies from an mmap that's
page-cache-hot after the startup scan), or kernel launch queuing hides it completely.

**What the steady-state per-kernel breakdown DOES suggest as real levers** (6.439s / 7 real steps,
tagged by `ce_backward_kernel` count = exactly one CE-backward per step):
- `rmsnorm_backward_act_kernel<bf16>` -- **11.0%** of GPU time, 255 calls (~36/step, matches ~2 norms
  x 16 layers + final). Unusually high for RMSNorm -- normally a cheap memory-bound op, not competitive
  with GEMMs/attention. Candidate #1 for a kernel-level look (fusion? vectorization? launch config?).
- `attn_bwd_dk_kernel` -- **10.4%**, 124 calls (= 16 layers x ~7-8 steps). CONFIRMS
  [[attention-kernel-throughput-bottleneck]]'s own note that DK stayed "bounded" (register pressure)
  after the DQ/DV/stats/fwd register-split work (commit 528c32a) got those four to 0 spill -- DK is now
  visibly the single largest attention-backward cost, i.e. the next natural target of that same
  register-split technique.
- A 7-call **FP32** `magma_sgemmEx_kernel` -- **6.8%** of time at ~62ms/call (vs ~0.9ms/call average
  for the bf16 CUTLASS GEMMs) -- almost certainly the LM head projection (VOCAB=32873, deliberately
  FP32 for numerical stability per [[conditioning-embedding-amplitude]]-adjacent precision-policy
  notes). Worth checking whether a tensor-core-eligible FP32 path (TF32, or a narrower stable
  approximation) exists without sacrificing the stability reason it's FP32 today.

**Caveat -- absolute per-step timing under nsys is NOT representative.** The 7 real steps in the 6.439s
steady window imply ~920ms/step under the tracer, vs ~31ms/step (32 win/s) measured in the SAME model's
untraced run moments before and after. This ~30x inflation is consistent with nsys's per-CUDA-API-call
interception overhead being roughly fixed-cost-per-call, and this workload issuing ~390-430 kernel
launches PER STEP (very fine-grained op-per-kernel design) -- a launch-rate-heavy workload pays far
more RELATIVE tracing tax than a few-huge-kernels workload would. The *relative* per-kernel-name
breakdown and the *idle-gap-structure* finding (no per-step gap) should both still hold under normal
(untraced) throughput, since they're about proportions/structure, not absolute durations -- but don't
quote "920ms/step" as a real number anywhere.

**STILL PENDING:** no `ncu` (Nsight Compute) deep-dive done yet on rmsnorm_backward / attn_bwd_dk /
the FP32 lm-head GEMM specifically (occupancy, memory throughput, warp stall reasons) -- that's the
natural next step to turn "these three are unusually expensive" into "here's the specific fix." Not
started this session; training was resumed (real run, unprofiled) rather than continuing to hold it
for further profiling, per the user's implicit priority (production progress over more investigation
once the concrete finding was in hand).
