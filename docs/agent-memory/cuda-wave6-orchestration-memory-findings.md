---
name: cuda-wave6-orchestration-memory-findings
description: Fable helicopter-view review of the full CUDA training step; ranked memory/scheduling backlog -- C1/B2/Wave8-primitive/B3/B5/A2/B4 all DONE (2026-07-04 review found 3 undocumented completions); only B1's actual exploitation + Waves 11/12 remain
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**CORRECTION (2026-07-04):** a routine "is this backlog still current" check before resuming this work
found that B3, B5, and A2 were ALREADY IMPLEMENTED in the codebase -- verified directly by reading the
current source, not from any memory record of the work. None of the three had a corresponding memory
update, so this file was stale relative to ground truth for some time. Lesson: verify a memory-recorded
backlog against the actual code before resuming it, don't assume "last recorded state" is current state.
The corrected status of every item, as of 2026-07-04:
- **B3 (memplan BF16-mirror accounting) -- DONE.** `sub0_cuda_train_footprint` (`backend_cuda.cu`) now
  calls `build_qkv_weights()` before its second `cudaMemGetInfo`, exactly the fix this memory specified.
- **B5 (dead F32 wqkv during BF16 training) -- DONE.** `wqkv_alloc()` returns early for BF16 builds;
  `build_qkv_weights()` writes the bf16 mirror directly from Wq/Wk/Wv, no F32 staging buffer allocated.
- **A2 (device-side grad-clip scale) -- DONE.** `grad_clip_scale_kernel` computes `gs` entirely
  on-device from `g_dev_normsq`; `adam_step_kernel` reads it via pointer, no host readback.
- **B4 (weight-decay mask) -- was PARTIAL (float->uint8 only), now FULLY DONE (2026-07-04).** Went
  further than uint8: eliminated the persistent mask buffer entirely. `include/sub0/layout.hpp` gained
  `DecayRange`/`NUM_DECAY_RANGES`/`DECAY_RANGES` -- a compact compile-time list of `[start,end)` ranges
  merging adjacent decay=true `PARAM_LAYOUT` entries (verified exhaustively offset-by-offset against
  every layout variant -- default/tied/gated/qk-norm -- at both toy and production d448 scale, 41M+
  assertions at production scale alone, in `layout_tests.cpp`). `backend_cuda.cu` copies this into
  `__constant__` memory once (`g_decay_ranges`) via `cudaMemcpyToSymbol` in `opt_alloc`; `adam_step_kernel`
  calls a tiny linear-scan `is_decay_param(i)` instead of reading a `PARAM_FLOATS`-long mask -- was 1
  byte/param persistent (~157 MiB at production scale), now zero. `memplan.hpp`'s `persistent_bytes`
  updated to match (dropped the `UINT8` term). Verified via the full `[cuda]` suite (including "CUDA
  AdamW step matches the CPU optimizer update") at both tied and untied production d448 configs, 100%
  green, assertion counts matching exactly what they were before this change.

**B1 -- DONE (2026-07-06, via a phased Fable dispatch).** Chunked the lm_head/tied-head-forward ->
cross-entropy -> lm_head/tied-head-backward chain over row-chunks of `M` (K=8 at production dims,
derived as `clamp(ceil(VOCAB/D_FF),1,8)` -- targets the chunked `[chunk_rows,V]` scratch buffer to land
at roughly the same byte-scale as the model's other D_FF-scale activations; clamped to 8 because the
marginal win beyond that is small relative to extra launches). `ce_backward_kernel` gained a
`row_offset` param (it was already fully row-independent per-block except the `b=m/T,t=m%T` window-index
math, which is exactly what needed the offset). Reused the existing accumulate-mode primitive from
Wave 8 (`launch_linear_bwd`'s `accumulate` flag; `launch_tied_head_bwd`'s `dTok` GEMM already hardcoded
beta=1.0f, needing zero changes). Verified: full `[cuda]` suite green across all 4 scale×tied
combinations (28/28 each); REAL measured VRAM footprint via `sub0_cuda_train_footprint` at production
dims batch=128 -- untied 4374->2528 MiB (42% reduction), tied 4248->2408 MiB (43% reduction), matching
the K=8 prediction closely. An actual ncu profile (not just wall-clock) of both new kernel shapes
confirmed no chunking-introduced inefficiency: `ce_backward_kernel` 95.25% achieved occupancy
(memory-bound, expected); the head-GEMM's occupancy matches its theoretical ceiling almost exactly
(register-pressure-limited by cuBLAS's own tile selection for this shape, not by chunking) -- the
uncoalesced-access findings in both are pre-existing (VOCAB=16517 alignment), present identically
whether chunked or not. Throughput: 5 tight repeated trials of the new code (23.38-23.44ms, <0.3%
spread) vs. one prior pre-chunking sample (25.06ms) -- not a true interleaved A/B (a `git stash`
incident mid-Phase-3 made further git-based isolation too risky to pursue, see
[[fable-subagent-for-big-risky-tasks]]), but combined with the ncu evidence this was judged sufficient
rather than chasing a cleaner comparison in an already-disturbed git state.

Of the original Wave 9/10/11/12 backlog, only Wave 11 (A4/C3/A3/A5, ~1% micro-opts each) and Wave 12
(B7, resume-time VRAM re-validation, robustness not perf) remain unstarted -- both low priority.

Follow-up to [[cuda-kernel-optimization-waves]] (waves 1-5, kernel-level fixes, all shipped). This is a
**read-only architectural review** (Fable subagent, `src/backend_cuda.cu` + `train_stage.cpp` +
`memplan.hpp` + `configurator.cpp`, 2026-07-02/03) — nothing below is implemented yet, this is the backlog
it produced. Headline: kernel shape is now in good order; the remaining leverage is almost entirely
**memory accounting and batch/shape scheduling**, not more kernel micro-opt or stream overlap (a fresh
end-to-end sync/overlap pass confirmed the earlier nsys 93%+-busy finding still holds — no big idle-gap win
exists, don't re-chase that).

**Suggested triage order (the agent's own ranking, by confidence x impact):**
1. **C1 — token-budget-constant batching under `vary_seq`.** `vary_seq` (on by default, `train_stage.cpp`)
   draws `T ∈ [64,512]` per step but runs a FIXED batch at every width, even though every buffer is sized
   for `cap × SEQ_LEN` rows — a T=64 step could legally run batch≈240 in the SAME allocation. Fixed per-step
   overhead (AdamW traffic, grad memset, mirror rebuild, ~3 syncs, ~550 launches) is ~20-25ms regardless of
   T, so short-T steps waste most of it. Estimated 5-10% average throughput, needs profiling to confirm.
   Zero extra memory needed. Ranked #1 by the agent.
2. **B2 — confirmed correctness bug in the configurator's VRAM clamp.** `tools/configurator.cpp:933/935/946`
   omits `act_bytes` when calling `train_resident_mb`/`max_batch_for_vram`, so it defaults to the F32
   footprint model even though this build bakes `ACT_DTYPE=BF16` (the runtime tuner passes `act_b=2`
   correctly elsewhere). Verified arithmetically: baked `DEFAULT_GPU_BATCH=30` is exactly the F32-model
   clamp of a stale cached `gpu_batch=256`; the BF16-correct clamp is ~42. Two errors (this one, plus no
   headroom subtraction against full VRAM spec) happen to nearly cancel at THIS config, so it's not visibly
   broken today, but won't cancel on other dims/cards (~30-40% systematic undersizing).
3. **B1 — chunked lm_head+CE removes the dominant per-window scratch buffer.** The `[M,V]` f32
   logits/dlogits buffer is 55.7% of all per-window training scratch (64.2 of 115.3 MiB at T=512). It's
   row-separable (lm_head GEMM -> CE fwd/bwd -> backward GEMMs, chunked over M in e.g. 2048-row pieces,
   same math, beta=1 accumulation) — would drop scratch to ~51.1 MiB/window, roughly doubling the
   VRAM-fit batch (30 -> ~60-65). Single biggest memory lever found. Needs a `memplan.hpp` term update too.
4. **B4 + B5 — easy MiB wins.** `g_dev_decay` is a full f32 0/1 mask = 627 MiB for one bit/param (already
   had a TODO); deriving decay from `PARAM_LAYOUT` segment boundaries inside `adam_step_kernel` frees all of
   it. The f32 fused `wqkv` (108 MiB) is dead weight during BF16 training (only inference reads it) —
   building the bf16 mirror directly from params instead would free it + cut per-step mirror-rebuild traffic.
5. **B3 — memplan undercounts ~216 MiB (BF16 weight mirrors) and the CI drift test can't see it** — the
   footprint probe (`sub0_cuda_train_footprint`) never calls `build_qkv_weights`, so the mirrors don't exist
   when the test's delta is measured; tolerance (128 MiB) currently masks it, silently absorbed by the
   tuner's 512 MiB headroom today. Fix: add the mirrors term to `persistent_bytes` + call
   `build_qkv_weights` before the probe's second `cudaMemGetInfo`.
6. **A2 — device-side grad-clip scale removes 2 of 3 per-step host syncs.** Only the clip scale `gs`
   actually needs the `normsq` readback; bias corrections `bc1/bc2` are host-computable from `t` alone. A
   one-thread kernel computing `gs` on-device, read by `adam_step_kernel` via pointer, collapses syncs
   #1(loss)+#2(normsq) into the final step-end sync. Small (~µs-ms) at T=512 but is the prerequisite for
   any future whole-step CUDA graph.
7. **A4/C3/A3/A5 — a profiled micro-batch**, roughly ~1% each: `bias_grad_kernel` at N=D_MODEL is a
   768-thread launch (same parallelism-starved shape waves 1-5 fixed elsewhere — GEMV-against-ones-vector
   or block-reduce would fix it); VOCAB=32873 (odd) as the GEMM N/ld for the 3 largest GEMMs might be
   defeating vectorized bf16 loads (pad to a multiple of 64, needs an A/B — same GEMM the earlier `force_tc`
   fix already flagged as marginal, circumstantial supporting evidence); full-param-blob grad memset zeros
   far more than the atomically-accumulated segments actually need; per-step QKV-mirror rebuild traffic
   (~0.9GB) could shrink once B5 lands.
8. **B7 — robustness, not perf:** resuming a checkpoint restores its saved `batch` with NO VRAM
   re-validation against the CURRENT card/dims — a checkpoint written at a since-changed tuned batch can
   silently spill to WDDM shared memory (the ~10x cliff the configurator clamp exists to prevent, bypassed
   on this path). Should clamp/warn via the existing `memplan::max_batch_for_vram` on resume.

**Explicitly NOT worth another wave** (per the agent, stated plainly rather than padded): stream
overlap/multi-stream (bounded ≤7% by the prior nsys finding, real complexity cost); training-step CUDA
graphs (measured 1.00x already, AND `vary_seq` makes launch shapes non-stationary — up to ~449 shape
variants — so a graph cache isn't viable unless `vary_seq` is first quantized to a few widths, which C1
would do as a side effect); more attention-kernel work (spill-guarded, waves-1-5 territory); more aggressive
activation checkpointing (already at the right point on the memory/recompute curve for 8GB — verified the
alternatives are strictly worse); pinned-memory H2D for ids/targets (~120KB/step, negligible).

**Wave 7 update (2026-07-03):** B2 (configurator VRAM-clamp act_bytes bug) is FIXED and committed
(436b826) -- verified the exact predicted numbers (F32-modeled clamp=30, matching the currently-baked
value; BF16-correct clamp=42) via a standalone call to the same memplan functions. C1 (token-budget
batching) turned out to be BIGGER SCOPE than the original estimate: `train_alloc(batch)`
(`backend_cuda.cu:1418`) sizes every training buffer at `batch * SEQ_LEN` rows, and `sub0_cuda_train_step`
calls `train_alloc(batch)` with the STEP'S OWN batch argument every call (`:2224`) -- so simply passing a
larger `batch_t` at a shorter T (the naive C1 approach) would trigger `train_alloc` to GROW to
`batch_t * SEQ_LEN` (assuming full-width rows), not reuse the existing footprint differently. The buffers
ARE already sized for the full `batch*SEQ_LEN` row budget regardless of T (that's WHY short-T steps
under-use them) -- but exploiting that safely needs `train_alloc`'s growth check changed from "batch
capacity" (`g_tr_cap >= batch`) to "row-product capacity" (`g_tr_cap_rows >= batch*T`), a real change to
allocation-lifecycle code shared by 6+ entry points (train_step, benchmark, footprint probe, tuner, etc,
all found via grep of `train_alloc`/`fwd_alloc` call sites). DEFERRED, not implemented -- this needs its
own properly-scoped wave (with the entry-point survey done above as a head start), not a quick pass. The
"needs profiling to confirm" caveat on the original 5-10% estimate is now compounded by "needs a real
refactor to attempt safely."

**Waves 8-12: persisted as a follow-up backlog for a future session (2026-07-03), not pursued further this
session** (quota/time reasons -- priority shifted to a TF32 investigation, then resuming real production
training). Original triage order, unchanged:
- **Wave 8 = B1**, chunked lm_head+CE. INVESTIGATED (2026-07-03) and DEFERRED -- worse scope/risk than C1,
  not better. Traced the actual code: `gemm()` (`backend_cuda.cu:1086`) hardcodes `beta=0.0f` (always
  overwrite) -- there is no accumulate mode anywhere in the GEMM layer today. Chunking lm_head+CE over M
  needs the backward dW/dbias GEMMs to ACCUMULATE across chunks (beta=1 on all but the first), which means
  adding accumulate-mode to `gemm()`/`gemm_t()`/`launch_linear_bwd()` -- the single most shared primitive in
  the file, used by every linear layer in the model, not just lm_head. `bias_grad_kernel` also currently
  overwrites (`dbias[o] = s;`, not `+=`) and would need the same treatment. `ce_backward_kernel` (rewritten
  carefully in Wave 3) would need a new row-offset parameter so its window/position math (`b = m/T`) stays
  correct per chunk instead of chunk-local. A subtle bug in the shared accumulate-mode change would silently
  corrupt gradients EVERYWHERE in the model, not just lm_head -- a bigger blast radius than C1's allocation-
  growth risk. DEFERRED for the same reason as C1: needs its own properly-scoped wave with careful
  before/after cuda_tests.cpp parity gating, not a quick pass.
- **Wave 9 = B4+B5+B3**, easy MiB wins (decay mask to 1 bit/param, drop dead f32 wqkv during BF16 training,
  memplan BF16-mirror accounting + footprint-probe fix). Smaller, more contained than B1 -- good next
  candidate after B1 is scoped.
- **Wave 10 = A2**, device-side grad-clip scale kernel (removes 2 of 3 per-step host syncs).
- **Wave 11**, profiled micro-batch: `bias_grad_kernel` at N=D_MODEL parallelism fix (A4), VOCAB-padding
  GEMM shape A/B (C3), scoped grad memset (A5), reduced QKV-mirror rebuild traffic (A3).
- **Wave 12 = B7**, resume-time VRAM re-validation for a restored checkpoint's batch (robustness, not perf).

**TF32 investigation (2026-07-03), closed:** two separate mechanisms share the "TF32" name in
`backend_cuda.cu`. (1) The baked `CUDA_TF32` knob feeds `gemm_compute()`'s choice between plain FP32
compute and `CUBLAS_COMPUTE_32F_FAST_16BF` (BF16-tensor-core, not literal TF32 despite the name) -- already
measured (a 2026-06 code comment: "TF32 gives NO speedup at this model's small-K GEMMs -- 0.98x vs FP32")
because cuBLAS's own heuristic already auto-promotes well-shaped GEMMs to tensor cores even under plain
compute; the ONE GEMM that didn't auto-promote (lm_head) was already fixed separately via the `force_tc`
flag (see [[lm-head-tensor-core-fix-and-graceful-stop]]) -- that's the real captured win. Correctly stays
off. (2) `set_handle_tf32()`/`cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` -- literal TF32 via the legacy
cuBLAS handle-level math mode -- is DEAD CODE: it only affects calls that don't specify an explicit compute
type (the legacy `cublasSgemm` family), but this file exclusively uses `cublasGemmEx` with an explicit
compute type everywhere (grep-verified: zero `cublasSgemm` calls exist, only comments describing the
migration away from it). Toggling it has zero effect on any GEMM this project runs. Minor cleanup candidate
(remove `set_handle_tf32`/`apply_math_mode` and the `SUB0_TUNING` sweep over "tf32", since it provably can't
matter) -- not a performance opportunity, low priority, not acted on.

**Fable agent outcomes (2026-07-03), reviewed and merged/deferred:** two isolated-worktree Fable
agents were dispatched on C1 and Wave 8 (see [[dq-dk-register-spill-fix]] for what else happened this
session); both hit their session's time/quota limit mid-task and returned no written summary, so their
actual work was reviewed directly from the worktrees.
- **C1 -- COMPLETE (2026-07-03, commit `3bbd46f`).** The foundation (row-budget capacity refactor,
  `0720cd9`/`f269442`) plus the actual scheduler on top: `batch_t = tokens_per_step / seq_t`, GPU-side
  `sub0_cuda_train_reserve` now wired into `GpuTrainer::enable()`. The earlier "host-buffer overflow"
  landmine concern turned out to be a NON-issue on closer analysis: `GpuTrainer::ids`/`targets` are
  TOKEN-indexed (not window-indexed), and every admitted (batch_t, seq_t) pair satisfies
  `batch_t*seq_t <= batch*SEQ_LEN` by construction (that IS the token budget) -- so the existing
  `batch*SEQ_LEN` sizing already covers it, no resize needed. Only the WINDOW-indexed arrays
  (`starts`/`win_len`/`cpu_starts`, indexed purely by b) needed resizing to `max_batch_t`. Throughput
  (wps) and `tokens_seen` now accumulate real per-step `batch_t`/`batch_t*seq_t` instead of assuming a
  fixed batch. Verified via CPU-only path (fresh + resume, `tokens_seen` matched the expected
  integer-rounding-loss pattern exactly) while a real GPU job was active elsewhere, THEN re-verified on
  the real GPU path once it freed up (full `[cuda]` suite + a real `vary_seq` training run, both clean).
- **Wave 8 -- accumulate-mode primitive COMPLETE + tested + committed (2026-07-03, commit `c8efac0`);
  the actual lm_head/CE chunking is still NOT done.** Picked the worktree's uncommitted groundwork back
  up: `gemm()` gained a `beta` parameter (default 0, so every existing call site is still overwrite-only
  and unaffected), `bias_grad_kernel`/`launch_linear_bwd` gained an `accumulate` flag threading `beta`
  into the dW GEMM + `dbias[o] = accumulate ? dbias[o]+s : s` (dX GEMM stays overwrite-only, correctly --
  chunking is over M/rows, dX has no cross-chunk overlap to accumulate). Given the original risk note
  ("a subtle bug here would silently corrupt gradients EVERYWHERE in the model"), added the dedicated
  test BEFORE any real caller exists: `sub0_cuda_test_accumulate_check` splits a `[M,in]x[M,out]`
  backward into two row-chunks (chunk 2 accumulate=true into the same buffers) and diffs against a
  single full-M `launch_linear_bwd` reference call -- measured max relative diff 3.2e-4, gated in
  `cuda_tests.cpp` at `< 1e-3`. **Still deferred/not wired into anything real**: no lm_head/CE chunking
  logic exists yet (that needs `ce_backward_kernel`'s row-offset parameter too, per the original scoping
  note below), so this is groundwork only -- the primitive is proven safe, not yet exploited.

**Endorsed, not flagged as a problem:** current checkpointing strategy (save residuals+rinv, recompute
a/qkv/att/fbuf/ff1/gact) — correct point on the curve for 8GB. Shared memory is nowhere occupancy-limiting
(largest static use ~25KB, well under the 48KB limit). One stale doc comment found (`TODO(mem)` at
`backend_cuda.cu:1406` says BF16 activation storage "would roughly halve..." — it already happened, comment
should be deleted/updated whenever that area is next touched).
