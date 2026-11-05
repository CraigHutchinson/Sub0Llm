---
name: hybrid-cpu-gpu-execution-design
description: "DONE + committed (2026-07-16): source-routed hybrid CPU/GPU training is LIVE in train_stage.cpp -- content-embed windows (scratch-mix/op-mix) run on a persistent CPU worker thread, every other window runs on GPU via GpuTrainer::backward_only(), gradients weighted-merged into one host-side AdamW step, params re-uploaded to device every step. Verified on real RTX 5070 hardware: full test suite green, loss curve matches pure-CPU baseline, ~2.3-2.6x faster. NOT the same as (and much smaller than) the Phase 3 full-HYBRID device-content-embed effort in cmake/Backends.cmake."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Context: [[meanpool-alternatives-prior-art-and-math]] (why this matters now -- content-embed forces the
WHOLE training step onto CPU today, `gpu_train = !content_embed_active && ...` in `train_stage.cpp`),
[[scratch-content-embedding-meanpool-vs-charencoder]]. User asked to "consider and continue" hybrid
execution so CPU-only mechanisms (content-embed) and CUDA-accelerated ones can be interleaved rather than
forcing an all-or-nothing choice, in service of keeping the option to defer expensive CPU-only verification
runs (see the "parked" item below) until acceleration exists, without giving up production-code progress
in the meantime.

## What already exists (checked before proposing anything new)

`cmake/Backends.cmake` already has a `SUB0_COMPUTE=HYBRID` mode SLOT -- described as "CPU + CUDA, autotuned
split [Phase 3 -- not yet]", and configuring with it today is a hard `FATAL_ERROR` ("not implemented yet").
Its own comment frames "hybrid" as an AUTOTUNED WORKLOAD split (throughput optimization across the whole
run), and separately notes: **generation/eval ALWAYS run on the CPU engine regardless of compute mode** --
GPU mode only accelerates the TRAINING loop's fast path (`GpuTrainer`/`sub0_cuda_train_step`); a "full GPU
engine" (device-resident gen too) is explicitly still Phase 3/unbuilt. So content-embed's CPU-only
requirement for generation is not a special case -- gen is CPU-only for every model today, hybrid or not.
The real opportunity is specifically about TRAINING.

`GpuTrainer::enable()` uploads ALL params (including `tok_emb`) to the device ONCE; each `step()` call only
passes token ids for that step's windows -- there's no per-step re-upload of specific embedding rows in the
existing device path.

## Why a naive "patch device tok_emb rows per step" approach doesn't fit content-embed's actual semantics

Content-embed's `ScratchBindings` are threaded PER-WINDOW (`win_binds[b]`), not per-token-id globally: two
DIFFERENT windows in the SAME batch can bind the SAME scratch-slot id to DIFFERENT fragment content (they
sample different documents with different `doc_bindings`). The device embedding table, by contrast, is
id-indexed and SHARED across every window in a batch -- there is no per-window distinction at that level.
So "just patch a few device tok_emb rows before the GPU step" is not sufficient on its own; it would need a
REAL per-window device-side embedding-lookup kernel (accepting a per-window binding table, replicating
`is_scratch_slot`/`ScratchBindings` logic on-device) to be correct -- which is effectively building the full
CUDA content-embed port this project has always deferred, not a shortcut around it.

## The tractable design: SOURCE-ROUTED hybrid, not per-token hybrid

The existing blend-source machinery already tags every window in a batch with which `BlendSource` it came
from (`src_idx[b]`, `train_stage.cpp`'s per-window fill loop). Only windows drawn from a
content-embed-ACTIVE source (scratch-mix and/or op-mix, when `content_embed_active`) need CPU. Every OTHER
window in the SAME batch (base corpus, spell-mix, and scratch-mix/op-mix windows when content-embed is
off) needs nothing special and could run on GPU exactly as today.

**Proposed shape** (design only, not implemented):
1. After the existing per-window sampling loop assigns `src_idx[b]`, partition the batch's window indices
   into two groups: `gpu_idx` (sources that don't need content-embed) and `cpu_idx` (sources that do).
2. Dispatch both sub-batches within the same step, gradients merged before the shared optimizer step --
   see "How to actually get overlap" below for the corrected mechanism (the first pass at this design
   assumed CUDA kernel launches would overlap for free; checked against the real code and that assumption
   was wrong for this specific API, see below).
3. Merge: both paths' gradients need to land in a form the SAME `AdamW::step()`/Muon dispatch can consume
   as one update -- today the GPU and CPU training paths are mutually exclusive (`if (gpu_train) ... else
   ...`), not combined. Making them combinable (GPU gradient readback added into the same reduction the
   CPU path's `reduce_gradients()` already performs, before `opt.step()`) is the real engineering content
   of this design -- non-trivial, but a bounded, well-scoped integration task, not a new kernel.
4. `gpu.step()`'s existing per-call `batch_t` already varies per step (the token-budget scheduler already
   does this), so a smaller sub-batch size per call is very likely already supported without device-side
   changes -- worth confirming, not assuming, before starting.

### How to actually get overlap (CORRECTED 2026-07-16, checked against the real code)

The first pass at this design assumed "CUDA kernel launches are async, so calling `gpu.step()` then doing
CPU work overlaps for free." **Checked this against `GpuTrainer::step()` (`train_stage.cpp`) and it's
wrong for the CURRENT API**: `sub0_cuda_train_step(...)` writes the step's loss into a HOST `double*`
argument before returning -- it CANNOT do that without an internal blocking sync (an implicit
`cudaStreamSynchronize`-equivalent) first. So `gpu.step()` as written today is a synchronous, blocking
call from the host's point of view; calling it and then doing CPU work afterward is just sequential
execution on one thread, not real overlap (still useful -- both sub-batches get processed within one step
instead of forcing everything to CPU -- but not the wall-clock win "concurrent" implies).

Two real ways to get GENUINE overlap, in order of how much `backend_cuda.cu` needs to change:
- **(Easier, no CUDA API change) A separate host thread for the CPU sub-batch.** Launch the CPU
  sub-batch's `train_batch(...)` on its own `std::thread` BEFORE calling the (blocking) `gpu.step(...)` on
  the main thread. While `gpu.step()` blocks waiting on the device, the host CPU is otherwise idle from
  CUDA's point of view (a blocked thread on a stream-sync doesn't consume CPU cycles), so the separate
  thread's `train_batch` (itself already OpenMP-parallel internally) can freely use the CPU cores without
  fighting the (idle-while-waiting) GPU-step thread for them. Needs: confirming `train_batch`'s existing
  thread-local engine state (`ensure_thread_built()` etc.) is safe to invoke from a NEW host thread
  concurrently with the main thread's CUDA calls (should be -- CPU engine state and the CUDA backend don't
  share globals today, but verify, don't assume), and a `std::thread::join()` before the gradient merge.
- **(More efficient, needs backend_cuda.cu changes) Split the CUDA call into launch/finish.** Add e.g.
  `sub0_cuda_train_step_launch(...)` (issues the kernels, returns immediately, no loss readback) and
  `sub0_cuda_train_step_finish(...)` (syncs, reads the loss back) -- call launch() first, do the CPU
  sub-batch's work on the SAME host thread while the device runs, call finish() right before the gradient
  merge needs the GPU loss/gradients. True async pipelining, no extra host thread, but requires touching
  the CUDA backend's step API, not just `train_stage.cpp`.

Start with the separate-host-thread approach if/when this is implemented -- it gets real overlap without
touching `backend_cuda.cu` at all, and the launch/finish split remains available as a later refinement if
the thread-per-step overhead turns out to matter.

**Payoff, concretely**: today, ANY step where `content_embed_active` is true forces the ENTIRE batch onto
CPU, even though `sources[kBaseSource].weight` (and any non-content-embed source) typically still claims a
majority share of the blend (whatever `--scratch-mix`/`--op-mix` don't claim). Source-routed hybrid would
let that majority-share run on GPU every step, with only the CONTENT-EMBED-active minority staying on CPU
-- a real, substantial throughput recovery for the common case (partial, not 100%, scratch/op-mix blend),
without needing the full device-side content-embed port.

## Sequencing relative to the rest of the roadmap

This is a genuinely separate, sizable engineering task -- bigger than any single encoder-candidate spike,
smaller than the full Phase 3 HYBRID effort `Backends.cmake` already flags. Per the user's explicit
reprioritization (2026-07-16): checkpoint persistence for learned slot-encoders and production wiring
(both pure CPU-side engineering, no CUDA dependency) come FIRST, since they make progress regardless of
whether/when this hybrid design gets built. The PARKED item this unblocks: a bigger-scale (e.g. d448)
CPU-only spike verification of whatever encoder wins the A/B round (Hash / ConvPool / HRR) was explicitly
deferred by the user until acceleration exists for that specific verification run -- source-routed hybrid
is the more realistic near-term path to that acceleration than waiting for a full device content-embed
port, since it only needs the (already-tagged) base-corpus share of a blended run to reach the GPU, not
the content-embed windows themselves.

## MAJOR CORRECTION (2026-07-16, later): the gradient-merge mechanism, verified against real code + real GPU

The "separate host thread, no CUDA API change" plan above gets WALL-CLOCK overlap but does NOT by itself
answer how to COMBINE the two sub-batches' results into one canonical model update -- checked this and the
first-pass design had a real gap: `sub0_cuda_train_step` performs forward+backward+AdamW/Muon update
ENTIRELY ON-DEVICE in one call, exposing no separable gradient. GPU-mode keeps `g_dev_params` as the
device-resident canonical copy; CPU-mode keeps `g_param_data` as the host-resident canonical copy -- these
are TWO SEPARATE parameter+optimizer-state copies with no existing path to merge an update across them.

**The fix, verified correct**: `sub0_cuda_backward(ids, targets, batch, T, out_grad, out_loss, lengths)`
ALREADY EXISTS (built for the CPU/GPU gradient-parity tests) -- forward+backward ONLY, no auto-applied
optimizer step, reads the reduced gradient back to a HOST float buffer. This is exactly the missing
primitive. Design becomes HOST-canonical:
1. GPU sub-batch: `sub0_cuda_backward(...)` -> gradient in a host buffer, no device-side param update.
2. CPU sub-batch: normal `sub0::train_batch(...)` (on a separate thread, concurrent with step 1) ->
   accumulates into the host's own `g_param_grad` exactly as today.
3. Both are PROVEN per-call MEAN-normalized over their own sub-batch size ("CUDA gradient scale is
   invariant to duplicating the batch" test; `backward_device`'s own "invM = 1/M... equal the CPU
   train_batch grad" comment) -- so combine via `combined[i] = (n_cpu/n_total)*cpu_grad[i] +
   (n_gpu/n_total)*gpu_grad[i]`, NOT a plain average.
4. ONE host-side `opt.step()` (the SAME call the pure-CPU path already makes) applies the combined update
   to the host's canonical `g_param_data`.
5. `sub0_cuda_upload_params(sub0::params_ptr())` re-syncs the device's now-stale copy before the NEXT
   step's GPU sub-batch forward pass -- a real per-step full-param upload cost (unlike pure-GPU mode,
   which uploads once at `enable()`), not yet measured, worth profiling once wired in.

**PROVEN on real hardware (2026-07-16)**: `tests/cuda_tests.cpp` "CUDA+CPU hybrid split: weighted-merged
gradient matches an undivided CPU batch" (commit `6d3c718`) -- partitions one batch into an uneven 3/5
split, computes each half on its respective backend, combines with the weighting above, checks against a
plain undivided-CPU-batch reference. Run on a real RTX 5070 (gsm8k build, d128, GPU-enabled): rel-L2 =
0.0087, cos = 0.999963 -- matching the tightness of the existing GPU/CPU parity tests. Full CUDA suite
(32/32) and the rest of the suite (232/232) green, no regressions. **No `backend_cuda.cu` changes needed**
-- `sub0_cuda_backward` already provided exactly the right primitive.

## Live-loop wiring: DONE (2026-07-16), including a self-run /cpp-review pass that caught 2 real bugs

`train_stage.cpp` now computes `gpu_available = gpu.enable(batch, opt.step_count())` unconditionally, then
`gpu_train = !content_embed_active && gpu_available` / `hybrid_train = content_embed_active &&
gpu_available`. The per-step loop's `if (gpu_train) {...} else if (hybrid_train) {...} else {...CPU...}`
third branch partitions `src_idx[0..batch_t)` by blend source (scratch_source_idx/op_source_idx -> CPU,
everything else -> GPU), materializes each sub-batch, dispatches both, merges gradients with the proven
`(n_sub/n_total)` weighting, applies ONE host `opt.step()`, and re-uploads params to device every step (see
bug #2 below for why "every step" and not "only when the GPU was touched this step"). A new
`GpuTrainer::hybrid` flag makes `sync_to_host()` a no-op in hybrid mode (host is ALWAYS canonical there --
downloading device optimizer state would clobber the host's real Adam moments with the frozen values
`enable()` uploaded once at the start, since hybrid mode never runs an on-device optimizer step).

**`thread_local const ScratchBindings* g_scratch_binds` (backend_cpu.cpp) makes the concurrent CPU-thread +
GPU-main-thread split race-free** -- confirmed by reading the actual declaration, closing the "verify don't
assume" item the design doc originally flagged. The CPU sub-batch (OpenMP workers via `train_batch`) and
the GPU sub-batch (device buffers + a caller-owned host gradient buffer) touch entirely disjoint memory.

**Two real bugs a self-run `/cpp-review` pass caught before commit** (both fixed, both re-verified with a
real GPU run before landing):
1. **[MUST, AGENTS.md #1 -- no per-step heap/OS allocation in a hot path]** The first working version spawned
   a fresh `std::thread` for the CPU sub-batch EVERY training step -- exactly the class of per-step OS-level
   allocation cost AGENTS.md #1 names (citing the `newton_schulz5` per-call-`std::vector` regression as the
   precedent: "should have been caught during design, not after"). Fixed with a `CpuSubBatchWorker`: a
   persistent `std::jthread` spawned ONCE (only when `hybrid_train`), signaled per-step via
   `condition_variable_any::wait(lock, stop_token, pred)` (C++20 cooperative-cancellation pattern -- no
   hand-rolled stop-flag needed, `std::jthread`'s destructor's `request_stop()` wakes it cleanly).
2. **[MUST, real correctness bug]** The device re-upload after `opt.step()` was gated on `gpu_n > 0`
   ("only re-upload when this step actually touched the device"). But `opt.step()` advances the HOST params
   regardless of `gpu_n` -- a step that happens to route its ENTIRE batch to the CPU sub-batch (`gpu_n==0`,
   probability `(scratch_mix+op_mix)^batch_t` per step, rare but real over a multi-day run) would skip the
   upload, leaving the device holding an EARLIER step's weights. The NEXT step with `gpu_n>0` would then
   compute its gradient against stale params -- silent, no crash, just a quietly wrong gradient for one
   step. Fixed by dropping the `gpu_n > 0 &&` guard: re-upload unconditionally every hybrid step. Re-verified
   by deliberately forcing this exact path (`--scratch-mix 0.9 --batch 8`, small batch + high scratch_mix
   makes `gpu_n==0` steps near-certain within 200 steps) -- ran clean, no device_failed, loss descended
   5.57->1.47.
3. **[SHOULD, DRY]** The hybrid branch's CPU-sub-batch materialization loop originally duplicated the
   pre-existing plain-CPU branch's per-window fill loop (~30 lines: token copy, mask copy, content-embed
   binding lookup) almost verbatim, differing only in whether the destination index is `b` (plain) or a
   compacted `i` (hybrid). Extracted a shared `materialize_cpu_window(dst, b)` lambda both branches call.

**Verified on real RTX 5070 hardware** (gsm8k build, d128): full 269-test suite green; a 300-step
`--scratch-mix 0.3 --content-embed` smoke run's loss curve (6.52->3.20->1.94->1.65->1.55->1.36) tracks the
SAME run under `SUB0_TRAIN_CPU=1` (pure-CPU baseline: 6.52->3.20->1.94->1.67->1.55->1.35) almost exactly --
small deltas consistent with BF16 GPU-path numerical noise, matching the differential test's own cos=0.999963
-- while running ~2.3-2.6x faster (85-98K tok/s hybrid vs 33-41K tok/s pure-CPU). Committed.

## Remaining open items (lower priority, not blocking)

- The persistent-worker fix above eliminates per-step thread CREATION cost but each step still pays a full
  parameter re-upload (`sub0_cuda_upload_params`, PARAM_FLOATS floats) -- not yet profiled in isolation;
  worth checking whether it's a measurable fraction of a hybrid step once a larger model is exercised.
- Real concurrent-thread wall-clock overlap is confirmed working (no crashes/hangs across two stress runs)
  but has not been profiled in isolation (nsys/VTune) to confirm the CPU and GPU sub-batches are actually
  overlapping in wall-clock time rather than serializing for some non-obvious reason (e.g. a hidden lock in
  the CUDA driver). Throughput numbers above are consistent with real overlap (faster than pure-CPU by more
  than the GPU-sub-batch's share alone would explain) but that's inference, not a direct measurement.
- Once this unlocks GPU-accelerated verification: un-park the bigger-scale (d448) CPU-only spike test for
  the encoder candidates (see [[meanpool-alternatives-prior-art-and-math]]).

## UPDATE (2026-07-18): the deferred "full CUDA content-embed port" (line ~41 above) is now DONE

See [[cuda-binding-compose-caps-flip-merged]] -- device now composes scratch/persistent/sentinel-bound embed
rows itself (param-free MeanPool/Hash/HRR), `supports_binding_compose=1`, merged into
`claude/llm-transformers-1bit-qf3t3i`. This SOURCE-ROUTED hybrid design (CPU-thread + GPU-main-thread split,
gradient-merged) is still what's LIVE in production and remains correct/necessary for the learned-`enc_w`
encoders (CharEncoder/ConvPool) that binding-compose deliberately excludes. But for the param-free arms
specifically, the source-routing split is no longer the only option: once `train_stage.cpp`'s hybrid router
consumes the caps flip (still open, see the other memory), content-embed windows using MeanPool/Hash/HRR could
route to GPU too instead of forcing the CPU sub-batch -- narrowing this design's CPU-bound share further.
