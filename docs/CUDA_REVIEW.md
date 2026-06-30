# CUDA Backend Review — Allocations, Step Bring-up/Teardown, Throughput

Scope: `src/backend_cuda.cu` — the allocation lifecycle, the training step, the AdamW update, the
inference graph, and the tune measurement primitive. Triggered by the GPU tune showing ~17 s
"samples" and a 2.7 GB transient VRAM use; this documents what's actually going on and the rework.

## 1. Allocation lifecycle (no leak found)

Four resident pools, all grow-on-demand and all released by `sub0_cuda_shutdown`:

| Pool | What | Sizing | Freed by |
|---|---|---|---|
| `g_dev_params` | f32 weight blob (PARAM_FLOATS) | batch-independent | shutdown |
| `g_fwd` (`fwd_alloc`) | forward scratch + fused `wqkv[]` | grow-on-demand; `full=false` in training (dids+wqkv only) | `fwd_free` / grow |
| `g_tr` (`train_alloc`) | saved activations (residual stream + rinv per layer; a/qkv/att/fbuf/ff1/gact as SINGLE checkpoint buffers recomputed in backward) + grad temporaries | grow-on-demand for `batch` | `train_free` / grow / shutdown |
| optimizer (`opt_alloc`) | grad, m, vel, decay mask, normsq | batch-independent | `opt_free` / shutdown |

`shutdown` calls `invalidate_graph` + `fwd_free` + `train_free` + `opt_free` + frees params — so a
clean init→…→shutdown returns to baseline. **Confirmed: idle GPU = 0 MiB used / 7891 free.** The
2.7 GB seen during the tune was the *process's own* footprint (CUDA context ≈ 1 GB + cuBLAS
workspace + the footprint-check's resident batch-64 set ≈ 1.5 GB), released at exit — **not a leak
or an external hog** (a leak/memory smoke test now guards this; see tests below).

**Bug — VRAM ceiling under-counted:** the tune calls `sub0_cuda_train_footprint(64)` (which *leaves*
its batch-64 scratch allocated) and only *then* probes `sub0_cuda_free_vram_mb`. So the free-VRAM
budget is short by the batch-64 resident set (~1.5 GB) and the VRAM-fit batch ceiling (293) is
conservative. Fix: free the probe scratch before measuring, or probe before allocating it.

## 2. Step bring-up / teardown (where the per-step overhead is)

**Per optimizer step (`device_adam_step`)** does three things beyond the math:
1. **grad-norm clip → `cudaStreamSynchronize`** — accumulates ‖g‖² on device, copies the scalar to
   host, and **blocks the stream** to read it before the update. A hard per-step host↔device
   serialization (the "AdamW host sync" the commit log notes). Removable by clipping fully on-device
   (scale inside `adam_step_kernel` from the device `normsq`), eliminating the sync.
2. **`build_qkv_weights`** — rebuilds the fused `[Wq|Wk|Wv]` per layer every step because the
   projections just changed (N_LAYERS × [C,3C] copies ≈ 26 MB of D2D copies for this model). Correct,
   but it could be folded into the update or deferred to the next forward.
3. **`invalidate_graph`** — params changed, so the *inference* graph is dropped. Harmless for training
   (the train path is eager), but means generation right after training always recaptures.

**Per tune measurement (`time_train_step`)** re-uploads the synthetic params, rebuilds QKV and zeroes
the Adam moments **every call**. Fine for a one-off, wasteful when the tuner calls it across the batch
ladder — hoist the param upload / QKV build / moment zero out of the per-point path (the timing is
data-independent, so the device state only needs setting once).

**Grow-realloc (`fwd_alloc`/`train_alloc`)** frees + reallocates the batch-dependent buffers and
invalidates the graph when the batch increases. Correct and rare (only on a larger batch), so not a
hot-path concern.

## 3. Why a "sample" is ~17 s (and training is 6.2 s/step)

Two separate effects, neither a bug in isolation:
- **Cold clock / cold cuBLAS (measurement):** `adaptive_time` warms up only 2 steps; the GPU clock
  ramps and cuBLAS autotunes over *many* steps, so a 2-step measurement reads the cold rate (~17–19 s)
  while sustained training reaches ~6.2 s/step (~10k tok/s). The batch-64 ≈ batch-256 timing
  (951 vs 3376 tok/s) is the fixed cold-start/overhead amortizing over more tokens. **The tune
  therefore *under-reports* throughput** — more warmup gives an accurate (but slower) sample.
- **Sustained step (real ceiling):** even warm, 6.2 s/step at batch 256 is ~3–6 % of the card's
  bf16 FLOPs. Two causes: (a) `d_model=448` makes small GEMMs that underutilize the SMs; (b)
  activation **checkpointing recomputes** a/qkv/att/fbuf/ff1/gact in backward — a memory-for-compute
  trade that adds roughly a forward's worth of work to every backward. This is the throughput lever
  (~10 h/epoch on the 1 GB smoke comes straight from it).

## 4. Rework recommendations (ranked)

1. **Drop the per-step host sync** — clip on-device (scale from the device `normsq` inside the adam
   kernel). Removes the only hard serialization in the step; lets the stream pipeline.
2. **Profile the step** (per-phase cudaEvent timing: forward / backward / adam) to attribute the 6.2 s
   between the recompute and the small GEMMs — decide whether to make checkpointing *optional* at
   large batch (when memory is not the binding constraint) so backward stops recomputing.
3. **Fix the VRAM-probe order** so the batch ceiling isn't under-counted (§1).
4. **Hoist `time_train_step` per-call setup** so the tuner's samples reflect steps, not re-uploads.
5. **More tune warmup** (or a longer probe) so a sample reflects the *sustained* clock — trades
   sample time for accuracy; pair with the per-test cap already in place.

## 5. Tests added (this review)

- **VRAM leak / smoke trace** — baseline free → init + train scratch alloc → shutdown → free returns
  to within tolerance of baseline (catches a buffer that `shutdown` forgets to release).
- **Batch-grow + graph re-capture** — forward at batch 4, then batch 8: both match the CPU, exercising
  `fwd_free_batch` + `invalidate_graph` + recapture on a grow.
- **Sustained vs cold step** — a warm (many-iter) throughput reading reported alongside a 2-iter one,
  to make the cold-clock gap visible and guard against a regression that inflates the step.
- (Existing `memplan` footprint-drift test @ batch 32/64/128 already guards the per-step memory model.)

Items needing `backend_cuda.cu` edits (rework #1, #2, #3, #4 + per-phase profiling instrumentation)
are left for a coordinated change since that file has an in-flight WIP (the VRAM-probe edit).
