---
name: gpu-tune-benchmark-vs-production-discrepancy
description: real sub0llm tune --backend gpu run on the d448 shape measured only ~62-65K tok/s at every batch tried (64/128/256/385) -- roughly HALF the ~141K tok/s the SAME shape achieved in real sustained production training at batch 96; tune result NOT applied, cache cleared, root cause still open
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**What happened:** once the d448 TinyStories GPU training run plateaued (see
[[tinystories-d448-plateau-result]]) and the GPU was free, ran the backlog item from
[[gpu-batch-undertuned-vram-headroom]]: rebuilt `ts_cuda` with all this session's fixes (needed the
MSVC environment properly initialized -- `cmd.exe`/Bash mangled `vcvars64.bat` invocation, PowerShell
calling the `.bat` directly worked), then `sub0llm-tune.exe --backend gpu`.

**Real tuner result:** VRAM-fit ladder capped at 385 (only 6367 of 6879 MiB free usable -- less free
VRAM than the 8151 MiB spec, some already in use elsewhere on the system). Batches tried: 64, 128, 256,
385. ALL FOUR landed in a tight, flat band of **62,258-64,741 tok/s** (confirm-phase medians:
64/128/256/385 all within ~4% of each other). Winner reported: batch=256, tf32=0, 64,576 tok/s median.
No `[refine]` phase samples were actually printed (the phase header appeared but logged nothing) --
worth checking separately whether refine is silently short-circuiting.

**The real discrepancy:** the SAME d448/L11/H7/seq256 shape, in ACTUAL sustained production training
(the run that had just plateaued), measured **~140,000-150,000 tok/s at batch=96** throughout its
2.5-epoch run (train.log, e.g. "552 win/s (141222 tok/s)") -- roughly **2.2x faster** than the tuner's
best measurement, at a SMALLER batch the tuner's ladder didn't even test (its ladder was 64/128/256/385,
skipping 96). This is not a small margin-of-error gap; it holds at EVERY batch the tuner tried, so it
is not "96 happens to be a lucky value" -- the isolated tuner harness itself is measuring something
substantially slower than real steady-state training achieves.

**Investigated but NOT conclusively root-caused:** compared `time_train_step()`'s benchmark loop
(`backend_cuda.cu` ~3979, calls `forward_train`+`backward_device`+`device_adam_step` directly, ids/
targets uploaded ONCE before the timed loop) against `sub0_cuda_train_step()` (the real production
entry point, ~3210, calls `run_fwd_bwd` which does the SAME `forward_train`+`backward_device` internally
but ALSO re-uploads ids/targets via `cudaMemcpyAsync` EVERY step) -- structurally very similar, no
obvious "production has a secret faster path" found by reading the code. Two remaining hypotheses,
neither confirmed:
1. **cuBLAS algorithm re-selection tax**: the tune sweep ping-pongs between DIFFERENT batch sizes
   (different GEMM M-dimension) within a single process/session, while real production always uses the
   SAME batch every step. If cuBLAS(Lt)'s internal heuristic/plan cache re-selects algorithms on every
   shape change and that cost doesn't fully amortize within each batch's short (~0.3-0.9s) sample
   window, the tuner would systematically under-measure steady-state throughput at ANY batch it visits
   mid-sweep -- consistent with the flat, uniformly-slow curve observed.
2. Some other warmup/steady-state gap between the tuner's few-hundred-ms timed windows and real
   training's many-minutes steady state (thermal, clock-boost ramp, driver-level batching) -- less
   likely to explain a clean 2.2x given [[thermal-confounds-ab-wallclock-testing]]'s own past findings
   were smaller-magnitude, but not ruled out.

**Action taken:** did NOT apply/bake this tune result (`cmake --build --preset native` was never run
after the tune). Cleared `out/build/ts_cuda/generated/sub0_tune.cache` back to empty (backed up the
suspicious content to `/tmp/suspicious_tune_cache_backup.txt` first) so a FUTURE `sub0llm-configure` run
on this build dir falls back to the new `gpu_batch_estimate()` heuristic (448 for this shape) instead of
silently baking in a measured-but-unreliable batch=256 that contradicts real evidence -- baking a tune
result that might be a genuine regression vs the already-working batch=96 would have been worse than
leaving it untuned.

**Next step, not yet done:** root-cause the benchmark-vs-production gap properly (ncu or a controlled
same-batch-repeated-many-times tune-harness variant to isolate the cuBLAS-reselection hypothesis) before
trusting ANY `sub0llm tune --backend gpu` result on this codebase again -- until this is understood, a
tune result should be treated as suspect, not authoritative, and cross-checked against real sustained
training throughput before being applied.

**Two of three hypotheses ruled out (2026-07-08, user asked to cross-check both):**
1. **Dims/flags mismatch -- RULED OUT.** Compared `out/build/ts_cuda/generated/sub0_corpus.hpp` (as
   used by the tune) field-by-field against the plateaued run's `meta.txt`: D_MODEL 448, N_LAYERS 11,
   N_HEADS 7, SEQ_LEN 256, D_FF 1792, VOCAB 16517, USE_TERNARY/GATED_FFN/TIED_EMBEDDINGS/QK_NORM all
   false, POS_ENCODING Rope -- EXACT match on every single field. Also checked GEMM_DTYPE/ACT_DTYPE:
   both BF16 in the current (post-reconfigure) build; could not directly confirm this matches the
   ORIGINAL run's baked precision (its own header was overwritten by this session's reconfigure before
   the question was asked), but BF16 is this project's standing AUTO-resolved default on a BF16-capable
   card and no `--prec-*` override was used in the reconfigure, so a mismatch here is unlikely though
   not 100% excluded.
2. **System not actually idle -- RULED OUT, but was a genuinely valid concern.** Checked timestamps
   after the fact: the CPU-bound `sub0llm report` sample-battery (24 OpenMP threads) was confirmed STILL
   RUNNING for the tune run's entire duration (tune finished 04:02, report didn't finish until 04:07) --
   a real, uncontrolled confound in the first tune run. Re-ran the tune on a VERIFIED idle system
   (`nvidia-smi` confirmed 0% GPU util / only baseline desktop-compositor VRAM before starting, `ps aux`
   confirmed no sub0llm/nvcc/cl process running) -- result was essentially UNCHANGED (batches 64-385
   still all landed in the same ~61-65K tok/s band, winner now batch=385 at 64,875 tok/s vs the first
   run's batch=256 at 64,576 -- a difference well within run-to-run noise, not a contention signature).
   So the ~2.2x gap vs production's real ~141K tok/s at batch 96 is REAL and reproducible, not an
   artifact of either dims or contention.
3. **Also checked: does production use a hidden faster path the tuner's harness misses?** No --
   `GpuTrainer::step()` (train_stage.cpp:722) calls the exact same `sub0_cuda_train_step()` the tuner's
   `time_train_step()` approximates (both bottom out in `forward_train`+`backward_device`+
   `device_adam_step`); unlike decode, there is no CUDA-graph-captured fast path for the full training
   step. Ruled out by code reading, not yet by direct A/B measurement.

**Still unresolved, deprioritized behind CUDA feature parity (Muon/gated-FFN) per user direction
(2026-07-08).** The cuBLAS-reselection-tax hypothesis (tune ping-pongs batch sizes; production never
changes shape) remains the leading unconfirmed candidate. A clean next test, if picked up later: call
`sub0_cuda_train_benchmark`/`time_train_step` repeatedly at a FIXED batch=96 (no ladder, no shape
changes) for many iterations and compare directly against production's sustained rate at the same batch
-- isolates the "isolated single-shape harness" case from the "ping-ponging shapes" case cleanly.
