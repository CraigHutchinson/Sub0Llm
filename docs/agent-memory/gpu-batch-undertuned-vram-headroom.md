---
name: gpu-batch-undertuned-vram-headroom
description: DONE (2026-07-08, fffef3b) -- untuned GPU builds now get a VRAM-scaled DEFAULT_GPU_BATCH (gpu_batch_estimate) instead of the CPU data-parallel width fallback; live run (still on the old binary) still needs a real re-tune once GPU is free
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Fix implemented same day (`fffef3b`), in direct response to the user's own framing:** "We should
have a separate default for the gpu setup - ideally based on the estimated vram we should estimate the
likely best being normally the largest multiple of some factor." Added
`sub0::config::gpu_batch_estimate(dims, vram_mb, hard_cap, act_bytes, headroom_mb=512, align=32)` in
`config_util.hpp`: the largest batch that fits a headroom-reserved VRAM budget
(`memplan::max_batch_for_vram`, the SAME primitive the tuner's own search bounds itself with), rounded
DOWN to a multiple of `align` (32, the warp size -- a safe divisor of the tensor-core tile shapes
cuBLAS/CUTLASS use) so the estimate lands on a clean GEMM-friendly boundary rather than an arbitrary
binary-search remainder. `TuneDefaults` gained `gpu_batch_from_cache` (mirrors the existing
`tf32_from_cache` pattern) so the configurator can tell "a real tune result" from "the derived
CPU-width fallback" and only substitutes the new estimate in the latter case -- never overrides an
actual tuned value. Verified end-to-end via the real configurator binary (disposable GPU-enabled
`vram_check` build dir): at d448/L11/H7/seq256 (this project's own TinyStories production shape)
against the real 8151 MiB card, the estimate is **512** vs the old CPU-width fallback's **96** -- a
5.3x jump. Unit-tested (`gpu_batch_estimate` test case, 9 assertions: unknown-VRAM passthrough,
align-multiple, headroom actually shrinks the budget, BF16 doubling the fit, monotonic in VRAM, zero
when VRAM <= headroom).

**Caveat, explicitly flagged in both the code comment and to the user:** this is a heuristic STARTING
POINT, not a guarantee of the true optimum -- the tuner's own real search already found throughput
non-monotonic in batch on this hardware (dips at 512, recovers by 768), and coincidentally 512 is
EXACTLY what the new estimate landed on for the d448 shape above (different model/corpus than the
dip was measured on, so not necessarily the same trough, but worth remembering when interpreting a
first real tune run against this default). `sub0llm tune --backend gpu` still finds the true optimum;
this just makes the PRE-tune starting point vastly more sensible than a number that had nothing to do
with the GPU at all.

**Observation (2026-07-08, user):** the live d448 TinyStories GPU training run (`4dba86b`) shows only
2523/8151 MiB VRAM used (`nvidia-smi`) despite 100% GPU utilization -- "there could be some performance
on the table potentially if we used more of the free VRAM."

**Root cause, confirmed via code + on-disk evidence, not guessed:** `meta.txt`/the training log show
`batch=96`. `out/build/ts_cuda/generated/sub0_system.hpp` bakes `DEFAULT_GPU_BATCH = 96` with its own
comment: "tuned device-training minibatch; CPU data-parallel width until `tune` sets it." 96 =
`DEFAULT_THREADS(24) * DEFAULT_WINDOWS_PER_THREAD(4)` exactly -- the CPU-oriented fallback, unrelated to
GPU VRAM/throughput at all. Confirmed `sub0llm tune` was never actually run for this specific build:
`out/build/ts_cuda/generated/sub0_tune.cache` is 0 bytes, timestamped the same minute as the corpus
config regen (created empty by configure, never populated by a real tune run). This matches
`scripts/workflow.ps1`'s own documented, deliberate default: tuning is OPT-IN (`-Tune` flag) --
"training runs fine on the un-tuned defaults, and tuning adds its own search time" -- so this is
expected behavior for how the run was started, not a bug, but it does mean the batch in use is NOT the
throughput-optimal one the tuner's own search would find.

**The tuner itself is already well-built for this** (`train_stage.cpp` ~1762-1890, `sub0_tune_stage`
GPU path): computes a real VRAM-fit batch ceiling from ACTUAL free VRAM (`sub0_cuda_free_vram_mb()`,
not just the baked card-total spec, since the driver/context reservation isn't visible to the pure
footprint model), builds a doubling ladder 64->128->256->...->ceiling, then runs the same robust
joint-grid + top-basin-refine + median-confirm search the CPU sweep uses (replaced an old "stop at
first plateau" heuristic after finding a real non-monotonic case: throughput dips at batch 512 below
384 before climbing higher again at 768 on this hardware -- so "more VRAM used" does NOT necessarily
mean "faster," but there's a real chance a properly-tuned batch beats 96 by a meaningful margin, since
the tuner never got to search past the CPU-width fallback here at all).

**Still open -- the LIVE d448 run itself is unaffected by the fix above** (it's already running the old
binary; a rebuild would only apply to the NEXT `sub0llm-configure` + build). `ts_cuda` must not be
rebuilt/retuned while d448 is actively training (would kill the process and lose the resume point, or
at minimum race on the GPU context). Actionable next steps, once GPU capacity is free (this run
plateaus, or between runs):
1. Run `sub0llm tune --backend gpu` (or `--backend all`) for the actual d448/L11/H7/vocab16517 shape on
   an idle GPU, see what batch it actually picks and the measured tok/s vs the current ~140K tok/s at
   batch 96 -- this is the concrete, direct answer to "how much is on the table." (A rebuild+reconfigure
   without tuning would ALSO pick up the new gpu_batch_estimate() default of ~512+ automatically, but a
   real tune run is still the way to find the true optimum, not just a better starting guess.)
2. If there IS a real gain, consider making `-Tune` the default (or at least standard practice) for the
   upcoming multi-size Chinchilla investigation runs (see [[cuda-parity-backlog-deferred]]'s
   downstream-dependency note) -- those are exactly the kind of dedicated, hours-long runs where paying
   the tune search's one-time cost upfront clearly pays for itself, unlike a quick smoke test.
3. Worth an `ncu` pass too if a wider batch DOESN'T help much: given [[training-step-nsys-occupancy-findings]]
   found the fineweb_edu training step already near 100% GPU-busy with ~390-430 launches/step (a
   fine-grained, launch-heavy kernel design), a low VRAM ceiling might be less about raw batch size and
   more about the per-step kernel-launch-count itself -- in which case a training-step CUDA-graph
   capture (the SAME technique already applied to decode, see
   [[hardware-efficiency-iteration-standing-task]]'s Phase 2) could be the real lever, not just a bigger
   batch. Don't assume either explanation without measuring -- this is a hypothesis to check, not a
   conclusion.
