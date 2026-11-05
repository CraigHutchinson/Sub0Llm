---
name: gpu-large-batch-access-violation
description: "NEW BUG (2026-07-16), found incidentally while verifying hybrid CPU/GPU training at d256: plain GPU training (gpu.step()/sub0_cuda_train_step, NOT the new hybrid path) crashes with an access violation (0xC0000005) at batch=768 and batch=960, but runs fine at batch<=512. Reproducible, unrelated to content-embed/hybrid work. Root cause NOT investigated yet."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Found while doing the "un-park the bigger-scale spike" verification for [[meanpool-alternatives-prior-art-and-math]]/
[[hybrid-cpu-gpu-execution-design]] -- not something the hybrid work introduced, confirmed by reproducing it with
`--content-embed` OFF (plain `gpu_train` path, `GpuTrainer::step()` -> `sub0_cuda_train_step`, code untouched this
session).

## Repro

Build: fresh `out/build/ce256` (`-DSUB0_COMPUTE=GPU`, no `-DSUB0_D_MODEL` -- dims set via
`sub0llm-configure.exe --corpus data/tinystories.txt --dmodel 256 --layers 6 --heads 4`, matching
[[tinystories-sweep-saturation]]'s own calibrated d256 row). RTX 5070, CUDA 13.2.

```
sub0llm-train.exe <dir>/model --steps 40 --batch 960   -> exit -1073741819 (0xC0000005, access violation),
                                                            crashes before printing ANY step line
sub0llm-train.exe <dir>/model --steps 40 --batch 768   -> same crash
sub0llm-train.exe <dir>/model --steps 40 --batch 512   -> WORKS, 448K tok/s, clean 40-step run
sub0llm-train.exe <dir>/model --steps 40 --batch 128   -> WORKS, 417K tok/s, clean 40-step run
```

## 2026-07-27: still live, and the AUTO-TUNER PICKS A CRASHING BATCH

Re-hit while setting up the GQA A/B, on a DIFFERENT config -- `d384 L6 H8 seq256` (gated-ffn, tied,
qk-norm), TinyStories, same RTX 5070 / CUDA 13.2. So it is not d256-specific.

```
--batch 704 (what --batch 0 AUTO-SELECTS here)  -> SIGSEGV, exit 139, no step line printed
--batch 512                                     -> clean run
--batch 256                                     -> clean run, ~265-298K tok/s
```

Two things this adds beyond the 2026-07-16 entry:

1. **The DEFAULT path is broken, not just an exotic large batch.** `gpu_batch_estimate()` chose 704 for
   this config, so a plain `sub0llm-train` with no `--batch` segfaults out of the box on this machine.
   That reframes the severity: previously it read as "don't ask for a huge batch", now it is "the
   out-of-box command crashes". A cheap mitigation while the root cause is open: clamp the auto-estimate
   to the known-good ceiling (<=512 observed on both configs) rather than letting it pick freely.
2. **NOT caused by the GQA work**, established rather than assumed: it reproduces identically at
   `--kv-heads 8` (plain MHA), which `890a0ec` verified bit-identical to the pre-GQA backend, and the
   crash boundary sits between 512 (ok) and 704 (crash) in both the MHA and GQA builds.

**Measurement trap worth remembering**: piping the trainer through `tail` makes `$?` report TAIL's
status, so the crash first looked like a clean `exit 0` with no output. Always capture the trainer's own
exit code (`cmd > log 2>&1; echo $?`) before concluding anything about how a run terminated.


Threshold is somewhere in (512, 768] -- not bisected further. **`--batch 0` (auto) picked 960 at this d256/L6/H4
shape and would have crashed a real run silently** -- this is a real production risk, not just an edge case, since
the auto-tuned default is the one most runs will actually use.

## What's ruled out

- **Not hybrid/content-embed related**: reproduces identically with `--content-embed` absent, i.e. via the
  ORIGINAL, this-session-untouched `gpu.step()`/`sub0_cuda_train_step` path.
- **Not a VRAM-fallback case**: no "batch N did not fit VRAM -- falling back to..." warning printed before the
  crash at batch=960 -- `sub0_cuda_train_reserve(960)` itself reported success; the crash happens after that,
  inside the training step.
- Crashes with ZERO step output -- happens on/before the FIRST step, not a gradual OOM/corruption that surfaces
  later.

## Not yet investigated

- Exact threshold between 512 and 768.
- Whether this is d256-specific (interacts with L6/H4/ff704's specific tensor shapes) or a pure batch-size
  ceiling that would also hit other dims (e.g. does d128/d448 crash at batch=768+ too? d448's own earlier
  calibration this session ran fine at batch=480, never tested higher there).
- Root cause: could be a fixed-size device scratch buffer sized for a smaller batch than what
  `sub0_cuda_train_reserve`/`max_batch_for_vram` actually admits (an off-by-something in `memplan.hpp`'s sizing
  math), a kernel launch config (grid/block dims) overflowing at this batch, or something else entirely --
  genuinely unknown, needs a debugger/cuda-memcheck session, not guessed at here.

## How to apply

Before trusting an AUTO-TUNED batch size on a fresh dims/corpus combination for a real run, sanity-check with a
short (~40 step) run first -- exactly the calibration step that caught this. If `sub0_cuda_free_vram_mb()`-driven
auto-sizing keeps landing on values that crash, the auto-sizer's ceiling (`GPU_VRAM_MB`/`max_batch_for_vram`) may
need a correctness cap added, not just a VRAM-fit cap -- see [[gpu-batch-undertuned-vram-headroom]] for the
existing (opposite-direction) auto-batch tuning work this would sit next to.
