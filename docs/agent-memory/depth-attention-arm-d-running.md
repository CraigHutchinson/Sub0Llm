---
name: depth-attention-arm-d-running
description: "THE 5-ARM PLAN (2026-07-30): arm D running; arm E built and REQUIRED (the 2x2's missing cell, so the INTERACTION term is measurable); then all arms to natural plateau. Carries the pinned params and the smoke-test lesson"
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T07:08:50.087Z
---

Depth attention Stages 1 (CPU) and 2 (CUDA) are DONE and green -- see [[depth-attention-stage0-done]].
This note is the EXPERIMENT plan. `docs/DEPTH_ATTENTION.md` 6a/6b carries the same design in the repo.

## The arm matrix -- arm D alone cannot answer the question (user, 2026-07-30)

Depth attention is NOT loop-specific: it mixes V across whatever depth axis exists. So D-vs-B conflates
"helps looping specifically" (the hypothesis it was scoped for) with "helps anything". The arms are a
2x2 and the predicted quantity is the **interaction term**:

|                | no depth attn | depth attn |
|----------------|---------------|------------|
| shallow 10 exec | C `ls_C_shallow10_s1b` | **E** `ls_E_depth4_flat` |
| looped 16 exec  | B `ls_B_loop10x6_s1`   | **D** `ls_D_depth4_s1` |

plus A `ls_A_deep16_s1` (16 real layers, 9.66M) as the real-depth reference. B/C/D/E are all 7.23M.

```
(D - B) - (E - C) > 0   -> depth attention pays off MORE under looping (hypothesis holds)
(D - B) - (E - C) ~= 0  -> a generic gain, nothing to do with the loop
```
Without E a positive D-B is unattributable -- the SAME hole arm C closed one level down.

**Arm E matches D's STRIDE, not its slot count** (E: 10 exec -> 3 slots; D: 16 -> 4). Equal stride =
equal append DENSITY, which is the mechanism. Matching slot count would need stride 3 for E, making its
mechanism denser than D's and confounding the interaction term with how often the cache is written.

## Fixed tokens and plateau are DIFFERENT questions (user, 2026-07-30)

All arms so far ran a pinned 8586 steps. Arm D reached arm B's FINAL value ~11% early in tokens -- the
signature of faster early convergence that WASHES OUT, which a fixed-token comparison structurally
cannot separate from a better model. So run every arm to its natural plateau.

Two things make this much cheaper than it sounds:
1. Evals land every 477 steps and **8586 = 477 x 18 is itself an eval point**, so a plateau run CONTAINS
   the token-matched comparison as one row of its own log. One set of runs, both answers.
2. **Resuming is exact, verified not assumed**: `lr_schedule(step, peak, warmup)` (train_stage.cpp ~485)
   is a pure function of the global step -- linear warmup then horizon-free inverse-sqrt -- and its own
   comment says "recomputed from the global step so resume is exact". A/B/C/D resume from 8586 under a
   larger cap and follow the identical trajectory. Plateau-stopping is the NATIVE mode; the fixed
   `--steps` was the token-matching override.

Budget: epoch = 4769 steps; plateau plausibly 3-5 epochs. Resume A/B/C/D (~6-15k more steps each) + E
from scratch ~= **30-55 GPU-hours**, less whatever the patience-guarded detector terminates early.
Do this BEFORE a second seed: a consistent 5-arm ordering beats a second seed on an incomplete design.

## Pinned parameters -- must match across arms or the comparison is void

```
--corpus data/cosmopedia.txt --vocab 16384 --dmodel 192 --heads 6 --kv-heads 3 --seq 512
  arm D: --layers 10 --loop-middle-layers 6 --loop-repeats 2 --depth-attn-stride 4   (16 exec, 4 slots)
  arm E: --layers 10 --loop-middle-layers 0 --loop-repeats 1 --depth-attn-stride 4   (10 exec, 3 slots)
-> vocab 16508 on every arm (verified identical, not assumed)

SUB0_FIXED_SEQ=1
--batch 448 --lr 0.00748332 --seed 1 --subset-seed 1 --corpus-fraction 0.4
(token-matched runs add --steps 8586 = 1,969,422,336 tokens; plateau runs drop it for a large cap)
```
`corpus_fraction 0.4` / `subset_seed 1` were READ OFF arm B's own config.json, not assumed --
[[loopsplit-3arm-batch-confound]] is what happens when these drift.

## SMOKE TEST FIRST -- it caught a defect that would have killed the run 1.6h in

A 60-step throughput probe never reaches an eval (eval warmup 2385), so it does not exercise eval. A
40-step run + `sub0llm report` did, and exposed that `sub0_dev_caps()` advertised `supports_decode = 1`
for a path CUDA does not implement under depth attention AND that `gpu_decode_try_enable()` never read
the bit. `report` runs a sample battery at EVERY eval, so this was never a `gen`-only problem. Fixed and
pinned in `ec1d8cf`; see [[caps-bit-nothing-reads-is-not-a-guard]] for the generalized lesson.

## Measured pillars (arm D, stride 4)

* **Memory**: fits at batch 448 -- but at **7793 / 8151 MiB**, and ~300 MiB of that nominal total is
  system-reserved, so the trainer is effectively AT the ceiling. "Fits" is true and paper-thin. Stride 1
  would have needed ~4.2 GB more and does not fit.
* **Throughput**: 95,346-95,588 tok/s, stable across the probe and the first ~6,500 steps, vs arm B's
  111,415 -- about -14%. **Report from that CLEAN window.**

### The mid-run 2.7x collapse was a VRAM SPILL (user's diagnosis, MEASURED -- my first one was wrong)

The run step-changed to ~35,100 tok/s at step ~6547 and never recovered. **Measured** via
`Get-Counter '\GPU Adapter Memory(*)\Shared Usage'`: the dGPU held 7,793 MiB dedicated **plus 718 MiB
SHARED (system RAM over PCIe)**. WDDM had migrated part of the working set under pressure, and a migrated
resource stays migrated -- hence the clean step-change with no recovery.

I first diagnosed power clamping. That was wrong, in three ways worth remembering:
1. **35 W at "100% utilization" means memory-STALLED SMs, not power capping.** Utilization counts time
   with a kernel resident, not useful work. A power-capped GPU sits AT its limit (~80-115 W here).
2. Battery at 79% ON AC is a **charge-limit policy** (battery care), not discharge. I read a normal
   value as corroboration.
3. **`nvidia-smi --query-compute-apps` lists only CUDA COMPUTE contexts.** Desktop/editor GRAPHICS
   allocations use the same VRAM pool and are invisible to it, so "only the trainer holds GPU memory"
   was never what that query answered. **Asking a narrow query and reading it as a broad one.**

Operationally: at stride 4 this config is ONE desktop allocation away from spilling, and the failure is
SILENT -- no OOM, no error, just a 2.7x slowdown. For the plateau runs that is 30-55 GPU-hours vs
80-150. Free VRAM (and VERIFY the counter) before starting them. Quality is unaffected either way
(val_nelbo is deterministic given seed and data order); only wall-clock is.

**Gap worth closing**: the trainer knows both `sub0_dev_free_mem_mb()` and memplan's predicted footprint
but only reacts when `train_reserve` FAILS -- and a spill never fails. A thin-headroom warning at
startup, plus a mid-run shared-usage check, would turn this from a mystery into a diagnosis. Same class
as [[caps-bit-nothing-reads-is-not-a-guard]]: a condition the code can detect but nobody asks it about.

## What to read

Not mean NELBO alone -- the **context-length gain** from `sub0llm report`, which on the clean arms tracks
execution count (A 14.2% / B 13.6% / C 13.3%, [[loopsplit-3arm-clean-result-looping-works]]) rather than
parameter count. And state the interaction term explicitly, not just D-vs-B.

Caveat on any single-seed verdict: B-vs-C (0.0160) sits inside this project's historical seed noise
([[gqa-ab-three-pillar-result]]: 1.73x the signal, rankings flipping between seeds).

## Note for `gen`

CUDA single-token decode does not implement depth attention; `sub0_dev_caps().supports_decode` reports 0
on such a build so decode consumers fall back to the CPU automatically. Train and report are fine.
