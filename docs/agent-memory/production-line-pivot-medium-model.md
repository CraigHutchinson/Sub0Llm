---
name: production-line-pivot-medium-model
description: Abandoned the 768-dim fineweb_edu production line in favor of a medium (D_MODEL=448) model trained on the FULL fineweb_edu corpus for higher throughput / feasible completion time
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Decision (2026-07-03):** the original 768-dim/16-layer fineweb_edu production line (paused at step
51816, `models/fineweb_edu_native`) is ABANDONED. User's reasoning: training a smaller model
(D_MODEL=448, L=11, H=7, matching `smoke_gpu`'s dims exactly) on the FULL fineweb_edu corpus gives
much higher throughput and a real chance of completing in about a day, vs. the 768-dim model being
both slower per-step and undertrained relative to its capacity at any timeframe that fits.

**The medium-model run (`models/fineweb_edu_workflow`, built in `out/build/workflow`) is now the
actual production line.** State as of the pivot: step 543607, batch=64 (checkpoint-saved; the build's
own tuned `DEFAULT_GPU_BATCH=198` differs -- resuming will show the new resume-override log line
firing exactly as designed: "batch 198 -> 64, ..."), status=stopped (graceful), git_sha=b032430 (an
early commit from this same session -- the run has already been through several stop/resume cycles
during this session's work, predating most of the later fixes).

**Cleanup performed:** the abandoned 768-dim checkpoint and its stray earlier-session leftovers
(several auto-named `sub0llm_fineweb_edu_d768...` dirs from earlier false starts, ~65GB total) were
moved (not deleted -- kept reversible) from `models/` to a new `models_archive/` sibling directory:
- `models_archive/fineweb_edu_native_768d_step51816/` (the real one, 30GB)
- `models_archive/sub0llm_fineweb_edu_d768l16h8sq512v32872r_4762486/` (11150 steps)
- `models_archive/sub0llm_fineweb_edu_d768l16h8sq512v32872r_e226d64/` (0 steps, essentially empty)
- `models_archive/sub0llm_fineweb_edu_d768l16h8sq512v32873r_bd6af5b/` (unknown/partial)
- `models_archive/sub0llm_fineweb_edu_d768l16h8sq512v32873r_fcea638/` (1653 steps, 27GB)

**`out/build/native`, the build dir tied to the 768-dim line, is no longer needed for production** --
it was mid-reconfigure back to fineweb_edu when this pivot happened (a CLI-explicit
`--dmodel 768 --layers 16 --heads 8 --seq 512` invocation was in flight to counter
`data/fineweb_edu.txt.model`'s pinned "medium size for faster training" default, see below). That
in-flight invocation had a further bug -- explicit dmodel/layers/heads/seq pins DID take effect via
CLI, but `--vocab` left at its 0/auto default silently inherited the `.model` file's PINNED vocab
(16384) instead of truly re-deriving from the corpus -- would have produced a config that still
mismatched the original checkpoint (vocab=32873 there). Not relevant to fix now given the pivot, but
worth knowing if `out/build/native` or the 768-dim line is ever revisited: pin `--vocab` explicitly
too, don't trust `--vocab 0` to bypass a `.model` file's pinned value.

**`data/fineweb_edu.txt.model` is a deliberately shared, per-corpus pinned-config file** (NOT an
auto-derived cache) -- its active (uncommented) block currently pins the "medium size for faster
training" dims (448/11/7/256/16384), with the original 768/16/8/512/32768 block present but
commented out. This file is shared across ANY build pointed at `data/fineweb_edu.txt` with no
explicit CLI dim overrides -- do not edit it without the user's say-so, since the medium-model pin is
intentional and load-bearing for their current strategy.

**Rebuild note:** `out/build/workflow`'s `CMakeCache.txt` has `SUB0_CORPUS=tinystories.txt` (stale/
unrelated to what's actually generated there) and `SUB0_AUTO_CONFIGURE=OFF` -- its real generated
config (`out/build/workflow/generated/*`, confirmed D_MODEL=448/L=11/H=7/SEQ=256/VOCAB=16489) was
populated by a manual `sub0llm-configure.exe` invocation outside the CMake cache mechanism, same
pattern as `out/build/native`'s drift earlier this session (see [[dq-dk-register-spill-fix]]). Because
AUTO_CONFIGURE is OFF here, a plain `cmake --build out/build/workflow --target sub0llm-train sub0_tests`
does NOT risk triggering an unwanted reconfigure -- confirmed safe, this is how the rebuild in this
session was done.

**Status at end of this session:** `out/build/workflow`'s `sub0llm-train`/`sub0_tests` were rebuilt
with all 12 commits from this session (dq/dk register-spill fix, memplan BF16-mirror accounting,
resume-time VRAM re-validation, on-device grad-clip scale, dead F32-wqkv removal, checkpoint-save
robustness, resume-override logging, ETA-to-next-epoch) and fully re-verified: full `[cuda]` suite
green (41,434,859 assertions, 23 cases) at the EXACT production dims (D_MODEL=448, HD=64, matching
`smoke_gpu` throughout this session), plus the register-spill test and the mixed train/infer parity
test explicitly. Training was NOT restarted -- user chose to start the actual resume themselves when
ready, so the model directory is still `status=stopped` at step 543607. See [[cuda-wave6-orchestration-memory-findings]]
for the CUDA optimization work itself, [[train-diagnostics-backlog]] for the diagnostic fixes.
