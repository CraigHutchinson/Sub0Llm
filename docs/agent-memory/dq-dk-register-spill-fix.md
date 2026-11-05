---
name: dq-dk-register-spill-fix
description: Fixed the cuda_tests.cpp dq_spill/dk_spill register-budget test failure at HD=64 by lowering attn_dq_lanes' threshold and giving DK the same warp-shuffle channel-split DQ already had
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Follow-up to [[cuda-kernel-optimization-waves]]. User feedback that triggered this: "we shouldn't
have pre-existing failures - we should make tasks to fix issues as we find them" -- previously the
`dq_spill`/`dk_spill` cuda_tests.cpp failure had been repeatedly waved off as pre-existing/unrelated
noise instead of getting an actual fix task.

**Root cause:** `attn_dq_lanes<HD>()`'s trigger (`3*HD > 255`) undercounted real per-thread register
pressure. At HD=64, `3*HD=192` looks comfortably under the 255-register hardware cap by the raw-array
formula, but `cudaFuncGetAttributes` (via `sub0_cuda_attn_regcheck`) measured BOTH
`attn_bwd_dq_kernel` and `attn_bwd_dk_kernel` compiling to exactly 255 regs + spill (56B / 40B) --
loop scalars, shared-memory staging pointers, and induction variables beyond the three raw HD-sized
arrays push real usage past the formula's estimate.

**Fix (commit b032430, `src/backend_cuda.cu`):**
1. Lowered `attn_dq_lanes<HD>()`'s threshold to `3*HD >= 192` (empirically grounded: HD=32 measured 0
   spill, HD=64 didn't) -- engages the ALREADY-VALIDATED lane-split mechanism (2 threads share a
   query/key, each owns HD/2 channels, partial dot products combined via `__shfl_xor_sync`) one tier
   earlier.
2. Gave `attn_bwd_dk_kernel` the SAME split, previously believed impossible -- the old doc comment
   said dk "cannot split further" since `dk_j` needs both `kr` (score) and `vr` (dp) simultaneously.
   That framing was comparing DK to DV (which drops an entire array), not to DQ -- but DQ's `s`/`dp`
   pair has the IDENTICAL shape (two independent half-width dot products combined into one scalar via
   warp shuffle), just with q/k roles swapped (DK owns the key, streams query/dout; DQ owns the query,
   streams key/value). The pattern transplanted mechanically: new `attn_dk_lanes<HD>()` (same formula,
   kept separate from `attn_dq_lanes` in case the two kernels' pressure diverges later), a rewritten
   `attn_bwd_dk_kernel` with lane/HALF/stage_off structure mirroring `attn_bwd_dq_kernel`, and a new
   `grid_dk` in `launch_attn_bwd_t` (block covers `kAttnTileQ/DK_LANES` keys, mirroring `grid_dq`).

**Verified, not just theorized:** rebuilt and ran the actual test suite (vcvars64 + ninja, real RTX
5070 GPU) on two configs:
- **smoke_gpu (D_MODEL=448, D_HEAD=64)**: register-spill test now passes (dq: 164 regs/0B spill, was
  255/56B; dk: same treatment). Full `[cuda]` suite (41,434,827 assertions, 21 test cases) passes,
  including flash-attention backward parity (reldiff 0.0697, well under the 0.15 bf16-noise bound) --
  confirms the DK rewrite didn't just eliminate spill, the gradient math is still correct.
- **ts_gpu (D_MODEL=192, D_HEAD=32)**: register-spill test still passes with 0 spill, LANES=1
  (unaffected path), confirming no regression at the small-HD size.
- **native (D_MODEL=768, D_HEAD=96, the production fineweb_edu config)**: the register-spill/regcheck
  test DID run at real HD=96 (before the user repurposed the `native` build dir for a tinystories
  experiment, see below) and measured dq=226 regs/0B spill, dk=222 regs/0B spill -- the fix works at
  HD=96 too. The only "failure" was a STALE test assertion (`if constexpr (3*D_HEAD>255) CHECK(dk_spill
  > 0)`, expecting the OLD accepted-spill behavior) -- fixed in a follow-up commit
  (`test(cuda): dk_spill is now unconditionally 0, matching dq/dv/fwd`) to just `CHECK(dk_spill == 0)`
  unconditionally. **NOT yet re-run**: a full `[cuda]` gradient-parity suite pass at real HD=96 against
  the CORRECTED test file (the HD=96 run above predates that fix, so it saw the old stale assertion).
  The user then ran their own tinystories experiment in `out/build/native` (AUTO_CONFIGURE=ON in that
  build dir's cache silently reconfigures on any build touching the generated config), so `native`
  currently points at D_HEAD=32 dims, not the production 768/8/96 ones -- re-verify with the corrected
  test file once `native` is pointed back at fineweb_edu, or before resuming production training,
  whichever comes first. **No risk to the production checkpoint**: `models/fineweb_edu_native/` (the
  MODEL directory) is fully separate from `out/build/native/generated/` (the BUILD directory) --
  checkpoints/corpus/tokenizer are bundled/pinned into the model dir specifically so a build-tree
  reconfigure can't touch a live or paused run (see `bundle_into_model_dir` in `train_stage.cpp`);
  confirmed via `ls`/`meta.txt` that the production files are untouched (still `d_model=768,
  steps=51816`, unmodified since the pause).

**Confirmed at the ACTUAL current production dims (2026-07-07, via Fable's hardware-efficiency Phase 1
audit):** the config referenced above as "native/production" (D_MODEL=768, D_HEAD=96) was later archived
-- see [[production-line-pivot-medium-model]] -- and D448/L11/H7/D_HEAD=64 is now the real production
line. ncu `--set full` on `attn_bwd_dk_kernel` at these actual production dims confirms
`Local Memory Spilling Requests = 0 byte` -- the fix holds at the dims that matter now. Same profile
also found the NEXT bottleneck at this kernel: 170 registers/thread -> block-limit 4 -> only 8.33%
theoretical occupancy (achieved 7.86%, ~94% of its own ceiling) -- register pressure was traded for zero
spill, and that trade is now the design's own limiting factor. Flagged (not yet acted on) as the
highest-risk/most-uncertain "big refactor" candidate in [[hardware-efficiency-iteration-standing-task]]
-- deserves its own scoped investigation/prototype, not a quick tune.

**Build mechanics note (for next session):** each PowerShell tool call is a fresh shell -- vcvars64
must be re-imported every time before `cmake --build` touches a `.cu` file, or nvcc fails with
"Cannot find compiler 'cl.exe'". Pattern used:
```powershell
$vcvars = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
(cmd /c "`"$($vcvars.FullName)`" && set") | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2] } }
```
Also: `sub0_tests.exe` needs its build dir's `sub0_backend_cuda.dll` on PATH at runtime (not just build
time), e.g. `$env:PATH = "D:\...\out\build\<dir>;" + $env:PATH`.
