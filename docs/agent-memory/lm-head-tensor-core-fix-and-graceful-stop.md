---
name: lm-head-tensor-core-fix-and-graceful-stop
description: LM head GEMM forced onto bf16 tensor cores (was falling back to a CUDA-core SGEMM despite the rest of the network already using tensor cores); + a new Ctrl+C/console-close graceful-stop that checkpoints before exit
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**LM head fix (2026-07-02), following on from [[training-step-nsys-occupancy-findings]]:** root-caused
via code reading (`launch_linear`/`gemm_compute()` in backend_cuda.cu) that `launch_linear_t` (the
bf16-storage GEMM path) has ZERO call sites in the whole file -- EVERY GEMM in the model, including
QKV/attn-out/FFN, goes through the FP32-storage `launch_linear`/`gemm()`/`gemm_compute()` path. The
`CudaTf32` knob is baked `false` for this build (`CUDA_TF32 = false` in sub0_system.hpp), so
`gemm_compute()` requests plain `CUBLAS_COMPUTE_32F` for every GEMM -- yet the nsys trace showed most
of them (QKV/attn-out/FFN) landing on bf16 tensor-op CUTLASS kernels anyway (cuBLAS's own
`CUBLAS_GEMM_DEFAULT` heuristic auto-promotes "nice" shapes even under plain COMPUTE_32F). The lm_head
GEMM's awkward VOCAB-wide N (32873, not tile-friendly) was the ONE GEMM that heuristic didn't promote,
falling back to `magma_sgemmEx_kernel` (a general FP32 CUDA-core SGEMM, ~60-100ms/call vs <1ms/call for
the promoted ones).

**Fix:** added a `force_tc` bool parameter threaded through `gemm()` -> `gemm_compute(force_tc)` ->
`launch_linear()`/`launch_linear_bwd()`, defaulting false (every existing call site unchanged). Set
`force_tc=true` at the 4 lm_head call sites only (forward x3 -- decode/forward/train-forward -- +
backward x1, which does 2 internal GEMM calls). This explicitly requests
`CUBLAS_COMPUTE_32F_FAST_16BF` (FP32 storage, bf16-tensor-core multiply, FP32 accumulate) for lm_head
regardless of the global TF32 knob, since the REST of the network is already effectively running at
this precision level anyway (proven by the trace) -- lm_head was the outlier, not a case for extra
caution.

**Validated on ts_gpu** (idle build, native was mid-production-training so untouchable): full
cuda_tests.cpp suite, 4341414/4341416 assertions pass (2 pre-existing unrelated failures, see below).
The CPU/GPU parity test at the new precision needed a tolerance bump (1e-2 -> 3e-2, matches the
already-established looser bounds elsewhere for BF16 paths) -- comment at the call site
(tests/cuda_tests.cpp, "CUDA forward stays correct across a batch grow") explains why.

**Deployed to native (production) 2026-07-02 19:xx**, resumed from the step-3834 checkpoint (no data
lost -- see [[training-step-nsys-occupancy-findings]]'s process-management notes, same care taken).

**A/B validated properly (2026-07-02, per [[thermal-confounds-ab-wallclock-testing]]):** the FIRST
with-fix run opened at 35 win/s then dropped to a 17-19 win/s steady state, which looked like the fix
was a real regression (power/thermal throttling from higher tensor-core utilization). The user
correctly pushed back on trusting a single before/after pair and asked for interleaved retesting.
Results across trials, same checkpoint each time (temporarily reverted force_tc to false for the B
runs, rebuilt, tested, restored):
  - A trial 1 (with fix): 35 (burst) -> 17, 18, 18, 19, 19, 19 win/s steady
  - B trial 1 (without fix): 32, 31, 32 win/s steady (88 deg C, ~2000-2100 MHz, ~75-78 W)
  - A trial 2 (with fix): 35, 35, 36 win/s steady (86 deg C, ~2070-2090 MHz, ~79 W) -- clearly the best
    of all three, at comparable temp/power to B trial 1
A trial 1's 17-19 win/s steady state was the outlier, not the fix. It followed immediately after a
long stretch of nsys/ncu/rebuild churn (see [[training-step-nsys-occupancy-findings]]), and A trial 2
-- run later, same checkpoint, same code -- did not reproduce the drop at all. **Conclusion: the fix is
a genuine, reproducible ~12-15% steady-state win (36 vs 31-32 win/s), confirmed by the interleaved
retest.** Left A trial 2 running as production (already the correct config + checkpoint, no need to
restart again).

**Side effect: found and fixed a REAL test-isolation bug**, not just a coincidental flaky test -- see
[[raii-cleanup-preferred-over-manual]] and [[cuda-padded-window-gradient-mismatch]] (now resolved).

**New feature, same session: graceful-stop for training (user request).** Ctrl+C / a console window
closing / `taskkill` WITHOUT `/F` (all deliver the same Windows console control event) now save a
checkpoint at the step reached instead of losing everything back to the last periodic save.
Implementation (train_stage.cpp): `SetConsoleCtrlHandler` registered via an RAII `ConsoleCtrlGuard`
(per [[raii-cleanup-preferred-over-manual]]) sets a `std::atomic<bool> g_graceful_stop`; the training
loop polls it once per step (right after `rs.step = step;`) and, if set, does the SAME
sync+save+checkpoint+prune+write_meta sequence the existing 60-minute crash-resistance tick uses, then
sets `stop=true` to exit the loop. `write_meta` gets a new "stopped" status (was previously conflating
this with "plateaued" via a shared bool) and the exit path skips the SEQ_LEN-token sample-generation
preview (real time at a large model) so the process exits promptly, inside Windows' grace period
before a force-kill.

**Testing limitation, honestly noted:** could NOT end-to-end verify actual OS signal delivery in this
sandboxed session -- a background process I launch isn't attached to a real interactive console the
way Ctrl+C or a non-forced `taskkill` expect (`taskkill` without `/F` against such a process errors
outright: "This process can only be terminated forcefully"). Verified the code compiles and reasoned
through the logic; the mechanism itself (`SetConsoleCtrlHandler`) is a standard, well-established
Windows API, low risk of being subtly wrong. **Recommend the user verify once** in a real interactive
terminal: start a training run directly, press Ctrl+C, confirm `train.log` shows "[graceful stop
requested @ step N -- saving before exit]" and a new `.ckpt` appears at that step.
