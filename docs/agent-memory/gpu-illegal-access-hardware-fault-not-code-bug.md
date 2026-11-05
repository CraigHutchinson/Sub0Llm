---
name: gpu-illegal-access-hardware-fault-not-code-bug
description: A CUDA "illegal memory access" cascade during d128 combined-feature training (gated-ffn+tied+qk-norm+Muon) was NOT a software bug -- it was a real, recurring NVIDIA driver/hardware fault (nvlddmkm Event 153) on this laptop GPU, confirmed via compute-sanitizer (0 errors) + non-reproduction + OS event log correlation. The silent-failure-detection gap this incident exposed is now FIXED -- see [[gpu-failure-detection-hardening]]
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**What happened (2026-07-08):** during the first run of the multi-size Chinchilla sweep (d128/L4/H2,
full feature set: gated-ffn + tied-embeddings + qk-norm + Muon, all four landed this same session),
training hit `cuda error: ... an illegal memory access was encountered` starting right after step 1288
(seed 42), then cascaded into thousands of repeated identical errors for the rest of the run (every
subsequent CUDA call fails identically once a context is corrupted) before the plateau detector,
running on stale/corrupted loss values, declared a false "plateau" at step 7084. Initially treated this
as a likely real bug in the newly-landed combined-feature CUDA path (see [[cuda-parity-backlog-deferred]]),
since gated-ffn's own combined-offset hazard WAS tested against tied+qk-norm together, but Muon was
never exercised in the same run as all three simultaneously over many real training steps -- a
plausible untested combination.

**User's correction: check reproducibility before assuming a code bug -- could be an undervolt/hardware
symptom.** This was the right call. Investigation:

1. **compute-sanitizer (memcheck) on the exact failing config (same seed, same corpus, past step 1289):
   ERROR SUMMARY: 0 errors.** Ran clean straight through to step 1400 with no illegal access at all.
   Rules out a simple deterministic OOB/bounds bug -- sanitizer's memcheck instruments every device
   memory access and would have caught a real out-of-bounds read/write.
2. **Exact same config rerun at full speed (same seed 42, uninstrumented): completed 3000 steps with
   ZERO cuda errors.** The original crash did not reproduce at all, let alone at the same step. A
   genuine deterministic software bug (fixed seed, fixed data order, fixed kernel code) would be
   expected to reproduce identically every time on the same hardware -- this one didn't.
3. **Windows Event Log correlation: `nvlddmkm` (NVIDIA display driver) Event ID 153, "Error occurred
   on GPUID: 200", fired at 16:12:40 local time** -- within roughly a minute of the original crash's
   tail activity. This is an OS/driver-level fault event, not something an application-level CUDA
   kernel bug would trigger on its own.
4. **Historical frequency: 30 occurrences of this exact event since mid-May 2026**, clustered and
   sporadic (not a one-off) -- a real, recurring characteristic of this machine's GPU under sustained
   load, not a fluke tied to this session's code. Consistent with [[thermal-confounds-ab-wallclock-testing]]'s
   already-documented finding that this RTX 5070 Laptop GPU is power/thermal constrained.

**Conclusion: hardware/driver-level fault (thermal or power/voltage related), not a bug in the
gated-ffn + tied-embeddings + qk-norm + Muon combined CUDA path.** The combined-feature CUDA parity
work does not need a fix; [[cuda-parity-backlog-deferred]] and [[muon-cuda-parity-in-progress]] stand as
previously verified.

**Real gap found regardless of root cause -- FIXED 2026-07-09, see [[gpu-failure-detection-hardening]]
for the full writeup.** At the time of this incident, the training loop had no way to detect that a GPU
step silently failed: `SUB0_CUDA_CHECK_VOID` (backend_cuda.cu) printed the error and `return;`d out of
the CURRENT function, but the train_stage.cpp driver loop had no signal that the step didn't actually
happen -- it just proceeded to the next step, which immediately failed the same way (the CUDA context
stays corrupted for the rest of the process), while the plateau detector ran on stale/garbage loss
values and could declare a confidently-wrong "success." Now fixed: `GpuTrainer::step()`/`sync_to_host()`
propagate real failure, and the training loop stops immediately without touching the device again on
any of the three device-touching points in the loop, live-verified via a temporary forced-failure test.

**Practical mitigation used for the rest of the multi-size sweep:** grep each run's own log for
"cuda error" before trusting its plateau result; discard and retry if found.
