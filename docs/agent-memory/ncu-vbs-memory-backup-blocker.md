---
name: ncu-vbs-memory-backup-blocker
description: Nsight Compute (ncu) fails with "driver resource unavailable" against the production d768 model but works fine on smaller models -- root cause is VBS blocking ncu's system-memory-backed kernel-replay fallback, not VBS blocking counters outright
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Root cause, isolated 2026-07-02 via bisection (not guessed):** ncu profiling `sub0llm-train.exe` against
the production fineweb_edu model (d768, ~5GB resident VRAM) reliably fails with `Failed to create counter
availability image (error = 20)` / "driver resource was unavailable" -- but the EXACT SAME command against a
minimal CUDA vector_add sample, AND against the first ~80 kernels of a full production-size training run,
succeeds cleanly. This ruled out the earlier working hypothesis (Windows VBS/Memory Integrity blocking GPU
performance counters outright, which was the original diagnosis before this session's deeper dig).

**What's actually happening:** ncu's kernel-replay mode needs to snapshot/restore the GPU memory a target
kernel touches, across the multiple passes full metric collection requires. Once free VRAM gets tight
enough (this app pre-allocates its ENTIRE ~5GB param+activation arena upfront -- zero-alloc design, no
incremental growth), ncu can't find enough ON-DEVICE room for that snapshot and falls back to staging
through SYSTEM RAM -- printing `Backing up device memory in system memory. Kernel replay might be slow.`
right before the failure. That specific host-staged fallback path is what VBS/HVCI blocks -- not the
counter-collection mechanism itself. Confirmed by: (1) `--replay-mode application` (ncu's own suggested
workaround, avoids the fallback path entirely by re-running the whole app instead of snapshotting memory)
STILL failed at the same point -- so it's not purely about kernel-replay's memory backup; (2) a fresh,
non-resumed run capped at just 3 steps on the SAME production model reproduced the identical failure at the
identical kernel (`embed_act_kernel`, launch #81) -- ruling OUT step count / resume state as the variable,
leaving VRAM footprint as the one thing that actually differs from the small-model case.

**Practical fix: use a smaller model for ncu work.** A "medium" reference model (~1.1GB resident, d448/L11/
H7) profiles every kernel cleanly with `--set full`, including the exact ones that fail on production.
See [[cuda-kernel-optimization-waves]] for the model that was built and kept around for this purpose --
prefer reusing it over rebuilding a new one each time. The mechanism (kernel shape/stall pattern) transfers
even though absolute numbers won't scale 1:1 to d768.

**Dead end explored and ruled out:** `ncu.bat` vs `ncu.exe` (direct exe invocation) made no difference --
same failure either way, so it's not a wrapper/quoting issue. A `|`-alternation regex in `-k` DOES break
when handed to `ncu.bat` (goes through cmd.exe's parser, which splits on the pipe even inside quotes) --
unrelated to the VBS issue, just run one `-k` capture per kernel name instead of trying to combine them.
