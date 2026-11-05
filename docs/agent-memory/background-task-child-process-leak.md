---
name: background-task-child-process-leak
description: "MEASUREMENT HAZARD (2026-07-27): stopping a background shell does NOT kill sub0llm-train children it launched with `&`. Two leaked trainers (32 GB working set each, on a 63 GB machine) silently contaminated later throughput measurements and produced two published-then-retracted diagnoses. Assert the process count before trusting any GPU/throughput number."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-27T22:17:47.824Z
---

While gating a fineweb_edu run I stopped a batch-sweep script mid-flight. The script launched each
trainer as `./sub0llm-train.exe ... &`, and stopping it killed only the bash wrapper. **Two
`sub0llm-train.exe` processes stayed alive**, each with a ~32 GB working set on a 63.4 GB machine (free
RAM had fallen to 0.3 GB), both holding the GPU.

Every measurement taken afterwards was contended and therefore meaningless. I published two confident
diagnoses off that contaminated data before catching it:

1. "Sustained throughput on the on-demand corpus path is 20-60K tok/s, not 240K" -- attributed to
   shuffle-buffer exhaustion and disk-bound refills.
2. "Pre-tokenization did not fix it, so the bottleneck is random access into a 27.5 GB token file
   exceeding page cache" -- attributed to RAM pressure.

Diagnosis 2 was measured against a leaked concurrent trainer and is void. Diagnosis 1 may still hold
(that probe appears to have run alone) but cannot be asserted until re-measured clean. Both were
plausible, internally consistent and reached by real arithmetic -- which is exactly why the
contamination was not obvious from the numbers.

**Why:** a shell's job control does not propagate a kill to grandchildren; the harness stops the command
it launched, not the process tree beneath it. GPU work is especially exposed because two trainers do not
error -- they quietly halve each other and look exactly like a performance cliff.

**How to apply:**
- Before ANY throughput/VRAM measurement, assert the expected process count and an idle GPU:
  `powershell -NoProfile -Command "(Get-Process sub0llm-train -ErrorAction SilentlyContinue | Measure-Object).Count"`
  plus `nvidia-smi` showing ~14 MiB / 0-1%.
- After stopping any background script that launched trainers, kill the tree explicitly:
  `Get-Process sub0llm-train | Stop-Process -Force`, then re-check free RAM.
- Prefer scripts that run trainers in the FOREGROUND sequentially over `&` + `wait`, so a stopped script
  cannot leave orphans.
- Treat "throughput degraded over time" as a contamination signal FIRST and a pipeline finding second.
  A real pipeline limit and a leaked competitor are indistinguishable in a tok/s trace.

Related: [[thermal-confounds-ab-wallclock-testing]] (the other reason a wall-clock A/B on this machine
lies) and [[check-captured-logs-before-process-archaeology]].
