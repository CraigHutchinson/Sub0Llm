---
name: cpu-threading-model
description: "One shared CPU topology module; data-parallel training pins P-cores-first, default P-cores/2"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

P/E core detection + thread pinning live in ONE place: `include/sub0llm/core/cpu_topology.hpp`
(`detect_cpu_topology`, `pin_current_thread`, `compute_pin_set`). `gemma_detect_cores()` delegates
to it. `diffusion/src/host_detect.cpp` keeps its own superset enumeration (physical cores + cache +
RAM for the constexpr HostSpec emitter) but its P/E rule (max EfficiencyClass == perf) MUST stay
consistent with cpu_topology so a specialized binary's `-verify` agrees with runtime pin sets.

Data-parallel training: `diffusion/include/sub0diff/train/parallel.hpp` ParallelTrainer — W worker
replicas, barrier-phased SYNC/COMPUTE/REDUCE, per-worker autograd+RNG+ctx (lockless hot path),
mean-gradient reduction into master. Also the effective-batch convergence lever. ch29 `--threads`
(0=auto). ch29b target = same source, 2nd name, to build/test while a long run holds the exe lock.

**Measured (2026-06-13, 857K model, concurrent full run active):** 4 pinned P-core workers
~6500 masked-tok/s vs 3344 serial (~1.95×); held-out NELBO falls monotonically (averaging correct).
8 workers REGRESSED to 3500 — contends with the in-flight training. **Why default = P-cores/2:**
compute-bound workers on E-cores/SMT siblings contend more than they contribute, and /2 leaves room
for a concurrent run. For a DEDICATED machine, raise `--threads` toward the P-core count.
Constexpr goal: monomorphized build folds HostSpec core counts at compile time, detection only in
`-verify`. See [[carry-forward-chapter-niceties]], [[optimization-target-is-hardware-ceiling]].
