---
name: b21-decode-concurrency-plateau
description: "CLOSED by B29 -- Sub0Llm's ParallelExperts decode-thread ceiling was real DRAM-bandwidth contention, not scheduling; fix was fewer threads (pinned to 1), not more"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-11T09:37:38.252Z
---

**CLOSED 2026-09-11 by B29** (`docs/INDEPENDENT_REVIEW_BACKLOG.md`), merged to `main` `b79c85c`,
independently reverified. Root cause resolved, `MOE_DECODE_THREADS` (`src/backends/cpu/internal.hpp`)
pinned to compile-time `1`, decode ~2-9% faster as a direct result.

**Root cause, finally decisive**: real DRAM-bandwidth contention. One resolve
(`moeq::ExpertCache::resolve` → dequant → 2x transpose → FFN) moves ~80.2 MB of real memory traffic per
expert (real byte accounting from `benchmarks/moe_expert_bench.cpp` against the real 48-layer sidecar) in
~5.5-5.6 ms — a SINGLE thread already sustains ~14.5-14.6 GB/s, i.e. ~44-52% of this host's own
independently-measured ~28-33 GB/s real achievable DRAM bandwidth ceiling
([[backbone-precision-staged-bf16-then-native-quant]]/`docs/BACKBONE_PRECISION.md` §2c). Post-B28 (the
`dequantize_iq2_xxs` fix, see that memory) live-engine wall-clock re-instrumentation found the 10
requested `ParallelExperts` threads run FULLY SERIALIZED in steady state — zero measured overlap, not the
earlier partial ~6-then-2-3 plateau this file originally described (that earlier finding was itself
confounded by IQ2_XXS's now-fixed 3-9x per-op cost inflation, which varied unpredictably by which format a
given thread happened to draw). Direct A/B on the real model confirmed fewer threads is not merely
"no worse" but measurably FASTER (10 threads 3.75-3.79 s/token vs. 1 thread 3.42-3.47 s/token, ~9%,
reproduced independently at a smaller ~1.8% margin) — concurrent large sequential DRAM streams thrash
row-buffer locality worse than proportional bandwidth-sharing would predict. Correctness unaffected at any
thread count (`forward_row_via_run` combines in original selection order regardless of completion order,
confirmed bit-exact/parity-0 at 1/3/10 threads).

**This session's combined MoE-decode throughput result** (B27 diagnosis → B28 dequant fix → B29 thread
fix, all independently reverified before merging): established baseline ~5.5-6.0 s/token → ~3.57-3.67
s/token, a real ~35-40% reduction. See `docs/INDEPENDENT_REVIEW_BACKLOG.md` B27/B28/B29/B30/B31 for full
detail — B30 is the closing synthesis (decode now sits close to, not exactly at, the real bandwidth-bound
floor; no more free/cheap levers of this kind remain) and B31 names the one real remaining structural
lever (fusing the resolve pipeline's ~3 redundant DRAM round-trips into fewer — a genuine redesign, not
started, scoped Large).

## CORRECTION 2026-09-22 (B40) — the B21/B29 conclusion was a measurement artefact
ensure_thread_built() pinned the MAIN thread to one logical CPU before any OpenMP region; LLVM libomp places
every team inside its master's affinity mask, so decode's "10 threads" all ran time-sliced on ONE core. That IS
the "zero overlap, back-to-back" signature, and why more threads were slower. Not DRAM contention.
Fix: initial thread pinned to the P-core SET (cpu_affinity.hpp pin_current_thread_p_set). After the fix:
routed experts 185 -> 39 ms at --moe-decode-threads 10; decode 0.474 -> 0.314 s/token; bit-identical.
Lesson: a team that doesn't scale is a placement question before a bandwidth question — check the master's mask.
