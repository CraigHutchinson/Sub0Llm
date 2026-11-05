---
name: host-cpu-arrow-lake-hx
description: "The dev/training machine is an Intel Core Ultra 9 275HX — 24 physical cores (8 P + 16 E), no SMT"
metadata: 
  node_type: memory
  type: reference
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

The training/dev box CPU is an **Intel Core Ultra 9 275HX** (Arrow Lake-HX, TSMC N3B), verified against Intel ARK + Windows `Win32_Processor`:

- **24 physical cores = 8 P-cores + 16 E-cores; 24 threads (NO Hyper-Threading)** — every logical CPU is its own core.
- P-core: 2.7 GHz base / 5.4 GHz turbo, 3 MB L2 each. E-core: 2.1 / 4.6 GHz, 4×4 MB shared L2.
- **L3 (Smart Cache) = 36 MB**, L2 total = 40 MB. DDR5-6400 (~100 GB/s dual-channel). TDP 55 W.

Consequences for threading work:
- `detect_cpu_topology()` reports perf=8, efficiency=16, no SMT siblings. Compute-bound pools can use ALL 24 cores (P-cores first via [[cpu-threading-model]]), not just the 8 P-cores — don't cap sweeps/defaults at `n_perf_cores()`.
- Total physical cores = `n_perf_cores() + eff_primary.size()` = 24 here.
- 36 MB L3 sizing matters: per-worker weight replicas that sum past ~36 MB spill to DRAM and make the data-parallel trainer memory-bandwidth-bound — share read-only weights instead (one L3-resident copy). See [[cpu-threading-model]].
- Sustained AVX load thermally throttles; benchmark with warmup + best-of-N interleaved, not a single cool→hot sweep.

**DRAM bandwidth, measured 2026-09-22 (post-PSU swap), streaming AVX2 read, 2 GiB:** 1 P-core 30.2 GB/s,
2 → 49.9, 4 → 70.9, 8 P-cores **79.0**, 16 → 90.7, 24 → **91.2 GB/s**. The "~30 GB/s ceiling" in
BACKBONE_PRECISION.md §2c and every roofline B27-O1 is the SINGLE-CORE number — READ FIRST before any
roofline claim. Decode's bf16 backbone GEMVs ran single-threaded, so they could never see the real ceiling.
Details: docs/optimization/profile_post_o1.md.
