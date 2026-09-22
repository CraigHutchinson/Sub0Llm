# Profile and roofline at the O1 operating point — and the DRAM ceiling correction

**Measured 2026-09-22**, real 48-layer BF16 artifact, `--moe-quant-dot 1`, O1 AVX2 kernel, 6 tokens,
background load 4-5%. Taken with the permanent profiler (`sub0llm-configure --profile-phases 1`;
`include/sub0/phase_profile.hpp`). It replaced the hand-written scaffold the post-B35 profile needed.

## 1. Two profiles, because first-touch and steady state differ

The engine memory-maps the 37 GiB sidecar. In a fresh process, the first touch of every expert page is a
**soft page fault**, even when the page is already in RAM, and Windows services those faults serially
per process.

| | first-touch (`--decode-only`) | steady state (default path) |
|---|---:|---:|
| decode s/token | 0.80–0.90 | **0.735** |
| page faults during decode | 128 K/token (237 K on token 1) | ~13 K/token |
| MoE: routed experts | 267 ms | **144 ms** |

The default path runs `forward()` first, which faults in exactly the expert pages decode then reads, so
its `forward_one` loop is the **steady state** that long generation converges to as experts are reused.
First-touch faults cost ~0.1–0.2 s/token. They also explain why `--moe-decode-threads 8` bought nothing
under `--decode-only` (1 thread 268 ms, 8 threads 277 ms): the work being serialised was the kernel's
fault handling. Pre-faulting the sidecar at load is an open lever for first-token latency (not briefed).

## 2. Steady-state phase split

| Phase | ms/token | Share | Bytes/token (from shapes) | Achieved |
|---|---:|---:|---:|---:|
| **Mixer** (36 GDN + 12 QSA, bf16) | **338** | **45.9%** | ~5.3 GB | ~15.8 GB/s |
| MoE: routed experts (O1 kernel) | 144 | 19.5% | ~0.78 GB | ~5.4 GB/s |
| MoE: router + shared + combine (bf16) | 99 | 13.5% | ~0.6 GB | ~6 GB/s |
| lm_head (bf16, 248,320 × 2,560) | 80 | 10.9% | 1.27 GB | ~15.9 GB/s |
| Gated Residual | 75 | 10.2% | small | latency-side |
| unattributed (the parity comparison in the timed loop) | 0.6 | 0.1% | | |

The routed-expert kernel now runs at its microbenchmark speed; it is no longer the bottleneck.
**~70% of decode is the bf16 backbone GEMV path, and all of it runs on ONE core.**

## 3. The ceiling correction

Streaming-read bandwidth, AVX2 loads, 2 GiB buffer, best of 3 (scratch `bw.cpp`):

| Threads | GB/s |
|---|---:|
| 1 (P-core) | **30.2** |
| 2 | 49.9 |
| 4 | 70.9 |
| 8 (all P-cores) | **79.0** |
| 16 | 90.7 |
| 24 | **91.2** |

**The "~30 GB/s DRAM ceiling" used since B27 (`BACKBONE_PRECISION.md` §2c) is the SINGLE-CORE
figure.** The machine sustains ~79 GB/s on its P-cores and ~91 GB/s on all cores. Every roofline in this
thread measured decode against the wrong denominator. The error was invisible because decode itself only
ever used one core for the backbone.

At ~8 GB of weight traffic per token, the bandwidth floor is **~90–100 ms/token**. Decode at 735 ms is
~7–8x from it. The goal is not met, and the gap is structural (O0 in the layered review): a
single-threaded backbone cannot exceed 30 GB/s however good its kernel is.

## 4. Next levers, in order

1. **O2 — threaded backbone GEMV.** Split each projection's output columns across the P-cores. Every
   output element keeps its own sequential sum, so the result is bit-exact and the `forward`/`forward_one`
   parity check stays at exactly 0. Targets the mixer, lm_head, shared expert and router (~70%).
2. **Vectorized bf16 GEMV** (widen by shift, FMA into vector accumulators). A single core reaches only
   ~16 of its 30 GB/s today. It compounds with O2.
3. **FP8 backbone (B33/B37, parked).** Halves backbone bytes. Parked when widening was the cost on a
   single core; with threads and AVX2 widening the balance may change (§13: 1 of 3 attempts spent).
4. **Routed experts across threads** (`--moe-decode-threads`), re-measured at STEADY state. The
   first-touch sweep above says nothing about it.
5. Pre-fault the sidecar at load (first-token latency only).
