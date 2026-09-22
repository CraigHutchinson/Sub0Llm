# O2 — one GEMV primitive for the bf16 backbone: vectorized, then threaded

**Status:** merged 2026-09-22. **Decode 0.757 → 0.313 s/token (2.42x), bit-identical.** Threading
worked on attempt 2, once B40 (every decode OpenMP team confined to one core) was found and fixed.
**Gates:** `G-HASH` (bit-exact: decode `816c4a54ad49b8cf`), `G-SUITE-*` exact, `G-QUALITY` (L2 unchanged at
0.23252 on the real artifact, by construction), `G-PERF`.

## 1. Context — the profile that justifies this

`../profile_post_o1.md`: at the O1 operating point (0.735 s/token steady state) **~70% of decode is the bf16
backbone GEMV shape** (mixer 46%, router + shared 13.5%, lm_head 11%), and **all of it ran on ONE core**.
Mixer and lm_head reached ~16 GB/s; the router + shared expert only ~6 GB/s. The same session measured the
DRAM ceiling at 30 GB/s single-core but **79 GB/s on 8 P-cores, 91 GB/s all-core**, so the single-core
roof, not the machine's, was what was being approached.

## 2. Problem

The shape `y[o] = sum_i x[i]*W[i*out + o]` (row-major `[in, out]`, input-major) was hand-written at six
sites: `decode.cpp`'s `linear_row`, `qsa::linear_row`, `gdn::forward`'s in/out projections,
`moe::expert_ffn_row`, `moe::router_topk_row`. None was explicitly vectorized (the bf16 weights go through
the `Bf16CPtr` widening proxy), and none was threaded.

## 3. What was built (iteration 1)

`include/sub0/gemv.hpp`: `gemv::axpy<Threads>(x, W, in, out, y)`, used at all six sites.

- **Vector path** for `const float*` and `Bf16CPtr`: bf16 widened by zero-extend and shift (which IS
  `bf16_widen`); **four weight rows per pass** into an 8-lane `y` group (one load and one store of `y` per
  four FMAs), each row read contiguously. A first cut tiled 32 outputs in registers across ALL rows, which
  would have touched one 64-byte line per row at a stride of `out*2` bytes (a new page, and likely a TLB
  miss, per access). It was caught in review and never measured.
- **Bit-exact by construction:** each output keeps its own sequential sum over `i` with one FMA per term
  (the scalar loops already contracted to FMA; verified: 14 `vfmadd`, 0 `vmulps`/`vaddps`). Lanes and
  threads split only across outputs. Confirmed: decode hash `816c4a54ad49b8cf` unchanged, suites exact
  (28,969,623 / 147; 141,609 / 257), and real-artifact L2 exactly 0.23252 at every thread count.
- **Threads:** `sub0llm-configure --decode-gemv-threads N` (default 1). A template argument, so only
  decode passes it; the batched `forward()` and every test keep 1.

## 4. Results so far

Steady state (default path, 6 tokens, profiled), one run per point:

| `--decode-gemv-threads` | s/token | mixer | router + shared | lm_head |
|---|---:|---:|---:|---:|
| (pre-O2) | 0.735 | 338 | 99 | 80 |
| **1** | **0.543** | **256** | **27** | **52** |
| 4 | 0.741 | 369 | 58 | 69 |
| 8 | 0.783 | 407 | 67 | 67 |
| 24 | 0.879 | 516 | 81 | 62 |

- **Vectorized, one thread: −26% in this single-run sweep; −14% by the interleaved A/B (§6)**, which is
  the number to quote. The router and shared expert are 3.7x faster.
- **Threaded: worse, and monotonically worse with more threads**, against a bandwidth measurement that
  says the opposite. That makes it a scheduling problem, not a bandwidth one. Candidates, to be separated
  by `sub0llm-bench-gemv` rather than by the decode: OpenMP fork/join or wake latency over ~300 parallel
  regions per token; unpinned threads on E-cores (a static split runs at the pace of its slowest thread);
  short strided per-thread segments on the small projections.

## 5. Threading attempt 2: the cause was placement, not bandwidth (B40)

`sub0llm-bench-gemv` (the real shapes, in isolation) showed threading **working**: GDN `in_qkv` 20 → 69
GB/s and lm_head 21.5 → 68 GB/s at 8 threads. So the primitive was sound, and something in the decode
process was different. It was `ensure_thread_built()`: it pinned the main thread to ONE logical CPU before
any OpenMP region, and libomp places every team inside its master's affinity mask. Reproduced in the bench
by pinning the master the same way: 8 threads at 15.7 GB/s, slower than one thread. Initializing OpenMP
first does not help. Pinning the master to the P-core **set** restores 51–62 GB/s. Fixed in `backend.cpp`
with `cpu_affinity.hpp`'s `pin_current_thread_p_set`. The same artefact explains B21 and B29 (see B40 in
`../../INDEPENDENT_REVIEW_BACKLOG.md`).

With the fix (steady state, 6 tokens, profiled, one run per point):

| config | decode s/token | mixer | routed experts | lm_head | GR |
|---|---:|---:|---:|---:|---:|
| GEMV 1, MoE 1 | 0.584 | 278 | 148 | 59 | 70 |
| GEMV 8, MoE 1 | 0.470 | 156 | 182 | 21 | 84 |
| GEMV 8, MoE 2 | 0.395 | 155 | 94 | 25 | 92 |
| GEMV 8, MoE 5 | 0.327 | 148 | 49 | 22 | 82 |
| **GEMV 8, MoE 10** | **0.314** | 148 | **39** | 25 | 79 |

L2 is 0.23252 at every point (bit-identical).

## 6. The gate: interleaved A/B, sandboxed, 3 runs each (3 tokens; quality at 6)

| arm | median s/token | runs | L2 | argmax |
|---|---:|---|---:|---:|
| pre-O2 (`4ae267a`, fused) | 0.757 | 0.688, 0.777, 0.757 | 0.23252 | 4/6 |
| vectorized only (GEMV 1, MoE 1) | 0.593 | 0.560, 0.593, 0.593 | 0.23252 | 4/6 |
| **O2 final (GEMV 8, MoE 10)** | **0.313** | 0.274, 0.324, 0.313 | **0.23252** | 4/6 |

(The vectorized-only row is from its own A/B, whose pre-O2 arm read 0.690.) G-HASH `816c4a54ad49b8cf`,
suites 28,969,623 / 147 and 141,609 / 257, all unchanged.

**Recommended real-axes decode flags:** `--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 10` (O3 adds `--decode-omp-spin 1`).
Both thread axes stay default 1 in the configurator, because they are machine-shaped (P-core count).

## 7. Next constraint

At 0.314 s/token the split is mixer 148 ms (47%), **Gated Residual 79 ms (25%, still serial)**, routed
experts 39, lm_head 25, router + shared 22. Against the ~100 ms bandwidth floor (~8 GB/token at
79 GB/s) that is ~3x left. GR and the mixer's serial parts (GDN recurrence, QSA attention) are next; GR
is latency-side, so re-profile its internals before choosing a lever.
