# O10 -- sub-phase roofline of decode on the real Qwen4 48-layer model

**Status:** measurement and analysis only. No lever is implemented; the profiling hooks are the only code change.
**Measured:** 2026-09-29, host Arrow Lake-HX (8P+16E, AVX2/AVX-VNNI), quiet (no clang/ninja/sub0llm processes at
start of each series), `--tokens 6`, default real-axes build plus `--profile-phases 1` (native backbone, O8 row split,
O9 act-super, 8 GEMV threads, 8 expert threads, OMP spin).

## 1. What was added

`prof::Phase` gained 18 sub-phases. They are exclusive-time, like the existing phases, and compile to nothing when
`PROFILE_PHASES` is false.

| Region | Mechanism | Why |
|---|---|---|
| GDN stages (in-quant, in GEMV, b/a, conv, recurrence, gated norm, out gather+quant, out GEMV) | `gdn::forward` takes a trailing `Probe` template parameter (default `NoStageProbe`) and calls `probe(gdn::Stage::X)` at each stage start | The stages live inside one engine-free `*_math.hpp` function; timing at the decode.cpp call boundary could only see the whole call |
| GR `mix` stages (quantize, down, elementwise, up) | Same, `gr::MixStage` | Same |
| GR hc_norm, inject gate, combine; QSA indexer, q/k/v projections, attention+o-proj | `PhaseScope` at the decode.cpp call boundaries | These are separate calls already, so no hook is needed |

`NoStageProbe` (`include/sub0/stage_probe.hpp`) is an empty type with a `constexpr` no-op call operator, so a default
build and every other caller (batched `forward()`, CUDA reference, tests) compile the marks away. The profiling
implementation is `prof::StageProbe<Stage, N>`, a stage-to-phase map that calls `switch_to`. The probe is passed by
value; nothing is allocated (AGENTS.md section 1).

Gates for the diff: neutral `sub0llm_tests` **29,510,661 / 147**, fingerprints `5a7382ea70d3913b` / `7f44bdae18c313dd` /
`d1625d19ed2258f1`; `sub0llm_frontend_tests` **230,939 / 300** (both unchanged, no cases added). `--stage ppl` on the real
build: **ppl 14.7035** without the profiler and **14.7035 (identical mean NLL 2.6881, top-1 0.4876)** with
`--profile-phases 1`, so the hooks are value-neutral. Long-run decode over the ppl text: 6.86 tok/s without, 6.95 with (noise; no
visible profiler cost).

## 2. Warm phase table (ms/token, mean of the two warm runs)

Run 1 of each series is discarded as cold. Start times (local): first series 11:27:53 / 11:28:44 / 11:29:27 (coarse
table); final-build series 11:33:59 (discarded) / 11:34:45 / 11:35:31. Run 2 and 3 agree within 0.3 ms per row.
Total 0.114-0.116 s/token (about 8.7 tok/s short-run). Positions 0-5 only (see caveats).

| Group | Sub-phase | ms/token | % |
|---|---|---:|---:|
| GDN (33.2) | in-proj act quantize | 0.2 | 0.2 |
| | in-proj qkv+z GEMV | 19.45 | 17 |
| | in-proj b,a (+ sigmoid/softplus) | 0.95 | 0.8 |
| | conv1d + SiLU | 1.0 | 0.9 |
| | recurrence (state update) | 2.2 | 1.9 |
| | gated RMSNorm | 0.2 | 0.2 |
| | out-proj gather+quantize | 0.6 | 0.5 |
| | out-proj GEMV | 8.55 | 7.5 |
| Gated Residual (20.2) | hc_norm | 1.1 | 1.0 |
| | act quantize (down+up) | 0.5 | 0.4 |
| | down GEMV | 7.95 | 6.9 |
| | up GEMV | 7.4 | 6.5 |
| | mix elementwise (silu, sigmoid, sum) | 1.9 | 1.7 |
| | inject gate | 1.2 | 1.0 |
| | combine (write) | 0.1 | 0.1 |
| QSA (18.1) | indexer (project + select) | 9.45 | 8.2 |
| | q|gate/k/v projections | 5.55 | 4.8 |
| | attention + o-proj | 3.05 | 2.7 |
| Other | MoE routed experts | 28.15 | 24.5 |
| | MoE router + shared + combine | 7.85 | 6.8 |
| | lm_head (+ final norm) | 6.5 | 5.7 |
| | unattributed | 0.8 | 0.7 |

GDN 33.2 + GR 20.2 + QSA 18.1 match the earlier coarse phases (36 / 21 / 19). Attribution is complete (0.7 % unattributed).

## 3. Roofline

Roof: 79 GB/s DRAM across the P-cores (91 all-core; 30 GB/s from one core). Bytes are decimal MB per token, from the
real formats (section 13c of `BACKBONE_NATIVE_QUANT.md`: GDN in_qkv/in_z Q5_K, GDN out Q6_K, GR Q8_0, QSA q|gate/k/v/o
Q5_K, lm_head Q4_K) and shapes (d 2560, 36 GDN layers, 12 QSA layers, 96 GR calls = attn + ffn x 48 layers).
Bytes per element: Q5_K 176/256 = 0.6875, Q6_K 210/256 = 0.8203, Q8_0 34/32 = 1.0625, Q4_K 144/256 = 0.5625, bf16 2.

| Sub-phase | Arithmetic | MB/token | ms | GB/s | % of 79 | Floor at 79 GB/s |
|---|---|---:|---:|---:|---:|---:|
| GDN in qkv+z | 36 x (10240 + 6144) x 2560 el x 0.6875 = 36 x 28.84 | 1038 | 19.45 | 53.4 | 68 | 13.1 ms |
| GDN out-proj | 36 x 2560 x 6144 x 0.8203 = 36 x 12.90 | 464 | 8.55 | 54.3 | 69 | 5.9 |
| GR down (Q8_0) | 96 x 10240 x 320 x 1.0625 = 96 x 3.48 | 334 | 7.95 | 42.0 | 53 | 4.2 |
| GR up (Q8_0) | 96 x 320 x 10240 x 1.0625 | 334 | 7.4 | 45.2 | 57 | 4.2 |
| QSA q|gate,k,v | 12 x (12288 + 512 + 512) x 2560 x 0.6875 = 12 x 23.43 | 281 | 5.55 | 50.7 | 64 | 3.6 |
| QSA o-proj (inside the 3.05 ms attn phase) | 12 x 2560 x 6144 x 0.6875 = 12 x 10.81 | 130 | <= 3.05 | >= 42.5 | -- | 1.6 |
| lm_head | 248320 x 2560 x 0.5625 | 358 | 6.5 | 55.0 | 70 | 4.5 |
| **QSA indexer projection** | 12 x 640 x 2560 x 2 B (bf16) | 39 | 9.45 | **4.2** | **5** | 0.5 |
| GDN b,a | 36 x 2 x 2560 x 48 x 2 B | 17.7 | 0.95 | 18.6 | -- | 0.2 |
| GDN recurrence | 36 x 48 x 128 x 128 x 4 B, read + write | 227 | 2.2 | ~103 | >100 | 2.9 |

Aggregate over the seven native GEMV rows plus the o-proj: 2,940 MB in about 58.5 ms = 50 GB/s, against a 37 ms floor.

Bandwidth-bound versus not:
- **Bandwidth-bound, near the practical plateau:** the K-quant GEMVs (53-55 GB/s) and the recurrence. The recurrence
  streams 3.1 MB of f32 state per GDN layer, read and written; the 103 GB/s figure exceeds the read-only roof, so part of the state must be
  served from cache. It cannot go faster except by shrinking the state (changes numerics).
- **Not bandwidth-bound:** QSA indexer projection (4 GB/s), GR Q8_0 GEMVs (42-45 GB/s, below the K-quants on the same host), and
  every small serial op below.
- **Quantize is negligible:** GDN in 0.2 + GDN out 0.6 + GR 0.5 = 1.3 ms (1.1 %).
- **Serial elementwise around GR and GDN:** hc_norm 1.1 + mix elementwise 1.9 + inject gate 1.2 (96 calls each, 11-20 us per call, all single-threaded) = 4.2 ms; GDN b,a 0.95 and conv 1.0 add ~2.
- **OpenMP fork/join:** counted **771 parallel regions per token** (temporary atomic counter in each `omp parallel` site, 4,626 regions over 6
  `forward_one` calls; counter removed, not committed). Empty 8-thread region cost measured in isolation: **19 us with libomp's default blocktime,
  0.8-1.3 us with infinite spin** (this build has `DECODE_OMP_SPIN` on). So pure fork/join is about 771 x 1.3 us = **~1 ms/token (0.9 %)**;
  it would have been ~14.6 ms/token without spin. The residual cost of small regions is tail imbalance and serial gaps, which shows as the 13-28 us per
  region of the tiny GDN b,a and conv phases.
- **More GEMV threads do not help.** Rebuilt with `--decode-gemv-threads 12` and `16` (3 runs each, first discarded): in-proj 20.1 / 19.2 ms vs 19.45 at 8; GR down 9.5-10.2 / 9.1-10.0 vs 7.95; total 0.127 s/token for both, worse than 0.115 at 8 (router+shared rose 7.85 to 10.5-11.1 ms). The GEMVs are on a per-run plateau of ~53 GB/s that thread count does not move.
  Start times 11:37:46-11:39:17 (12) and 11:40:11-11:41:39 (16).

## 4. Ranked levers (expected ms/token saved)

1. **Thread the QSA indexer projection: about -8.5 ms (7 %).** `qsa::indexer_project_row` is a serial bf16 axpy
   (single thread, no `Threads` parameter, called as `indexer_project_row<USE_SIMD_REDUCE>` in decode.cpp). It reads 3.3 MB per QSA layer at 4 GB/s
   when one core alone can do 30 GB/s and eight can do 79. Route it through `gemv::axpy<DECODE_GEMV_THREADS>` (the same primitive the other bf16 projections use, documented bit-exact:
   same i-ascending sum per output) and it should drop to ~1 ms. Evidence: 9.45 ms for 39 MB. Cheapest and highest-value item; expected bit-exact.
   Caveat: the 9.45 ms also contains `indexer_select_row`, which is negligible at 6 positions (at most one pooled block). A split of the two scopes at long context is not measured.
2. **A faster Q8_0 kernel for Gated Residual: about -3 to -6 ms.** GR is 15.35 ms of GEMV at 42-45 GB/s, below the K-quants (53-55) on the
   same host, and O9's per-256 fold refuses Q8_0 (`super_fusable`). `gemv_rows<Q8_0Plane,true>` materialises a `WeightGroup` copy and does one
   `hsum256_epi32` plus scalar float FMAs per 32 weights (the earlier bench measured native Q8_0 at 35.6 GB/s at 8 threads vs bf16 axpy at 54). A llama.cpp-style kernel
   (sign trick with AVX-VNNI `dpbusd`, float vector accumulator across blocks, one hsum per row, no group copy) removes that per-block cost. At K-quant efficiency (54 GB/s) the saving is 3.4 ms; at 70 GB/s, 6.2 ms.
   The up projection (10240 rows of only 320 elements) also pays a per-row epilogue, so it needs a tiled row loop. Not bit-exact with the current float accumulation order, so it needs its own G-PPL.
3. **Thread or fuse the small serial ops: about -4 to -5 ms.** hc_norm (1.1), mix elementwise (1.9: ~10.5k `exp` per call, scalar), inject gate (1.2), GDN b,a (0.95), conv (1.0, scalar `exp` in SiLU over 10240 channels).
   With ~1.3 us fork cost against 11-28 us of work per call, 8 threads or vectorised exp should cut each by 60-80 %. Alternative: fold hc_norm's scale into the down-projection's activation quantize (one pass fewer over the 10240-wide vector).
4. **K-quant GEMV plateau at ~53 GB/s (up to -21 ms if the whole 79 GB/s roof were reachable): unproven, needs a benchmark first.** 12 and 16 threads give nothing, so the limit is not thread count.
   Candidates to test with `sub0llm-bench-gemv` on the sidecar's actual memory: TLB reach (the sidecar sits in 4 KB pages; large pages need a privilege), software prefetch distance, and whether the sidecar stream competes with the routed-expert mapping. Expect a fraction of the gap at best; do not budget the full 21 ms.
5. **Fork/join fusion: -1 to -2 ms.** Pure fork/join is ~1 ms with spin; fusing b,a into the qkv/z region (they share the input) and conv+recurrence+gate-norm per head across one region would remove ~3 x 36 regions and their tails. Small; do it only as a by-product of item 3.
6. **Recurrence state in bf16/fp16: -1.1 ms, changes numerics.** Bandwidth-bound at the f32 state size; not worth the quality risk yet.
7. **Overlapping independent GEMVs within a layer: no expected gain.** Bandwidth is not saturated at 8 threads yet 12-16 threads (equivalent extra memory-level parallelism) gain nothing, so concurrency between GEMVs is unlikely to. Not recommended.

Order of the tail of the token, outside this brief's scope but the largest single bucket: **routed experts 28.15 ms and router+shared 7.85 ms** (43.3 ms with lm_head) are 31 % of the token. Byte counts here are unverified (the .moeq averages 1.62 MB/expert-layer, giving 780 MB at 10 experts x 48 layers, i.e. 28 GB/s; O7 quotes ~0.25 GB), but either figure is far below the roof: O7 territory.

Best case if items 1-3 land: 114.7 - (8.5 + 4 + 4.5) = about 97.5 ms/token, roughly 10 tok/s short-run.

## 5. Unverified and caveats

- **Short context only.** Six positions: QSA attention and indexer select are near zero. The long-run ppl decode is 6.9 tok/s (about 144 ms/token) against 8.7 tok/s (115 ms) here, so ~29 ms/token of long-run cost is not in this table (attention, select, page faults over 2,418 positions). The attempt to take a 120-position table failed (both runs launched together and did not produce output). The phase table needs a long-context run before the QSA rows are trusted for long generations.
- The indexer scope contains both `indexer_project_row` and `indexer_select_row`; the "select is negligible here" statement is inferred from the position count, not separately timed.
- The recurrence 103 GB/s exceeds the 79-91 GB/s measured roofs; cache residency of part of the state is the likely reason but not measured.
- Region count (771) is one process, one 6-token run; MoE team regions are counted as one each.
- Fork/join microbenchmark was a standalone empty loop, not the in-situ cost; imbalance tails are inferred from the small phases' per-call times.
- Expert-phase byte counts (above) are unverified.
- The build for the coarse table (first series) and the final table differ only by the quantize/GEMV split; both agree.

## 6. Follow-ups

Next brief: O11 = thread the indexer projection (lever 1), measured with the O10 table as its baseline and `--stage ppl` for value neutrality (expected bit-identical). Then lever 2 (Q8_0) and lever 3 together, per the combination-matrix rule (AGENTS.md section 13).

## Lever 1 done: O11, threaded QSA indexer projection (2026-09-29, primary agent)

`qsa::indexer_project_row` now takes `Threads` and a caller-owned `qk_buf` (`idx_qk_out()` floats). It
projects the whole q|k tensor with one `gemv::axpy<Threads>` call, then copies the query and key parts out.
Decode passes `DECODE_GEMV_THREADS` and a `thread_local` buffer. The batched `qsa::forward` takes its buffer
from `scratch_floats`, which grew by `idx_qk_out()`. There were three test call sites; the first build missed
them because the consumer grep covered `include/` and `src/` only.

- **Bit-exact.** The real-model G-PPL gives ppl 14.7035, identical. Neutral: `sub0llm_tests` 29,510,661 / 147
  with all fingerprints unchanged; `sub0llm_frontend_tests` 230,939 / 300, which covers the QSA fixture tests.
- **Phase.** Warm `--tokens 6` profiles give `QSA: indexer` **9.45 -> 0.8-1.2 ms/token**. The same session
  measured other phases 5-10% slower than this doc's baseline (GDN in-proj 21.7-23.7 against 19.45 ms), and
  totals of 0.121-0.133 s/token. The host was in its slow state, so no absolute tok/s claim is made from
  that session. The phase delta is the measured result.
