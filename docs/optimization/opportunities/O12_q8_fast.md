# O12 -- vector-accumulator Q8_0 GEMV for the native backbone (`--backbone-q8-fast 1`)

Status: **merged, default ON (auto) for real-axes builds since 2026-10-07.** G-PPL PASS on `ppl_blend_v2`
(S7); the Gated Residual up projection and the shared expert's down projection are 11-17% faster and
long-run decode about 4% (S8). Not bit-exact. `--backbone-q8-fast 0` restores the per-block path.

## 1. What it does

`sub0llm-configure --backbone-q8-fast 1` emits `constexpr bool BACKBONE_Q8_FAST` (a configure error unless
`--backbone-quant-dot 1`; `-1` = auto = **off** until promoted, see `resolve_decode_defaults`). With it on, every
native Q8_0 plane -- Gated Residual down/up and the shared expert's down projection -- runs
`bbqd::detail::gemv_q8_0_fast` instead of `gemv_rows<Q8_0Plane,true>`.

## 2. Why the old path was slow (verified in the code, not taken from O10)

Per 32-weight block `gemv_rows<Q8_0Plane,true>` (a) `memcpy`s the block into a `WeightGroup`, (b) runs
`dot32_avx2`, which widens to int16 and ends in a full `hsum256_epi32`, and (c) folds into one scalar float
`acc`. The K-quant super kernels keep vector state across a superblock and reduce once per row. Confirmed.

A cache-resident microbenchmark (one thread, ~90 KB of real-format blocks, so no DRAM) then showed the byte dot is
**not** the bottleneck: replacing the per-block scale with a constant made the same loop ~2x faster (60 vs 32 GB/s).
The cost is the per-block **scale path** (f16 convert, multiply by the activation scale, broadcast to 8 lanes) on the
shuffle-limited ports. Variants tried in that harness (GB/s, 640-wide rows, cache-resident):

| variant | GB/s |
|---|---:|
| old per-block path | ~20 |
| vector accumulator, per-block scale broadcast (the O10 recipe) | 32 |
| 4 accumulators | 41 |
| broadcast-load scale path | 40-51 |
| **4-block `vphaddd` fold, one 4-lane scale vector, one 4-lane FMA** | **55-62** |
| 4-block, per-block splat via `vpermilps` | 54 |
| 4-block, scales packed through GPRs | 54 |

## 3. Kernel design (`dot_row_q8_0_fast`)

Four blocks per step. Their four 8-lane int32 dots are merged into ONE 4-lane vector of per-block sums with two
levels of `vphaddd`; the four f16 scales convert with one `vcvtph2ps` and multiply one 4-wide load of the
activation scales; one 4-lane FMA folds all four blocks into a float accumulator. One horizontal reduction per
ROW. A 1-3 block tail reuses the same shape with zero dots and zero scales in the missing lanes. The per-block
weight scale is still applied per block, as Q8_0 requires.

Byte dot, three exact-equivalent forms (`Q8Isa`): `Avx2` (`|w|`/`sign(x,w)` then `vpmaddubsw`+`vpmaddwd`; safe:
`|-128| = 128` as u8 and activations are +-127, so pair sums <= 32512), `Vnni` (same, then `vpdpbusd`), `VnniInt8`
(one `vpdpbssd` on the raw signed operands; this host has AVX-VNNI-INT8). `kQ8FastIsa` picks `VnniInt8` where
available, else `Avx2`.

**Bit-exactness: NOT bit-exact** against today's path (the integer dots are exact, but the float sum is reassociated
and `w_scale*x_scale` is rounded first). The three `Q8Isa` forms ARE bit-identical to each other (tested).

## 4. Seam and consumers (AGENTS.md S10)

ONE seam: a `bool Q8Fast = false` template parameter on `bbqd::gemv_plane<Threads, Q8Fast>` (both overloads),
forwarded through `detail::gemv_plane_dispatch`/`gemv_plane_avx2`. Inert for non-Q8_0 planes and on a non-AVX2
build. Baked at compile time (no runtime branch, AGENTS.md S2).

| Consumer | Change |
|---|---|
| `gr::mix` (`gated_residual_math.hpp`) | new `bool Q8Fast` template parameter after `Threads` (default false); both native GEMVs pass it |
| `decode.cpp` GR call sites (layer loop, exit/top) | `gr::mix<DECODE_GEMV_THREADS, BACKBONE_Q8_FAST>` (2 sites) |
| `decode.cpp compute_shared` | shared down projection `gemv_plane<DECODE_GEMV_THREADS, BACKBONE_Q8_FAST>` |
| unchanged, verified inert | shared gate/up (Q5_K/Q6_K), lm_head (Q4_K), GDN, QSA; `tools/sub0llm-qwen4-forward.cpp` and `backend.cpp` call `gr::mix` without a `Native`; tests' explicit `gemv_plane_avx2(...)` calls compile unchanged (defaulted template arg) |

Classification: inference arithmetic only, shape-neutral -- does NOT join `ARCH_FINGERPRINT`/`PARAM_FLOATS` (same
as `BACKBONE_ACT_SUPER`). No checkpoint or sidecar change.

## 5. Kernel bench (real Q8_0 shard bytes, `sub0_backbone_quant_dot_bench --only-q8 --q8-row-elems N`)

GB/s of bytes read, DRAM-streamed pools of real Q8_0 tensors (the three row widths are the real roles):

| row width (role) | thr | old | **fast** | bf16 |
|---|---:|---:|---:|---:|
| 10240 (GR down) | 1 | 13.4 | **21.0** | 19.4 |
| | 8 | 42.6 | **47.9** | 43.8 |
| 640 (shared down) | 1 | 13.2 | **19.9** | 21.8 |
| | 8 | 29.3 | **37.9** | 36.8 |
| 320 (GR up) | 1 | 9.4 | **19.6** | 26.3 |
| | 8 | 39.5 | **48.4** | 62.1 |

(bf16 moves 2 bytes/weight against Q8_0's 1.06, so equal GB/s is ~1.9x the weights/s for Q8_0.) 1-thread, DRAM:
old 11.2 / 10.3 / 9.5 GB/s, fast 19.6-20.6 GB/s for every `Q8Isa` (memory-bound: all three forms tie); cache-
resident the ordering is VnniInt8 56 > Vnni 47-49 > Avx2 47 GB/s, hence the `kQ8FastIsa` choice. Software prefetch
(512/576 bytes ahead) measured within noise (18.1 vs 18.2 GB/s 1-thread); not kept -- the single-core stream is
already at this host's ~19-20 GB/s stride ceiling for this layout.

## 6. Correctness

- Unit (`tests/backbone_quant_dot_tests.cpp`, `[q8fast]`, 4 cases / 241 assertions): every tail shape (1..8 blocks) and
  the real widths 320/640/2560/10240 against an independent scalar **double** reference on the same quantized
  activation, tolerance 1e-5 of the row's sum of |terms| (stated, not fitted); worst-case operands (all -128 weights,
  +-127 activations); all `Q8Isa` forms bit-identical; `gemv_plane<T,false>` == today's path exactly; `<T,true>` bit-exact
  across 1/2/4 threads on 37 rows; inert for non-Q8_0; Plane overload == loose overload; REAL Q8_0 shard bytes.
- Neutral build (`--dmodel 196 --layers 11 --heads 7 --kv-heads 7 --seq 256`; only diff is `BACKBONE_Q8_FAST = false`):
  `sub0llm_tests` **29,510,661 / 147**, fingerprints `5a7382ea70d3913b` / `7f44bdae18c313dd` / `d1625d19ed2258f1`;
  `sub0llm_frontend_tests` **231,180 / 304** = 230,939 / 300 + 241 / 4 (all new cases). Builds confirmed successful
  before every count.

## 7. G-PPL (real 48-layer artifact, `ppl_blend_v2`, 9,631 tokens)

| | base | q8fast |
|---|---:|---:|
| perplexity | 11.2956 | 11.3589 |
| mean NLL | 2.4244 | 2.4300 |
| top-1 | 51.91% | 51.77% |

Paired dNLL q8fast-base **+0.0056 nats/token, 95% CI -0.0041..+0.0153**, top-1 agreement 88.1%: **PASS**. The
gate needs the CI upper bound at or below +0.03. Measured twice with identical results (2026-09-29 and
2026-10-07); the path is deterministic.

**Why v2.** On the 2,418-token `ppl_blend_v1` this change was inconclusive by 0.0002 nats. A throwaway control
that ran the same fast kernel with a different accumulator split, changing rounding order only, moved
perplexity as far as the kernel itself did (per-token std ~0.5 nats). MoE routing over the IQ1_S experts turns
rounding noise into discrete expert flips, so v1's CI half-width (~0.02) was mostly that noise. v2 halves it.

## 8. Throughput (primary agent, 2026-10-07)

Long-run decode from one `--stage ppl --ppl-speed-rounds 2` run: every arm built before any run, a cooldown
before each, the two 2,000-token rounds in rotated order.

| Run | base tok/s | q8fast tok/s |
|---|---:|---:|
| scoring run, 9,631 tokens | 7.83 | 8.29 |
| speed round 1, 2,000 tokens | 6.93 | 7.45 |
| speed round 2, 2,000 tokens | 7.28 | 7.16 |
| mean | 7.35 | 7.63 (+3.9%) |

q8fast lost one of the three pairings, so the long-run gain is small against this host's run-to-run spread.
The phase profile is the direct evidence (`--profile-phases 1`, `sub0llm-qwen4-forward --tokens 6`, four
alternating runs per arm, the first of each discarded; ms/token):

| Phase | base | q8fast |
|---|---|---|
| GR: up GEMV | 8.4, 7.6, 8.4 | 6.8, 6.6, 6.9 |
| MoE: router + shared + combine | 8.8, 9.3, 8.9 | 7.9, 7.8, 8.2 |
| GR: down GEMV | 10.3, 27.0, 14.0 | 9.1, 5.9, 9.6 |
| whole token, s | 0.120, 0.132, 0.126 | 0.118, 0.106, 0.119 |

The up projection and the shared expert's down projection improve in every run. The down projection's row
swings from 6 to 27 ms between runs of the same binary in both arms, so it shows no reliable delta here;
the swing itself is unexplained (the `.bbq` is a file mapping whose residency nothing guards, see
`O13_storage_convergence.md` C2).

## 9. Follow-ups / unverified

- Not unit-testable in isolation: `compute_shared` and the two `gr::mix` decode call sites (covered only by the
  real-artifact runs above; `gr::Native` has no test today).
- Iteration status (AGENTS.md S13): three design passes on the kernel (per-block splat -> 4 accumulators -> 4-block
  hadd fold; ISA variants; prefetch). Two-rows-at-a-time (sharing the activation loads) was not tried: the DRAM
  numbers show all forms already tie at the stream ceiling.
- Promoted 2026-10-07: `resolve(backbone_q8_fast, backbone_quant_dot != 0)` in `resolve_decode_defaults`.
