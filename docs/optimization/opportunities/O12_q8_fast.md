# O12 -- vector-accumulator Q8_0 GEMV for the native backbone (`--backbone-q8-fast 1`)

Status: **built, default OFF (auto resolves OFF), awaiting primary-agent review.** G-PPL is **inconclusive by
0.0002 nats** on the letter of the gate (S5); decode is ~6-7% faster long-run and the GR down projection ~30%
faster warm. Not bit-exact.

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
  `sub0_tests` **29,510,661 / 147**, fingerprints `5a7382ea70d3913b` / `7f44bdae18c313dd` / `d1625d19ed2258f1`;
  `sub0_frontend_tests` **231,180 / 304** = 230,939 / 300 + 241 / 4 (all new cases). Builds confirmed successful
  before every count.

## 7. G-PPL (real 48-layer artifact, `ppl_blend_v1`, 2,418 tokens)

| | base | q8fast |
|---|---:|---:|
| perplexity (base arm matches the pinned 14.7035) | 14.7035 | 14.8559 |
| mean NLL | 2.6881 | 2.6984 |
| top-1 | 48.76% | 48.88% |

Paired dNLL q8fast-base **+0.0103 nats/token, 95% CI -0.0096..+0.0302**, top-1 agreement 86.8%. The gate passes if
the CI upper bound is <= +0.03: **0.0302 -- INCONCLUSIVE by 0.0002.** (The reversed-order pass prints PASS because it
reports base-minus-q8fast; that direction is not the question and is not evidence.)

**Noise floor, measured, not assumed.** The per-token difference has std ~0.5 nats and mean |d| ~0.25 nats even though
the arithmetic differs only by float reassociation -- the MoE routing over IQ1_S experts turns rounding noise into
discrete expert flips. To check that this is the model's noise floor and not a defect, a THROWAWAY control (not
committed) ran the SAME fast kernel with a different accumulator split, changing rounding only: ctl-vs-q8fast
mean +0.0029, std 0.535; ctl-vs-base mean +0.0133, std 0.509 (ctl ppl 14.8997). Two independent reassociations both
land ~+0.01 above base with the same per-token spread as q8fast itself, so q8fast's shift is indistinguishable from
rounding noise -- but note both sit above base, so a small systematic effect cannot be excluded at 2,418 tokens.
More tokens is the gate's own remedy (needs a larger fixture than `ppl_blend_v1`).

## 8. Throughput

Long-run decode (tok/s from the ppl runs, same process, ~2,400 tokens):

| | base | q8fast | delta |
|---|---:|---:|---:|
| pass 1 (base first) | 6.19 | 6.64 | +7.3% |
| pass 2 (q8fast first) | 6.26 | 6.66 | +6.4% |

`--profile-phases 1` `sub0llm-qwen4-forward --tokens 6`, separate build dirs per arm, interleaved, first pair of each
batch discarded, `busy_procs=0` at every start. The runs are dominated by page-cache state (a 22.8 GiB working set):
cold runs show GR down at 20-37 ms in either arm, so medians are unreliable and the warm band is what is reported.

| GR sub-phase (ms/token) | base warm | q8fast warm |
|---|---|---|
| down GEMV | 8.7, 9.0 (also 13.8, 20.8, 28.2 cold) | 6.1, 6.2, 6.4, 7.2 (also 27.0 cold) |
| up GEMV | 7.8, 8.0, 8.5, 8.8, 10.2 | 7.0, 7.3, 7.3, 7.4, 8.0 |
| implied GB/s (338 MB/token each) | down ~38-39, up ~40-43 | down ~47-55, up ~45-48 |
| forward_one s/token (warm) | 0.114, 0.116, 0.118 | 0.109, 0.112, 0.123 |

Down: roughly -25% to -30% warm. Up: -10% to -15%. Both stay below the ~79 GB/s roof; the 1-thread stream ceiling
(S5) and per-call OpenMP overhead on 3.5 MB planes are the remaining gap.

## 9. Follow-ups / unverified

- Not unit-testable in isolation: `compute_shared` and the two `gr::mix` decode call sites (covered only by the
  real-artifact runs above; `gr::Native` has no test today).
- Iteration status (AGENTS.md S13): three design passes on the kernel (per-block splat -> 4 accumulators -> 4-block
  hadd fold; ISA variants; prefetch). Two-rows-at-a-time (sharing the activation loads) was not tried: the DRAM
  numbers show all forms already tie at the stream ceiling.
- Promotion: change `resolve(backbone_q8_fast, false)` to `backbone_quant_dot != 0` in `resolve_decode_defaults`.
