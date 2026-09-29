# O9 -- per-256 `ActSuper` native kernels in decode (`--backbone-act-super 1`)

Status: **merged 2026-09-29, default OFF, recommended ON for the real model.** G-PPL PASS, reproduced
bit-for-bit by the primary agent; long-run decode about **+14%** in the primary agent's own passes (S6).

## 1. What it does

`sub0llm-configure --backbone-act-super 1` emits `constexpr bool BACKBONE_ACT_SUPER` (default false; a configure
error unless `--backbone-quant-dot 1`). With it on, every native role whose real plane is Q4_K/Q5_K/Q6_K and
passes `bbqd::super_fusable` (256-aligned rows) quantizes its activation into `bbqd::ActSuper` and runs
`bbqd::gemv_plane_super<Threads>` (docs/BACKBONE_NATIVE_QUANT.md S14). Anything else keeps the per-32
`ActBlocks` path, decided per plane, per call, from the plane's own `type_raw`/`row_elems`.

## 2. The seam (AGENTS.md S10 consumer list)

One seam: `bbqd::super_ok(const Plane&, const ActSuper*)` (eligibility) plus `bbqd::gemv_plane_super(const Plane&,
...)` (the Plane overload of the existing kernel entry). Each `Native` struct gains a parallel `ActSuper*` per
`ActBlocks*`; a null pointer means "never super", so leaving them unset is exactly the old behaviour. The
engine-free math headers hold no compile-time flag: `decode.cpp` sets the pointers under
`if constexpr (BACKBONE_ACT_SUPER)`, the same way it gates `BACKBONE_QUANT_DOT`.

| Consumer | Roles | Real format | Path under toggle |
|---|---|---|---|
| `gdn::Native` (`gdn_math.hpp`) | in_qkv, in_z / out | Q5_K / Q6_K | super, checked per plane |
| `qsa::Native` (`qsa_math.hpp`) | q\|gate, k, v / o | Q5_K | super, checked per plane |
| `decode.cpp compute_shared` | shared gate, up | Q5_K (layer-2 outlier Q6_K) | super, checked per plane |
| `decode.cpp` lm_head | LmHead | Q4_K | super |
| `gr::Native`, shared-expert down | Gated Residual, down | Q8_0 | unchanged (`super_fusable` refuses Q8_0) |

`gated_residual_math.hpp` is deliberately not extended (AGENTS.md S8: a field nothing could read).
Each distinct activation is quantized once per representation actually needed (both forms, once each, only when
sibling planes differ). Scratch is `thread_local`, reserved in `kv_reset` (S1); `ActSuper` carries its own
per-16 sums, so no `Gsum16` is involved on the super path. Not touched: `moe_quant_dot.hpp`, O8, the sidecar.

## 3. Correctness

- Default-off: `sub0_tests` 29,510,661 / 147, fingerprints `5a7382ea70d3913b` / `7f44bdae18c313dd` /
  `d1625d19ed2258f1`; `sub0_frontend_tests` 228,198 / 295 before the new cases, 230,939 / 300 after (+2,741 / +5,
  all new cases).
- New cases: `bbqd` Plane overload == loose overload exactly, `super_ok` truth table; `gdn::Native` super path
  == direct `gemv_plane_super` on the same bytes for in_qkv/in_z/out, and a Q8_0 in_z falls back to per-32 exactly
  in the same call (toggle-off equivalence); `qsa::Native` q|gate and o == direct super calls, Q8_0 v falls back.
- Not unit-tested in isolation: `compute_shared` and the lm_head site (they live inside `decode.cpp`); covered by
  the real-artifact runs below only.

## 4. Measured (real 48-layer artifact, flags F = `--moe-quant-dot 1 --decode-gemv-threads 8
--moe-decode-threads 8 --decode-omp-spin 1 --backbone-quant-dot 1 --moe-row-split 1`)

G-PPL (`--stage ppl`, `ppl_blend_v1`, 2,418 tokens):

| | base | super |
|---|---:|---:|
| perplexity | 14.8488 | 14.7035 |
| mean NLL | 2.6979 | 2.6881 |
| top-1 | 48.64% | 48.76% |
| long-run tok/s, pass 1 (base first) | 5.54 | 7.11 |
| long-run tok/s, pass 2 (super first) | 5.34 | 7.13 |

Paired dNLL super-base **-0.0098 nats/token, 95% CI -0.0318..+0.0122**, top-1 agreement 86.6%: **PASS**
(indistinguishable from zero, as for the native backbone itself, S19 of BACKBONE_NATIVE_QUANT.md). The reversed
pass prints "inconclusive" only because super is then the reference arm (+0.0098, CI -0.0122..+0.0318).
Base perplexity reproduces S19's 14.8488 exactly.

Phase profile (`--profile-phases 1`, `--tokens 6`, 3 rounds, order base/super, super/base, base/super; ms/token):

| run | total s/tok | GDN | QSA | routed | router+shared | GR | lm_head |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 base (cold) | 0.147 | 46.9 | 21.8 | 35.6 | 8.6 | 24.4 | 9.2 |
| r1 super (cold) | 0.129 | 34.6 | 18.4 | 38.3 | 8.3 | 20.9 | 8.0 |
| r2 base | 0.134 | 47.0 | 20.4 | 27.6 | 8.8 | 21.3 | 7.8 |
| r2 super | 0.121 | 35.6 | 18.6 | 28.6 | 8.7 | 21.6 | 7.5 |
| r3 base | 0.137 | 47.8 | 21.2 | 28.2 | 8.8 | 21.0 | 9.2 |
| r3 super | 0.125 | 36.7 | 19.8 | 29.3 | 8.7 | 22.2 | 7.1 |

Warm runs: GDN 47.4 -> 36.2 ms (-24%), QSA ~-8%, lm_head ~-15%, total 0.136 -> 0.123 s/token (~-9.5%, 7.4 -> 8.1
tok/s). Six-token forward-vs-forward_one L2 0.292615 -> 0.292952 (small). One host, no per-run clock log; the
r1 rows are cold.

## 5. Caveats and next steps

- One quality run on one 2,418-token blend; the CI half-width (~0.022 nats) bounds what is excluded.
- Weight-side error is unchanged; only the activation grid is coarser (2-3x per-dot error, S14e), which G-PPL
  does not see at this sample size. A larger blend or a second text set would tighten it before recommending.
- Per AGENTS.md S13 a positive result needs no parking; the toggle stays default-off until the primary agent
  decides. VNNI variant remains parked (S14d).

## 6. Independent verification (primary agent, from merged `main`)

Two more `--stage ppl` passes, run by the primary agent on a quiet host from `main` at 555fa12 (the
`wp5c_full48` build dir, reconfigured and rebuilt for each arm by the script):

| pass | order | base tok/s | super tok/s | gain |
|---|---|---:|---:|---:|
| 3 | super, base | 5.49 | 6.15 | +12% |
| 4 | base, super | 5.94 | 6.91 | +16% |

- Quality reproduces exactly: perplexity 14.7035 and 14.8488, the same dNLL of -0.0098 with the same CI, and 86.6%
  top-1 agreement. That is expected, because the whole path is deterministic.
- Pass 3 prints "inconclusive" only because its first arm (super) became the reference. Pass 4 prints PASS.
- **Speed.** Super is faster in all four passes, but the size of the gain differs by who ran it:
  - the authoring agent measured +28-33% (base 5.34-5.54, super 7.11-7.13);
  - the primary agent measured +12-16%.
  - Both agree that base runs at about 5.3-5.9 tok/s; they disagree on super.
  - The phase profile in S4 (GDN -24%, a warm total of -9.5% on six tokens) is closer to the smaller figure.
  - **The claim carried forward is the conservative +14%** (mean of passes 3-4: base 5.72, super 6.53 tok/s).
- **Recommendation.** Add `--backbone-act-super 1` to the recommended real-model flags. The CI's upper bound
  (+0.012) is well inside the gate's +0.03, and the speed-up is positive in every pass. The toggle itself stays
  default-off (AGENTS.md S4), because the neutral build carries no sidecar to read.
