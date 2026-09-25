# O7 — faster routed-expert (MoE) kernels: SuperCache + gather, VNNI evaluated and parked

**Status:** pass 1+2 implemented, default (bit-exact, folded into `gemv_plane`'s own dispatch, no new
toggle); pass 3 (VNNI) implemented, tested, **parked** — measured roughly parity-to-slightly-worse, not
wired into the default path. Real-artifact decode timing NOT run (out of this package's scope; the
primary agent runs it at merge time per the brief).
**Expected gain:** the design doc (`docs/SPECULATION_NGRAM_MOE_DESIGN.md` §2) puts routed experts at
~46 ms/token, ~22% of decode, compute-bound at ~5 GB/s against a ~79 GB/s roof — this package narrows
that compute-bound gap on IQ1_S/IQ2_XXS specifically (§2 below); IQ4_NL is untouched (already
memory-side per O1).
**Risk:** low for pass 1+2 (bit-exact by construction, checked not assumed); pass 3 carries no risk
because it ships unwired.
**Complexity:** S/M
**Gates:** `G-HASH`, `G-SUITE-ENGINE`, `G-SUITE-FRONTEND` (all verified on the neutral build, this
package); `G-QUALITY`/`G-PARITY`/`G-PERF`/`G-COMPETITOR` are real-artifact gates the primary agent runs.

## 1. Context — the profile this package starts from

O1's own per-format microbench (`docs/optimization/opportunities/O1_iq2xxs_sign_fold.md` §10, "Next
constraint" — measured 2026-09-22, `sub0llm-bench-moeqd`, this host, post-O1 AVX2 kernel):

> Known candidates from the microbench: IQ1_S/IQ2_XXS grid lookups (4 table loads + `set_epi64x` per
> group, compute-side at ~15% of AVX2), IQ4_NL memory-side (~8.5 GB/s single-core)...

Reproduced at the start of this package (`sub0llm-bench-moeqd --seconds 0.6`, this host, one thread,
real Qwen4 plane shapes 640×2560 / 2560×640, before any O7 change):

| format | shape | mode | µs/plane | GB/s | % DRAM roof (30 GB/s, 1-core) | GMAC/s | % AVX2 int8 ceiling |
|---|---|---|---:|---:|---:|---:|---:|
| IQ1_S   | gate/up | hot    | 126.8 | 2.52  | 8.4%  | 12.92 | 11.5% |
| IQ1_S   | gate/up | stream | 139.9 | 2.29  | 7.6%  | 11.71 | 10.5% |
| IQ1_S   | down    | hot    | 108.1 | 2.96  | 9.9%  | 15.15 | 13.5% |
| IQ1_S   | down    | stream | 149.3 | 2.14  | 7.1%  | 10.98 | 9.8%  |
| IQ2_XXS | gate/up | hot    | 124.1 | 3.40  | 11.3% | 13.20 | 11.8% |
| IQ2_XXS | gate/up | stream | 125.1 | 3.38  | 11.3% | 13.10 | 11.7% |
| IQ2_XXS | down    | hot    | 136.3 | 3.10  | 10.3% | 12.02 | 10.7% |
| IQ2_XXS | down    | stream | 134.9 | 3.13  | 10.4% | 12.15 | 10.8% |
| IQ4_NL  | gate/up | hot    | 50.5  | 18.26 | 60.9% | 32.46 | 29.0% |
| IQ4_NL  | gate/up | stream | 110.7 | 8.33  | 27.8% | 14.80 | 13.2% |
| IQ4_NL  | down    | hot    | 74.6  | 12.36 | 41.2% | 21.97 | 19.6% |
| IQ4_NL  | down    | stream | 108.2 | 8.52  | 28.4% | 15.14 | 13.5% |

IQ1_S/IQ2_XXS sit at 8–13% of the AVX2 int8 ceiling regardless of stream/hot (compute-bound, confirming
O1's own read); IQ4_NL's stream numbers (~8.3–8.5 GB/s) sit well below its hot numbers, confirming it is
memory-side, not compute-side — so this package leaves IQ4_NL's kernel untouched and targets IQ1_S/IQ2_XXS.

## 2. Problem — two concrete inefficiencies, found by reading the code next to the profile

Both are in `include/sub0/moe_quant_dot.hpp`'s `Iq1SPlane`/`Iq2XxsPlane` (post-O1 AVX2 kernel):

1. **Redundant per-group scale decode.** IQ1_S (50-byte block) and IQ2_XXS (66-byte block) each pack
   256 elements — 8 consecutive 32-wide `GROUP`s — under ONE `f16 d`. `fields()` re-reads and
   `group()`/`group_v()` re-converts that same `d_bits` (`gguf::f16_to_f32`/`_cvtsh_ss`) on **every one**
   of those 8 calls. This is the exact shape `docs/BACKBONE_NATIVE_QUANT.md` §12b already fixed for the
   backbone's K-quant kernels (`KScaleTable`, "decode ONCE per superblock, not once per 32-element
   group").
2. **Scalar-gather codebook/sign lookups.** `group_v()`'s four-lane grid/sign lookup is four independent
   scalar table reads packed with `_mm256_set_epi64x` — a serial chain of GPR loads and vector inserts
   rather than a single vector memory operation.

## 3. Reference study (AGENTS.md §5) — llama.cpp's own AVX2 kernels, quoted

Read directly from `D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-cpu\arch\x86\quants.c`
(`ggml_vec_dot_iq2_xxs_q8_K`, `ggml_vec_dot_iq1_s_q8_K`) — not paraphrased, and re-derived onto this
project's own conventions before use, per AGENTS.md §5.

**IQ2_XXS** (`ggml_vec_dot_iq2_xxs_q8_K`, AVX2 arm):

```c
memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                       signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
```

**IQ1_S** (`ggml_vec_dot_iq1_s_q8_K`, AVX2 arm, non-BMI2 fallback — the BMI2 arm only batches the
`qs|qh` bit-pack via `_pdep_u64`, it still finishes with the same `_mm256_set_epi64x` grid read):

```c
const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                        iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
```

**What this means for this package, worked through rather than assumed:** llama.cpp's own AVX2 kernel
uses the identical `_mm256_set_epi64x`-of-four-scalar-loads shape this project's pre-O7 kernel already
used — it is **not** a case of "port the reference's faster technique". Replacing it with
`_mm256_i32gather_epi64` (§4b below) is a genuine departure from the reference, evaluated on its own
merits against this host's gather throughput, not inherited from llama.cpp. Two further things confirmed
by this read, matching what `moe_quant_dot.hpp`'s own file header already documented and worth
re-stating because they are exactly the axes AGENTS.md §5 asks to re-derive rather than assume: (a)
llama.cpp's `mul_add_epi8` (`ax=sign_epi8(x,x); sy=sign_epi8(y,x); maddubs_epi16(ax,sy)`) is the same
magnitude/sign split `detail::gemv_avx2`'s `vpsignb`+`vpmaddubsw` shape already uses — nothing to port
there either; (b) llama.cpp's own IQ2_XXS kernel processes TWO 32-groups per outer-loop iteration with
two independent `__m256i` accumulators specifically to break the dependency chain — this project's
kernel processes one group at a time with a single running `__m256` accumulator (O1's own design,
unchanged here); dual-accumulator batching is named as an untried lever in §7, not attempted this pass.

## 4. Implementation

### 4a. Pass 1 — `SuperCache`: hoist the per-superblock `d` decode

New `SuperCache` struct (`moe_quant_dot.hpp`, ahead of `Iq1SPlane`): `block` (which 256-element
superblock is cached), `d` (portable path's converted scale), `d_f` (AVX2 path's). Each plane gets a
`refresh(p, SuperCache&)` that recomputes only when `p`'s own superblock (`p / 256`) differs from the
cached one, and new `group_cached()`/`group_v_cached()` methods that use the cached scale instead of
re-converting `d_bits`.

**Why this is provably bit-exact, not merely close:** `d` is a pure function of the SAME 16-bit pattern
every one of the 8 times a superblock's groups ask for it. Half-precision → single-precision is a
lossless, deterministic IEEE-754 conversion (every `binary16` value is exactly representable in
`binary32` — no rounding step exists to reorder), so caching the result changes only how many times the
conversion runs, never the value it produces.

**Reset discipline, checked not assumed:** the cache is a local, default-constructed
`SuperCache sc;` at the top of each ROW in `gemv_fast`/`gemv_avx2_fast`, not persisted across rows. This
matters concretely: the down projection's rows are 640 elements = 2.5 IQ1_S/IQ2_XXS superblocks, so
consecutive rows begin at ALTERNATING phase (superblock-relative offset 0, 128, 0, 128, ...) — nothing
may be assumed to carry over from the previous row. `tests/moe_quant_tests.cpp`'s "O7" cases exercise
both real shapes (640×2560 AND 2560×640) for exactly this reason.

### 4b. Pass 2 — `_mm256_i32gather_epi64` in place of `_mm256_set_epi64x`-of-four-scalar-reads

`group_v_cached()` for both formats replaces the scalar-load-then-pack shape with one gather:

- **IQ2_XXS grid** (byte-aligned index — `aux8[l]` IS byte `l` of `aux[0]`): a single `vpmovzxbd`
  (`_mm_cvtepu8_epi32(_mm_cvtsi32_si128(aux[0]))`) zero-extends all four index bytes directly into the
  gather's index lanes, no scalar shift/mask at all.
- **IQ2_XXS signs** and **IQ1_S grid** (7-bit / 11-bit fields, not byte-aligned): the index is still
  built the scalar way (`_mm_set_epi32` of four computed indices — a shared `index()`/`sign_index()`
  static helper, so the SAME formula both the scalar `grid()`/`signs()` and the gather path read), but
  the TABLE READ itself becomes one `_mm256_i32gather_epi64` instead of four independent scalar loads.

**Bit-exact by construction:** a gather is a set of loads, not an arithmetic operation — it reads the
identical table at the identical per-lane index the scalar form did, so it cannot itself introduce any
numeric difference. `tests/moe_quant_tests.cpp`'s "O7" case for `gemv_avx2_fast` requires **exact**
agreement (not a tolerance) against the pre-O7 `gemv_avx2`, over real plane shapes, for exactly this
reason.

### 4c. Dispatch — folded into the default, old kernels kept as the reference (AGENTS.md §13)

`gemv_best` (what `gemv_plane` — and so `expert_ffn_row_quant` and decode's `--moe-quant-dot` path —
actually calls) now dispatches to the NEW `gemv_fast`/`gemv_avx2_fast`, not the old `gemv`/`gemv_avx2`.
Per this package's own numerics rule ("a bit-exact kernel may replace the default outright"), this is
folded straight in rather than gated behind a new toggle — a second toggle for a value-preserving
reordering would just be surface nothing reads (AGENTS.md §8). The pre-O7 kernels (`gemv`, `gemv_avx2`,
and each format's own `group()`/`group_v()`) are UNCHANGED and stay in the tree: they are the correctness
reference the new "O7" tests check against, and `tests/moe_quant_tests.cpp`'s existing `[moequant]` cases
(O1's own AVX2-vs-portable tolerance test, the B35 activation-error tests, etc.) all still exercise them
transitively through `gemv_plane` — nothing about those cases changed, and they still pass unmodified.

### 4d. Pass 3 (evaluated, parked) — AVX-VNNI (`vpdpbusd`)

`detail::gemv_avx2_vnni<Plane>` (gated `#if defined(__AVXVNNI__)`, which `-march=native` defines on this
host) replaces the per-group `vpmaddubsw` + `vpmaddwd`-against-`ones` pair with one
`_mm256_dpbusd_avx_epi32`, which computes the same unsigned-times-signed 4-way dot AND the pairwise
int32 reduction in a single instruction. `docs/BACKBONE_NATIVE_QUANT.md` §12d/§14d found the analogous
fold roughly PARITY for the K-quant kernels because that kernel also needed a SEPARATE integer multiply
to fold in a per-sub-block scale after the dot (so VNNI saved nothing net); this kernel has no such extra
step (the group's own float scale is applied once, outside the dot, identically with or without VNNI),
so the instruction-count argument looked cleaner here — worth an independent measurement rather than
assuming the backbone's verdict transfers.

**Measured (see §5c): NOT a clear win — roughly parity to slightly worse.** Kept in the tree,
individually tested for exact agreement against `gemv_avx2_fast` (`tests/moe_quant_tests.cpp`, "O7 pass
3"), and NOT wired into `gemv_best()` — AGENTS.md §13's "park, never revert": the next person who wants
to push on VNNI resumes from a tested, correct starting point instead of re-deriving the fold.

## 5. Measurements

All kernel numbers: `sub0llm-bench-moeqd`, this host, one thread, real Qwen4 plane shapes (gate/up
640×2560, down 2560×640), `--seconds 0.6`–`0.8`. Checksums (the bench tool's own smoke check) were
IDENTICAL across every pass reported below, at every format/shape — consistent with the bit-exactness
claimed for passes 1+2 and confirmed independently by the exact-agreement unit tests in §6.

### 5a. Pass 1+2 vs the §1 baseline (clean measurement, no host contention)

Two independent runs (`--seconds 0.6`, `--seconds 0.8`) both showed the same direction and similar
magnitude; the table below is the `--seconds 0.8` run.

| format | shape | mode | baseline µs | pass 1+2 µs | Δ |
|---|---|---|---:|---:|---:|
| IQ1_S   | gate/up | hot    | 126.8 | 103.7 | **−18.2%** |
| IQ1_S   | gate/up | stream | 139.9 | 106.1 | **−24.2%** |
| IQ1_S   | down    | hot    | 108.1 | 106.3 | −1.7% (noise) |
| IQ1_S   | down    | stream | 149.3 | 110.0 | **−26.3%** |
| IQ2_XXS | gate/up | hot    | 124.1 | 102.0 | **−17.8%** |
| IQ2_XXS | gate/up | stream | 125.1 | 106.6 | **−14.8%** |
| IQ2_XXS | down    | hot    | 136.3 | 107.3 | **−21.3%** |
| IQ2_XXS | down    | stream | 134.9 | 116.6 | **−13.6%** |
| IQ4_NL  | (all)   | —      | — | — | unchanged by construction (code path untouched — `kHasSuper = false`) |

IQ1_S/IQ2_XXS move **14–26%** faster per plane across both shapes, in both memory modes (hot AND
stream), consistent with a genuinely compute-side fix rather than a cache-residency artifact. IQ1_S
down/hot is the one cell inside this host's own run-to-run noise band (§5d) rather than a real
improvement — reported rather than rounded up.

IQ4_NL's own numbers moved too in the raw logs (e.g. down/hot 74.6 → ~53 µs across different runs), but
that code path is byte-for-byte UNCHANGED (`Iq4NlPlane::kHasSuper = false` routes it straight through the
untouched `group()`/`group_v()`) — the movement is host measurement noise (confirmed: re-running the
UNMODIFIED pre-O7 header on this host, §5d, reproduces the same spread), not a real effect, and is
reported as such rather than claimed.

### 5b. Pass 1+2 GB/s and % of ceiling (same run)

| format | shape | mode | GB/s (was) | GB/s (now) | GMAC/s (was) | GMAC/s (now) |
|---|---|---|---:|---:|---:|---:|
| IQ1_S   | gate/up | hot    | 2.52 | 3.09 | 12.92 | 15.80 |
| IQ1_S   | gate/up | stream | 2.29 | 3.02 | 11.71 | 15.44 |
| IQ2_XXS | gate/up | hot    | 3.40 | 4.14 | 13.20 | 16.07 |
| IQ2_XXS | down    | hot    | 3.10 | 3.94 | 12.02 | 15.27 |

Still well short of the AVX2 int8 ceiling (~14–16% now, up from ~11–14%) — the remaining gap is named
in §7, not claimed as closed.

### 5c. Pass 3 (VNNI) — preliminary, and why it is reported as such

Measured immediately before the primary agent's timing hold notice arrived (a controlled real-model
decode A/B needing a quiet host). Two comparisons were taken:

| format | shape | mode | pass 1+2 µs | VNNI µs |
|---|---|---|---:|---:|
| IQ1_S   | gate/up | hot | 100.8–103.7 | 109.6–110.9 |
| IQ2_XXS | gate/up | hot | 102.0–105.6 | 108.4–118.8 |
| IQ2_XXS | down    | hot | 107.3–119.7 | 109.0–126.6 |

VNNI is **slower, not faster**, on every cell measured — the opposite of the naive instruction-count
argument in §4d. The immediately-following "default fast kernel, re-measured" run (taken right as the
hold notice was arriving) itself showed elevated numbers (IQ1_S gate/up hot 122.0 µs, vs 100.8–103.7 µs
on the two earlier clean runs) — a sign host contention was already starting, which the VNNI numbers
above may partly reflect too. **Reported honestly as preliminary, not re-verified after the hold, per
the coordinator's own instruction not to trust a number taken during/near the hold.** The qualitative
verdict (VNNI does not clearly help this kernel, unlike the naive instruction-count argument) is
consistent across every one of the measurements taken, though, and matches `docs/BACKBONE_NATIVE_QUANT.md`
§14d's own finding for a related kernel (VNNI's fusion advantage is real for a pure dot product but
buys little once the surrounding shape is already lean) — enough to park it with reasonable confidence,
not enough to publish a precise percentage. **This is attempt 1 of 3 for the VNNI mechanism specifically
(§4d/§7); it is not "IQ1_S/IQ2_XXS kernels have had 3 passes" — passes 1 and 2 (SuperCache, gather) each
individually cleared AGENTS.md §13's bar on their own first measurement (clean, positive, reproduced
twice) and did not need a second/third iteration to be judged.**

### 5d. Host noise, characterized directly

Re-running the UNCHANGED pre-O7 header (temporarily swapped back in via `git show main:...`, rebuilt,
re-measured, then restored — the same technique used for the G-HASH control in §6) showed the SAME
~74→~53 µs-range spread on IQ4_NL's own untouched code path across different points in this session,
confirming the swings above are host-level noise rather than anything this package touched. This host's
kernel-level noise floor is therefore at least ±20–30% run to run for hot, cache-resident cases — a
sharper instrument than the real-artifact decode number (which the primary agent's own controlled A/B
is a separate, more careful measurement of) but not noise-free itself. Recommend re-running §5c's VNNI
comparison, with 3+ interleaved samples, before treating its direction as final.

## 6. Correctness

**Every shipped kernel is bit-exact by construction, and this is checked, not merely argued:**

- `tests/moe_quant_tests.cpp`'s new "O7" cases (`gemv_fast` vs `gemv`, `gemv_avx2_fast` vs `gemv_avx2`)
  `REQUIRE` **exact** float equality (`==`, not a tolerance) between the O7 kernel and its pre-O7
  reference, over both real plane shapes (640×2560, 2560×640) and all three formats.
- A dedicated mutation test ("SuperCache actually gates the value") forges a `SuperCache` that claims to
  already hold the current superblock's data but carries a wrong scale, proving `group_cached()` really
  does read from the cache (not a no-op that always refreshes) and that `refresh()` really does detect a
  genuine superblock boundary and correct itself.
- The parked VNNI kernel (`gemv_avx2_vnni`) has its own exact-agreement case against `gemv_avx2_fast`,
  so it stays a TESTED park, not an untested one.
- All PRE-EXISTING `[moequant]` cases (O1's AVX2-vs-portable tolerance test, the B35 activation-error
  isolation tests, the refusal/geometry-guard tests, etc.) pass **unmodified** — they exercise the new
  default kernel transitively through `gemv_plane`/`expert_ffn_row_quant` and all still pass, confirming
  the swap is invisible above that seam.
- The bench tool's own checksum (sum of every output row) was IDENTICAL across baseline, pass 1+2, and
  the VNNI arm at every format/shape — a second, independent confirmation (not the gate itself, per the
  tool's own header comment, but corroborating).

## 7. Gates

**Neutral build (this package's own responsibility; real-artifact gates are the primary agent's, per the
task brief's explicit "do NOT run real-model decode timing"):**

- **`G-HASH`**: the neutral d196check config (`sub0llm-configure --corpus data/gsm8k.txt --dmodel 196
  --layers 11 --heads 7 --kv-heads 7`, reproduced locally and diffed byte-identical against the
  checked-in `d196check` generated header) is **structurally unreachable** from this change:
  `src/backends/cpu/decode.cpp` gates every `moeqd::` call with
  `if constexpr (USE_MOE_QUANT && MOE_QUANT_DOT)`, and the neutral config has `NUM_EXPERTS = 0` /
  `MOE_QUANT_DOT = false`, so the whole branch — and every template this package touches
  (`gemv_fast`/`gemv_avx2_fast`/`gemv_avx2_vnni`/`group_cached`/`group_v_cached`) — is discarded, never
  instantiated, in that build. **Verified directly, not just argued**: `sub0_tests.exe` was built and run
  BOTH with this package's `moe_quant_dot.hpp` and with the byte-identical pre-O7 original (`git show
  main:include/sub0/moe_quant_dot.hpp`, swapped in, rebuilt, tested, then restored) — the neutral build's
  own printed fingerprints (`forward hash=5a7382ea70d3913b`, `grad hash=7f44bdae18c313dd`, `decode
  hash=d1625d19ed2258f1`) were **IDENTICAL** in both cases. **These do not match the values recorded in
  `docs/optimization/kpi_gates.json` (`816c4a54ad49b8cf` etc.)** — that is a PRE-EXISTING drift on
  `main`, reproduced identically whether or not this package's changes are present, and NOT something
  this package caused (confirmed by the same before/after swap). Flagged for the primary agent; not
  fixed here (out of this package's file scope, and `kpi_gates.json` is not listed as mine to touch).
- **`G-SUITE-ENGINE`**: **29,510,661 assertions / 147 cases**, exact match to the stated gate, identical
  with and without this package's changes (same swap-test as above).
- **`G-SUITE-FRONTEND`**: baseline (this package's starting point, confirmed before any edit)
  **148,922 / 285**; after this package's 4 new `[moequant]` cases (`gemv_fast` exact-agreement,
  `gemv_avx2_fast` exact-agreement, the SuperCache mutation test, and the parked VNNI kernel's own
  exact-agreement case): **206,563 assertions / 289 cases**. `[moequant]` alone: 15 → 19 cases (this
  package added 4, matching the file diff).
- **PRE-EXISTING `[moequant]` cases**: unchanged pass/fail, exercised through the new default kernel.

**Real-artifact gates (`G-PARITY`/`G-QUALITY`/`G-PERF`/`G-COMPETITOR`) were deliberately NOT run** — the
task brief is explicit that another agent is running a controlled A/B on the real 48-layer artifact and
that the primary agent runs this package's own real-model numbers at merge time. Expectation, stated as
a prediction rather than a measurement: since passes 1+2 are bit-exact, `G-QUALITY`/`G-PARITY` should be
**unaffected** (same L2 `0.2938`/argmax `4/6` at `--moe-quant-dot 1` the O1 merge already recorded) —
only `G-PERF` should move, and only in the direction of a real (if here still kernel-level, not
decode-level) improvement on IQ1_S/IQ2_XXS.

## 8. Risks

- **Host noise (§5d) makes the kernel-level numbers directionally solid but not precisely quotable** —
  reported as ranges/directions, not single point estimates, per the honesty this doc's own §5 tries to
  hold to.
- **Pass 3's measurement window overlapped the start of a timing hold.** The qualitative verdict (VNNI
  does not clearly help) is consistent across every sample taken and matches a related precedent
  (`BACKBONE_NATIVE_QUANT.md` §14d), but the exact percentage is not trustworthy and should be
  re-measured with 3+ interleaved clean samples before being cited as a number rather than a direction.
- **The real end-to-end decode win is unverified by this package** — kernel-level µs/plane improvements
  on two of three formats (IQ1_S, IQ2_XXS) at 14–26% do not automatically translate 1:1 into a decode
  s/token number; the real-artifact A/B is what settles that, and is explicitly out of this package's
  scope.

## 9. Follow-ups / compounding opportunities

- **Dual-accumulator batching** (§3's own reference-study finding): llama.cpp's IQ2_XXS kernel processes
  two 32-groups per outer-loop step with two independent accumulators specifically to break the serial
  dependency chain a single running `__m256` accumulator creates. This project's kernel still processes
  one group at a time; trying two interleaved accumulators per row (still bit-exact, since it is only a
  reassociation-free instruction-level-parallelism change if the SAME single accumulator sequence is
  preserved, or a new tested arm if not) is a concrete, unattempted next pass and the most likely
  candidate to close more of §5b's remaining ~84–86% gap to the AVX2 ceiling.
- **Remaining scalar bit-twiddling for IQ1_S's grid index and IQ2_XXS's sign index** (§4b: still built
  the scalar way, only the table READ is vectorized) — batching two sub-groups' index computation with
  BMI2 `_pdep_u64` the way llama.cpp's own IQ1_S BMI2 arm does (§3) is a concrete unattempted lever,
  worth trying alongside dual-accumulator batching rather than in isolation, per AGENTS.md §13's own
  "measure combinations, not just isolated arms."
- **Multi-row processing to amortize the activation load** (named in the task brief, not attempted this
  pass): `x.qs`/`x.scale`/`x.gsum` are re-read from memory once per (row, group) pair; processing 2+ rows
  per group iteration would reuse one activation load across multiple weight rows. This changes the loop
  SHAPE, not just the unpack — a bigger, riskier change than passes 1/2, and the natural "pass 3" to
  attempt for real (in place of, or alongside, VNNI) if this thread continues.
- **IQ4_NL remains the largest single MoE-dot slice** (O1's own §9: "39.1% of MoE dot time... unexamined")
  and is untouched by this package (it has no superblock to hoist and llama.cpp's own scalar-gather
  shape doesn't apply — it already uses `vpshufb`). Its stream-mode numbers (~8.3–8.5 GB/s, well under
  its own hot-mode ~12–18 GB/s) mark it as memory-bound, not compute-bound, so the lever there is
  bandwidth-side (prefetch, multi-row streaming) rather than a repeat of this package's technique — named
  here as a DIFFERENT next step, not this package's own unfinished business.
- **Re-profile once wired**: per `OPTIMIZATION_PROCESS.md` §5, re-profile the MoE phase split (IQ1_S vs
  IQ2_XXS vs IQ4_NL share of the 46 ms/token) once this package's kernels are exercised in a real decode
  run — this doc's §1 baseline is a kernel microbenchmark, not a decode-phase profile, and the two should
  be reconciled before naming the next MoE lever.
