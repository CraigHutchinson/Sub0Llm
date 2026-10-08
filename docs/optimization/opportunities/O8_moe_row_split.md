# O8 — row-split scheduling for the routed MoE experts in decode

**Status:** merged 2026-09-26, default off (`--moe-row-split 1`, `MOE_ROW_SPLIT`), and **recommended on
for the real model**. Bit-exact end to end (§6). **Long-run decode 5.16 → 5.71 tok/s (+11%)**, measured
in both arm orders (§7).

**Expected gain (brief estimate):** the routed-expert phase from ~44 to ~27 ms/token, roughly
6.6 → 7.5 tok/s (see the brief's own problem statement). **Risk:** low — bit-exact by construction,
touches only decode's scheduling, no kernel or format change. **Complexity:** M. **Gates:** `G-HASH`,
`G-SUITE-ENGINE`, `G-SUITE-FRONTEND` (neutral, verified §5); `G-PARITY`, `G-QUALITY`, `G-PPL`, `G-PERF`
are real-artifact gates (§6–§7).

## 1. Context — the profile this package starts from

At the recommended real-axes flags (`--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 8
--decode-omp-spin 1 --backbone-quant-dot 1`), decode runs ~150 ms/token (~6.6 tok/s). The routed-expert
phase (`ParallelExperts`, `moeqd::expert_ffn_row_quant`) is ~43.6 ms/token, the largest single phase
after GDN (O7 §11's own measured baseline, 2026-09-26).

`ParallelExperts` hands out WHOLE experts across `MOE_DECODE_THREADS` threads via `#pragma omp for
schedule(dynamic)` over the `k`-loop (`n = EXPERTS_PER_TOK = 10` iterations). At 8 threads, two threads
each draw two experts and the phase waits on the slower of those two draws instead of ~10/8 experts'
worth of work; at 10 threads (the older recommendation), two experts land on E-cores, which O4/O7's own
measurements put at ~2x slower per expert on this host. Either way, the work granularity (one whole
expert = one gate GEMV + one up GEMV + one down GEMV, ~4.4 ms combined at the real axes) is coarser than
the thread count can evenly divide.

The phase moves only ~0.25 GB/token (~9 GB/s against this host's ~79 GB/s P-core DRAM ceiling,
`docs/host-cpu-arrow-lake-hx.md`) — compute-bound, so it should scale with cores rather than bandwidth,
which is the premise this package tests directly rather than assumes.

## 2. Problem — the real per-plane-type findings, measured not assumed

The routed-expert phase's three GEMVs per expert are gate `[D_FF=640, D_MODEL=2560]`, up (same shape),
and down `[D_MODEL=2560, D_FF=640]`. Splitting any of them by ROW RANGE across threads needs the
row-sliced GEMV to be bit-identical to the same rows read out of a whole-plane call — which the brief
flagged as a real hazard: a plane's row `r` begins at byte `r * plane_bytes(type, row_elems)` **only if**
`row_elems` is a whole number of that quantized format's own blocks (IQ1_S/IQ2_XXS: 256-element
super-blocks; IQ4_NL: 32-element blocks — `include/sub0/gguf.hpp`'s `block_spec`). `plane_bytes()`
returns 0 when it isn't, which is the exact signal this package uses to fall back to a whole-plane
chunk rather than guess.

**What the real sidecar actually contains** (`D:/ModelWeights/Sub0Llm-Qwen4-full48-bf16/
qwen4_full48_q_bf16.bin.moeq`, parsed directly — header + the full 73,728-entry descriptor table, all
48 layers × 512 experts):

| role | `type_raw` observed | format | count | row_elems | aligned to the format's block? |
|---|---|---|---|---:|---|
| Gate | 19 | IQ1_S (256-elem super-block) | 17,408 | 2560 (D_MODEL) | yes (2560 / 256 = 10) |
| Gate | 16 | IQ2_XXS (256-elem super-block) | 7,168 | 2560 | yes |
| Up | 19 | IQ1_S | 17,408 | 2560 | yes |
| Up | 16 | IQ2_XXS | 7,168 | 2560 | yes |
| Down | 20 | IQ4_NL (32-elem block) | 24,576 (ALL of them) | 640 (D_FF) | yes (640 / 32 = 20) |

Counts sum exactly (17,408+7,168 = 24,576 = 48×512 for each of gate/up/down). Two findings this
disproves/confirms against the brief's own stated hazard:

- **Down is ALWAYS IQ4_NL in this real model — never IQ1_S/IQ2_XXS.** The brief's hazard ("D_FF=640 is
  not a multiple of 256, so down under IQ1_S/IQ2_XXS can't be row-sliced") is a real, general property of
  the FORMAT, but it never actually fires for THIS sidecar: IQ4_NL's 32-element blocks divide 640 evenly
  (640/32 = 20), so down IS row-sliceable here, at every layer and every expert, with no fallback ever
  taken.
- **Gate/up are always row-sliceable too.** Both observed formats (IQ1_S, IQ2_XXS) share the same
  256-element super-block size, and D_MODEL=2560 is a whole 10 super-blocks regardless of which format a
  given layer uses — `plane_bytes(type, 2560)` is nonzero for every format this sidecar contains, and for
  IQ4_NL as well (2560/32=80), so a build with mixed gate/up formats would still be fully sliceable.

**Net effect: on this real model, the general "compute whole when unsliceable" fallback is implemented
and unit-tested (§4), but never actually taken in production.** It exists for correctness on a
hypothetical sidecar that assigns down to IQ1_S/IQ2_XXS (the format spec does not forbid it), not because
this one does it.

## 3. Design

Three phases per layer, matching `moeqd::expert_ffn_row_quant`'s own three steps exactly, so every output
float is produced by the SAME kernel on the SAME bytes in the SAME per-row order as the non-split path:

1. **gate + up.** Work items are `(selected expert k, plane Gate|Up, row range)`. Each selected expert's
   `D_FF=640` rows are split into `ROW_SPLIT_GU_CHUNK_ROWS=64`-row chunks (10 chunks/stream at the real
   axes) and scheduled `#pragma omp for schedule(dynamic)` across the `MOE_DECODE_THREADS` team — 20
   streams (`10 experts × {gate,up}`) × 10 chunks = 200 work items at the real axes, well above the
   thread count, so `schedule(dynamic)` can rebalance across whatever real per-format cost spread exists
   (O1's own per-format bench).
2. **Barrier** (the implicit one at the end of an `omp for`), then per selected expert `k`:
   `h = silu(gate[k]) * up[k]` (640 floats) and quantize `h` into that expert's OWN `ActBlocks`. Small
   against the GEMVs either side of it (`EXPERTS_PER_TOK × D_FF` = 6,400 elements vs `D_FF × D_MODEL` =
   1,638,400 per GEMV), so a plain `schedule(static)` split over `k` is enough.
3. **Barrier**, then **down.** Work items are `(k, row range)` over `D_MODEL=2560` output rows, split into
   `ROW_SPLIT_DN_CHUNK_ROWS=128`-row chunks (20 chunks/expert), `schedule(dynamic)`, writing directly into
   `routed_out` — the same buffer `moe_math.hpp`'s own phase 2 (the fixed-k-order weighted sum) reads
   from next. That phase is completely unchanged; this package only ever replaces phase 1.

### 3a. The hook, not a new mechanism

`include/sub0/moe_math.hpp`'s `forward_row_via_run_ex` already has two optional, `requires`-detected
hooks on the caller's `RunExperts` object: `prefetch()` (B36) and `compute_shared()` (O5). This package
adds a THIRD, `run_rows(d, topk_idx, n, routed_out)`, which — when present — REPLACES phase 1's
`run_experts(...)` call entirely, instead of adding work around it the way the other two do:

```cpp
if constexpr (requires { run_experts.run_rows(d, topk_idx, d.experts_per_tok, routed_out); }) {
    run_experts.run_rows(d, topk_idx, d.experts_per_tok, routed_out);
} else {
    run_experts(d.experts_per_tok, ffn_scratch, g_scratch, [&](int k, float* ffn, float* g) {
        compute_expert(k, topk_idx[k], routed_out + k * d.hidden_size, ffn, g);
    });
}
```

It has to REPLACE rather than wrap because the existing `compute_expert` callback's contract is "produce
one whole expert's output", which cannot express "here is a quarter of one expert's gate projection" —
the row-level GEMV kernel knowledge (plane formats, block alignment) belongs in `moe_quant_dot.hpp`/
`decode.cpp`, not in the engine-free, format-agnostic math core (the same reasoning `compute_shared`'s
own comment already gives for why ITS native path lives on the caller's object rather than a `Native`
struct in `moe_math.hpp`). `SerialExperts`/`ParallelExperts` have no such method, so every existing
caller — including `op_moe`'s batched path, which never touches this file — takes the exact same
`if constexpr`-eliminated path as before this package, unchanged.

### 3b. `RowSplitExperts` (decode.cpp)

`RowSplitExperts : ParallelExperts` inherits `prefetch()`/`compute_shared()` unchanged (both are
orthogonal to how the ROUTED experts' compute is scheduled — B36's I/O pipelining and O5's native
shared-expert path do not care) and adds `run_rows`. It is only ever constructed when
`MOE_ROW_SPLIT && USE_MOE_QUANT && MOE_QUANT_DOT` all hold — a compile-time `if constexpr` at the ONE
call site in `Model::forward_one` picks between `RowSplitExperts{ParallelExperts{l}}` and the existing
`ParallelExperts{l}`, since the runner is a template argument of `forward_row_via_run_ex` and so which
one gets instantiated must be a compile-time decision. The per-expert `compute_expert` lambda itself is
written ONCE (named `compute_one_expert`) and passed to both arms of the `if constexpr` — `RowSplitExperts`
never actually calls it (its own `run_rows` replaces phase 1 entirely), but the function signature still
requires a `compute_expert` argument, so this avoids a second copy of that ~60-line lambda.

### 3c. Row-range GEMV via the PUBLIC `moeqd::gemv_plane`/`plane_bytes` API — no kernel changes

The brief explicitly scoped `moe_quant_dot.hpp`'s kernels and `kO7Kernels` out of this package. The row
range is computed WITHOUT touching that file at all:

```cpp
bool gemv_plane_range(const moeqd::EncodedPlane& p, int row_elems, int r0, int r1,
                       const moeqd::ActBlocks& x, float* out) {
    const std::uint64_t row_bytes = moeqd::plane_bytes(p.desc.type_raw, row_elems);
    if (row_bytes == 0)   // unsliceable -- the chunk list only ever schedules THIS as [0, n_rows)
        return moeqd::gemv_plane(p.desc.type_raw, p.bytes, r1 - r0, row_elems, x, out);
    const std::uint64_t off = static_cast<std::uint64_t>(r0) * row_bytes;
    const std::uint64_t len = static_cast<std::uint64_t>(r1 - r0) * row_bytes;
    if (off + len > p.bytes.size()) return false;
    return moeqd::gemv_plane(p.desc.type_raw, p.bytes.subspan(off, len), r1 - r0, row_elems, x, out);
}
```

**Why this is bit-exact, not an approximation.** `Iq1SPlane`/`Iq2XxsPlane`/`Iq4NlPlane` each hold only a
`const std::uint8_t* plane` base pointer and derive every byte address from the ELEMENT POSITION `p`
passed to `fields()`/`group()` — `blk = plane + (p / kSuperElems) * kBlockBytes` (IQ1_S/IQ2_XXS) or
`blk = plane + (p / 32) * 18` (IQ4_NL). When `row_bytes = plane_bytes(type, row_elems)` is nonzero,
`row_elems` is by definition a whole number of that format's blocks, so `r0 * row_bytes` is EXACTLY the
byte offset of element `r0 * row_elems` from the plane's true start — i.e. `raw.subspan(off, len)`'s own
row 0 IS the original plane's row `r0`, addressed by the identical unpacker reading the identical bytes.
`gemv_plane()` itself is not modified or duplicated: the row-range call and the whole-plane call are
LITERALLY THE SAME FUNCTION, called with a different `(raw, n_rows)` pair that happens to describe a
sub-range of the same underlying bytes. There is no new arithmetic anywhere in this package — only which
subset of an unmodified kernel's output rows a given call produces, and which thread makes that call.

## 4. Correctness

**Unit test** (`tests/moe_quant_tests.cpp`, `[o8]`), two cases, added to `sub0llm_frontend_tests`:

1. *"row-sliced gemv_plane agrees EXACTLY with the whole-plane result, real sidecar bytes, at several
   thread counts."* Opens the real `.moeq` sidecar (env override `SUB0_QWEN4_MOEQ_PATH`, graceful
   `WARN`+skip if absent — same pattern as `backbone_quant_dot_tests.cpp`'s real-GGUF test), scans the
   first 8 layers for a real IQ1_S gate plane and a real IQ2_XXS gate plane (found, not assumed — the
   test fails loudly if a format it expects is absent from the scanned window) and reads the real down
   plane's own format (IQ4_NL, confirmed by the file rather than hardcoded). For each real plane: builds
   a random activation, computes the reference via a single whole-plane `gemv_plane()` call, then computes
   the SAME plane via `o8_gemv_plane_range` (decode.cpp's `gemv_plane_range` reproduced independently in
   the test file — deliberately, so a shared-helper bug can't hide from both sides;
   `[[independent-reimplementation-catches-identity-swap-bugs]]`) split into row chunks and run across
   **1, 2, 4, and 8 real `std::thread` workers** (a static round-robin split of the chunk list — genuine
   concurrency, not a simulated schedule; `sub0llm_frontend_tests` links only `sub0llm_frontend`, not the full
   OpenMP runtime, so `std::thread` is what proves both the math AND the absence of a cross-chunk race).
   `REQUIRE(split[r] == reference[r])` for every row, every plane, every thread count — exact equality,
   not a tolerance.
2. *"an unsliceable plane ... falls back to a single whole-plane chunk, and still agrees with
   gemv_plane."* Synthetic bytes (`make_iq_blocks`, the file's existing fixture generator) at
   `row_elems=640` under IQ1_S and IQ2_XXS — the exact shape §2 found the real sidecar never actually
   produces — confirms `plane_bytes(type, 640) == 0`, that the chunk builder emits exactly one chunk
   covering `[0, n_rows)`, and that the fallback's own output still matches `gemv_plane`'s direct
   whole-plane call exactly.

**End-to-end**: real 48-layer artifact, `sub0llm-qwen4-forward --tokens 6`, native backbone on
(`--backbone-quant-dot 1`). L2-relative logit diff **0.292615** identical with and without
`--moe-row-split 1` (§6), `max |forward - forward_one|` identical (5.12861 at row 2), argmax agreement
identical (3/6). `G-PARITY`'s own hard threshold (`max |forward - forward_one| == 0.0`) is already waived
project-wide for `--moe-quant-dot` (the int8-activation error, not this package); this package adds no
new deviation from THAT waived value. §6's perplexity run then confirmed the same claim over 2,418 real
tokens, exactly, not just 6.

**A real gap caught in self-review, fixed before this was considered done.** The first implementation
built each expert's row-chunk work list straight from the sidecar's `Desc` without re-checking that
`Desc.in_f`/`out_f` actually match this build's `d.hidden_size`/`d.d_ff` — the same
`moeqd::detail::plane_ok` check `expert_ffn_row_quant`'s own reference path runs before it will compute
anything. Without it, a sidecar built at different axes could pass `gemv_plane_range`'s own byte-bounds
check (a too-small read still fits inside a too-large `raw` span) while silently computing from the
wrong bytes, instead of the loud abort the non-split path already gives for that mismatch. Not observed
in practice — the real sidecar's axes match this build exactly, which is why neither the unit test nor
the real-artifact runs caught it — but the reference path guards it and this one now does too
(`RowSplitExperts::run_rows`, re-checked once per selected expert per layer, before the row-chunk lists
are built).

## 5. Neutral gates (default build unaffected — AGENTS.md §4)

Built and run from `out/build/o8` (neutral recipe `--corpus data/gsm8k.txt --dmodel 196 --layers 11
--heads 7 --seq 256`, `SUB0_COMPUTE=CPU`, `-DSUB0_NATIVE=ON`, clang++ 22.1.6):

| suite | assertions/cases | fingerprints |
|---|---|---|
| `sub0llm_tests` | **29,510,661 / 147** (exact) | forward `5a7382ea70d3913b`, grad `7f44bdae18c313dd`, decode `d1625d19ed2258f1` — all unchanged |
| `sub0llm_frontend_tests` | **228,198 / 295** = 208,910/293 (unchanged baseline) + 19,288/2 (the two new `[o8]` cases, real sidecar bytes exercised not skipped) | — |

`MOE_ROW_SPLIT` is `if constexpr`-gated at its one call site and requires `MOE_QUANT_DOT` (which itself
requires `MOE_QUANT_EXPERTS`, off by default) — structurally unreachable from the neutral build, and the
exact assertion/fingerprint match above confirms it, not just the `if constexpr` argument.

## 6. Real model, bit-exact

`sub0llm-qwen4-forward --model <artifact> --tokens 6`, recommended flags
(`--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 8 --decode-omp-spin 1
--backbone-quant-dot 1`):

| arm | L2-relative logit diff | max \|forward − forward_one\| | argmax agreement |
|---|---:|---:|---:|
| base (`--moe-row-split` unset) | 0.292615 | 5.12861 (row 2) | 3/6 |
| split (`--moe-row-split 1`) | **0.292615** (identical) | **5.12861** (identical) | 3/6 |

Perplexity gate (`scripts/run_perf_suite.py --stage ppl --label O8`, fixture `ppl_blend_v1`, 2,418
real tokens, paired per-token NLL, same recommended flags):

| arm | perplexity | mean NLL | top-1 |
|---|---:|---:|---:|
| base | 14.84875 | 2.697916 | 0.4864 |
| split | **14.84875** (identical) | **2.697916** (identical) | 0.4864 |

**Paired delta: +0.0000 nats/token, 95% CI [+0.0000, +0.0000], `se=0.0`, top-1 agreement 100.0%** — every
one of the 2,418 tokens' NLL matched EXACTLY, not merely on average (`se=0.0` is what a truly zero
per-token difference produces, not a coincidentally-small mean over a real spread). `G-PPL`: **PASS**.
This is the strongest correctness evidence in this package: it is the SAME bit-exactness claim as the
6-token L2 check above, now confirmed over 2,418 real tokens of real text instead of 6.

## 7. Throughput

Measured by the primary agent on a quiet host: two `--stage ppl` passes, whose long-run decode speed
over the same 2,418 tokens is the throughput measure here. The arm order was reversed between passes.
Recommended flags plus `--backbone-quant-dot 1`.

| pass | order | base tok/s | split tok/s | gain |
|---|---|---:|---:|---:|
| 1 | base, split | 5.12 | 5.74 | +12% |
| 2 | split, base | 5.20 | 5.67 | +9% |
| **mean** | | **5.16** | **5.71** | **+11%** |

Split wins in both orders, so arm order and thermal drift do not explain it. The authoring agent's own
first pass, run while another agent was building, gave +15% by the same measure.

**A contradicting number, and why it is not used.** The authoring agent also ran
`--stage perf --tokens 3`, six interleaved rounds: base 0.160, split 0.171 s/token median (−7%). Its
per-arm spreads were 24.5% and 43.3%, several times the effect being measured, over only three decoded
tokens per run. That run cannot resolve an 11% difference either way. Long runs amortize start-up,
average thousands of expert selections instead of 30, and are what generation actually does.

The estimate in the brief (~44 → ~27 ms on the routed-expert phase, about 6.6 → 7.5 tok/s short-run) was
not confirmed or refuted phase by phase: no `--profile-phases` table was taken before the agent's session
ended. The end-to-end +11% is the measured claim.

## 8. Thread-count findings

Measured 2026-09-29, after the merge: row-split on, `--moe-decode-threads` 8, 12 and 16, two long-run
`--stage ppl` passes with the order reversed. Background load was ~11%, above the usual 5% bar, so only
within-pass comparisons are read.

| long-run tok/s | 8 | 12 | 16 |
|---|---:|---:|---:|
| pass 1 (8, 12, 16) | **5.31** | 4.88 | 5.13 |
| pass 2 (16, 12, 8) | 5.15 | **5.49** | 4.85 |
| mean | 5.23 | 5.19 | 4.99 |

- **No gain from a larger team.** The best arm flips with the order, and the ~0.3 tok/s spread is as large
  as any difference between arms. Sixteen is, if anything, worse.
- Row chunks let E-cores take work without holding a whole expert hostage, but that does not turn into
  measurable speed at this chunk size. Eight threads, all P-cores, stays the recommendation: it is the
  simplest option and none of the alternatives beats it.
- **Bit-exact across thread counts:** perplexity 14.8488 in every arm, paired dNLL exactly 0, top-1
  agreement 100%.

## 9. `cpp-review` pass

- **Authoring agent:** renamed the O8-prefixed symbols to `RowSplit*` (WIP 5/n), and re-validates plane
  geometry before splitting (WIP 6/n).
- **Primary agent, at merge:** `gemv_plane_range`'s fallback for a plane whose rows cannot be addressed
  (`plane_bytes == 0`) computed the whole plane from row 0 whatever range it was given. `run_rows` only
  ever passes such a plane one full `[0, n_rows)` chunk, so the result was correct, but a future partial
  range would silently have written rows `[0, r1-r0)` where `[r0, r1)` belongs. It now refuses any range
  not starting at row 0. No behaviour change, since that branch is never taken on this sidecar (§2).
- **Build note:** the neutral `d196check` build needed its generated config regenerated (same recipe)
  to pick up the new `MOE_ROW_SPLIT` constant. The regenerated header differs from the old one only by
  that line.

## 10. Risks & mitigations

- **Numerical**: none beyond what `--moe-quant-dot` already accepts — this package adds no new
  arithmetic, only a different row/thread assignment for an unmodified kernel (§3c's bit-exactness
  argument, checked by §4's tests, not merely argued).
- **The "unsliceable plane" fallback path is real but unexercised by the current real model** (§2) — kept
  because the format itself does not forbid an IQ1_S/IQ2_XXS down plane, and because leaving it out would
  silently corrupt output the moment a re-quantized sidecar picked a different format for that role. It is
  unit-tested directly (§4, case 2) rather than left as an un-exercised branch.
- **Scratch memory**: `RowSplitScratch` is `EXPERTS_PER_TOK`-wide fixed arrays (gate/up accumulators,
  per-expert `ActBlocks`, chunk lists), sized once from compile-time constants (`D_FF`, `D_MODEL`,
  `EXPERTS_PER_TOK`), a plain (non-`thread_local`) global — same reasoning as `g_moe_act_q`'s own comment:
  `forward_one` is single-threaded at the call level, so only this token's own `MOE_DECODE_THREADS` team,
  inside ONE parallel region, ever touches it. No heap allocation in the per-token path (AGENTS.md §1) —
  `ActBlocks::quantize` allocates only on a width change, which never happens after the first token since
  `D_FF` is a compile-time constant.
- **FTZ/DAZ**: set per OpenMP worker inside `run_rows`'s own parallel region, identically to
  `ParallelExperts::operator()` — an intermediate going subnormal would otherwise compute DIFFERENT
  floats on a worker that skipped it, not merely slower ones.

## 11. Follow-ups / compounding opportunities

- If §7's A/B confirms the estimated win, re-measure O7's SuperCache+gather kernels (parked default-off,
  `docs/optimization/opportunities/O7_expert_kernels.md`) ON TOP of row-split scheduling — O7's own §10/§11
  found it slower under whole-expert scheduling on both 8 and 10 threads; a finer row-granularity work
  distribution changes which cores actually run the gather-heavy inner loop and is worth re-testing rather
  than assuming O7's verdict still holds (AGENTS.md §13's "measure combinations, not just isolated arms").
- §8's thread-count sweep (8 vs 12 vs 16) may make E-cores newly productive for balanced row work where
  they were not for whole-expert work (O7 §10/§11's own E-core findings were about WHOLE experts landing
  on E-cores, not row chunks) — if a wider team pays off, `--moe-decode-threads`'s own recommended default
  should be revisited alongside this package, not in isolation.
- The chunk-row constants (`ROW_SPLIT_GU_CHUNK_ROWS=64`, `ROW_SPLIT_DN_CHUNK_ROWS=128`) were chosen to
  give `schedule(dynamic)` several chunks per thread at the real axes, not tuned against a sweep — a
  follow-up pass could sweep chunk size directly against the real decode phase time.
