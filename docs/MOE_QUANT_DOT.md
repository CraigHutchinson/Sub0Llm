# B35 — fused quantized dot products for the MoE resolve path

**Status: BUILT AND MEASURED (2026-09-21), on `feature/b35-quant-dot`, not merged.** Sections 1-5 below
are the original design pass and are left exactly as they were written, so what was predicted can be
read against what was measured. Section 6, at the end, is what was actually built and what it does.

**The one-line result: decode 3.63 → ~1.47 s/token on the real 48-layer artifact, a ~2.4x speedup, at a
logit cost (L2-relative 0.294, argmax 4/6) between this project's own BF16 and FP8 precedents.**

---

**Design pass, as originally written.** Design pass for `docs/INDEPENDENT_REVIEW_BACKLOG.md`
B35, which was filed deliberately deferred ("needs its own design pass before implementation"). This is
that pass. It reaches one non-obvious conclusion that changes what B35 should build first, and it rests
on a correction to how B24 Phase 2's own verdict has been generalized.

---

## 1. The reframing: B35 belongs on the MoE path, not the backbone

B35 was filed around the **backbone** (`Q5_K`/`Q6_K`/`Q8_0`), mirroring the `ggml_vec_dot_q5_K_q8_K`
kernels that reading llama.cpp's source surfaced. That framing is backwards for this engine, for two
independent reasons:

1. **The backbone is not quantized-resident here; the experts are.** `sub0llm-transplant` fully
   dequantizes every backbone tensor to F32 offline, once, and the engine stores the result as F32/BF16/FP8
   (`PARAM_DTYPE`, B24/B33). So applying a quantized-dot-product kernel to the backbone would *first*
   require a new resident format and a transplant-format change. The routed experts need none of that:
   they are **already** sitting in memory as their native `IQ1_S`/`IQ2_XXS`/`IQ4_NL` GGUF bytes in the
   `.moeq` sidecar (WP4e). The technique applies there with **zero offline format change**.
2. **The MoE path is the dominant cost.** B30's own accounting puts the resolve path at the large
   majority of decode's per-token time (~2.6-2.9 s of the current ~3.38 s/token), against a backbone
   stream term of ~300-350 ms (`docs/BACKBONE_PRECISION.md` §2c).

So: **do the experts first.** The backbone variant stays a later, separate question, and it inherits a
prerequisite (a quantized resident backbone format) that the expert path simply does not have.

## 2. The correction: B24 Phase 2's "2a beats 2b" does not transfer here

B24 Phase 2 (`docs/BACKBONE_PRECISION.md` §2c) measured dequantize-once-resident (2a) beating
dequantize-inline-per-read (2b) by **5-30x**, and recommended never building an inline per-token dequant
path. That measurement is real and its verdict stands **for the case it measured** — the backbone, at
100% (dense) access density, where a resident buffer is read every token and the dequant cost amortizes
across all of those reads.

**That amortization does not exist on the MoE path, at all.** `src/backends/cpu/internal.hpp`'s own
`MOE_DECODE_SLOTS` comment establishes it, and not as an estimate:

> a decode row's hit rate against that cache is **provably zero**, not merely small:
> within one (token, layer) the selected experts are a TOP-K, so their indices are distinct by
> construction [...] the key is (layer, expert), so nothing carries across layers either [...] across
> tokens, a full token issues 480 resolves, which round-robins any pool [...] several times over.

Every dequantized expert plane is written, read **exactly once**, and discarded. There is nothing to
amortize the materialization against — it is pure overhead. §2a of `BACKBONE_PRECISION.md` already drew
the sparse-vs-dense distinction in its own table ("resolve-pool hit rate: provably zero in decode" vs
"would be 100% on every access"); what it did not do is re-run the 2a-vs-2b crossover under the sparse
case. Under zero reuse the ledger inverts: **fusing wins on the expert path precisely because 2a's own
premise is absent there.**

This is the one claim in this document most worth attacking before building on it.

## 3. What it buys, in real bytes

Per resolve, from this sidecar's own real census (73,728 planes / 24,576 experts / 39,845,888,000 B
payload) and `moe_expert_bench.cpp`'s own measured figures:

| | encoded read | dequant write | FFN read-back | total |
|---|---:|---:|---:|---:|
| **today** (dequantize → f32 planes → dot) | 1.55 MiB | 18.75 MiB | 18.75 MiB | **39.05 MiB** |
| **fused** (dot directly against encoded bytes) | 1.55 MiB | — | — | **1.55 MiB** |

**25.3x less DRAM traffic per resolve.** Across a token's 480 resolves: 18.30 GiB → 0.72 GiB, i.e. a pure
bandwidth term of ~655 ms → ~26 ms at this host's own measured ~30 GB/s (`BACKBONE_PRECISION.md` §2c).

And it attacks the *other* term too. B27 measured the resolve as compute-dominated, with the dequant
itself the largest single component — and this engine's dequant is `gguf::to_f32`'s **scalar** per-element
decode. A fused kernel does that unpacking inline, in registers, as integer SIMD. That is the same double
attack (bandwidth *and* compute) that B32's llama.cpp comparison exposed as the real source of its ~4-6x
lead, and it is why this is the only remaining lever this session has identified that is not a pure
compute-side change on a bandwidth-bound workload (which B28's prefetch, B33's FP8, and B34's SIMD each
independently showed does not help — see those entries).

## 4. Shape of the build

- **Activation side.** Quantize the row once per (layer, projection) into a `q8_K`-equivalent block form,
  then reuse it across all `EXPERTS_PER_TOK` selected experts and all output rows of that projection. Cost
  is O(hidden_size) per layer against O(hidden_size × d_ff × experts) of dot-product work — negligible,
  but it must genuinely be hoisted, not recomputed per row.
- **Weight side.** Three kernels, one per real format this sidecar actually contains: `IQ1_S` (34,816
  planes), `IQ2_XXS` (14,336), `IQ4_NL` (24,576). All three have real AVX2 reference implementations in
  the local llama.cpp checkout (`ggml_vec_dot_iq1_s_q8_K`, `ggml_vec_dot_iq2_xxs_q8_K`,
  `ggml_vec_dot_iq4_nl_q8_0`, in `ggml/src/ggml-cpu/arch/x86/quants.c`) to read as a correctness and
  technique reference. Do not vendor them; this project's own `gguf.hpp` already owns the block layouts
  and must stay the single definition of each format (AGENTS.md §3).
- **Seam.** `expert_ffn_row_source` is the natural insertion point: it already consumes the planes in
  GGUF source order (B31), which is exactly the order a block-quantized dot product wants. The fused path
  replaces `dequantize_expert_source` + `expert_ffn_row_source` as a pair, behind the same kind of
  default-off `constexpr` toggle B36/B37/B38 established, so the existing path stays the reference.

## 5. Correctness

This is **not** bit-exact and must not claim to be: an integer dot product against quantized operands is
a genuinely different (and, for the activation side, lossier) computation than dequantize-then-f32-dot.
Gate it the way B24 gated BF16 — a real logit comparison against the existing path on the real 48-layer
artifact, reporting L2-relative diff and argmax agreement honestly, with the existing precedents as the
yardstick (BF16 ~0.199, FP8 ~0.43). The activation quantization is the new error source with no precedent
in this codebase, so it deserves its own isolated measurement before the end-to-end number is trusted.

`forward`/`forward_one` parity is a separate matter: if only `forward_one`'s path is fused, that invariant
(exactly 0 since B24, and the thing B34 broke and had to repair) **will** break. Either fuse both paths or
scope the toggle so the parity check compares like with like — decide this before writing code, not after
the gate fails.

---

## 6. What was actually built

### 6a. The shape it took

`include/sub0/moe_quant_dot.hpp` (new, engine-free — `gguf.hpp` + `moe_math.hpp` + `moe_quant.hpp` only),
behind `--moe-quant-dot` → `constexpr bool MOE_QUANT_DOT`, default `false`, following B36/B37/B38's own
idiom exactly: one localized configurator option, one generated `constexpr`, one `if constexpr` at the
consumer, and a configure-time refusal if it is set without `--moe-quant-experts 1`.

Three things exist per format and nothing else does:

| | what it owns |
|---|---|
| `Iq1SPlane` / `Iq2XxsPlane` / `Iq4NlPlane` | ONLY that format's block layout: how to find the group's scale, and how to get its 32 weights out as signed int8. ~20 lines each. |
| `detail::gemv<Plane>` | the row walk, the group walk, the integer multiply-accumulate, the scale fold, and IQ1_S's delta term (`Plane::kHasDelta` compiles it away for the other two). Written ONCE. |
| `detail::dot_group` | the inner 32-element int8 dot. Written once, called by everything. |

The activation side is `ActBlocks`: int8 quants, one float scale per 32 elements, and one int32
sum-of-quants per 32 elements (IQ1_S's `dl * (grid + delta)` needs the sum as a separate term — the job
`block_q8_K::bsums` does in llama.cpp). Quantized **once per (token, layer)** in `decode.cpp`, before
`forward_row_via_run_ex`, and reused by all ten selected experts. Only the down projection's own input is
re-quantized per expert, because `silu(gate(x))*up(x)` differs per expert — 640 elements against
640×2560 dot terms.

`src/backends/cpu/decode.cpp`'s MoE lambda gained one `if constexpr` arm; the existing
`ExpertCacheSource`+`expert_ffn_row_source` arm is untouched, and B36's pipelined-I/O staging works with
both. The fused arm also skips `cache.allocate()` entirely — the fused path never materializes an f32
plane, so it never needs the 18.75 MiB of slot storage either.

### 6b. No raw intrinsics — a measurement, not a preference

B34/B38 established that this project's float reductions do not auto-vectorize because Clang will not
reassociate floating-point addition. **Integer addition has no such barrier.** Compiled at `-O3
-march=native`, the portable 32-element int8 dot emits, with no scalar fallback anywhere:

- `vpmovsxbw` + `vpmaddwd` + `vpaddd` for IQ2_XXS and IQ4_NL — the same instruction shape
  `ggml_vec_dot_*`'s hand-written `_mm256_maddubs_epi16` produces;
- `vpmovsxbd` + `vpmulld` + `vphaddd` for IQ1_S, where the weight bytes arrive as four register-resident
  qwords rather than through memory — 8 lanes per multiply instead of 16, still fully vector.

`#pragma clang loop vectorize(enable)` was tried on that loop and changed neither shape, so it is not
carried. Raw AVX2 intrinsics would have bought at most the narrower multiply on one of three formats and
would have cost a second, unshared copy of the kernel per format.

### 6c. The three block layouts: `gguf.hpp` and ggml AGREE

Cross-read field by field against `ggml/src/ggml-common.h`'s struct definitions and
`ggml/src/ggml-quants.c`'s `dequantize_row_iq1_s` / `dequantize_row_iq2_xxs` / `dequantize_row_iq4_nl`
scalar references (the AVX2 kernels were read for technique, the scalar ones for semantics):

| format | block | `gguf.hpp` | ggml | agree |
|---|---|---|---|---|
| IQ1_S | 256 elems | 2 + 32 + 16 = 50 B | `ggml_half d; uint8 qs[32]; uint16 qh[8]` = 50 B | yes |
| IQ2_XXS | 256 elems | 2 + 64 = 66 B | `ggml_half d; uint16 qs[32]` = 66 B | yes |
| IQ4_NL | 32 elems | 2 + 16 = 18 B | `ggml_half d; uint8 qs[16]` = 18 B | yes |

Including the two details most likely to be silently wrong: IQ4_NL's nibble split is **byte j holds
element j (low) and element j+16 (high)**, not adjacent pairs — both sources agree; and IQ1_S's `qh[ib]`
packs three high grid bits per 8-element group at `[3l+2:3l]`, a 3-bit scale at `[14:12]`, and the delta
sign at bit 15 — both sources agree. Nothing to report as a defect in either.

### 6d. One deliberate divergence from llama.cpp: GROUP = 32, not q8_K's 256

llama.cpp quantizes the activation into `block_q8_K` — one float scale per 256 elements — because its
kernels serve a general `[n × k]` GEMM whose `k` is a multiple of 256. **This engine's real shapes are
not.** The down projection's rows are `d_ff = 640` elements, so row `j` begins at element `j*640`, a
*half-block* offset inside a 256-element IQ1_S/IQ2_XXS super-block for every odd `j`. Every one of the
three formats is internally structured in 32-element groups, though, and `2560 % 32 == 640 % 32 == 0`, so
32 is the only granularity at which this engine's real row shapes and all three formats simultaneously
align. A per-32 activation scale is also strictly more accurate than a per-256 one — a free consequence,
not the reason.

### 6e. Correctness: the new error source, isolated first

Section 5 above required the activation quantization to get its own measurement before the end-to-end
number was trusted. It did (`tests/moe_quant_tests.cpp`, four new cases):

1. **The weight side adds no error at all.** Feed an activation constructed to int8-quantize *losslessly*
   (integers, with a ±127 planted in every group so the scale is exactly `1.0f` and `lrintf` is the
   identity) and the fused dot agrees with `gguf::to_f32`'s own dequantized dot to **2.0e-8 (IQ1_S),
   8.7e-8 (IQ2_XXS), 4.2e-7 (IQ4_NL)** — float rounding only. This is the case that would catch a
   block-layout error, which random data cannot: a shifted field still produces finite, plausibly-scaled,
   "close-ish" output.
2. **The activation side alone**, on a Gaussian row: **0.48% / 0.94% / 3.3%** relative error on a single
   256-term dot.
3. **A whole fused expert FFN** vs the dequantize path on the same bytes: **~0.9-1.1% L2-relative**.
4. **An unfusable format is refused, not guessed at** — and so is a span shorter than the geometry
   requires (the kernels index by arithmetic and cannot otherwise notice), and so is a plane whose
   descriptor disagrees with the build's axes (the test swaps gate and down and requires a refusal,
   since a swapped pair would otherwise produce finite, plausibly-scaled, wrong output).

### 6f. End-to-end on the real 48-layer artifact

`forward` (unfused — `op_moe` keeps the f32 resolve) vs `forward_one` (fused), same binary, same weights:

| | L2-relative logit diff | argmax agreement |
|---|---:|---:|
| default build (`--moe-quant-dot 0`) | **0** | 6/6 |
| fused build (`--moe-quant-dot 1`) | **0.2938** | 4/6 |
| *precedent: BF16 backbone (B24)* | *~0.199* | *5/6* |
| *precedent: FP8 backbone (B33)* | *~0.43* | *—* |

Stated plainly: this is a **real** quality cost, larger than the entire BF16 backbone change and smaller
than FP8's. It is not materially worse than FP8, so it is not a stopper — but it is not small either, and
the whole 0.2938 is B35's own contribution, since the same build's default arm reproduces exactly 0.

`forward()`'s own per-row logit statistics are byte-identical between the two arms, which is the direct
check that the batched path really is untouched.

### 6g. Throughput

Interleaved A/B on the real artifact, `--tokens 6`, nothing else running on the host (checked before each
batch — this session has had real 2x contention from sibling agents, and there was none here):

| arm | s/token, run by run | steady state |
|---|---|---:|
| default | 3.746, 3.613, 3.563, 3.593, 3.795 | **~3.66** |
| fused | 1.858, 1.771, 1.576, 1.436, 1.460, 1.513, 1.574 | **~1.51** |

The fused arm's first two samples are visibly still warming (it touches 25x fewer sidecar bytes, so its
page-cache behaviour differs from the default arm's); the last five are stable. **~2.4x.** Against B32's
external reference — real llama.cpp at 0.59-0.94 s/token on this host and model — the gap closes from
~4-6x to **~1.6-2.6x**.

The last sample in each row was taken *after* the review pass that introduced `ExpertPlanes` and the
geometry guard, and the L2-relative logit diff is bit-identical across that refactor (0.293822 before and
after) — i.e. the cleanup changed the shape of the code and nothing about the numbers.

### 6h. The real-axes build recipe, recorded because reconstructing it cost real time

This session's handover recipe is **incomplete**, and the failure mode is a `load_model` rejection with
no hint as to which axis is wrong. Omitting any of these produces `PARAM_FLOATS 2570717696 /
NUM_PARAMS 626` instead of the correct `4915107200 / 1074`; `--loop-middle-layers 4` is in the handover
recipe and is **wrong** (the real model has `LOOP_MIDDLE_LAYERS = 0`). The working invocation:

```
sub0llm-configure --dmodel 2560 --layers 48 --heads 24 --seq 128 --kv-heads 2 --head-dim 256 \
  --rotary-dim 64 --rope-theta 10000000 --gdn-full-attn-stride 4 --gdn-key-heads 16 \
  --gdn-value-heads 48 --gdn-key-head-dim 128 --gdn-value-head-dim 128 --hc-count 4 --hc-lowrank 320 \
  --qsa-indexer-n-heads 4 --qsa-indexer-kv-heads 1 --qsa-indexer-head-dim 128 \
  --qsa-indexer-budget 2048 --qsa-indexer-compress-ratio 4 --tie-embeddings 0 --d-ff 640 \
  --moe-quant-experts 1 --num-experts 512 --experts-per-tok 10 --prec-param 1 --vocab-exact 248320
```

It was recovered by differencing `out/build/wp5c_full48/generated/sub0_corpus.hpp` (in the main checkout)
against a fresh configure — which is the general technique worth remembering, not just this recipe.

### 6i. What section 2's claim looks like after the fact

Section 2 named its own inversion of B24 Phase 2's verdict as "the one claim in this document most worth
attacking before building on it." It survived: on a path with provably zero reuse, fusing wins, by 2.4x.
B24 Phase 2's measurement was never wrong — it was answering a different question, at a different access
density, and the mistake would have been generalizing it.

### 6j. The next lever, named rather than attempted

B27 measured the resolve as compute-dominated. Having removed 25x of the DRAM traffic, what is left is the
unpack-plus-MAC itself. IQ1_S is 47% of all planes (34,816 of 73,728) and is the one format whose inner
dot lands on the wider `vpmulld` shape rather than `vpmaddwd`, because its weight bytes reach the dot
through registers rather than memory. Narrowing that multiply — or arranging for those four grid qwords to
arrive through memory the way the other two formats' do — is a bounded, directly measurable next step, and
is not attempted here.
