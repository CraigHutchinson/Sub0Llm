# O5 phase 1 — keep the backbone resident in its native GGUF quantization

**Status: PHASE 1 (census + design + isolated kernels + tests + microbenchmark). No engine, transplant
tool, `decode.cpp`, `moe_quant_dot.hpp` or `gemv.hpp` code was touched.** Everything this document
describes lives in three new files (`include/sub0/backbone_quant_dot.hpp`,
`tests/backbone_quant_dot_tests.cpp`, `benchmarks/backbone_quant_dot_bench.cpp`) plus one new census tool
(`tools/sub0llm-backbone-census.cpp`); nothing in `src/` or `tools/sub0llm-transplant.cpp` includes any of
them. A later phase reviews this design and does the actual wiring.

**Integration verdict (2026-09-22, §8a): the kernels are correct but not yet fast.** The unpack is
compute-bound at 1–5 GB/s per thread for the K-quants, so at one thread they lose to the real bf16
`gemv::axpy`. The fit win (§7) is real; the throughput win needs phase 2's streaming unpack.

**Phase 2a status (2026-09-22): PAUSED mid-implementation, design verified but not yet ported into the
real header.** See §12 below for the checkpoint — the Q4_K anomaly is explained (disassembly, not
denormals), a streaming/vectorized-unpack design is built and checked bit-exact in a scratch harness
(not yet in `include/sub0/backbone_quant_dot.hpp`), and a first throughput number is in hand for Q4_K
only. Tasks 0 (bench fix), 4 (OpenMP threading), 5 (tests), 6 (suite gate), and the doc/header write-up
are NOT done yet.

Mirrors `docs/MOE_QUANT_DOT.md`'s own structure and rigour, applied to the BACKBONE instead of the routed
experts.

---

## 1. The question, and the correction to how B24 Phase 2 has been read

`docs/BACKBONE_PRECISION.md` S2c measured, on the real backbone tensors, that **dequantize-once-into-a-
resident-buffer (2a) beats dequantize-inline-per-read (2b) by 5-30x**, and recommended never building an
inline per-token dequant path for the backbone. That measurement is real and correct for what it
measured. **It is not the question this document answers.**

B24 Phase 2's own comparison was ALWAYS-inline-scalar-dequant (`gguf::to_f32`, an element-by-element
`std::vector`-writing decode with no fused reduction) vs. a resident f32/bf16 buffer read by a plain AXPY.
That is not what B35 built for the MoE path, and it is not what this document proposes for the backbone
either: **a FUSED integer dot product**, where the weight bytes are unpacked into registers and
multiply-accumulated directly against an int8-quantized activation, never written to a scratch float
buffer at all. B35's own finding (`docs/MOE_QUANT_DOT.md` S2) was that fusing inverts the 2a-vs-2b verdict
on a ZERO-reuse access pattern (the routed experts, resolved once and discarded). The backbone's access
pattern is the opposite extreme — **100% reuse is not the right word either: every backbone weight is
read exactly once per token, with no cross-token amortization, same as the routed experts** — decode is
`SEQ_LEN`=1, so there is no "read the resident buffer 48 times" to amortize a promote cost against. The
resident/promote-once buffer never gets reused within a token's own compute; it only avoids re-decoding
across DIFFERENT tokens, and each token pays a full weight stream either way (native-quant bytes, or the
promoted buffer's own bytes).

So the real question B24 Phase 2 did NOT ask, because it measured a different (unfused) alternative on
the 2b side: **does a FUSED int8-activation x native-quant-weight dot product cost less, per token, than
streaming a resident bf16 buffer of the same logical weights?** That is what this document's census,
kernels, and microbenchmark measure.

---

## 2. Census — every real backbone tensor, from the real GGUF header/tensor tables

Read via `tools/sub0llm-backbone-census.cpp` (new, engine-free, `gguf.hpp` only — header/tensor-table
reads, never a bulk payload load) against the real three-shard `D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\
UD-IQ1_S\` artifact.

**A real defect this census caught in itself, worth recording**: a first pass excluded the n-gram/PLE
table by matching a `"ple_"` PREFIX, which matched nothing — the real tensor is named
`per_layer_token_embd.weight`, a single **51.2-BILLION-element** IQ4_NL tensor (would have added ~27 GiB
to every total below). Caught because the "grand totals" bf16 figure (104.52 GiB) did not match
`BACKBONE_PRECISION.md`'s own documented 9.16 GiB — a sanity check against an independently-derived
number, not code inspection, is what surfaced it. Fixed by matching `"per_layer_token_embd"` as a
substring, plus a second real exclusion this same process found (`"blk.1.ple_*"` — six small per-layer
n-gram sub-tensors, ~35 MiB total, also out of `tools/sub0llm-transplant.cpp`'s own scope per
`docs/WP4_SCOPE.md` S5). After both fixes: **0 unclassified tensors, 0 misaligned tensors**, and the
bf16 total (9.15 GiB) and f32 total (18.31 GiB) match `BACKBONE_PRECISION.md`'s own already-published
figures almost exactly (9.16/18.31), which is the independent corroboration that the census is now
complete and correctly scoped.

### 2a. Per-format totals (1073 tensors, 4,914,858,880 elements)

| Format | Tensors | Elements | Native bytes | bytes/elem | Consumers (representative) |
|---|---:|---:|---:|---:|---|
| F32 | 553 | 78.3M | 298.70 MiB | 4.000 | norms, router (`ffn_gate_inp`), `ssm_alpha`/`ssm_beta`, `hc_*_inject`, biases |
| Q8_0 | 242 | 714.3M | 723.83 MiB | 1.0625 | Gated Residual up/down (`hc_attn_up/down`, `hc_ffn_up/down`), MoE shared-expert down |
| Q4_K | 2 | 1271.4M | 682.03 MiB | 0.5391 | `token_embd.weight` (embed), `output.weight` (untied lm_head) |
| Q5_K | 212 | 2219.7M | 1455.35 MiB | 0.5250 | GDN in-proj (`attn_qkv`, `attn_gate`), QSA (`attn_q/k/v/output`), MoE shared gate/up |
| Q6_K | 40 | 611.5M | 478.34 MiB | 0.6250 | GDN out-proj (`ssm_out`) |
| BF16 | 24 | 19.7M | 37.50 MiB | 2.000 | QSA indexer (`indexer.q_proj`/`k_proj`) |

**Grand totals**: 4,914,858,880 elements. Native bytes **3,854,307,840 (3.59 GiB)**. Today's resident bf16
**9,829,717,760 (9.15 GiB)**. f32-equivalent **18.31 GiB** (matches `BACKBONE_PRECISION.md`'s own figure
exactly). **Native-quant / bf16 ratio: 0.392** — i.e. native residency is **~2.55x smaller than today's
bf16**, and **~5.1x smaller than f32**.

**0 tensors flagged misaligned** (ne0 not a multiple of the format's own block size) — every real
backbone contraction dim (2560, 6144, 10240, 12288, 320) is a multiple of 32, and the K-quant/Q8_0 block
sizes (256 and 32 elements respectively) both divide 32-multiples cleanly whenever the row itself is also
a multiple of the block size — checked directly by the census, not assumed.

**Consumer map** (GGUF name -> engine consumer), transcribed from `include/sub0/transplant.hpp`'s own
`dest_gguf()` table (AGENTS.md S5 — re-derived from the project's single source, not guessed), not
duplicated in full here; see the tool's own `rules()` table for the complete 38-entry mapping. The four
formats this phase's kernels target (Q8_0/Q4_K/Q5_K/Q6_K) account for **3,339.55 MiB of the 3,854.31 MiB
native total (86.6%)** — F32 and BF16 tensors (336.20 MiB, 8.7%) are already small and are NOT
candidates for this lever (F32 has no smaller native form; BF16 already IS the resident target format).

### 2b. Which tensors are on decode's critical path, every token

Every Q8_0/Q4_K/Q5_K/Q6_K tensor above is read on **every decode token, every layer** except:
`token_embd.weight` (a GATHER of one row, not a GEMV — out of scope for a row-range GEMV kernel; see S8),
and layers whose mixer branch (GDN vs QSA, `GDN_FULL_ATTN_STRIDE`) is not active for that layer. `output.
weight` (the untied lm_head, Q4_K, 635.7M elements) is the single largest per-token dense GEMV in the
whole backbone — every decode token computes the full `[2560 -> 248320]` projection.

---

## 3. Layout, and the affine-K-quant generalization this design needed

### 3a. DOT, not AXPY — same reasoning as `moe_quant_dot.hpp`

A GGUF tensor is `[out][in]`, contiguous over `in` (`ne0`) — the natural per-output-row form is a DOT
product, threaded by splitting **output rows** across threads. The engine's existing bf16 backbone path
(`param_store.hpp`'s `Bf16CPtr`, consumed by `*_math.hpp` kernels via `op_linear`) is instead an AXPY:
`for each input element t: for each output o: Y[o] += X[t]*W[t,o]`, over a `[in,out]` layout. **These are
different access patterns, and a later wiring phase has to reconcile them, not this one** — see S7.

### 3b. Q8_0 is pure-scale; Q4_K/Q5_K/Q6_K are AFFINE, not pure-scale

B35's three MoE formats (IQ1_S/IQ2_XXS/IQ4_NL) are all `value = scale * (q + delta)` — one multiplicative
term. The backbone's K-quant formats are genuinely different:

| Format | Value formula | Sub-block width | Notes |
|---|---|---:|---|
| Q8_0 | `d * q` | 32 | pure scale, no bias — identical shape to B35's formats |
| Q4_K | `d*sc*q - dmin*m` | 32 | 8 sub-blocks/256-superblock, `sc`/`m` via `gguf::k_scale_min` |
| Q5_K | `d*sc*q - dmin*m`, `q` gains a high bit from `qh` | 32 | same affine shape as Q4_K, plus `qh` |
| Q6_K | `d*sc*(q-32)` | **16** | zero-centered, HALF of `moeqd::GROUP` |

An affine weight needs an affine dot: `sum_j x[j]*(scale*q[j]+bias) = scale*sum_j(x[j]*q[j]) +
bias*sum_j(x[j])`. The second term is exactly why `moeqd::ActBlocks` already carries `gsum` (an int32
per-32-group sum of the quantized activation, originally added for IQ1_S's own additive delta) — **this
design reuses `ActBlocks`/`gsum` completely unchanged**, per AGENTS.md S3's "check whether an existing
field already discriminates it" discipline applied to a design choice, not a file format: no second
quantized-activation type was invented.

### 3c. GROUP = 32, reused from `moeqd`, not re-derived

Q4_K/Q5_K's own native sub-block width IS 32 (confirmed against `gguf.hpp`'s already-S5-verified
`dequantize_q4_k`/`dequantize_q5_k`), so their sub-block boundary and `moeqd::GROUP`'s activation-group
boundary coincide exactly — a single `(scale, bias)` pair covers a whole activation group, no splitting.

**Q6_K's own native sub-block is 16, not 32.** Re-deriving its actual bit layout from
`gguf::dequantize_q6_k` (not assumed to be "just half of 32"): the format is genuinely interleaved — each
128-element half splits into four 32-wide, mutually disjoint, l-contiguous **strips** (`y0+l`, `y0+l+32`,
`y0+l+64`, `y0+l+96` for `l` in `[0,32)`), and WITHIN one strip's own 32-wide contiguous output range, the
first 16 elements (`l<16`) and last 16 (`l>=16`) each carry their OWN `sc` entry (`sc[2*strip+0]` and
`sc[2*strip+1]` respectively). A 32-aligned GROUP therefore lands entirely within one strip and needs
**two** `(scale, bias)` pairs, not one. `WeightGroup` (the kernel's shared per-group return type) carries
`scale_lo/bias_lo` (elements `[0,16)`) and `scale_hi/bias_hi` (elements `[16,32)`) unconditionally — Q8_0/
Q4_K/Q5_K simply set both pairs equal, at the cost of two redundant float reads the microbenchmark (S6)
measures rather than assumes is free. This keeps `gemv_rows` written ONCE for all four formats rather than
forking the row walk per format.

### 3d. Consequences for every call site (worked through, not asserted)

- **Backbone GEMV becomes per-output-row, not per-output-column.** `op_linear`'s current AXPY loop reads
  one input element and touches every output column; a native-quant DOT reads one whole output row and
  touches every input element. Threading changes from "split columns across threads" to "split rows".
- **`op_tied_head`-style weight tying is unaffected here** — this project's real Qwen4 axes run
  `--tie-embeddings 0` (untied), so `output.weight` (Q4_K) is its own separate tensor with its own native
  bytes, not reusing `token_embd.weight`'s. A tied-embedding build would need the SAME row read twice for
  two different purposes (a row-gather for `token_embd`, a full GEMV for the head) — named as an open
  question for a build that re-enables tying, not resolved here (S8).
- **Backward/training is unaffected by construction**: `param_store.hpp`'s own `static_assert` already
  requires `PARAM_DTYPE != F32` to imply `FORWARD_ONLY` (no honest f32 span for AdamW to read/write). A
  native-quant `PARAM_DTYPE` would inherit that same constraint, not introduce a new one — the training
  path this project actually still exercises (default F32) is provably untouched by this whole class of
  change, exactly as BF16/FP8 already established.
- **Threading is bit-exact across any row split**, checked directly (S5's "threading" test case): each
  output row's own dot is computed from that row's own plane bytes and one shared, read-only activation —
  no partial-sum accumulation across threads, so splitting `[0,n_rows)` into any partition of contiguous
  ranges reproduces the single-threaded result bit for bit. This is the same property `moe_quant_dot.hpp`
  relies on implicitly (each `gemv` call is single-row-independent already); this design makes it an
  explicit, tested parameter (`gemv_plane(..., row_lo, row_hi)`).

---

## 4. Activation quantization

Reuses `moeqd::ActBlocks` verbatim: symmetric round-to-nearest int8, one float scale AND one int32 sum
per 32-element group (`GROUP=32`), quantized once per (token, layer) and reused across every projection
that reads the same input row within that layer — the same hoist B35 already established (quantizing is
`O(hidden_size)` against `O(hidden_size x out_rows)` of dot work, negligible).

No new activation-quantization mechanism was needed or built. The only question worth checking was
whether K-quant's 256-element super-block (with its OWN internal 16-or-32-element sub-block scale
structure) needs a coarser or finer activation group than `GROUP=32` — it does not: every sub-block width
this backbone census found (32 for Q4_K/Q5_K/Q8_0, 16 for Q6_K) divides 32 evenly, so `GROUP=32` is
simultaneously fine enough to align with the narrowest sub-block (Q6_K's 16, handled via the two-pair
`WeightGroup`, S3c) and coarse enough to keep the per-group scale overhead the same B35 already measured
negligible.

---

## 5. Quality plan

### 5a. The weight side is exact (checked, not assumed)

`tests/backbone_quant_dot_tests.cpp`'s lossless-activation case (S6 below) isolates the weight decode from
activation quantization entirely: feeding an activation that int8-quantizes losslessly (integers with a
planted +/-127 in every group, so the quantizer's scale is exactly 1.0), the fused dot must match
`gguf::to_f32`'s own dequantized dot to float-rounding scale. **Measured worst-case relative disagreement
across all four formats: 2.9e-7 to 8.0e-7** (synthetic bytes, real 2560-element rows) and **6.9e-7 to
6.9e-9 on real sidecar bytes** (S6c) — float-rounding only, no layout error survives this check. This is
the same discipline `moe_quant_dot.hpp`'s own B35 gate used, and it is what caught two real defects during
this phase's own development (S6b).

### 5b. The activation side is the one new error source (measured on REAL weights, not synthetic bytes)

A real, honest finding from this development pass: **a synthetic-random-byte K-quant block gives an
unrealistically inflated error for Q6_K specifically.** A first version of the Gaussian-activation-error
test used fully-random `sc`/`d` fields (same idiom B35's own `make_iq_blocks` uses for its pure-scale
formats) and measured **12.1%** relative error for Q6_K — vs. **0.12%** on the model's own real weight
bytes for the same tensor. The reason: fully-random bytes give K-quant's affine `bias` term (`-32*d*sc`)
an unrealistic dynamic range no trained weight distribution actually produces, and Q6_K's bias is the
largest of the three affine formats relative to its own scale. **Fixed by moving the error-SIZE
measurement onto real sidecar bytes** (the committed test's `find_real_picks()`), keeping synthetic random
bytes strictly for the lossless LAYOUT check (S5a), where random data is exactly what is wanted.

**Measured on real backbone tensors (Gaussian activation row, `N(0,1)`, real weight bytes), one large
tensor per format:**

| Format | Real tensor | row_elems | Activation-quantization relative error |
|---|---|---:|---:|
| Q8_0 | `output_hc_down.weight` | 10240 | **0.20%** |
| Q4_K | `output.weight` | 2560 | **3.85%** |
| Q5_K | `blk.0.attn_gate.weight` | 2560 | **1.99%** |
| Q6_K | `blk.0.ssm_out.weight` | 6144 | **0.12%** |

For context, this project's own precedent (`docs/MOE_QUANT_DOT.md` S6e): B35's int8-activation error on
the MoE path was 0.48%-3.3% (single-dot, Gaussian) and produced a 0.29 end-to-end logit L2-relative diff
against the BF16 backbone's own 0.199. The backbone numbers above (0.12%-3.85%) sit in the SAME band as
B35's — **Q4_K's 3.85% is the outlier worth flagging**, plausibly because it is the format with the fewest
mantissa-equivalent bits among the four (0.5391 bytes/elem, the lowest of any format here) applied to two
very large, statistically dense tensors (`token_embd`/`output.weight`, both 635.7M elements) — named as
the one number worth re-checking first if this design is revived, not glossed over.

### 5c. Parity with the batched `forward()` path

**Not addressed by this phase, and flagged rather than silently deferred**: this phase's kernels are
isolated (S "what this phase does not do"). Wiring them in would need the SAME "fuse both paths or scope
the toggle so parity compares like with like" decision `docs/MOE_QUANT_DOT.md` S5 already made explicitly
for B35 — `forward()`/`forward_one()` bit-exact parity (0 since B24) is a real, tested invariant this
project gates on, and a backbone fusion that only touched `forward_one()` (decode) would break it the same
way an unscoped B35 would have. Named here as the load-bearing open question for phase 2, not resolved.

### 5d. End-to-end logit comparison

**Not run** — this phase does not wire the kernels into `decode.cpp`, so there is no `forward_one()` path
to compare against the F32/BF16 reference on the real 48-layer artifact. The isolated numbers in S5a/S5b
are the substitute evidence this phase can offer; a phase-2 wiring pass should re-run
`sub0llm-qwen4-forward --dump-logits` against the existing BF16 (~0.199) and FP8 (~0.43) precedents,
following B24/B33/B35's own established gate exactly.

---

## 6. Kernels — `include/sub0/backbone_quant_dot.hpp`

One shared row walk (`gemv_rows<Plane, UseAvx2>`), four per-format unpackers (`Q8_0Plane`, `Q4KPlane`,
`Q5KPlane`, `Q6KPlane`), two integer-primitive implementations (portable, AVX2), no heap allocation in any
per-call path (AGENTS.md S1 — every kernel writes only caller-supplied spans/pointers).

### 6a. Portable vs. AVX2 — a genuine second path, not just relying on auto-vectorization

`moe_quant_dot.hpp`'s own B35 finding was that a plain int8 loop already auto-vectorizes under `-O3
-march=native` (no reassociation barrier the way float addition has), so it deliberately carries no raw
intrinsics. An explicit AVX2 path was built alongside it anyway, so the two can be differentially tested:
`dot32_avx2`/`dot16_avx2`/`sum32_avx2`/`sum16_avx2` use real AVX2/SSE4.1 intrinsics
(`_mm256_cvtepi8_epi16` + `_mm256_madd_epi16` for the 32-wide primitives, the SSE4.1 128-bit equivalents
for the 16-wide ones Q6_K needs). Both paths are checked to agree EXACTLY (not merely closely) against
each other — integer arithmetic has no precision to lose to reassociation, so any disagreement is a real
bug, never "SIMD noise" (S6b, the test that caught the second real defect below).

### 6b. Two real defects found and fixed during this phase's own development

1. **Q6_K double-subtracted its own zero-point.** `Q6KPlane::group()` first stored `(nib|(hi<<4)) - 32` in
   `wg.q` AND ALSO set `bias_lo/bias_hi = -32*scale`, applying the format's `-32` affine offset twice.
   Caught by the lossless-decode test (S5a): every element of every Q6_K group came out offset by exactly
   one `bias` term — a constant, structured, "still finite, still plausibly scaled, still wrong" signature,
   not noise. Fixed by storing the RAW unsigned 6-bit code (`0..63`) in `wg.q` and letting `bias` alone
   carry the `-32` term, matching Q4_K/Q5_K's own convention.
2. **`dot16_avx2`/`sum16_avx2` silently read only 8 of their claimed 16 lanes.** `_mm_cvtepi8_epi16` sign-
   extends only the LOW 8 bytes of its 128-bit input (its own output is a full 128-bit register of 8
   int16 lanes, with no room for the other 8) — a first version loaded all 16 bytes via `_mm_loadu_si128`
   but converted only that low half, computing an 8-wide dot for what claimed to be Q6_K's 16-wide
   sub-block. Caught by the AVX2-vs-portable EXACT-agreement test (S6a): Q8_0/Q4_K/Q5_K (all 32-wide,
   unaffected by this bug) matched exactly; Q6_K did not, isolating the defect to the 16-wide primitives
   specifically. Fixed by converting the high 8 bytes separately via `_mm_srli_si128(v, 8)` before its own
   `_mm_cvtepi8_epi16`, matching both halves.

Both were reverted-and-reproduced during authoring to confirm the test that found each one actually catches
its own bug (not passing "by accident" alongside an unrelated fix) — the round-trip AGENTS.md's own
gate-verification discipline asks for.

### 6c. Row-range threading entry point

`gemv_plane<UseAvx2>(type_raw, raw, n_rows, row_elems, x, out, row_lo, row_hi)` — `row_lo`/`row_hi`
default to the full range; a caller threads by splitting `[0, n_rows)` into disjoint ranges and handing
each to an independent worker, each writing into its own slice of a shared `out` buffer. Checked bit-exact
across an uneven 3-way split (S "threading" test case).

---

## 7. Memory-fit numbers

| Resident format | Backbone bytes | vs. today (bf16) |
|---|---:|---:|
| F32 (pre-B24) | 18.31 GiB | +100.1% |
| **BF16 (today, B24)** | **9.15 GiB** | baseline |
| **Native quant (this design)** | **3.59 GiB** | **-60.8%** |

A native-quant-resident backbone would free **~5.56 GiB** relative to today's bf16 residency. Peak working
set is not re-measured in this phase (no engine wiring exists to measure it against); the S0Q1 sidecar's
own real MoE working-set precedent (`docs/BACKBONE_PRECISION.md`/`MOE_QUANT_DOT.md`) suggests the win
should track the byte reduction closely, since these kernels add no new large per-call buffers (S1's own
no-heap-allocation discipline).

---

## 8. Bandwidth — measured, with a roofline read from the microbenchmark

`benchmarks/backbone_quant_dot_bench.cpp` (new, plain `main()`, engine-free) measured 1/4/8-thread
portable and AVX2 GEMV throughput at the four largest real backbone tensors per format (row count capped
at 1024 for the two huge Q4_K tensors to keep the tool's own runtime bounded; row_elems and every byte
read stay the tensor's own real values), against a **bf16-promote DOT baseline** — `sub0::bf16_widen`
(the exact branchless-shift function `Bf16CPtr::operator[]` calls) run over the identical row-dot shape.

**Scope caveat, stated precisely rather than glossed over**: the engine's ACTUAL resident bf16 path is an
AXPY kernel over `[in,out]` (S3a), not a DOT over `[out,in]` — `include/sub0/gemv.hpp`, the name this
phase's own task brief used for that kernel, does not exist in this checkout to benchmark directly. What
this microbenchmark isolates is the PER-ELEMENT PROMOTE/DECODE cost at the SAME access shape: does a
native-quant block-decode cost more CPU per element than bf16's cheap widen, at the same row length — the
decode-cost side of the ledger, not the AXPY-vs-DOT layout question S3d discusses qualitatively instead.

**1-thread results** (µs/call, GB/s of native bytes moved):

| Format | Tensor | row_elems | Portable | AVX2 | bf16-dot baseline |
|---|---|---:|---:|---:|---:|
| Q8_0 | `output_hc_down.weight` (320 rows) | 10240 | 140.2 us / 24.83 GB/s | 133.5 us / 26.08 GB/s | 2529.9 us |
| Q4_K | `output.weight` (1024-row cap of 248320) | 2560 | 1153.5 us / 1.28 GB/s | 1205.7 us / 1.22 GB/s | 1970.3 us |
| Q5_K | `blk.3.attn_q.weight` (1024-row cap of 12288) | 2560 | 443.4 us / 4.06 GB/s | 519.2 us / 3.47 GB/s | 2382.7 us |
| Q6_K | `blk.2.attn_qkv.weight` (1024-row cap of 10240) | 2560 | 473.3 us / 4.54 GB/s | 435.8 us / 4.93 GB/s | 2074.9 us |

At 8 threads, portable/AVX2 GB/s roughly triples for Q5_K/Q6_K/Q8_0 (to ~5-13 GB/s) while the bf16
baseline improves by a similar factor, keeping the RELATIVE gap in the same direction throughout. **In
every format and every thread count measured, the fused native-quant kernel's own µs/call is LOWER than
the bf16-promote-dot baseline's** — a surprising direction worth being honest about rather than
over-claiming: the bf16 baseline in this benchmark is a plain, unstructured scalar float-accumulate loop
(`acc += x[i]*bf16_widen(row[i])`), and `docs/optimization/O2_backbone_gemv.md`'s own precedent
(`simd_reduce.hpp`'s header comment) already established that a plain scalar float reduction does NOT
auto-vectorize under this project's toolchain without the multi-accumulator-plus-pragma restructuring —
while this header's own INTEGER dot products vectorize on their own (no such barrier for integer
addition). **This comparison therefore likely overstates the native-quant kernel's true advantage over a
well-optimized bf16 AXPY** (one built with `simd::dot`'s own multi-accumulator shape) rather than against
a naive scalar loop. Flagged as the load-bearing open question for phase 2's own measurement, not
resolved here: re-run this comparison against an actually-optimized bf16 kernel (or the real
`op_linear` AXPY path, once `gemv.hpp` exists) before trusting the magnitude of this result, though the
DIRECTION (native-quant is not paying a decode-cost tax bf16 avoids) is unlikely to reverse given the
integer-vectorization argument holds regardless of which bf16 baseline is used. **§8a re-measured this
against the real kernel: at one thread the direction DOES reverse for the K-quants.**

**A second honest limitation**: Q4_K's numbers are measured at a 1024-row cap of a 248,320-row tensor —
representative of the PER-ROW cost, but not of the tensor's own full-row aggregate time (which would be
~242x larger). Q4_K's own real per-token cost (the full `output.weight` GEMV, every decode token) is
therefore NOT directly read off this table without that scaling, named here rather than left implicit.

### 8a. Re-measured at integration against the real bf16 kernel (2026-09-22)

This phase's branch was cut from `2df23b0`, before O2 merged `include/sub0/gemv.hpp`; that is why the
kernel "did not exist in this checkout". On `main` it does, so the comparison §8 deferred was run at merge
time, on the same host, both tools built in `d196check` (another track's build was running; ~13% total
CPU load, so treat single digits as noise). The shape is directly comparable: Q6_K `blk.2.attn_qkv` is
2560 → 10240, exactly `sub0llm-bench-gemv`'s `gdn in_qkv` row.

| kernel, 2560 → 10240 | bytes | 1 thread | 8 threads |
|---|---:|---:|---:|
| bf16 `gemv::axpy` (DRAM-streamed, persistent OpenMP team) | 52.4 MB | **2623 µs** (20.0 GB/s) | **1048 µs** (50.0 GB/s) |
| Q6_K `bbqd` AVX2 (1024 rows × 10, cache-resident) | 16.4 MB | ~4060 µs (5.3 GB/s) | ~2880 µs (spawn-bound, see below) |

Per-thread throughput of the phase-1 kernels (compressed bytes, cache-resident, AVX2, 1 thread):
**Q8_0 ~26 GB/s, Q6_K ~5.3, Q5_K ~4.3, Q4_K ~1.1.** The reading:

1. **The K-quant kernels are compute-bound, not bandwidth-bound**, at a quarter or less of one core's
   ~20 GB/s bf16 streaming rate, and Q4_K at a twentieth. The integer dot vectorizes. The per-group
   UNPACK does not: `group()` rebuilds a `WeightGroup` element by element (nibble select, high-bit OR)
   and recomputes `k_scale_min` and two `f16_to_f32` for every 32 elements. §8's integer-vectorization
   argument covered the dot, which was never the cost.
2. **At one thread they lose to bf16 despite reading 3.2x fewer bytes.** The cache-resident setup
   flatters them, so this is an upper bound on their speed.
3. **The 8-thread arms here do not measure scaling.** The bench spawns fresh `std::thread`s on every
   ~250 µs call. If the kernels scaled linearly on a persistent team, Q6_K/Q5_K would reach ~42/34 GB/s:
   roughly 2.5x faster than bf16 at this shape, still ~2x from the ~79 GB/s P-core roof. Q4_K would still
   lose to bf16 (~8.8 GB/s: lm_head ~40 ms against bf16's ~24 ms).
4. **Q4_K is anomalous.** It is the simplest K-quant format, yet it runs 4x slower than Q5_K through the
   same row walk. That must be explained before it is optimised.

**Consequence for phase 2:** the lever is real (the bytes are 2.4–3.2x fewer, and the memory fit follows
from §7), but it is only realised with an unpack that streams. The gate is **≥ 10 GB/s of compressed bytes
per thread, streamed from DRAM**, so that 8 P-cores saturate the roof. That means a superblock-at-a-time
AVX2 unpack with scales hoisted per 256 elements, `maddubs`-shaped since the weights are unsigned and the
activations signed (this host also has AVX-VNNI and AVX-VNNI-INT8, confirmed from clang's
`-march=native` macros), plus a load-time repack if the GGUF block order fights
it. Before any of that, the bench must be fixed to compare against `gemv::axpy` with a DRAM-sized tensor
pool and a persistent team, the way `sub0llm-bench-gemv` does.

---

## 9. Resident format — how it gets onto disk

**Not decided in this phase, named as the real fork for phase 2, following B24 Phase 2's own precedent for
exactly this kind of choice:**

- **(a) A backbone sidecar, `.moeq`-shaped.** Extend `moeq::Store`'s own S0Q1 format (or a sibling
  `S0B1`-tagged one) to also carry the backbone's Q8_0/Q4_K/Q5_K/Q6_K planes, keyed by tensor role rather
  than `(layer, expert, which)`. Additive to the existing sidecar contract (AGENTS.md S3's "prefer an
  additive, gracefully-degrading format... over reshuffling existing fields") — an OLD `.moeq` reader
  encountering a NEW-shaped sidecar must refuse cleanly (a version/magic check), not silently misread.
- **(b) A second, wholly separate sidecar file.** Cleaner separation of concerns (the backbone's access
  pattern, S2a's own table, is fundamentally different from the experts' — dense vs. sparse), at the cost
  of a third file alongside `<model>.bin`/`<model>.bin.moeq`.

**Existing-artifact compatibility, traced per AGENTS.md S3, not assumed**: `model_file.hpp`'s
`ParamDtype` enum (`F32=0, BF16=1, FP8=2`) would need a fourth value for a native-quant backbone — this is
NOT a shape change to the existing `S0L5` header/blob (the blob's `PARAM_FLOATS` count is unchanged; only
what the bytes AT that count mean changes), so it follows exactly the precedent B33/B37 already set for
FP8: a new enum value, a new `param_dtype_bytes`/`param_dtype_name` case, and `engine_core.cpp`'s own
file-size-based dtype discriminator (the authoritative signal, per that code's own established
comment) gains a fourth candidate-size branch. **The one genuine complication a native-quant `ParamDtype`
would add that BF16/FP8 did not**: those two are FIXED-WIDTH per element (2 or 1 bytes, uniformly), so a
file-size-based discriminator is a single division. Native quant is NOT fixed-width per element — Q8_0/
Q4_K/Q5_K/Q6_K each have a different bytes-per-element ratio, and this project's real backbone MIXES all
four per-tensor (S2a's own table). A native-quant `PARAM_DTYPE` therefore cannot reuse the flat
`param_t[PARAM_FLOATS]` arena shape at all — it needs a per-tensor type tag alongside a per-tensor byte
offset, structurally closer to `moeq::Desc`'s own `{type_raw, in_f, out_f, off, bytes}` than to
`param_store.hpp`'s single homogeneous array. **This is the load-bearing reason a native-quant backbone is
a genuinely new storage shape, not a fourth `param_t` alias** — named here so phase 2 does not
underestimate the change's real shape by analogy by to BF16/FP8's much smaller step.

---

## 10. Phase 2 — what wiring would need, named rather than attempted

1. **Dispatch by tensor type at each call site.** Every `op_linear`/`op_tied_head`-style backbone
   consumer would need an `if constexpr`/runtime-dispatch branch reading each tensor's own `type_raw`
   (never its role, per the per-tensor-mixed-quantization discipline S2a's table already follows) and
   calling `bbqd::gemv_plane` instead of the existing AXPY loop for that tensor.
2. **A configurator flag** (e.g. `--backbone-quant-dot`, mirroring `--moe-quant-dot`'s own
   default-off `constexpr bool` pattern, AGENTS.md S2) gating the whole path, with a configure-time
   refusal if set without a native-quant-resident backbone sidecar (S9) to read from.
3. **What stays bf16**: F32 tensors (routers, norms, biases — no smaller native form exists) and any
   tensor whose row width does not divide `GROUP=32` (none found in the real census, S2a, but the refusal
   path — `fusable()` returning `false` — must be honoured by a real caller, not just this phase's tests).
4. **The AXPY-vs-DOT reconciliation (S3d)** is the single largest open engineering question this document
   does not resolve: either `op_linear`'s own loop shape changes for native-quant tensors specifically
   (row-threaded DOT instead of column-threaded AXPY), or an activation-side transpose/restructuring
   bridges the two — both are real, non-trivial engine changes outside this phase's scope.
5. **Parity and end-to-end quality gates** exactly as S5c/S5d name: `forward()`/`forward_one()` bit-exact
   parity decided explicitly (fuse both or scope the toggle), and a real `--dump-logits` comparison against
   the BF16 (~0.199) and FP8 (~0.43) precedents on the real 48-layer artifact.

---

## 11. What this phase deliberately does not do

- Wire `bbqd::` into `op_linear`, `decode.cpp`, or any production call site.
- Change `sub0llm-transplant.cpp`, `moe_quant_dot.hpp`, `gemv.hpp` (which does not exist in this
  checkout), or any existing math header.
- Decide the on-disk sidecar format (S9) beyond naming the fork.
- Measure end-to-end logit quality or real decode throughput (both require engine wiring this phase does
  not do).
- Add Q4_K support to anything outside this header — it was added here BEYOND the task's named three
  formats (Q5_K/Q6_K/Q8_0) because the census (S2a) found it real and substantial (682.03 MiB, 17.7% of
  native bytes, the two largest single tensors in the whole backbone).

---

## 12. Phase 2a — checkpoint (2026-09-22, PAUSED mid-implementation)

Work stopped here on a pause request before the streaming kernels were ported into the real header, the
bench was fixed, or the tests were updated. This section records what is verified so the next session
resumes from evidence, not from scratch (AGENTS.md's own "a parked toggle is a finding banked" spirit,
applied to an in-flight task rather than a finished one).

### 12a. The Q4_K anomaly (§8a point 4), explained — NOT denormals

Hypothesis tested first, because this project has a real precedent at almost the same magnitude
(`docs/host-cpu-arrow-lake-hx` memory note: FTZ/DAZ fixed a measured 4.4x slowdown, close to Q4_K's own
"4x slower than Q5_K"): **REJECTED**. A standalone scan of every superblock's `d`/`dmin`/`d*sc`/`dmin*m`
term across 10,240 real `output.weight` superblocks found zero subnormal values, and forcing
`_MM_SET_FLUSH_ZERO_MODE`/`_MM_SET_DENORMALS_ZERO_MODE` ON changed the measured time by <1% (1239.5 vs
1246.2 µs/call). Also ruled out: the anomaly is NOT data-dependent — real `output.weight` bytes and
freshly-random synthetic bytes of the identical shape (1024×2560, Q4_K) time within 0.5% of each other
(1404.5 vs 1398.3 µs/call), and the cost scales linearly with row count (128 rows: ~205 µs, 1024 rows:
~1400 µs), ruling out a paging/allocation artifact tied to the specific 248,320-row tensor.

**Actual cause, found by reading the generated assembly (`clang++ -O3 -march=native -S`) for
`gemv_rows<Q4KPlane,true>` and `gemv_rows<Q5KPlane,true>` side by side**: Clang auto-vectorized
`Q5KPlane::group()`'s 32-element unpack loop (which does MORE work — nibble extract AND a qh high-bit OR)
into ~15 AVX2 instructions including `vgf2p8affineqb`, but did NOT vectorize `Q4KPlane::group()`'s
simpler nibble-only unpack loop at all — it compiled to 64 sequential scalar `movzx`/`and`/`mov`
instructions (32 for the low-nibble branch, 32 for the high-nibble branch, selected by a hoisted
loop-invariant branch). Same compiler, same flags, same `-O3 -march=native`, functionally similar
loops — LLVM's vectorizer / idiom-matcher simply did not fire for the simpler case. This is an
empirically confirmed compiler-heuristic quirk, not a property of the Q4_K format itself, and it
independently justifies this phase's plan (don't rely on autovectorization for the unpack — write
explicit AVX2 intrinsics), rather than needing a Q4_K-specific workaround.

**A second, smaller finding from the same assembly read, affecting all four formats equally**: none of
`Q8_0Plane::group()`/`Q4KPlane::group()`/`Q5KPlane::group()` were inlined into `gemv_rows` — the
generated code has a real `call` instruction (plus a `vzeroupper` state-transition before it) on every
group, paying full calling-convention overhead (stack frame, register spill/fill) on top of whatever the
callee itself costs. The streaming design below eliminates this by writing one self-contained per-row
function per format instead of calling a struct-returning `group()` once per 32 elements.

### 12b. Streaming design, verified bit-exact in a scratch harness — NOT yet ported into the real header

Design (matches the task brief's own shape): one 256-element superblock at a time, `d`/`dmin`/the 8
sub-block `(sc, m)` pairs decoded ONCE per superblock (`KScaleTable`, was: once per 32-element group, an
8x-redundant `f16_to_f32` + `k_scale_min` + pointer-arithmetic cost), and the nibble/high-bit unpack done
via explicit AVX2 intrinsics instead of relying on the compiler:

- `nibble_lo(bytes) = bytes & 0x0F` (byte-safe AND, trivial).
- `nibble_hi(bytes) = (bytes >> 4 as 16-bit lanes) & 0x0F` — llama.cpp's own idiom; re-derived and
  confirmed safe by hand (a 16-bit lane right-shift of 4 moves each byte's own high nibble into that
  same byte's position in the shifted lane without leaking bits from the neighbour byte, worked through
  bit-by-bit rather than assumed).
- Q5_K's single `qh` bit (`(qh[l] >> bit_idx) & 1`, `bit_idx == sub` exactly — re-derived algebraically
  from the scalar reference's `bit = (hi_nibble?2:1) << (2*(sub/2))`, not copied) is extracted via
  mask-then-compare (`and` with `1<<bit_idx`, `cmpeq` against the same constant, `and` with a 16-broadcast)
  rather than a runtime-count lane shift, specifically to avoid the cross-byte-contamination hazard a
  16-bit lane shift by a runtime count can have when the masked value isn't provably confined to one
  byte's own bit range — branch-free and byte-safe by construction, verified against the scalar reference
  rather than trusted by inspection.
- Q6_K's 2-bit `qh` field (`(qh[l] >> shift) & 3`, `shift = strip*2 ∈ {0,2,4,6}`) is extracted via
  mask-then-compile-time-immediate-shift, selected by a 4-way `switch` on `shift` so every shift the
  compiler emits is a known-safe immediate (each case hand-verified not to cross a byte boundary, the
  same way `nibble_hi` was).
- Q6_K also needs a per-16 (not per-32) activation sum for its `bias_lo`/`bias_hi` split that
  `moeqd::ActBlocks::gsum` (per-32) doesn't carry. Phase 1's `group()`-based path recomputed this from
  scratch on EVERY (row, group) pair via `sum16_avx2` — an O(n_rows) redundant recompute of a value that
  only depends on the activation column, not the row. Designed (not yet perf-measured with caching): a
  small `Gsum16` helper built ONCE per `gemv_plane` call (per activation row, shared across all output
  rows), mirroring `ActBlocks`' own "resize only when the width changes" allocation-free-in-steady-state
  contract rather than touching `moe_quant_dot.hpp` (out of scope for this phase).

**Verification method**: a standalone scratch program (`clang++ -O3 -march=native`, not part of the
build) implements the four pieces above and compares the result against
`bbqd::gemv_plane<true>(...)` (today's `group()`-based AVX2 path, unchanged) row by row, requiring EXACT
equality (`==`, not a tolerance) — the same discipline this file's own §6b used for the two defects phase
1 caught. Result: **Q4_K, Q5_K, and Q6_K all bit-exact** against the existing AVX2 reference, both on
multi-superblock synthetic bytes (4 rows × 2560 elements, 10 superblocks/row) and on 32 real rows of the
real `output.weight` (Q4_K) tensor (0/32 mismatches). One real bug was caught and fixed during this
verification: Q6_K's scale table (`sc`) must be offset by `half*8` within its 16-entry array — re-derived
from `gguf.hpp`'s `dequantize_q6_k`, which advances `sc += 8` at the end of each outer `half` iteration
(easy to miss since it is a plain pointer increment at the bottom of a loop, not inline with the indexing
that uses it); the first draft of the scratch kernel omitted it and every row mismatched until fixed.

**First throughput number** (scratch prototype, cache-resident real `output.weight` bytes, single
thread, NOT yet run through the fixed DRAM-streamed bench from task 0): **Q4_K went from phase 1's 1.1
GB/s to ~9.4–10.5 GB/s** — roughly 9x, at or just under the phase-2 gate of ≥10 GB/s/thread compressed
bytes. Q5_K and Q6_K were queued for the same measurement when the pause request arrived; no number for
them yet.

### 12c. Explicit next step to resume from

1. Port the four verified scratch functions (in this session's scratchpad,
   `streaming_verify.cpp` — `dot_row_q4_k`, `dot_row_q5_k`, `dot_row_q6_k`, plus the `KScaleTable`/
   `Gsum16`/`nibble_lo`/`nibble_hi`/`qh_bit_to_hi4`/`qh_2bits_to_hi4`/`dot32_avx2_reg`/`dot16_avx2_reg`
   helpers they use) into `include/sub0/backbone_quant_dot.hpp` under `detail::`, gated on
   `row_elems % 256 == 0` (true for every real backbone tensor per the census, §2a) with a fallback to
   the existing `gemv_rows<Plane,true>` for the general case.
2. Restructure the public entry point to auto-dispatch (moeqd's own `kAvx2Kernels`/`gemv_best`
   convention, not a caller-chosen `UseAvx2` template bool — the task brief's own explicit instruction),
   keeping `detail::gemv_plane_portable`/`detail::gemv_plane_avx2` as the two differential-test targets
   (mirroring `tests/moe_quant_tests.cpp`'s own "O1" test shape).
3. Add a `Threads`-templated OpenMP entry point mirroring `gemv::axpy<Threads>` exactly, including its
   `omp_in_parallel()` nesting guard.
4. Fix `benchmarks/backbone_quant_dot_bench.cpp` per task item 0: real `gemv::axpy` baseline (link
   OpenMP the way `sub0llm-bench-gemv` does in root `CMakeLists.txt`), a pool of distinct real tensors
   exceeding 3x L3 so every call streams from DRAM, persistent OpenMP team (no per-call `std::thread`
   spawn), 1/2/4/8/16 threads.
5. Update `tests/backbone_quant_dot_tests.cpp`: replace every `gemv_plane<false>`/`gemv_plane<true>`
   call site with `detail::gemv_plane_portable`/`detail::gemv_plane_avx2` (or the new auto-dispatching
   `gemv_plane`, per what each case is actually checking), add a streaming-vs-portable exact-agreement
   case, a `Gsum16` correctness case, and a threading bit-exactness case for the new `Threads`-templated
   entry point.
6. Re-run the diagnostic scripts already in this session's scratchpad
   (`denormal_diag.cpp` — FTZ/DAZ hypothesis, rejected; `anomaly_diag2.cpp` — data-vs-algorithm
   isolation) are NOT needed again; their findings are recorded in §12a above and don't need
   re-verification, only the streaming kernels themselves need porting and re-testing in-tree.
7. Only after 1–5: run `sub0_frontend_tests` (must stay 141,609/257 outside `[backbonequant]`), the
   `[backbonequant]` tag alone, the fixed bench at real shapes, write up final numbers in this section
   (replacing "not yet measured"), and run `cpp-review` over the diff before calling phase 2a done.
