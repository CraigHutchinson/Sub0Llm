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

**Phase 2a status (2026-09-22): DONE — streaming kernels ported, bench fixed, threaded, tested.** See §12
for the full write-up: the Q4_K anomaly explained (disassembly, not denormals), the llama.cpp reference
study, three measured optimization passes (AGENTS.md §13), and the honest gate result — the streaming
kernels clear the ~10 GB/s/thread bar only approximately (within run-to-run noise) and do NOT clear the
~60 GB/s/8-thread aggregate bar, but DO beat the real `gemv::axpy` bf16 kernel on WALL-CLOCK time per row
at 8 threads for every format (the metric that answers "is a real token faster"), because native bytes
read are 1.9–3.6x fewer even though the native kernel's own GB/s is lower. Parked at this state per
AGENTS.md §13 (a genuine 3-pass mechanism, not reverted); the concrete next lever is named in §12f.

**Phase 2b-1 status (2026-09-22): DONE — the `S0B1` resident-format sidecar and its loader, resolving §9's
own fork.** See §13 for the full write-up: `include/sub0/backbone_quant.hpp` (new), the writer wired into
`tools/sub0llm-transplant.cpp` behind an opt-in `--backbone-quant` flag (default off, byte-identical
existing output confirmed), `tests/backbone_quant_tests.cpp` (new). Storage/loader only — still nothing in
`src/` reads this sidecar; phase 2b-2 (the parallel, separate work package finishing
`backbone_quant_dot.hpp`'s kernel wiring) is what will eventually consume it. A real finding from actually
running this against the file (AGENTS.md §9): the backbone is not uniformly one format per role the way
§2a's aggregate table suggests — layer 2 is a genuine per-layer outlier (its `ffn_gate_shexp`/
`ffn_up_shexp` are Q6_K, every other layer's are Q5_K), the same phenomenon §3a-bis already documented for
the routed experts.

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

**bytes/elem corrected at integration (2026-09-22):** the K-quant rows first read 0.5391/0.5250/0.6250.
The true per-element cost is each format's block size over 256 elements — Q4_K 144/256 = 0.5625, Q5_K
176/256 = 0.6875, Q6_K 210/256 = 0.8203 — which is what this table's own MiB and element counts already
implied. Only the derived column was wrong; the byte totals, the 3.59 GiB figure and every fit claim
come from the totals and are unaffected. The bf16-to-native ratio per format is therefore 1.88x (Q8_0),
2.44x (Q6_K), 2.91x (Q5_K), 3.56x (Q4_K), not the 2.4–3.9x first stated.

| Format | Tensors | Elements | Native bytes | bytes/elem | Consumers (representative) |
|---|---:|---:|---:|---:|---|
| F32 | 553 | 78.3M | 298.70 MiB | 4.000 | norms, router (`ffn_gate_inp`), `ssm_alpha`/`ssm_beta`, `hc_*_inject`, biases |
| Q8_0 | 242 | 714.3M | 723.83 MiB | 1.0625 | Gated Residual up/down (`hc_attn_up/down`, `hc_ffn_up/down`), MoE shared-expert down |
| Q4_K | 2 | 1271.4M | 682.03 MiB | 0.5625 | `token_embd.weight` (embed), `output.weight` (untied lm_head) |
| Q5_K | 212 | 2219.7M | 1455.35 MiB | 0.6875 | GDN in-proj (`attn_qkv`, `attn_gate`), QSA (`attn_q/k/v/output`), MoE shared gate/up |
| Q6_K | 40 | 611.5M | 478.34 MiB | 0.8203 | GDN out-proj (`ssm_out`) |
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
mantissa-equivalent bits among the four (0.5625 bytes/elem, the lowest of any format here) applied to two
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

## 12. Phase 2a — streaming AVX2 kernels, ported, benched, threaded, tested (2026-09-22)

This section records the complete phase 2a pass: what changed in `include/sub0/backbone_quant_dot.hpp`,
the llama.cpp reference study behind it, three measured optimization passes (AGENTS.md §13), the fixed
DRAM-streamed bench, the new tests, and an honest read of where the numbers land against the stated gate.

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

### 12b. Streaming design, in `include/sub0/backbone_quant_dot.hpp`

One 256-element superblock at a time, `d`/`dmin`/the 8 sub-block `(sc, m)` pairs decoded ONCE per
superblock (`detail::KScaleTable`, was: once per 32-element group, an 8x-redundant `f16_to_f32` +
`k_scale_min` + pointer-arithmetic cost), and the nibble/high-bit unpack done via explicit AVX2
intrinsics instead of relying on the compiler:

- `nibble_lo(bytes) = bytes & 0x0F` (byte-safe AND, trivial).
- `nibble_hi(bytes) = (bytes >> 4 as 16-bit lanes) & 0x0F` — llama.cpp's own idiom (§12c); re-derived and
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
  only depends on the activation column, not the row. `detail::Gsum16` is built ONCE per
  `gemv_plane_avx2` call (per activation row, shared across all output rows that call handles), mirroring
  `ActBlocks`' own "resize only when the width changes" contract rather than touching `moe_quant_dot.hpp`
  (out of scope for this phase). **Not yet hoisted to a caller-owned, cross-call-persistent buffer** —
  there is no decode-loop call site yet to own it across tokens; named as phase 2b's job if this path is
  adopted.

`Q8_0Plane`/`Q4KPlane`/`Q5KPlane`/`Q6KPlane::group()` and `gemv_rows` are UNCHANGED from phase 1 — they
remain the correctness reference (`detail::gemv_plane_portable`) and the fallback for the (real-tensor-
absent) case where `row_elems % 256 != 0` (`detail::gemv_plane_avx2` still calls `gemv_rows<Plane,true>`
there). The public `gemv_plane<Threads = 1>` auto-dispatches on `bbqd::kAvx2Kernels` (moeqd's own
`kAvx2Kernels`/`gemv_best` convention, not a caller-chosen `UseAvx2` template bool) and, for `Threads >
1`, splits `[row_lo, row_hi)` across a persistent OpenMP team exactly the way `gemv::axpy<Threads>` does,
including its `omp_in_parallel()` nesting guard.

### 12c. Reference study — llama.cpp's own AVX2 K-quant kernels (AGENTS.md §5)

Read directly from `D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-cpu\arch\x86\quants.c`
(`ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q5_K_q8_K`, `ggml_vec_dot_q6_K_q8_K`, `ggml_vec_dot_q8_0_q8_0`),
not paraphrased from training data. Two techniques were taken from it and re-derived onto this project's
own conventions, not copied verbatim:

1. **The `utmp[4]` branch-free scale/min unpack** (Q4_K/Q5_K's AVX2 path):
   ```c
   static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
   uint32_t utmp[4];
   memcpy(utmp, x[i].scales, 12);
   utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
   const uint32_t uaux = utmp[1] & kmask1;
   utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
   utmp[2] = uaux;
   utmp[0] &= kmask1;
   ```
   llama.cpp feeds `utmp` straight into `_mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3..0]))` — one vector
   register holding all 8 scales then all 8 mins. This project's per-32 `ActBlocks` still needs each
   sub-block's scale/min as an individually-addressable `float` (not a shared vector register, since the
   activation-side scale differs per 32-group — see §12g), so `detail::KScaleTable::load` re-expresses
   the SAME bit-unpack with plain byte indexing into `utmp` instead, checked bit-exact against
   `gguf::k_scale_min`'s own byte-at-a-time reference for all 8 sub-block indices
   (`tests/backbone_quant_dot_tests.cpp`'s own "KScaleTable" case) before being trusted.
2. **The nibble/high-bit unpack shape** (`vpandn`/`vpsrlw`-style per-16-bit-lane bit tricks across all
   three K-quant kernels) confirmed the general APPROACH (explicit intrinsics, not a scalar loop left to
   the auto-vectorizer) but was NOT copied instruction-for-instruction: llama.cpp processes a whole
   256-element superblock per AVX2 call with a SINGLE combined horizontal reduction (only possible
   because its own activation is quantized per-256, `block_q8_K`, letting the K-quant's own integer scale
   fold into the accumulator before any float conversion). This project kept per-32 `ActBlocks`
   (§12g explains the tradeoff that decision makes), so `dot_row_q{4,5,6}_k_avx2` below do their own
   independent per-32-group unpack+dot+reduce, matching this project's own accumulation order (the SAME
   `gemv_rows`-established sequence, `sub = 0, 1, ..., 7`) rather than llama.cpp's per-256 one.

### 12d. Three measured optimization passes (AGENTS.md §13)

All numbers: real Qwen3.8-Flash-Next UD-IQ1_S shards, this host (Arrow Lake-HX, 8P+16E, 36 MiB L3), 1
thread, compressed bytes/sec. Passes 1–2 measured cache-resident (a standalone scratch harness, before
the bench fix); pass 3 is the first DRAM-streamed number and is what §12e cites going forward.

| Pass | Change | Q8_0 | Q4_K | Q5_K | Q6_K |
|---|---|---:|---:|---:|---:|
| 0 (phase 1 baseline) | `group()`-based, scalar unpack | ~26 | 1.1 | 4.3 | 5.3 |
| 1 | streaming design ported (superblock-cached scale, vector unpack) | 13.30 | 7.75 | 7.82 | 8.58 |
| 2 | + `KScaleTable` branch-free bit-unpack (§12c technique 1) | 9.76 | 9.62 | 9.81 | 8.49 |
| 3 | bench methodology fix (§12e) — DRAM-streamed, real number | 12.57 | 10.09 | 9.94 | 7.99 |

Pass 1→2 is a real, reproducible improvement for Q4_K/Q5_K (+24%/+25%); Q6_K and Q8_0 moved inside
run-to-run noise (±10–15% between repeated runs of the identical binary on this host, observed directly).
Pass 3 is not a kernel change at all — it is the bench fix itself (§12e) revealing that passes 1–2's own
numbers were partly measuring cache-residency and per-call OpenMP-region spin-up overhead, not the
kernels' true DRAM-streamed throughput; it is included as the third pass because it is the measurement
that finally isolates the mechanism's own ceiling from the harness measuring it.

**Per AGENTS.md §13, this is parked, not reverted.** The mechanism (streaming AVX2 unpack) delivered a
real 7–9x improvement over phase 1 at every format and stays in the tree, default-off in the sense that
nothing in `src/`/`tools/` includes this header yet (unchanged from phase 1 — see §11). §12g names the
concrete next lever for a 4th pass, should someone continue this thread.

### 12e. Threading and the bench fix (task 0)

`gemv_plane<Threads>` (§12b) is `bbqd`'s own `gemv::axpy<Threads>`-shaped entry point.
`tests/backbone_quant_dot_tests.cpp`'s "gemv_plane<Threads> is bit-exact across 1/2/4 threads" case checks
this directly on an uneven 37-row split (not divisible by 2 or 4) for all four formats.

`benchmarks/backbone_quant_dot_bench.cpp` was rewritten per this task's own item 0/step 4 findings:

- **Baseline**: the real `sub0::gemv::axpy<Threads>` (`include/sub0/gemv.hpp`), not a naive scalar loop —
  phase 1's own bf16 arm measured ~26 GB/s cache-resident, which was never the real kernel's own number.
- **DRAM-streamed**: both arms cycle through a POOL exceeding 3× this host's 36 MiB L3 (108 MiB) —
  native's pool is built from REAL tensors (or, for Q4_K's two 248,320-row planes, several disjoint
  real row-ranges of the SAME tensor), the bf16 pool from several distinct random matrices at the
  matching `(row_elems, out_dim)` shape, the identical technique `sub0llm-bench-gemv` already uses.
- **Persistent OpenMP team**: `#pragma omp parallel for num_threads(Threads)` per call (`gemv_plane<Threads>`
  and `gemv::axpy<Threads>` both), relying on libomp's own idle-thread pool rather than spawning
  `std::thread`s per call, the way phase 1's version did.
- **A second, real methodology finding during this pass**: the pool's PER-ENTRY row cap
  (`kMaxRowsPerEntry`) itself mattered. At 3072 rows/entry the bf16 BASELINE's own 8-thread number came
  in well under this host's documented ~50–65 GB/s range (found: ~44–58 GB/s) — both arms were paying a
  per-call OpenMP parallel-region spin-up on every pool entry, not a property of either kernel. Raising it
  to 16384 rows/entry brought the bf16 baseline into its documented range (61.51/68.06 GB/s at 8/16
  threads for the Q4_K-shaped baseline, §12f) and is what §12f's numbers were measured at. A real decode
  call is smaller than 16384 rows for most projections, so amortizing OpenMP region overhead across MORE
  than one call (persistent-team reuse across layers, not just within one call) is a real, unresolved
  question for phase 2b's own wiring — named here, not solved by this constant.

### 12f. Final DRAM-streamed numbers (real Qwen3.8-Flash-Next shards, this host, `--seconds 0.6`)

| Format | Pool | 1 thr | 2 thr | 4 thr | 8 thr | 16 thr |
|---|---|---:|---:|---:|---:|---:|
| Q8_0 (native GB/s) | 33 entries, 109.6 MiB | 12.57 | 19.82 | 31.37 | 35.56 | 36.61 |
| Q8_0 (bf16 `axpy` GB/s) | matching shape | 30.11 | 36.29 | 44.34 | 54.11 | 45.92 |
| Q4_K (native GB/s) | 5 entries, 112.5 MiB | 10.09 | 16.53 | 20.04 | 28.65 | 49.57 |
| Q4_K (bf16 `axpy` GB/s) | matching shape | 29.11 | 34.44 | 49.00 | 61.51 | 68.06 |
| Q5_K (native GB/s) | 15 entries, 111.3 MiB | 9.94 | 14.70 | 21.74 | 35.07 | 36.65 |
| Q5_K (bf16 `axpy` GB/s) | matching shape | 27.78 | 35.10 | 42.39 | 53.17 | 58.46 |
| Q6_K (native GB/s) | 9 entries, 110.7 MiB | 7.99 | 13.72 | 19.91 | 32.25 | 38.99 |
| Q6_K (bf16 `axpy` GB/s) | matching shape | 28.57 | 31.49 | 38.19 | 45.76 | 38.85 |

**Against the literal stated gate (≥10 GB/s/thread, ~60 GB/s aggregate at 8 threads)**: Q4_K clears the
1-thread bar (10.09); Q5_K and Q6_K sit just under it (9.94, 7.99) — within the ±10–15% run-to-run noise
this host shows on repeated identical runs, so "approximately at the bar" is the honest characterization,
not "clears it". **None of the four formats reach ~60 GB/s aggregate at 8 threads** (28.65–35.56 GB/s
measured); Q4_K reaches 49.57 GB/s at 16 threads, the closest any format gets.

**The metric that actually answers "is a real token faster", computed from the SAME table (wall-clock
time per output row, native vs. bf16 `axpy`, at the matching logical `(row_elems, out_dim)` shape)**:
despite lower raw GB/s, native reads 1.9–3.6x fewer bytes per row (§7's own ratio table), so total time
can still be lower. At **8 threads, native wins on wall-clock time for all four formats**: Q8_0 15% faster,
Q4_K 40% faster (1.67x), Q5_K 48% faster (1.93x), Q6_K 42% faster (1.72x). At 1 thread the result is mixed:
Q4_K 19% faster, Q5_K 4% faster, but Q8_0 27% SLOWER and Q6_K 47% SLOWER than `gemv::axpy` at 1 thread —
the per-superblock fixed cost (scale decode, 8 independent horizontal reductions, §12g) is a larger share
of the total at low thread counts, and for Q6_K's larger row width in particular has not yet been paid
down by DRAM-bandwidth savings the way it has by 8 threads.

### 12g. What's next, if this thread continues — an architectural ceiling, not a bug

The literal GB/s gate is not fully met, and the reason is visible in the kernel shape, not a mystery:
**each streaming kernel does 8 independent `hsum256_epi32`-style horizontal reductions per 256-element
superblock — one per sub-block — and that count is very likely irreducible under this project's own
choice to keep `moeqd::ActBlocks`' per-32-element activation scale** (§3b/§4's own design call, reused
verbatim rather than re-derived, per AGENTS.md §3). llama.cpp avoids this entirely: because its own
activation (`block_q8_K`) carries ONE scale per 256 elements, it folds each K-quant sub-block's own
INTEGER scale (0–63, a plain small int) into the int16/int32 accumulation via `_mm256_madd_epi16` BEFORE
any float conversion, and only converts to float and reduces ONCE per superblock, not once per sub-block.
This project's own per-32 activation scale is a `float`, not a small int, so an analogous fold is not
directly available without adopting llama.cpp's own coarser (per-256) activation granularity.

**This was evaluated, not overlooked, and deliberately deferred rather than adopted in this pass**: doing
so would mean building a new, `block_q8_K`-shaped activation-quantization scheme (a `bsums`-per-16-style
struct) alongside — not instead of, since Q8_0 still wants its own native per-32 granularity —
`moeqd::ActBlocks`, a materially bigger and riskier change within this phase's remaining time, and the
per-32 approach ALREADY gets close to the stated gate and (per §12f) already wins on real wall-clock time
at every format once enough threads are in play. Named here as the concrete 4th-pass lever for whoever
picks this back up: a `Q8_K`-shaped per-256 activation quantizer, used ONLY by the K-quant streaming
kernels (Q8_0 keeps `ActBlocks`, whose own per-32 granularity already matches its native block exactly),
would let the K-quant sub-block scale fold into the integer domain the way llama.cpp's own kernel does,
cutting 8 horizontal reductions per superblock to 1 — the single biggest remaining lever this analysis
found. The accuracy tradeoff (§4/§5b's own numbers, unchanged by this phase since the activation side was
not touched: Q8_0 0.20%, Q4_K 3.85%, Q5_K 1.99%, Q6_K 0.12%) would also need re-measuring against a
per-256 scheme before adopting it, per the same discipline §5b already established.

### 12h. Test counts (`sub0_frontend_tests`, this host, `out/build/o5p2`)

- `[backbonequant]` alone: **3,891 assertions in 10 test cases** (phase 1 was 2,848/6 — the 4 new cases
  are the non-256-aligned-row fallback, `gemv_plane<Threads>` bit-exactness, `KScaleTable` vs.
  `gguf::k_scale_min` exact agreement, and `Gsum16` vs. a direct re-sum).
- `~[backbonequant]`: **141,609 assertions in 257 test cases** — matches the required baseline exactly,
  confirming nothing in this phase leaked into any other test path.
- Full suite: **145,500 assertions in 267 test cases**.

## 12i. Integration re-verification (primary agent, 2026-09-22)

Rebuilt from the merge on a quiet host (no other agent running) and re-run independently.

**Gates reproduce exactly:** `[backbonequant]` 3,891 / 10; `~[backbonequant]` 141,609 / 257, unchanged
from main; full frontend suite 145,500 / 267.

**Throughput reproduces**, within this host's noise (my run / the branch's run, GB/s of compressed bytes
at 1 thread): Q8_0 14.5 / 12.6, Q4_K 9.1 / 10.1, Q5_K 8.7 / 9.9, Q6_K 7.9 / 8.0. At 8 threads:
35.8 / 35.6, 37.1 / 28.7, 34.9 / 35.1, 28.0 / 32.3.

**The wall-clock claim holds, and is the number that matters.** Same logical GEMV, same row width, native
against the real `gemv::axpy`, 8 threads, µs per output row derived from each arm's own bytes and GB/s:

| format | row_elems | native µs/row | bf16 µs/row | native speedup |
|---|---:|---:|---:|---:|
| Q8_0 | 320 | 0.010 | 0.012 | 1.21x |
| Q4_K | 2560 | 0.039 | 0.085 | 2.17x |
| Q5_K | 2560 | 0.050 | 0.097 | 1.93x |
| Q6_K | 6144 | 0.180 | 0.327 | 1.82x |

So the literal gate (≥10 GB/s/thread, ~60 GB/s at 8) is the wrong gate, and it was mine: it assumed the
kernel had to reach the bf16 path's bandwidth to win. It does not, because it moves 1.9–3.6x fewer bytes
to do the same arithmetic. **On the question phase 2 exists to answer — is a token faster — the answer at
8 threads is yes for all four formats.** The GB/s gate is retained as the measure of how much of the
DRAM roof is still unused (28–37 of ~79 GB/s), which is what §12g's fourth pass would target.

**Corrections applied at merge:** §2a's bytes/elem column (see the note there) and the 2.4–3.9x ratio,
which is really 1.9–3.6x once Q8_0's 1.88x is included.

**One finding recorded, not fixed:** `Gsum16` heap-allocates inside `gemv_plane_avx2`, per call and per
thread. That is an AGENTS.md §1 violation the moment a decode loop calls it, so phase 2b must pass in a
caller-owned buffer. A `TODO(phase 2b)` now sits at the allocation, because §12 of AGENTS.md exists
precisely because this class of thing merges quietly and is not found until a profile says so.

**What is still unproven:** everything end-to-end. No engine call site uses this header, so there is no
decode throughput number and no logit-quality number. The per-row win above is a kernel measurement;
whether it survives contact with the real decode depends on phase 2b's layout reconciliation (§3d), which
is unsolved.

---

## 13. Phase 2b-1 — the `S0B1` resident-format sidecar and its loader (2026-09-22)

Resolves §9's own fork: **(a)** — a sibling sidecar to `moeq`'s own `S0Q1`, tagged `S0B1`, not folded into
the existing `S0L5`/`S0Q1` formats. Storage and a read-only loader only, per this phase's own brief:
`include/sub0/backbone_quant.hpp` (new, engine-free), a writer wired into
`tools/sub0llm-transplant.cpp` behind an opt-in `--backbone-quant <path>` flag (default off), and
`tests/backbone_quant_tests.cpp` (new). Nothing in `src/` reads this sidecar yet — that is phase 2b-2's
job, a separate, parallel work package finishing `backbone_quant_dot.hpp`'s own kernel wiring, whose three
files (`include/sub0/backbone_quant_dot.hpp`, its tests, its bench) this phase deliberately did not touch.

### 13a. Format

`Header` (48 bytes: magic `S0B1`, version, `n_layers`, `n_tensors`, `data_off`/`data_bytes`,
`model_param_floats`) + a `Desc` table (40 bytes each: `role`, `layer`, `type_raw`, `in_f`/`out_f` =
GGUF's own `ne[0]`/`ne[1]`, `off`, `bytes`) + the payload, mmapped by `Store` exactly the way `moeq::Store`
already maps `S0Q1` (`FileMap`, RAII, never eagerly read). Both struct sizes are `static_assert`-pinned.

**Identity is `(Role, layer)`, not a GGUF name.** `Role` is a 15-member `enum class` — one member per
`transplant::Dest` this sidecar carries (`TokEmb`, `LmHead`, `GrAttnDown/Up`, `GrFfnDown/Up`,
`GrExitDown/Up`, `MoeSharedGate/Up/Down`, `QsaQGateProj`, `QsaKProj/VProj/OProj`) — mirroring `moeq::Which`'s
`(layer, expert, which)` precedent with a flat per-role enum instead (no expert axis here). `role_pattern()`
reads each role's own GGUF source-name pattern from `transplant::recipe_for()` directly (AGENTS.md §5's
single-source discipline) rather than a second hand-copied table.

**Sparse, not dense.** Unlike `moeq`'s dense `(layer, expert, which)` grid (every cell always present by
construction — a routed layer always has all 512 experts), a per-layer role here may genuinely be absent
at some layers (a QSA role on a GDN layer, and vice versa). `Store` builds a small
`std::unordered_map<role_key, index>` once at `open()` and `find(role, layer)` returns `nullptr` for a real
gap — checked directly (`tests/backbone_quant_tests.cpp`'s own deliberate-gap fixture) rather than assumed
safe by a dense-grid analogy.

**Byte order: GGUF's own, verbatim, untransposed** — the opposite of the `.bin` blob, and the reason this
is a genuinely separate format rather than a fourth `ParamDtype`. A GGUF tensor is `[out][in]`, the natural
DOT shape `backbone_quant_dot.hpp`'s kernels consume directly; the existing blob instead transposes every
2-D weight to this project's own `[in, out]` AXPY convention. `dequantize_expert`'s `transpose_out_in` step
is deliberately NOT applied anywhere in this sidecar's write or read path.

### 13b. Per-role pre-transform decisions (the real work of this phase)

Every `transplant::Dest` this sidecar could plausibly carry was checked against `transplant.hpp`'s own
`fold_for()` and `vperm_for()` tables — the project's single source for which destinations need a value
transform — not assumed safe by inspection:

- **`fold_for()` (RMSNorm `1+w`, `ssm_a`'s `-exp(A_log)`): never fires for any role in the sidecar.** Every
  `Dest` `fold_for` returns non-`None` for is a norm, a bias, or an F32 per-head-scalar GDN gate — already
  excluded on format grounds (§13c point 1) before the fold question is even reached. Checked mechanically
  by `tests/backbone_quant_tests.cpp`'s own "no role's transplant Dest carries a Fold" case, not left as a
  prose claim.
- **`vperm_for()` (GDN's grouped→tiled value-head reorder): fires for three otherwise-native-quant
  candidates, and this phase had to actually resolve it, not just note it.** `GdnInProjQkv`/`GdnInProjZ`
  (Q5_K) and `GdnOutProj` (Q6_K) all carry a real value-order permutation. Worked through by axis, not
  assumed uniform:
    - `GdnInProjQkv`/`GdnInProjZ` reorder the DESTINATION's column axis, which — because both are
      `Op::Transpose` — is the RAW GGUF tensor's own ROW axis. Reordering whole rows of an independently
      block-quantized tensor is, in principle, just a permuted row COPY: no re-quantization needed.
    - `GdnOutProj` reorders the destination's ROW axis, which is the raw GGUF tensor's own COLUMN axis —
      i.e. it reorders elements WITHIN each row, across the very axis K-quant super-blocks are computed
      over. That does NOT commute with raw quantized bytes: reassembling a validly-quantized permuted row
      would need real re-blocking, not a copy.
  Rather than build two different mechanisms (a row-copy path for two roles, a re-blocking path for the
  third) with no live kernel consumer yet to validate either against, this phase made ONE conservative
  decision for all three: **exclude every VPerm-affected role from the sidecar.** They stay in the existing
  bf16 blob. The row-copy option for the two `Cols`-axis roles is named as a scoped future increment, not
  attempted here.
- **Roles excluded on format grounds** (no `Fold`/`VPerm` question even reached): F32 tensors (norms, the
  dense router, biases, small per-head-scalar GDN gates — no smaller native form exists) and the QSA
  indexer (`indexer.q_proj`/`k_proj`, BF16 — already the resident target format).
- **Roles excluded because they already have their own sidecar**: the 512 routed experts per layer
  (`moe_quant.hpp`'s own `S0Q1`) — this design's role table does not duplicate them.
- **`QsaQGateProj`, a genuinely different kind of decision.** `attn_q.weight` supplies BOTH
  `Dest::QsaQProj` and `Dest::QsaGateProj` via `transplant::Op::PerHeadHalf` — a non-contiguous per-head row
  selection, not a slice or a transpose. Rather than pre-split the tensor at sidecar-write time (real risk
  of an off-by-one in the per-head row arithmetic with no kernel consumer yet to catch it against), the
  sidecar stores the WHOLE source tensor verbatim under one role; a future consumer applies
  `per_head_half_transpose`'s own row-selection formula (`h*2*head_dim + half*head_dim`) directly against
  this role's raw bytes. No pre-split, no permutation, no re-blocking risk.

### 13c. Real file, end to end (AGENTS.md §9) — the measured size, and the honest discrepancy

Ran `sub0llm-transplant-q48 --gguf D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S --param-dtype 1 --out
<...>.bin --backbone-quant <...>.bin.bbq` end to end (157.2s for the whole run, produced all three real
artifacts: 9.83 GB `.bin`, 39.85 GB `.moeq`, 1.99 GB `.bbq`; `D:` had 263 GiB free against a ~50 GB need).
Real measured sidecar:

| | Value |
|---|---:|
| Tensors considered | 388 |
| Tensors included | 388 (100% of considered — every candidate that passed the format filter was block-aligned) |
| **Native encoded bytes** | **1,993,630,720 (1.857 GiB)** |
| Same tensors' bf16 cost | 5,481,431,040 (5.105 GiB) — **2.75x** smaller |
| Q8_0 tensors | 242 (100% of the file's Q8_0 — no exclusions apply to this format) |
| Q4_K tensors | 2 (100% — `TokEmb` + `LmHead`) |
| Q5_K tensors | 142 (94 `MoeSharedGate`/`Up`, 48 QSA `K`/`V`/`O`/`QGateProj`) |
| Q6_K tensors | 2 (the layer-2 outlier's `MoeSharedGate`/`Up` — see §13d) |

**Discrepancy from §2a's 3.59 GiB prediction, stated plainly rather than rounded away: the measured
sidecar is 1.857 GiB, about 52% of that figure — and the right comparison is actually against §2a's own
Q8_0+Q4_K+Q5_K+Q6_K subtotal (3,339.55 MiB / 3.262 GiB, the "4-format" figure that table itself reports),
against which the measured result is 58.3%.** The gap is real and almost entirely explained by §13b's
VPerm exclusion, not a measurement error: §2a's own totals put Q6_K at 478.34 MiB and this sidecar carries
only ~2.6 MiB of it (the layer-2 outlier), and Q5_K at 1,455.35 MiB against this sidecar's ~493 MiB (GDN
in-proj — `GdnInProjQkv`/`GdnInProjZ` across the file's ~35 non-outlier GDN layers — accounts for
essentially the whole difference). Q8_0 and Q4_K, which have no VPerm-affected role, land at their full
predicted totals exactly (242/2 tensors, matching §2a's own per-format counts). **Net: the conservative
VPerm decision gives up roughly 1.4 GiB of the format's theoretical native-quant byte reduction** — real,
quantified, and the direct, traceable consequence of §13b's own reasoning, not a surprise.

Put against the whole backbone's current bf16 residency (9.15 GiB, §7): this sidecar's included tensors are
55.8% of that footprint by bf16 bytes, and shrinking them 2.75x saves **3.248 GiB** of the model's total
resident size if wired in — about 58% of §7's full four-format potential (5.56 GiB), the same fraction
§13b's exclusion gives up.

### 13d. Correctness gates — real numbers

- **Round-trip against the real shard (AGENTS.md §9), every included tensor, not a sample** (388 is small
  enough that a full sweep costs nothing next to the transplant's own ~157s): decode the sidecar's own
  bytes via `gguf::to_f32`, independently re-read and decode the SAME bytes straight from the GGUF shard via
  the tool's own `read_range`, compare bit for bit. **388/388, 0 mismatches.**
- **Cross-check against the `.bin` blob's own value, to bf16 rounding, one real tensor per format** — the
  gate the round-trip alone cannot give, since it would pass even if every role were paired with the wrong
  GGUF name, as long as the pairing were self-consistent. Uses a SEPARATE decode path (`PARAM_LAYOUT` offset
  arithmetic into the blob + the blob's own bf16 decode, not the sidecar's own code): `GrAttnDown` layer 0
  (Q8_0), `TokEmb` (Q4_K), `MoeSharedGate` layer 0 (Q5_K) — and, found only by actually running this against
  the real file (not assumed from §2a's aggregate table), `MoeSharedGate` layer 2 (Q6_K, the real per-layer
  outlier §13b's own comment names). **36 sample values (3 rows × 3 columns × 4 tensors), 0 mismatches** —
  so, unlike an earlier draft of this section assumed, Q6_K DOES get a real cross-check after all, because
  the writer discovers each tensor's actual `type_raw` by lookup rather than assuming a role's format from
  its name, and correctly picked up the layer-2 anomaly with no special case.
- **Refusals**, each with its own test in `tests/backbone_quant_tests.cpp`: a file that is not `S0B1` at
  all, a truncated header, a truncated descriptor table, a truncated payload, a version bump, a
  `model_param_floats` mismatch (an optional caller-supplied check — this header has no compile-time
  `PARAM_FLOATS` of its own, being engine-free, so the expected value is a parameter; 0 means "skip"), an
  out-of-range role, and a duplicate `(role, layer)` descriptor. All refuse cleanly with a non-empty `err`
  rather than reading garbage.
- **AGENTS.md §4 — zero effect on existing output when the flag is omitted.** Ran the real 4-layer
  transplant twice, identical arguments except one run added `--backbone-quant`: the resulting `.bin` and
  `.moeq` files are byte-for-byte identical (`cmp` confirmed) between the two runs.
- **Suites** (`out/build/o5p2b`, this host, `sub0_core.dll` beside the test binaries): `sub0_tests`
  **28,969,623 / 147**, exactly unchanged from `main` (this package touches nothing `sub0_tests` depends
  on — no engine, no `src/` file). `sub0_frontend_tests` **145,636 / 275**, i.e. `main`'s 145,500/267 plus
  exactly this package's own 136 assertions / 8 test cases in `[backbonequantsidecar]`, nothing else moved.

### 13e. What phase 2b-2 (kernel wiring) inherits, named rather than re-derived

- The `Store`/`Desc`/`Role` surface (§13a) is the seam a real `op_linear`/decode consumer would read
  from: `find(role, layer)` → `nullptr` or a `Desc`; `raw(desc)` → the exact byte span
  `bbqd::gemv_plane(type_raw, raw, n_rows, row_elems, ...)` wants, with `Desc::in_f`/`out_f` already named
  `row_elems`/`n_rows` in that kernel's own vocabulary.
- **The VPerm-excluded roles (§13b) are NOT in this sidecar at any layer.** A wiring pass must fall back to
  the existing bf16/f32 blob for `GdnInProjQkv`, `GdnInProjZ`, and `GdnOutProj` unconditionally — `find()`
  returning `nullptr` for those roles is the correct, permanent outcome, not a bug to chase.
- **`QsaQGateProj` needs the per-head-half row-selection arithmetic applied by the CONSUMER**, not
  pre-applied here (§13b) — `transplant::per_head_half_transpose`'s own row math
  (`h*2*head_dim + half*head_dim`) is the reference to port.
- The AXPY-vs-DOT reconciliation §3d already named as unresolved is unchanged by this phase — this phase
  only produces bytes and an identity lookup, it does not touch any engine call site's loop shape.

---

## 14. Pass 4 (kernel pass 4) — the §12g lever: fold the per-sub-block scale into the integer accumulator

§12g named the concrete next lever, unattempted at the time: §12's own streaming kernels pay 8 (Q4_K/
Q5_K) or 16 (Q6_K) independent `hsum256_epi32`-style horizontal reductions per 256-element superblock —
one per sub-block — because `moeqd::ActBlocks`' own per-32 float activation scale cannot fold a
sub-block's INTEGER weight scale into the integer accumulator the way llama.cpp's per-256 `block_q8_K`
activation does. This section builds that: a new per-256 activation type (`bbqd::ActSuper`, beside
`ActBlocks`, not a change to it), three implementations of the fold (portable / AVX2 / AVX-VNNI) checked
bit-exact against each other, and an honest measurement of both the speed win and the activation-error
cost it buys.

### 14a. Reference study (AGENTS.md §5) — quoted, not paraphrased

`D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-common.h`'s `block_q8_K`:

```c
// This is only used for intermediate quantization and dot products
typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;
```

`D:\Craig\llama.cpp-qwen4exp\ggml\src\ggml-cpu\arch\x86\quants.c`'s `ggml_vec_dot_q4_K_q8_K` (AVX2 arm,
elided to the two lines that matter here — the full kernel is quoted at length in §12c already):

```c
const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
...
const __m256i q4l = _mm256_and_si256(q4bits, m4);
...
__m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
p16l = _mm256_madd_epi16(scale_l, p16l);
...
sumi = _mm256_add_epi32(sumi, sumj);
...
__m256 vd = _mm256_set1_ps(d);
acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
```

The load-bearing fact, re-derived rather than assumed: `scale_l`/`scale_h` carry the RAW unsigned 6-bit
`sc` codes (0..63, from `utmp`, never multiplied by `d`), and `madd_epi16(scale_l, p16l)` folds that raw
scale into the int32 domain BEFORE any float conversion. `sumi` accumulates ALL EIGHT sub-blocks of one
superblock in the integer domain; `d = y[i].d * x[i].d` (activation `d` times weight `d`) is multiplied
in and reduced to float exactly ONCE, at the very end of the `nb` (superblock) loop — not once per
sub-block. This is possible ONLY because `y[i].d` (the activation scale) is the SAME single float for
the whole 256-element superblock; if it varied per 32-element sub-block (as `ActBlocks::scale` does), the
scale could not be folded into `p16l`'s own accumulation and would need its own per-sub-block float
multiply — exactly the shape §12's own kernels are stuck with.

The min-term fold (same kernel, elided):

```c
const __m128i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);
```

`y[i].bsums` (16 per-16-element sums) is pairwise-summed (`_mm_hadd_epi16`) into 8 per-32-element sums,
matched against the 8 raw `min` codes via `madd_epi16`, and folded with ONE `dmin` float multiply per
superblock — the min-term equivalent of the same trick.

`ggml_vec_dot_q6_K_q8_K`'s own affine term (Q6_K has no separate "min", only the constant `-32`
zero-point, so this is its WHOLE affine correction, quoted in full because this project's own Q6_K
kernel below follows it exactly):

```c
const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
const __m256i scales_16 = _mm256_cvtepi8_epi16(scales);
const __m256i q8sclsub = _mm256_slli_epi32(_mm256_madd_epi16(q8sums, scales_16), 5);
...
sumi = _mm256_sub_epi32(sumi, q8sclsub);
acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
```

`madd_epi16(q8sums, scales_16)` combines all 16 `(scale, bsum)` pairs in ONE instruction, `slli_epi32(...,
5)` multiplies by 32 (the zero-point) via a shift, and the whole `-32*sum(scale*bsum)` correction is
subtracted from the raw `sumi` before a SINGLE float multiply-add — the `q8sclsub` shape this project's
own `dot_row_q6_k_super_*` (§14c) reuses under the same name.

**How this project's own conventions were re-mapped, not assumed to match (AGENTS.md §5):**
- llama.cpp's weight tensor and this project's are the SAME GGUF bytes (Q4_K/Q5_K/Q6_K block layout is a
  format constant, not a convention llama.cpp invented) — no axis remap needed on the WEIGHT side, unlike
  the Newton-Schulz precedent AGENTS.md §5 cites.
- The ACCUMULATION and OUTPUT shape differ: llama.cpp's `ggml_vec_dot_*` computes ONE dot product (row
  vs row) inside a caller-side GEMM/GEMV loop; this project's `dot_row_q{4,5,6}_k_super_*` are already
  that same "one row's dot" shape (matching §12's own `dot_row_q{4,5,6}_k_avx2`), so no restructuring was
  needed there either — the fold is the only thing ported.
- **What was NOT ported**: llama.cpp's `get_scale_shuffle_k4`-based broadcast batching (two sub-blocks'
  scales broadcast into one register via `_mm256_shuffle_epi8` against a combined lookup table, processing
  64 raw elements per outer-loop step). This project's kernels already decode ONE sub-block's nibbles at a
  time (`nibble_lo`/`nibble_hi`, unchanged from §12), so each sub-block's own raw scale is already an
  individually-addressable scalar by the time it is needed — a plain `_mm256_set1_epi16`/`set1_epi32`
  broadcast suffices, at the cost of one broadcast per sub-block instead of amortizing two via a shuffle
  table. A deliberate simplicity-over-micro-optimization call (`include/sub0/backbone_quant_dot.hpp`'s own
  §14 file-header comment names this explicitly), not an oversight.
- llama.cpp's own quantized activation (`quantize_row_q8_K_ref`) was re-derived onto this project's
  caller-owned/reused-buffer convention (`ActSuper::quantize`, mirroring `ActBlocks::quantize` field for
  field: allocates only when the width changes) rather than copied — the RATIONALE for the /127 divisor
  (not /128, keeping every int8 product's headroom) is identical to `ActBlocks`' own, already established
  in this project, not re-derived from llama.cpp's own comment (which does not spell out that reasoning).

### 14b. `bbqd::ActSuper` — the new type, beside `ActBlocks`

`include/sub0/backbone_quant_dot.hpp`, mirroring `block_q8_K` field for field: `qs` (n int8 quants),
`d` (n/256 per-superblock float scales — ONE per 256, not per 32), `bsums` (n/16 per-16-element int16
sums, llama.cpp's own name — finer than `ActBlocks::gsum`'s per-32, needed for Q6_K's own 16-wide
sub-blocks). Used ONLY by the Q4_K/Q5_K/Q6_K kernels below; `Q8_0` stays on `ActBlocks` unconditionally
(`super_fusable()` refuses it) because its own native block already IS 32 wide — a coarser per-256 scale
would only cost it accuracy with nothing to fold, since Q8_0 has no per-sub-block affine scale at all.

### 14c. Three kernel implementations, one dot-product algebra

`detail::KScaleRaw` unpacks Q4_K/Q5_K's `(sc, m)` pairs the SAME branch-free way `KScaleTable` (§12b)
does, but keeps them as raw `uint8` codes rather than pre-multiplying by `d`/`dmin` — the fold needs the
raw integer.

- **`dot_row_q{4,5,6}_k_super_portable`**: the correctness reference. Per superblock: `isum`/`isum_min`
  accumulate in `std::int32_t` across all 8 (Q4_K/Q5_K) or 16 (Q6_K) sub-blocks BEFORE any float
  conversion; ONE `d_combined = x.d[s] * t.d` float FMA (plus one `dmin_combined` FMA for Q4_K/Q5_K, or
  one subtract for Q6_K's `q8sclsub`) closes out the superblock.
- **`dot_row_q{4,5,6}_k_super_avx2`**: `_mm256_maddubs_epi16` (unsigned weight nibble × signed
  activation) + `_mm256_madd_epi16` (broadcast raw scale, fold + reduce to int32) per sub-block,
  accumulated into a running `__m256i`/`__m128i` across the WHOLE superblock, with exactly ONE
  `hsum256_epi32` (or the 128-bit equivalent for Q6_K) at the end — down from §12's own 8 or 16.
- **`dot_row_q{4,5,6}_k_super_vnni`**: `_mm256_dpbusd_avx_epi32` (AVX-VNNI, `vpdpbusd`) computes the raw
  unsigned×signed dot directly to int32 (no int16 intermediate), then `_mm256_mullo_epi32` folds the raw
  scale in before accumulating — the lever the task brief named explicitly (§14e).

**Why all three are bit-exact, not merely close (checked, not assumed):** integer addition and
multiplication are associative and commutative exactly, with no rounding — `sum_sub sc[sub] *
dot(w_sub, x_sub)` evaluates to the identical `std::int32_t` regardless of which order or which SIMD
lane grouping computes the partial sums, as long as no term overflows int32 (worked through explicitly in
the header's own §14 comment: Q4_K/Q5_K's worst case is ~63.5M per superblock, Q6_K's ~260M — both three
orders of magnitude under `INT32_MAX`). The only floating-point step in any of the three implementations
is the SAME final `d_combined`/`dmin_combined` multiply-add, done in the same per-superblock sequential
order in all three — so `tests/backbone_quant_dot_tests.cpp`'s "AVX2 super path agrees EXACTLY with
portable" and "AVX-VNNI... agrees EXACTLY..." cases require bit-identical output, not a tolerance, and
both hold.

**A free correctness/§1 side-effect worth recording**: §12i flagged `Gsum16` (Q6_K's per-16 activation
sum cache) as heap-allocating inside `gemv_plane_avx2`, once per call and per thread. Pass 4's `bsums`
lives INSIDE `ActSuper` itself, built once by `ActSuper::quantize()` (called once per token/layer, the
same cadence `ActBlocks::quantize()` already has) — so `dot_row_q6_k_super_*` needs no per-call
allocation at all, and no analogous `TODO` is needed here.

### 14d. AGENTS.md §14 — three measured passes

**Pass 1 — ActSuper + AVX2(maddubs) + VNNI(dpbusd), dispatch defaults to VNNI where available.**
`gemv_plane_super<Threads>` picked VNNI whenever `kVnniKernels` was true (mirroring `kAvx2Kernels`'s own
convention one ISA tier up). Real Qwen3.8-Flash-Next UD-IQ1_S shards, this host, DRAM-streamed pool
(`sub0_backbone_quant_dot_bench`, `--seconds 0.4`), 8 threads, wall-clock speedup over the real
`gemv::axpy` bf16 kernel at the matching `(row_elems, out_dim)` shape:

| Format | OLD (§12, per-32 ActBlocks streaming) | NEW (pass 4, per-256 ActSuper, VNNI-default) | Task's own §12i target |
|---|---:|---:|---:|
| Q4_K | 1.78x | **2.59x** | 2.17x |
| Q5_K | 1.39x | **2.11x** | 1.93x |
| Q6_K | 1.41x | **3.21x** | 1.82x |

All three formats beat their own stated target on the FIRST pass. The mechanism is a clear win before any
tuning — this is the "a first milestone that merely works is proof the mechanism is viable" case AGENTS.md
§14 describes, not yet evidence of the CEILING.

**Pass 2 — is VNNI actually the better default? Measured, not assumed.** `time_super_kernel_1t` (new in
`benchmarks/backbone_quant_dot_bench.cpp`) calls `detail::gemv_plane_super_avx2`/`_vnni` DIRECTLY,
bypassing the dispatcher, isolating the two kernels' own 1-thread cost on identical real bytes. Four
independent runs (`--seconds 0.3`-`0.5`), all three formats (12 measurements total):

| Run | Q4_K VNNI/AVX2 | Q5_K VNNI/AVX2 | Q6_K VNNI/AVX2 |
|---|---:|---:|---:|
| A | 0.92x | 0.97x | 0.98x |
| B | 0.99x | 0.99x | 0.94x |
| C | 0.94x | 0.94x | 0.87x |
| D | 1.09x | 0.92x | 0.92x |

Mean ~0.95x — rough PARITY, never a clear win, occasionally a real regression, once (Q4_K, run D)
marginally ahead. **Mechanistic reading, not just a number**: `madd_epi16`'s own job is "multiply two
int16 operands AND reduce adjacent pairs to int32" in one instruction; VNNI's `dpbusd` computes the raw
unsigned×signed sum directly to int32, but the per-sub-block SCALE still has to be folded in afterward
via a separate `mullo_epi32` — so the instruction count does not actually drop (2 instructions either
way), and `vpmulld` is typically the higher-latency of the two paths' respective "extra" instructions on
x86. VNNI's fusion advantage is real for a PURE dot product; it is not free once a per-sub-block integer
scale still needs folding in a separate step. **Consequence**: `gemv_plane_super_dispatch` was changed to
prefer plain AVX2 over VNNI (simpler, needs no `-mavxvnni` availability, never measured clearly worse) --
`detail::gemv_plane_super_vnni` stays in the tree, individually callable and tested, parked rather than
reverted (AGENTS.md §14's own "never fully back a change out").

**Pass 3 — re-verify gates and re-measure the corrected default.** Full rebuild, full suite:
`[backbonequant]` **6,884 assertions in 17 test cases** (phase 2a's own 3,891/10 plus 7 new pass-4 cases);
`~[backbonequant]` **141,609 assertions in 257 test cases** — bit-for-bit the required baseline, confirming
nothing leaked; full `sub0_frontend_tests` **148,493 assertions in 274 test cases**. Real-artifact
DRAM-streamed re-measurement (default AVX2 dispatch, `--seconds 0.5`), 8 threads:

| Format | OLD (per-32, §12) | NEW (per-256, pass 4, plain-AVX2 default) | Target |
|---|---:|---:|---:|
| Q4_K | 2.11x | **2.50x** | 2.17x |
| Q5_K | 1.68x | **2.35x** | 1.93x |
| Q6_K | 1.50x | **2.87x** | 1.82x |

Still comfortably ahead of every stated target after the dispatch correction — the win is not an artifact
of the (now-abandoned) VNNI-first choice. Per-thread compressed-byte throughput (native/super GB/s,
`sub0_backbone_quant_dot_bench`'s own columns), same run: Q4_K 1→8 threads 11.71→44.93 GB/s, Q5_K
8.63→41.39, Q6_K 15.71→53.68 — all comfortably above §12's own per-thread numbers at every thread count,
though (like §12f) still short of the host's ~79 GB/s all-P-core roof, leaving headroom for a future pass.

**Run-to-run noise, stated plainly rather than glossed over**: this host shows real ±15-40% swings between
runs on both the native AND the bf16 baseline arms (e.g. Q5_K's 1-thread `bf16 GB/s` ranged 12.35-21.92
across the sessions this pass ran in) — consistent with the shared-host contention this task's own brief
warned about (a sibling agent writing a ~3.6 GiB sidecar file concurrently). The RATIO (native vs bf16,
same run) is far more stable than either arm's own absolute number, which is why every speedup claim above
is a same-run ratio, never an absolute GB/s compared across different runs.

### 14e. Activation-quantization error: OLD vs NEW, side by side, on real weights (not buried)

Measured identically to §5b/§12's own precedent: real backbone tensor bytes, a single Gaussian (`N(0,1)`)
activation row, SAME seed and SAME draw for both schemes so the comparison is apples-to-apples (not two
different random instances) — `tests/backbone_quant_dot_tests.cpp`'s own pass-4 real-bytes case.

| Format | Real tensor | OLD, per-32 `ActBlocks` (§5b's own baseline) | NEW, per-256 `ActSuper` | Ratio |
|---|---|---:|---:|---:|
| Q4_K | `output.weight` | 3.85% | **8.35%** | 2.17x |
| Q5_K | `blk.0.attn_gate.weight` | 1.99% | **6.50%** | 3.27x |
| Q6_K | `blk.0.ssm_out.weight` | 0.12% | **0.41%** | 3.48x |

**This is a real cost, reported plainly, not picked around.** A per-256 activation scale is coarser by
construction (one scale must now cover 8x more dynamic range than before), and the error roughly doubles
to triples across the three formats. For context against this project's own precedent
(`docs/MOE_QUANT_DOT.md` §6e, B35's own MoE-path numbers, and §5b's own end-to-end read): the OLD scheme's
own worst case (Q4_K 3.85%) sits in the same band as B35's MoE path (0.48%-3.3%), which produced a 0.29
end-to-end logit L2-relative diff against the BF16 backbone (shipped default-off, "a real tradeoff a user
can choose" per the B35 precedent, not "too costly to ship"). The NEW scheme's Q4_K number (8.35%) is
roughly 2.5x that precedent's own worst case — **plausibly, not confirmed, in the same general
neighbourhood as FP8's own 0.43 L2-relative/3-of-6-argmax-flip result** (shipped default-off, "not
recommended" — the one precedent this project has explicitly NOT recommended for exactly this reason).
**No end-to-end logit comparison was run this pass** (§5d's own gap — no engine wiring exists yet to
produce one), so this is an informed analogy, not a verified equivalence; a phase-2b wiring pass MUST run
the real `--dump-logits` comparison against the BF16 (~0.199) and FP8 (~0.43) precedents before this
scheme is trusted at inference quality, not merely at kernel-dot-product accuracy. **This is the primary
agent's call to weigh, per this task's own brief — not something this pass resolves or should resolve on
its own.**

The weight-decode side stays EXACT under the new scheme (as required, unconditionally): the
lossless-activation test (`tests/backbone_quant_dot_tests.cpp`'s own pass-4 case, reusing §5a/§12's own
`lossless_row()` helper — every 32-wide GROUP already plants an exact ±127, so every 256-wide SUPERBLOCK's
own amax is trivially also 127, checked directly) measures worst-case relative disagreement 8.9e-8 to
3.2e-7 against `gguf::to_f32` on real sidecar bytes — float-rounding only, no layout error, matching §5a's
own bar.

### 14f. Gates

- `[backbonequant]`: 6,884 / 17 (was 3,891 / 10 before this pass; +7 new cases, all pass-4).
- `~[backbonequant]`: 141,609 / 257 — **exactly** the required baseline, unchanged.
- Full `sub0_frontend_tests`: 148,493 / 274.
- New pass-4 test cases: `super_fusable()` accepts only Q4_K/Q5_K/Q6_K at a 256-aligned width; the fused
  super unpackers decode the SAME weights `gguf::to_f32` does (lossless activation); the AVX2 super path
  agrees EXACTLY with the portable super path; the AVX-VNNI super path agrees EXACTLY with both (gated on
  `SUB0_BBQD_VNNI`); `gemv_plane_super<Threads>` is bit-exact across 1/2/4 threads on an uneven split;
  `gemv_plane_super` refuses Q8_0 and an unaligned row, untouched on refusal; the AGENTS.md §9 real-bytes
  case (lossless decode + the OLD-vs-NEW error comparison above).

### 14g. What this pass does not do, named rather than left implicit

- No engine wiring, exactly per this task's own scope — `gemv_plane_super` is not called from `src/` or
  `tools/`, same as §12's own kernels.
- No end-to-end logit comparison (§14e's own gap) — the activation-error rise is real and measured at the
  dot-product level; whether it is acceptable at INFERENCE quality is the primary agent's call, informed
  by but not resolved by this pass.
- No further VNNI tuning beyond the batching alternatives considered and rejected in pass 2's own
  reasoning (§14d) — a genuinely different VNNI formulation (e.g. batching two sub-blocks' scale folds
  into one wider multiply) was considered but has no clear instruction-count win over the current shape,
  so it was not attempted as a fourth pass; named here as the concrete next step if VNNI is revisited.
- The per-thread GB/s gate (~79 GB/s all-P-core roof) is still not reached (§12f's own gate, restated in
  §14d) — 28-67 GB/s measured across formats and thread counts here, the same "real headroom, not yet
  claimed" reading §12i already gave the per-32 scheme.
