# B35 — fused quantized dot products for the MoE resolve path

**Status: DESIGN. No code changed by this document.** Design pass for `docs/INDEPENDENT_REVIEW_BACKLOG.md`
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
