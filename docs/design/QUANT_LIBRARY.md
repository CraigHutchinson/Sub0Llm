# Quant module library: formats, kernels and tests as pluggable modules

Status: DESIGN for the next increments; the first half describes what exists. Tracking: #18 (modules), #31-#41
(per-format), #74 (batching), #63/#66/#70 (kernel qualification). Supersedes the forward-looking parts of
`docs/QUANT_FORMATS.md` and `docs/QUANTIZED_RESIDENT_DENSE.md`, which stay as history.
Legend: **[V]** verified in the code today; **[P]** proposed.

## 1. What exists [V]

- `include/sub0/quant/format.hpp` defines the `FormatModule` concept: `type`, `name`, `block{elems,bytes}` and
  `dequantize(raw, n, out)`. `static_assert(FormatModule<IQ1S>)` style checks make a malformed module a build error.
- A module (e.g. `quant/iq1_s.hpp`) is self-contained and has the same anatomy: `Fields` (the one place the bit
  layout is read), small decode primitives (`index`, `grid`, `scale`, `delta`), the scalar `dequantize`, and an
  optional row kernel `dot_row<Native>(row, x, width)` that keeps float activations.
- `quant/all.hpp` lists the modules (`kModules`: Q2_K, IQ3_XXS, IQ2_XS, IQ2_S, IQ3_S, IQ1_M, IQ1_S, IQ4_XS, Q3_K).
  `gguf.hpp` builds its `kFormats`/`kDequant` tables from the nine legacy entries **plus** `kModules`, so adding a
  module adds the format to size rules and dispatch with no edit elsewhere. `tests/quant_registry_tests.cpp`
  iterates those tables, so every registered format gets the same contract checks automatically.
- Still legacy in `gguf.hpp`: F32/F16/BF16 plain types and Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ4_NL. PR #67 (draft)
  moves IQ2_XXS into a module and adds `fused-iq2`.
- Dense projection: `denseq::project(type, raw, x, y, scratch, kernel)` (`include/sub0/dense_quant.hpp`) decodes a
  row block-by-block and dots it with ONE float input vector. `Iq1Kernel {Baseline, Scalar, Native, NativeAll}`
  selects the variant; PR #67 adds a `fused-iq2` string. `sub0llm-dense-token` splits output rows over up to 8 OpenMP workers
  (`Layer::project_impl` in `tools/dense_qwen_support.hpp`).
- Independent oracles: `tools/reference/check_ggml_decoders.cpp`, `check_projection_dots.cpp`,
  `check_iq1s_dots.cpp` compare against the pinned llama.cpp (cf659f24) on real rows.

## 2. Kernel tiers (the vocabulary this library should use) [P]

| Tier | Meaning | Today |
|---|---|---|
| T0 reference | scalar `dequantize`, the oracle for everything above | all 9 modules + legacy formats |
| T1 generic | dequantise a block to float, float dot (`denseq::project`), optional AVX2 8-lane accumulate | every format |
| T2 format-native, float activations | a per-format row kernel that decodes in registers and keeps float inputs | IQ1_S (`dot_row<Native>`, PR #71 changes it), IQ2_XXS fused (PR #67) |
| T3 quantised activations | int8 (or int16) activation blocks, integer dots, one scale fold per group | exists for MoE experts (`moe_quant_dot.hpp`) and backbone roles (`backbone_quant_dot.hpp`); **not on the dense 27B path** |

Measured so far: moving the dominant formats from T1 to T2 took the model from 0.25 to 1.51 tok/s (#61, #62, #66).
T3 is what llama.cpp uses, and it is the only route to clearly exceed it on prefill (`PERFORMANCE_MODEL.md`, section 3).
Rule: a tier above T0 must match T0 within the stated numerical contract on real rows before it is selectable.

## 3. Where the next kernels go: by measured time, not bytes [V]

PR #67's HANDOVER text records the next measured projection costs: **IQ1_S 28.6%, IQ1_M 18.5%, Q3_K 16.4%**. Compare
with the share of file bytes: IQ1_M is 4.3% of the bytes but 18.5% of the time, because it runs through the slow
T1 path. Priority therefore follows the profile (`docs/performance/QWEN27B_PROJECTION_PROFILE.md`), re-measured
after each landing:

| Format | % of file bytes | Module | Best kernel now | Ticket |
|---|---|---|---|---|
| IQ1_S | 58.0 | yes | T2 | #31, PR #71 |
| IQ2_XXS | 11.9 | PR #67 | T2 fused (PR #67) | #32, #66 |
| Q3_K | 8.8 (output head) | yes | T1 | #41 |
| Q2_K | 7.8 (incl. token_embd gather) | yes | T1 | #40 |
| IQ1_M | 4.3 | yes | T1 | #34 |
| IQ2_XS | 4.2 | yes | T1 | #35 |
| IQ2_S | 3.1 | yes | T1 | #36 |
| IQ3_XXS | 1.1 | yes | T1 | #37 |
| Q8_0 | 0.4 | legacy | T1 | #33 |
| IQ4_XS, IQ3_S | <0.1 | yes | T1 | #38, #39 |

(Byte shares from the real UD-IQ1_S tensor table, 2026-11-06.)

## 4. Design changes to make iteration modular [P]

1. **One kernel selector, not an IQ1-named enum.** Replace `Iq1Kernel` (and the string `fused-iq2`) by a per-format
   named variant set: each module declares its variants (`baseline`, `native`, `fused`, ...) and a default; a
   single table maps (format, variant) to a row kernel. The generated default (`--dense-projection-kernel`) and the
   runtime override both name variants, and `sub0llm-dense-token` prints the resolved choice per format (it already prints
   `KERNEL selected=...`). Adding a kernel = a function in the module plus one row; AUTO stays the old value until
   qualification passes (AGENTS section 4).
2. **One decode core per format.** The dequantiser, the row kernel and the batch kernel must share `Fields`/decode
   primitives so a bit layout is transcribed once (IQ1_S already does this; make it the rule).
3. **A batch API next to the single-vector one** (this is #74's foundation, `PREFILL_BATCHING.md`):
   `denseq::project_batch(type, raw, X, n_tokens, Y, scratch, kernel)` that decodes each block once and dots it
   against all tokens. Each token keeps its own accumulator and adds in the same block and lane order as the
   serial kernel, so results are **bit-identical per token** to `project`. Formats opt in with a
   `dot_rows_batch` member; others get a correct generic fallback.
4. **Per-format contract test, generated from the module list.** Beyond today's registry checks add: (a) T1 equals T0
   on random and real blocks; (b) every declared variant equals its tier-0 reference within the documented limit
   (bit-exact for reassociation-free variants, the 0.001 relative-L2 gate otherwise); (c) batch equals serial per
   token, bit for bit; (d) no allocation inside the kernel (AGENTS section 1). A new format or variant cannot ship without
   all four because the test iterates the module list.
5. **Fixtures built from real tensors.** Each module gets a frozen set of rows taken from the real model
   (already the practice in `tools/reference`), kept small and committed with hashes.
6. **Unify the legacy formats.** Migrate Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL and (PR #67) IQ2_XXS into modules one
   commit each; delete the legacy table rows when the last one moves (#18).
7. **Backbone/expert stores stay separate.** `.bbq`/`.moeq` decide WHERE a tensor lives; modules decide HOW it is
   encoded. A store consumes a module's `block` and `Fields`; it must not re-implement a layout.

## 5. Checklists

**Add a format:** verify geometry against `ggml-common.h` at the pinned revision (do not recall it); write the module
(`Fields`, decode primitives, `dequantize`); add it to `all.hpp`; the registry tests run on it for free; compare against
the pinned ggml on real rows with `check_ggml_decoders`; add a fixture; only then a faster kernel.

**Add a kernel variant:** name it in the module; prove batch/serial equality and the numerical contract; measure alone
and combined with the other live variants (AGENTS section 13); three passes before parking; keep it toggleable.
Promotion to a default goes through the corpus gate (#63) and is recorded in the generated profile.

## 6. Open questions

- Int8 activation quantisation for I-quants: per-32 or per-256 activation scales, and the accuracy cost on this model
  (llama.cpp's choice is the comparator). Decide with a kernel-only experiment, then the corpus gate.
- Where the `kGroup=16` question lands for Q2_K/Q3_K (two scales per 32) when their T2 kernels are written.
- Whether CPU-feature variants (AVX2, AVX-512, VNNI) live inside a module behind `if constexpr` (current practice) or
  as separate translation units selected at configure time (the project prefers compile-time decisions).
