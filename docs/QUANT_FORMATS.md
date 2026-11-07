# Quantisation formats: one registry now, one module per format next

Status: step 1 implemented (registry + conformance tests). During the live Qwen3.8-27B work, the FormatModule contract/list and scalar decoders are being introduced incrementally with the actual file blockers. Current modules are Q2_K, Q3_K, IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, and IQ4_XS. GGML type IDs are pinned to upstream: Q3_K=11, IQ2_XS=17, IQ3_XXS=18, IQ2_S=22, IQ3_S=21, IQ4_XS=23, IQ1_M=29. A corrected full-file transplant dry run reconciles all 868 destinations (26,896,246,784 floats), with one synthesized LM-bias tensor, no unmatched source tensors, and zero per-tensor mismatches. Independent numerical parity against ggml and the full test suite remain gates before merge. IQ4_NL remains the planned pilot for migrating an existing Plane.
Context: docs/QUANTIZED_RESIDENT_DENSE.md (issue #15) needs about eight more formats, and the project
will keep meeting new ones. The goal is that supporting a new format is ONE self-contained module
plus ONE line in a list, with tests that cannot be forgotten.

## Where a format's code lives today (verified)

| Piece | Where | Extension cost |
|---|---|---|
| id, block geometry | `gguf.hpp` `kFormats` (was a `switch`) | one row |
| dequantiser (reference, scalar f32) | `gguf.hpp` `dequantize_*` + `kDequant` (was a `switch` in `to_f32`) | function + one row |
| IQ codebooks / grids | `gguf_quant_tables.hpp` | data file |
| dot kernel (int8 activations) | `moe_quant_dot.hpp` `Iq1SPlane`/`Iq2XxsPlane`/`Iq4NlPlane`; `backbone_quant_dot.hpp` Q4_K/Q5_K/Q6_K/Q8_0 | a Plane, hand-wired into callers |
| role-keyed resident stores | `.moeq` (experts), `.bbq` (backbone roles) | per store |
| tests | spread across `gguf_tests`, `moe_quant_tests`, `backbone_quant*_tests` | easy to miss one |

Step 1 (done): the two closed `switch`es became tables, a `static_assert` fails the build if a quantised
format has no dequantiser, and `tests/quant_registry_tests.cpp` iterates the tables so every registered format
gets the same contract checks (unique ids, sane geometry, zero blocks decode to zeros incl. a partial final
block, short buffers refused, deterministic decode, `to_f32` dispatch). Behaviour is unchanged: the frontend
suite's pre-existing assertions are identical to before (193,082 on `main`).

## Target: a format is a module

```
include/sub0/quant/
  format.hpp        TensorType, BlockSpec, the FormatModule concept, the registry (kFormats/kDequant derived from it)
  q2_k.hpp          Q2_K scalar dequantiser + block geometry
  iq3_xxs.hpp       IQ3_XXS scalar decoder + its codebook
  iq2_xs.hpp        IQ2_XS scalar decoder + its codebook
  iq2_s.hpp         IQ2_S scalar decoder + its codebook
  iq3_s.hpp         IQ3_S scalar decoder + its codebook
  iq1_m.hpp         IQ1_M scalar decoder + shared IQ1_S grid
  iq4_xs.hpp        IQ4_XS scalar decoder + shared IQ4_NL values
  q3_k.hpp          Q3_K scalar decoder + packed 6-bit scales
  iq1_s.hpp         everything for IQ1_S: geometry, tables it needs, dequantise, Plane (dot), self-description
  iq2_xxs.hpp  iq4_nl.hpp  q4_k.hpp ...
  all.hpp           the ONE list of modules (adding a format = one #include + one entry)
tests/quant/
  conformance.hpp   the generic checks above, parameterised over a FormatModule
  iq1_s_tests.cpp   per-format MATH only: values against an independent reference, plus its Plane vs dequantise
```

A module exports, as `constexpr` members of one struct (a concept, checked at compile time):

- `type` and `name`, `block` geometry (`elems`, `bytes`);
- `dequantize(raw, n, out)`: the scalar reference, the oracle for everything else;
- optional `Plane`: the dot-kernel decoder (`group(p) -> {int8[32], scale, delta}`), present only if the
  format can be dotted in place. Absent means "dequantise to bf16 at load" is the explicit fallback;
- `kGroup` (32 for every Plane today). Formats whose scales are per 16 elements (Q2_K/Q3_K sub-blocks) need
  either `kGroup = 16` or a two-scale group; decide when the first one lands, not before.

Build integration: the `all.hpp` list is the only thing the registry, the dispatcher, `memplan` and the tests
read. A format that is listed but lacks a required member fails to compile; a format not listed is invisible.
Plane-vs-dequantise agreement is a GENERIC test over modules that have a Plane, so a new Plane cannot ship
without it.

## Iteration plan (each step independently shippable and verifiable)

1. DONE: registry tables + generic conformance (this change).
2. Move ONE small format (IQ4_NL: 18 bytes per 32, one scale, no super-block) into `quant/iq4_nl.hpp` as a
   pilot: its geometry, dequantiser, and the existing `Iq4NlPlane`, behind the concept. Gate: all existing
   suites bit-identical (AGENTS 10: enumerate consumers repo-wide, run the full suite with the feature on).
3. Move the rest of the existing formats the same way, one commit each; delete the old switches/tables.
4. DONE initial modules for the missing formats (IQ1_M, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K,
   Q3_K); next add independent numerical parity checks against ggml using the real UD-IQ1_S tensors and, where
   there is a Plane, the generic Plane test. The block geometries needed are already derived from the real file
   (they sum to the file's size); verify each against ggml-common.h before implementing (AGENTS 5).

## Open questions

- Do dot kernels and dequantisers share one decode core per format (one transcription of the bit layout, two
  consumers), as `Iq1SPlane::Fields` already does for IQ1_S? Preferable: one place a layout is read.
- Where do per-format `.bbq`/`.moeq` role stores plug in: via the module list, or kept separate (stores are
  about WHERE a tensor lives, formats about HOW it is encoded)? Leaning separate.
- The CPU-feature variants (AVX2 `group_v`) belong inside the module, behind the same `if constexpr` pattern.
