# IQ2_XXS fused float projection — preparation

Tracking #66 / #18 / #30. Status on CrogBuntu1, 2026-11-08: implemented opt-in, row checks pass;
complete-model qualification and controlled performance remain pending. No speed claim or default change.
The existing #63 native-all qualification continues with preserved executables and frozen inputs.

## Hypothesis and implementation

The post-#62 diagnostic profile attributed 21.60% of complete decode wall time to IQ2_XXS.
Hypothesis: avoid decoded scratch stores/loads and scalar magnitude/sign expansion by converting packed
codebook groups directly into AVX2 float lanes before the existing eight-lane reduction. Activations
remain float. Finite weight values and lane accumulation order match prior native-all.

`quant/iq2_xxs.hpp` owns the format's geometry, explicit little-endian unaligned packed fields,
scalar decoder and fused float row cores. The registry and retained compatibility function forward
to that module. `Iq2XxsPlane` shares its packed fields/grid/sign/scale helpers; its integer-activation
algorithms remain the existing implementations. Scalar/generic fallback is retained.

The separate `fused-iq2` experiment enables that core for IQ2_XXS and retains native-all behavior for
other formats. The generated configurator accepts it explicitly; AUTO remains baseline pending #63.
This is a scheduling/implementation selection, not a weight-layout or model-semantic fingerprint axis.
Runtime strategy dispatch remains a proving-ground experiment override. TODO #29: after quality and
performance qualification, instantiate the generated selection with `if constexpr` in the production
engine adapter; preserve baseline experiments in the proving ground.

## Current evidence

[Raw row evidence](qwen27b_iq2xxs_row_evidence.json) pins sources, oracle executable and model/reference
identities. Against independent pinned ggml decoding, 1,494 original real rows across 11 types and
12 re-encoded real rows for four absent types pass. Decoder floats are bit-identical; fused-iq2 equals
prior native-all bit for bit for all 1,506 rows. IQ2_XXS has 177 real sampled rows, relative L2
0.000000615582 against independent double dots, within the unchanged per-format 0.00002 limit.

Independent synthetic wire/double-dot fixtures cover widths 256, 5,120 and 17,408, signed/zero/subnormal
and large finite half scales, every magnitude/sign codebook index, all scale nibbles, unaligned data,
tails and truncated/overflowing lengths. Generic, AVX2 and native builds pass; ASan/UBSan pass.
A separate Plane reconstruction check covers 22,784 groups in scalar/cached/SIMD/cached-SIMD forms.
These are short correctness checks; their elapsed times are not performance comparisons.

The cpp-review pass checked that mapped rows and input spans outlive all field/dot calls, that
Plane fields remain values, that no pointer/reference survives a resized output, and that the fused
core uses only caller-owned input/output plus register/stack temporaries. No actionable lifetime or
ownership finding remains. Compiler warnings are errors; executed row/tail paths also pass sanitizers.

The shared benchmark executable registers six new prior/fused row cases (79 total). Generic and
AVX2 list-only runs pass every correctness fixture. Its normal fixture checks
execute before listing or measuring. The extended real-row oracle additionally requires bit equality
with the prior vector reduction. The 18 Python script tests pass with the new chat override. The freshly rebuilt complete frontend
suite passes 220,980 assertions / 327 cases, identical to the pre-migration #60 baseline counts. The freshly rebuilt complete neutral engine suite
also passes the unchanged 44,292,784 assertions / 147 cases.

## Remaining acceptance

Run the tooling CI groups and review any subsequent C++ edits. Stage all binaries before
performance measurements. After the existing quality run releases the shared lock, measure three
interleaved complete-model rounds against native-all and clean native llama.cpp, with memory/load
and artifact identities. Run full-context and corpus qualification against the #63 cached oracle;
validate actual chat behavior. Publish complete evidence through a linked PR and merge only when the
required gates pass. Negative performance results need three passes and remain parked behind their
selector. Current row checks do not establish full-model quality, throughput or default promotion.
