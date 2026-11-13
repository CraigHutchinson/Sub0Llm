# Packed-weight integer-activation backend — 2026-11-13

Tracking #84, #18, #31/#32, #63 and concurrency #90. CrogBuntu1, Ryzen 9 8945HX,
Clang 21.1.8. This is an explicit proving-ground backend, not a default promotion.

## Execution and ownership

IQ1_S and IQ2_XXS weights remain in their original mapped GGUF blocks. Kernels fetch codebook
entries and expand a 32-element integer group into registers. There is no float weight array,
unpacked tensor copy or per-call heap allocation. The caller sizes activation scratch at startup,
quantizes once before splitting output rows, and shares that immutable prepared view among workers.
The same prepared row can also feed multiple weight matrices. Concurrent engine integration must give each session its own activation,
output, KV, GDN and RNG state. This increment tests projection-level isolation; sharing a mutable
Layer instance across sessions is not supported.

`ActivationRow` shares validation/storage logic between Q8_K-compatible and signed-int16 activation
rows. Both precisions have scalar, AVX2 and native VNNI kernels, with scalar/native row equivalence
and allocation tests. Moving a prepared row invalidates the source view explicitly.

References fetched before implementation: pinned llama.cpp
[activation encoder](https://github.com/ggml-org/llama.cpp/blob/cf659f240b232e15700b1721bbb43c1d973c52e9/ggml/src/ggml-quants.c#L2784),
[IQ1_S integer dot](https://github.com/ggml-org/llama.cpp/blob/cf659f240b232e15700b1721bbb43c1d973c52e9/ggml/src/ggml-cpu/quants.c#L1198)
and [IQ2_XXS integer dot](https://github.com/ggml-org/llama.cpp/blob/cf659f240b232e15700b1721bbb43c1d973c52e9/ggml/src/ggml-cpu/quants.c#L954).
Our rows follow GGUF output-row order. Activation scales are per 256 elements; the first signed
largest-magnitude input determines scale sign. Q8_K excludes -128; Q16 excludes -32768.

IQ1_S's offset is folded exactly into integer weights: `q + sign/8 = (8*q + sign)/8`, followed by
the odd group scale. This removes separate activation-sum correction loads. Q16 block sums can
exceed int32 for IQ2_XXS: native lanes remain bounded, then widen to int64 before horizontal summing.
Q16 final conversion to float rounds normally; it is checked numerically, not claimed bit-exact to float inputs.

A compiled copy of the actual pinned Q8_K C encoder matched all fields for 2,560,000 random activation
values. The initial multiply-plus-bias expression failed at a rounding boundary because FMA contracted
it. The implementation uses nearest-even rounding of the float product; an independent magic-bias
oracle and a concrete regression case cover the failure. Zero blocks clear sums as well as quants.

## Numerical evidence and limits

[Q8 model gate](packed_activation_evidence/q8-model-gate.json): **fails**, worst relative logit L2
0.0589081 against the unchanged 0.001 limit. Matching stock Q8_K encoding does not establish the
float-reference contract. `packed-q8k` remains an explicit diagnostic candidate and is not promoted.

[Q16 model gate](packed_activation_evidence/q16-model-gate.json): **passes all 16 positions**, worst
0.000193743 at the unchanged 0.001 limit. Peak process RSS 5,874,464 KiB (about 5.60 GiB), below 8 GiB.
Inputs are the first 16 frozen literature-0 IDs, compared with the preserved independent float oracle.
This does not replace the 128-position/four-domain and 4,096-target corpus qualification under #63.

[Real rows](packed_activation_evidence/real-rows.json): 792 IQ1_S and 177 IQ2_XXS rows per precision;
scalar/native exact on all rows; independent double-dot relative L2 <=1.21e-7 on reconstructed
integer activations. The validation tool dequantizes one row solely for its oracle; inference does not.
Synthetic gates cover widths 256/5120/17408, extrema, ties, invalid input/state, move ownership,
truncated rows, an adversarial >int32 block sum, no C++ allocation and concurrent independent sessions.

## Kernel iteration

Five calibrated interleaved rounds per pass, warm 32-row slices, native ISA. No compile/model job
overlapped timing. Raw [first](packed_activation_evidence/q8-pass1.json),
[second](packed_activation_evidence/q8-pass2.json), [third](packed_activation_evidence/q8-pass3.json)
and [Q16](packed_activation_evidence/q16-kernels.json) captures are retained.
The first two IQ2 float comparisons used generic NativeAll; only the third/Q16 use best FusedIQ2.

At width 5120, the third Q8 pass measures IQ1_S prepared 9.84 us versus float 10.20 us, and IQ2_XXS
7.82 us versus fused float 14.70 us. Preparation adds about 4.5 us for this small 32-row case.
Q16 measures IQ1_S prepared 10.30 us versus float 10.22 us (no meaningful gain), IQ2_XXS 10.17 us
versus float 14.62 us; including preparation gives 14.35/14.24 us. Preparation must be amortized over
real matrix row counts; these are not whole-model speedups. Pass 1 established correctness, pass 2
split accumulator chains without a gain, pass 3 folded IQ1_S's offset and improved its prepared time.

## Use and next gates

Build the existing explicit `sub0llm-dense-token` target with the real dense profile. Select
`--projection-kernel packed-q16 --projection-threads 8` for the accuracy-passing candidate;
`packed-q8k` is the lower-precision experiment. Other formats retain the existing fused-float fallback.
AUTO, previous artifacts and checkpoint formats are unchanged.

TODO #29/#63: proving-ground runtime dispatch remains an iteration convenience. After full corpus,
chat, RSS and controlled end-to-end qualification, bake the selected precision/format strategy into
the generated profile and production adapter with constexpr dispatch. No speculative configurator
flag is added before that adapter consumes it.

Next execution increment: group prepared activation rows across active sessions and reuse each decoded
weight group across them (#90/#74). Compile-time session capacity with runtime occupancy must preserve
per-session state and RNG. Compare aggregate accepted tokens/s and per-session latency against llama.cpp
at the same concurrency, context and CPU/memory budget. Multiple independent inference loops do not
establish the shared-weight batching win. MTP/tree proposals require target verification and acceptance-
adjusted accounting; different seeds alone are not a correctness oracle.

## Regression checks

Complete default build and unfiltered 523-test suite pass (two platform-dependent mapped-arena skips).
Neutral engine remains 44,292,784 assertions/147 cases with unchanged three fingerprints; frontend
remains 220,980/327. Generic and native tooling suites pass 11/11. Both precisions pass generic,
AVX2 and native checks; ASan/UBSan cover move ownership, allocation and concurrent session cases.
No numerical tolerance was relaxed. Code reviewed against cpp-review before publication.
