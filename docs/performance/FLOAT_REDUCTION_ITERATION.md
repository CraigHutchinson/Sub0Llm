# All-format float reduction iteration

Tracking #62, parent #30/#29/#18. After the IQ1_S gain, remaining scalar decode/dot paths dominate;
IQ2_XXS, IQ1_M and Q3_K lead the post-change profile. Hypothesis: eight-lane float accumulators across
all decoded blocks remove the serial sum dependency without changing activation precision or decoders.

The opt-in `--projection-kernel native-all` (alias of the existing `--iq1s-kernel` experiment selector)
keeps the fused IQ1_S implementation and vectorizes the remaining float reductions on AVX2. Other
ISAs retain the prior float reduction. Whole-block span validation precedes both paths. Unaligned
loads are explicit; scratch is caller-owned and reused. Defaults and prior measured binaries stay intact.

Before timing: generic/native fixtures cover all 15 registered quant formats at widths 256 and 5120,
four rows, unaligned spans and invalid/truncated input. The changed reduction is checked against
independent double accumulation. ASan/UBSan and seven CTest checks pass. Independent pinned ggml
checks 1,494 actual model rows across the 11 present formats; the four absent legacy formats (Q4_K,
Q5_K, Q6_K, IQ4_NL) additionally use 12 real source rows re-encoded by ggml. Those derivative rows are
explicitly distinguished from original model tensors. Every type passes a per-type double-dot L2
limit 0.00002, and decoder output remains byte-identical to ggml. Full-model 16-position logits pass
the unchanged 0.001 gate (worst relative L2 0.0000791262).

Staged kernel/model performance and memory results will be appended after controlled measurement.
Broader context/corpus quality and default promotion remain #63; short numerical checks do not close it.
