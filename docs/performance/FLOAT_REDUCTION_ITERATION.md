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

## Controlled model result

Measured 2026-11-07 on CrogBuntu1, Ryzen 9 8945HX, eight physical cores 0–7. Same model and raw
13-token prompt; eight additional greedy evaluations, EOS ignored for fixed-count measurement.
Three rotating measured rounds follow one discarded warm-up per arm. All nine measured samples
are untainted; the tracked tree and every input/binary/source identity remain fixed during sampling.

Broader context/corpus quality and default promotion remain #63; short numerical checks do not close it.

| Backend | Prompt tokens/s | Generated tokens/s | Setup + prefill s | Peak RSS GiB |
|---|---:|---:|---:|---:|
| Sub0 IQ1_S native | 0.738803 | 0.735826 | 17.663 | 5.569 |
| Sub0 all-format vector | 1.268824 | 1.264391 | 10.314 | 5.569 |
| llama native defaults | 16.942971 | 4.287226 | 1.299 | 6.103 |

Generation improves **1.718×** over the previous IQ1_S candidate; prompt processing improves **1.717×**.
The remaining generation gap to clean llama.cpp is **3.391×**; prompt processing remains **13.353×** behind.
Memory stays under the 8 GiB proving-ground budget. Relative to the historical original native baseline,
this is approximately 5.1× faster; that cumulative figure spans separate runs, not one interleaved comparison.

| Backend | Three prompt rates | Three generation rates | Generation robust CV |
|---|---|---|---:|
| Sub0 IQ1_S native | 0.736862, 0.738803, 0.740227 | 0.735067, 0.735826, 0.737647 | 0.153% |
| Sub0 all-format vector | 1.267693, 1.285280, 1.268824 | 1.264391, 1.270375, 1.261691 | 0.317% |
| llama native defaults | 16.942971, 16.683839, 16.975664 | 4.277850, 4.287226, 4.291826 | 0.159% |

The [raw evidence](qwen27b_float_reduction_evidence.json) preserves all samples, setup/wall/RSS,
quiet/load accounting, power and calibration, source commit `aa60d68`, commands and artifact hashes.
Reproduce using `scripts/compare_qwen_speed.py` and its embedded `experiment.manifest`, with updated
local paths and restored matching artifacts. The candidate is Clang 21.1.8 `-O3 -march=native -fopenmp`.
Kernel measurements use the shared rig’s contraction-disabled native build, a separate artifact validated
against its own fixtures. Five calibrated rounds cover 73 benchmark cases, including the prior controls.
No new automatic regression baseline was accepted.

The reference is clean pinned llama.cpp, with CPU repacking enabled, normal quantized activation dots,
F16 KV and automatic flash attention. It batches prefill; this Sub0Llm path remains serial and computes
all prompt-position output heads. Context is requested 128 and padded to 256 by llama.cpp. Raw-phase
rates exclude setup/tokenization. All three arms predict the same nine fixed-count IDs here. These
short tests do not establish long-context scaling, corpus quality or equivalence after arbitrary prompts.

## Kernel results

Warm-cache four-row slices at width 5120; medians of five calibrated interleaved rounds on CPU 0.
The larger IQ1_S packed path is retained; only the other formats’ reduction changes. Ratios below are
microkernel results and must not be substituted for model token rates.

| Format | Prior median ns | Vector median ns | Speedup |
|---|---:|---:|---:|
| Q8_0 | 16239.3 | 3920.7 | 4.142× |
| Q2_K | 17029.8 | 3522.3 | 4.835× |
| Q3_K | 25897.7 | 12485.8 | 2.074× |
| Q4_K | 16660.0 | 3138.3 | 5.309× |
| Q5_K | 17343.4 | 3804.9 | 4.558× |
| Q6_K | 30297.2 | 16969.0 | 1.785× |
| IQ2_XXS | 20787.9 | 7216.4 | 2.881× |
| IQ2_XS | 26701.4 | 13460.5 | 1.984× |
| IQ3_XXS | 26423.4 | 13175.6 | 2.005× |
| IQ1_S | 1483.1 | 1500.6 | 0.988× |
| IQ4_NL | 22064.4 | 11956.4 | 1.845× |
| IQ3_S | 26902.1 | 13629.2 | 1.974× |
| IQ2_S | 26819.1 | 13508.3 | 1.985× |
| IQ4_XS | 22856.2 | 9416.9 | 2.427× |
| IQ1_M | 27968.9 | 14506.2 | 1.928× |

This is a diagnostic post-change attribution, with no foreign-load rejection:

| Format | Complete decode wall share |
|---|---:|
| IQ1_S | 24.00% |
| IQ2_XXS | 21.60% |
| IQ1_M | 15.74% |
| Q3_K | 13.75% |
| IQ2_XS | 10.72% |
| IQ2_S | 7.32% |
| IQ3_XXS | 1.98% |
| Q2_K | 0.70% |
| Q8_0 | 0.13% |
| IQ4_XS | 0.09% |
| IQ3_S | 0.07% |

IQ1_S and IQ2_XXS now lead. Further candidates include multiple independent packed-dot accumulators
and fused decode/dot kernels for the remaining formats, followed by batching prompt projections and
avoiding unused output heads. Full context/corpus qualification and generated default promotion are
tracked in #63; the goal remains the complete [performance campaign](QWEN27B_CAMPAIGN.md).

## Chat candidate

The preserved default remains unchanged. On this machine, select the staged candidate explicitly:

```sh
OMP_NUM_THREADS=8 OMP_PROC_BIND=false OMP_DYNAMIC=false OMP_THREAD_LIMIT=8 \
  taskset -c 0-7 python3 scripts/qwen_chat_poc.py \
  --backend out/build/qwen-perf/sub0-vector-all --projection-kernel native-all
```

The generic and native fixtures, ASan/UBSan run, all 15 per-type independent row checks, unchanged
16-position gate and seven CTest checks pass. The projection header is consumed by standalone tools
and benchmark fixtures; no engine/MoE decoder or generated model layout changed in this increment.
The existing complete frontend/engine regression evidence belongs to #60 and is not presented as a
new full-engine test run for this tool-only reduction change.

The actual two-turn chat smoke passes: `Hi` → `Hi there!`, then `Repeat my first message.` → `Hi`,
both EOS, with prompt accounting 13 → 31 tokens. This is functional context evidence, not the
full-window/corpus qualification in #63.
