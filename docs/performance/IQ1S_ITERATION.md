# IQ1_S packed float-dot iteration

Tracking #60, parent #18/#30. The two-run profile in #58 attributes ~71% of dense 27B evaluation wall
time to IQ1_S projections. Hypothesis: fusing packed decode with the float dot, then accumulating
vectors, can remove scratch traffic and the serial accumulation dependency without quantizing activations.

The new IQ1_S module owns geometry, packed fields and decode. The legacy GGUF decoder and MoE Plane
forward to that shared core. Dense projection retains its original baseline and adds fused scalar and
AVX2 float candidates. On a target without AVX2, native falls back to fused scalar. Whole-block row,
span and scratch validation remain common to all paths. No allocation is added inside the row core.

`--iq1s-kernel baseline|scalar|native` is a temporary proving-ground evaluation knob, default baseline.
TODO(#60): after the independent quality/memory and controlled speed gates justify promotion, select
the supported proven kernel through the generated configuration; preserve an explicit baseline for
future experiments. Production chat artifacts are preserved during evaluation.

Reference layout was checked against the clean pinned llama.cpp `cf659f240b232e15700b1721bbb43c1d973c52e9`,
`ggml/src/ggml-quants.c::dequantize_row_iq1_s` (line 2666). Each 256-element block has an F16 scale,
32 low grid-index bytes and eight 16-bit group descriptors. Each group describes 32 contiguous input
columns; its four grid entries decode eight signed bytes each. Scale and delta equations, row-contiguous
GGUF output layout, and the 0..2047 grid indices are retained. SIMD changes the reduction order, not
activation precision or weight values.

## Verification before timing

- Generic and native benchmark fixtures compare all three kernels with an independently written
  packed-byte decoder and double-precision dot at 256, 5120 and 17408 input widths, 32 output rows,
  unaligned packed/input offsets, full codebook coverage across the fixtures, group signs/scales,
  zero/subnormal/normal/negative F16 scales, partial dequantization and truncated/partial-block refusal.
  Fixture vector relative L2 limit is 0.00002. Near-zero components can exhibit cancellation error
  even in the original scalar baseline; the vector metric avoids dividing by those individual values.
  The production logit gate remains 0.001.
- Independent unmodified ggml matches 1,494 sampled real-model blocks with zero differing values.
- `check_iq1s_dots.cpp` checks 792 real IQ1_S rows with ggml decoding plus double dots; all kernels pass.
- Fresh frontend suite: 220,980 assertions / 327 test cases pass. The complete engine suite passes
  44,292,784 assertions / 147 cases with a generated tokenizer and realistic 16,794-token vocabulary.
  An initial external-vocabulary engine build failed two tokenizer-dependent cases because that mode
  emits no tokenizer; the corpus-backed configuration resolves those failures without test changes.
  Seven performance CTest checks pass.
  The earlier local frontend executable was stale (203,457 / 317); it was rebuilt before this result.
- ASan/UBSan native fixture checks pass on the host. LeakSanitizer requires running outside the sandbox’s
  ptrace boundary; the successful host run retained leak checking.
- Migrated baseline is bit-identical to the prior native binary at all sixteen full-model positions.
  SIMD passes the unchanged independent gate (worst relative L2 0.0000794629).
  Fused scalar is bit-identical to the migrated baseline at all sixteen positions.

## Controlled performance result

Measured 2026-11-07 on CrogBuntu1, Ryzen 9 8945HX, eight physical cores 0–7. Same downloaded GGUF,
13 prompt tokens and eight additional greedy evaluations, EOS ignored for fixed-count measurement.
One discarded warm-up per arm, then three rotating rounds. All twelve measured samples are untainted.


| Backend | Prompt tokens/s | Generated tokens/s | Setup + prefill s | Peak RSS GiB |
|---|---:|---:|---:|---:|
| Sub0 IQ1_S fused native | 0.733382 | 0.730649 | 17.796 | 5.569 |
| Sub0 IQ1_S fused scalar | 0.344265 | 0.342910 | 37.829 | 5.569 |
| Sub0 prior native baseline | 0.248649 | 0.247915 | 52.352 | 5.569 |
| llama native defaults | 17.030096 | 4.239316 | 1.298 | 6.103 |

The native candidate improves prompt processing **2.949×** and generation **2.947×**.
The remaining clean llama.cpp generation gap is **5.802×**; prompt processing remains **23.221×** behind.
Peak RSS remains about 5.57 GiB, below the existing 8 GiB budget.

| Backend | Three prompt rates | Three generation rates | Generation robust CV |
|---|---|---|---:|
| Sub0 IQ1_S fused native | 0.733382, 0.731820, 0.735820 | 0.730649, 0.729162, 0.736161 | 0.302% |
| Sub0 IQ1_S fused scalar | 0.344265, 0.343441, 0.344309 | 0.342910, 0.342176, 0.343003 | 0.040% |
| Sub0 prior native baseline | 0.248019, 0.248659, 0.248649 | 0.248081, 0.247764, 0.247915 | 0.090% |
| llama native defaults | 17.030096, 16.899424, 17.041075 | 4.280760, 4.239316, 4.202221 | 1.297% |

The independent correctness records are bound to the final candidate SHA256
`610c762e87dbd8b477dc259c411f87ed46ab964a1aab3b8992ec833acaab1bc9`.
The experiment freezes source commit `e107997`, generated profiles, all executable/library/model hashes,
thread environment, power, governor and calibration. The source tree stayed clean throughout sampling.
Complete phase times, every raw sample, setup/wall/RSS, calibration and identities are retained in
[raw evidence](qwen27b_iq1s_evidence.json). Reproduce with the committed comparison driver and its embedded
`experiment.manifest`, updating local artifact paths. No automatic regression baseline was accepted.

This retains the baseline comparison’s limits: serial Sub0Llm prefill versus batched llama.cpp prefill,
float versus quantized activations, requested context 128 versus llama.cpp’s padded 256, short workload
and synthetic post-EOS continuation. All four arms produced the same nine predicted IDs here; that does
not establish corpus quality or full-window parity.

## Kernel measurements and remaining cost

Five interleaved calibrated rounds, CPU 0, warm-cache 32-row slices. Native benchmark uses Clang 21.1.8,
`SUB0MATH_NATIVE_ARCH=ON` and the shared rig’s contraction-disabled floating-point options. End-to-end
candidate is a separate `-O3 -march=native -fopenmp` build, validated independently above. Microkernel
ratios are not model token rates.

| Width | Baseline median ns | Fused scalar ns | Fused native ns | Native speedup |
|---:|---:|---:|---:|---:|
| 256 | 9386.9 | 5653.5 | 647.9 | 14.49× |
| 5120 | 187508.7 | 120441.1 | 11741.3 | 15.97× |
| 17408 | 637402.4 | 410567.1 | 40040.2 | 15.92× |

Post-change attribution is diagnostic, not a separate speed trial. Remaining decode shares are:

| Format | Complete decode wall share |
|---|---:|
| IQ2_XXS | 29.28% |
| IQ1_M | 16.21% |
| Q3_K | 15.32% |
| IQ1_S | 13.69% |
| IQ2_XS | 11.46% |
| IQ2_S | 7.57% |
| IQ3_XXS | 2.17% |
| Q2_K | 1.63% |
| Q8_0 | 0.20% |
| IQ4_XS | 0.11% |
| IQ3_S | 0.07% |

IQ1_S’s share falls from ~71% to ~14%. The next target is the remaining scalar float dot/decode path,
led by IQ2_XXS, IQ1_M and Q3_K. Preserve fused scalar as the bit-identical reference/fallback; evaluate
vector reduction for the remaining formats and direct packed kernels where the profile warrants them.
The final goal remains the [performance campaign](QWEN27B_CAMPAIGN.md), not this one-format gain.

## Try the candidate in the chat proving ground

The existing default backend stays preserved while broader context/quality qualification proceeds.
The chat launcher now forwards an explicit experimental kernel choice; omitting it preserves compatibility
with the original backend. On this machine:

```sh
python3 scripts/qwen_chat_poc.py --backend out/build/qwen-perf/sub0-iq1s --iq1s-kernel native
```

The measured candidate also completed the real two-turn chat smoke test: `Hi` → `Hi there!`,
then `Repeat my first message.` → `Hi`, both ending at EOS; prompt accounting grew from 13 to 31 tokens.
The launcher’s explicit kernel forwarding and default omission are covered by its regression tests.

For a fresh build, compile `tools/sub0llm-dense-token.cpp` against the generated dense Qwen profile
with OpenMP and native CPU flags; its frontend/simdjson link dependencies match the existing mapped PoC.
The benchmark executable and raw evidence retain exact command/artifact identities for the measured build.
Promote a proven kernel through generated configuration after broader qualification; the runtime knob
is explicitly temporary. This increment does not silently replace the preserved production artifact.
