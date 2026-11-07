# Qwen 27B backend speed comparison

Measured 2026-11-07 on CrogBuntu1; issue #56, parent #30/#29, quant prerequisites #18.
The clean native llama.cpp reference is **68.42× faster at prompt processing** and
**17.24× faster at generation** than the native Sub0Llm proving ground on this workload.
This measures the bounded dense 27B backend, not the older MoE engine.

## Results

Medians of three measured rounds after one discarded warm-up per arm. Rates exclude setup;
setup + prefill approximates backend first prediction latency, excluding tokenization and process startup.
Process wall includes setup, both phases and teardown. RSS is whole-process peak, including mapped weights.

| Backend | Prompt tokens/s | Generated tokens/s | Setup + prefill s | Process wall s | Peak RSS GiB |
|---|---:|---:|---:|---:|---:|
| Sub0 generic O2 | 0.2396 | 0.2389 | 54.315 | 88.010 | 5.569 |
| Sub0 native O3 | 0.2489 | 0.2485 | 52.303 | 84.636 | 5.569 |
| llama native F32 attention | 16.8976 | 4.2284 | 1.296 | 3.381 | 6.119 |
| llama native defaults | 17.0272 | 4.2836 | 1.290 | 3.345 | 6.103 |

Native compiler flags improve Sub0Llm generation by 4.04%.
This does not close the reference gap. The shipped chat binary is unchanged; these are separate instrumented builds.

| Backend | Phase | Three raw rates, tokens/s | Robust CV (1.4826 MAD / median) |
|---|---|---|---:|
| Sub0 generic O2 | prefill_tok_s | 0.239628, 0.242298, 0.239394 | 0.144% |
| Sub0 generic O2 | decode_tok_s | 0.238412, 0.241882, 0.238886 | 0.294% |
| Sub0 native O3 | prefill_tok_s | 0.248859, 0.249145, 0.248181 | 0.170% |
| Sub0 native O3 | decode_tok_s | 0.248580, 0.248534, 0.247507 | 0.027% |
| llama native F32 attention | prefill_tok_s | 16.897607, 16.745588, 16.921488 | 0.210% |
| llama native F32 attention | decode_tok_s | 4.226445, 4.267819, 4.228396 | 0.068% |
| llama native defaults | prefill_tok_s | 17.113873, 17.027200, 16.754427 | 0.755% |
| llama native defaults | decode_tok_s | 4.283641, 4.280292, 4.302381 | 0.116% |

## Workload and controls

- Same downloaded `Qwen3.8-27B-UD-IQ1_S.gguf`, SHA256 `3895b6eaa91e705c06ad1938d16c22e86f073c6a67df86260a1da79be3d1f887`.
- AMD Ryzen 9 8945HX, 64 GB RAM; CPU-only, eight distinct physical cores 0–7 on one CCD, performance governor. This is not a 16-core tuning sweep.
- Identical pretokenized 13-token non-thinking `Hi` prompt and eight additional greedy single-token evaluations. The first generated prediction belongs to prefill; generation rate is eight divided by the sum of those eight decode times.
- Fixed-count evaluation deliberately ignores EOS. Actual reply plus EOS agrees (`12675,1017,0,248046`, “Hi there!” plus EOS); synthetic continuation after EOS differs in the F32 reference. No all-token or corpus quality equivalence is claimed.
- Context requested 128 for both. llama.cpp pads its allocation to 256; Sub0Llm allocates 128. Only 21 positions are evaluated.
- llama.cpp batches all prompt tokens and computes last-position logits. Sub0Llm processes prompt tokens serially and computes the output head each time. Both timers include finite-logit scan and greedy argmax. Backend throughput therefore includes different execution strategies.
- Sub0Llm uses float activations and F32 attention. Stock llama.cpp uses optimized quantized activation dots and CPU repacking. The F32 reference arm forces F32 K/V and disables flash attention; default arm uses F16 K/V and automatic flash attention. Neither reference arm contains the diagnostic float-dot patch.
- Warm working weights, no eviction or memory pressure. Each arm starts a fresh process. Logging is silenced during timed phases.
- Shared Sub0Math `sub0perf` lock, power checks, calibration before/after, process RSS/output hashes, and sustained ten-second CPU quiet checks. Orders rotate between rounds. None of the twelve measured samples were tainted; all are retained. Rates are raw steady-clock measurements, not calibration-scaled scores.
- Sources, generated profile, binaries, shared libraries, model and correctness records are hashed before/after the experiment; tracked tree and HEAD must remain unchanged. Full commands, identities, per-evaluation timings, load accounting and calibration are in [the raw evidence](qwen27b_speed_samples.json).

## Correctness and limits

Both Sub0Llm builds passed the independent 16-position F32 logit gate at the unchanged relative L2 limit 0.001:
generic worst 0.0000787653; native worst 0.0000845936. That gate uses a separate diagnostic reference with
float activation dots, F32 cache and flash disabled. Its patched kernels are never used for speed results.
The clean performance reference is pinned at llama.cpp `cf659f240b232e15700b1721bbb43c1d973c52e9`.

This is a short-prompt, short-generation backend comparison, not long-context scaling, perplexity,
a chat service latency benchmark, or a regression-baseline acceptance. Three rounds describe local spread;
they do not establish broad confidence intervals. Hardware counters were unavailable (`perf_event_paranoid=4`).

## Local roof probes and next increment

Separate eight-core AVX512 probes measured median 1,386.89 FP32 GFLOP/s, 5,546.30 INT8 dot GOP/s,
and 32.894 GB/s for repeated sequential reads of a 1 GiB buffer. These are local measured probe ceilings,
not the old Intel host's roof or whole-machine theoretical limits. Source: `tools/reference/bench_host_roofs.cpp`.

Sub0Llm reports 5,764,050,576 encoded weight bytes and 51,247,841,280 logical projection FLOPs per evaluation.
At native generation speed this is approximately 1.433 GB/s of encoded weight payload and 12.737 logical GFLOP/s.
These are semantic proxies, not measured DRAM traffic or retired instructions; dequantization and non-projection
work are omitted. They cannot establish a hardware bottleneck by themselves.

The comparison prioritizes profiling the scalar float-activation/dequantization projection path, then adding
packed quantized kernels under #18, and batched prefill under #29/#30. Preserve the independent numerical
and memory gates for each increment. A compiler flag change alone is insufficient.

## Reproduction

Instrumentation source commit: `3eed3b371bb5a82947157afd31903b9bc9275218`.
Clang 21.1.8 was used throughout. Sub0Llm generic: `-O2 -fopenmp`; native: `-O3 -march=native -fopenmp`,
C++26 with the existing generated dense profile and frontend/simdjson libraries. Reference: Release,
`GGML_NATIVE=ON`, `GGML_CPU_REPACK=ON`, Clang C/C++, CUDA/Vulkan disabled, same pinned revision.
Build the standalone reference harness against the clean checkout's `llama` library; preserve its matching ggml shared libraries.

The committed evidence includes the exact four command arrays and required identity files. With those artifacts restored,
extract `experiment.manifest` to a local manifest and update paths for this checkout, retaining the model/reference identities.
Run the shared calibration target from the `perf` preset, then:

```sh
python3 scripts/compare_qwen_speed.py PATH_TO_MANIFEST --out NEW_EVIDENCE.json
```

The driver refuses dirty tracked sources, changed evidence identities, invalid counts, power changes and persistent foreign load.
Tests cover serial-versus-batched denominators, invalid samples, load subtraction and summary exclusions;
a C++ fixture independently validates the emitted native phase contract. Seven perf CTest checks passed,
including 303 shared workflow tests and the chat/speed tools tests.
