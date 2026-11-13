# Packed IQ1_S Q16 row/session tile

Issue #105, parent #90; quant prerequisites #18/#84 and goal #77. The accepted kernel keeps GGUF
weights packed and shares each decoded 32-value integer group across two active sessions, and each
activation load across two output rows. Capacity is compiled as two; runtime occupancy one uses the
same entry point without executing an inactive second lane. Caller-owned prepared activation rows
are immutable through the call. Output is a single contiguous session-major slab with explicit stride;
it must not overlap weights or activation storage. Each concurrent caller owns its output/scratch.

The codebook scaled by eight retains its 16 KiB size; it is not an unpacked model copy. IQ1's complete
256-value block fits int32: 256 × 9 × 15 × 32767 = 1,132,427,520. This proof does not apply to IQ2,
which retains the existing widened Q16 reduction. No weights or float tensors are cached by the tile.

## Correctness and review

Independent wire decoder/double-dot fixtures cover widths 256/5120/17408, two distinct inputs,
1/2/3/31/32 rows, stride padding, inactive outputs, 2→1 occupancy, scalar/256/512 and original/scaled
codebooks. Output is poisoned before each arm, so skipped stores fail. Inputs, extents and overflow
are refused before writes; the fixture also checks concurrent callers and zero C++ new/new[] hot
allocation. Generic, AVX2, native and native ASan/UBSan checks pass. The generic tooling suite passes
12/12. Astra's read-only review found no kernel arithmetic/bounds defect. Its fixture ownership,
stale-output and invalid-width findings were corrected and verified before publication.

The explicit `--projection-kernel packed-q16-tile` proving-ground option uses the 512-bit form when
compiled in, otherwise the 256-bit/portable form. IQ2 stays Q16 and other formats use the existing
fallback. Generated production defaults, ordinary float kernels and checkpoint formats are unchanged.
All 16 frozen literature logit arrays are byte-identical to the prior Q16 candidate. Independent float
comparison passes at worst relative L2 **0.00019374251767034764**, unchanged limit **0.001**.
Peak captured process RSS is 5,874,192 KiB (~5.60 GiB). Full corpus/context qualification remains #63.

## Kernel experiment

CrogBuntu1, Ryzen 9 8945HX, Clang 21.1.8, -O3 -march=native, AC power, pinned CPU 31. Five
calibrated shuffled/interleaved sub0perf rounds, quiet host, no overlapping build/model measurements.
One operation projects 32 output rows for all active sessions; timings are warm-cache microbenchmarks,
not model tokens/s or chat latency. The serial baseline invokes the existing Q16 kernel per session;
float uses the existing fused baseline. Preparation-inclusive arms prepare only active sessions.

| Width | Sessions | Serial Q16 µs | Tile256 µs | Tile512 µs | Serial + preparation µs | Tile512 + preparation µs |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 0.531 | 0.508 | 0.470 | 0.734 | 0.672 |
| 256 | 2 | 1.061 | 0.542 | 0.501 | 1.470 | 0.907 |
| 5120 | 1 | 10.284 | 9.313 | 8.966 | 14.338 | 12.994 |
| 5120 | 2 | 20.587 | 10.052 | 9.527 | 28.664 | 17.593 |
| 17408 | 1 | 35.286 | 31.788 | 30.450 | 48.984 | 44.458 |
| 17408 | 2 | 70.847 | 34.104 | 32.202 | 98.083 | 59.667 |

At width 5120, two-session prepared aggregate projection throughput improves **2.16×**, from
~97,148 to ~209,922 session projections/s. Including preparation it improves **1.63×**. The batch
wall duration is also the bound on completion of either session's projection within that call; these
numbers do not measure queueing, TTFT or service p95. Prepared one-session speed improves **1.15×**.
The 512-bit form wins on this host. Inspected two-row specializations have no YMM/ZMM stack accesses;
a larger session tile is deferred until engine-level evidence warrants extra register pressure.
Raw five-round timings/spread and calibration are in [kernels.json](q16_tile_evidence/kernels.json),
with [build identities](q16_tile_evidence/build.json).

## Whole-model comparison

Three rotating measured rounds after one warmup per arm, same downloaded model, 13 raw input IDs,
eight additional greedy evaluations, eight physical cores 0–7, warm weights and unchanged 8 GiB
budget. The codified `compare_qwen_speed.py` harness holds the shared sub0perf lock, validates source
and binary/input hashes before/after, checks power, CPU quiet windows and foreign load, records
calibration, native phase durations and RSS. All nine measured samples are untainted. No builds or
other model jobs overlap. Rates exclude setup. Backend phases exclude tokenizer and process startup.

| Backend | Prompt tok/s | Generation tok/s | Setup + prefill s | Peak RSS GiB |
|---|---:|---:|---:|---:|
| Sub0 Q16 serial | 1.8752 | 1.5601 | 7.010 | 5.602 |
| Sub0 Q16 tile | 1.8902 | 1.5802 | 6.955 | 5.602 |
| llama native defaults | 16.5304 | 4.1311 | 1.298 | 6.103 |

The one-session model gain is only **1.29%** in generation on this short workload;
three samples do not establish broad statistical confidence. llama.cpp remains **2.61×** faster
at generation and **8.75×** at prompt processing. The two-session kernel gain is not a measured
concurrent-model gain. All nine generated IDs match across these runs, including the deliberately
fixed-count continuation after EOS. This is neither long-context nor corpus-quality qualification.

llama.cpp is the clean pinned `cf659f240b232e15700b1721bbb43c1d973c52e9` CPU reference with default
attention/cache and CPU packing. It batches prefill and allocates padded context 256; Sub0 processes
prompt positions serially, elides unused intermediate LM heads, uses F32 attention and context 128.
Both evaluate only 21 positions. These execution strategies explain a substantial remaining prompt
gap; integer dots alone cannot remove it. Full commands, identities, phase samples, spread and method
are retained in [speed.json](q16_tile_evidence/speed.json). The logit [gate](q16_tile_evidence/model-gate.json)
and prior-Q16 [parity record](q16_tile_evidence/model-parity.json) are independent of speed verdicts.

## Reproduce and follow-up

```sh
cmake -S . -B out/build/q16-session-tile-native -G Ninja -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release -DSUB0_PERF_ONLY=ON -DSUB0MATH_NATIVE_ARCH=ON \
  -DSUB0MATH_PERF_ON_BUILD=OFF -DSUB0MATH_PERF_GIT_HOOKS=OFF
cmake --build out/build/q16-session-tile-native -j2
ctest --test-dir out/build/q16-session-tile-native --output-on-failure
python3 tools/perf/bench_check.py run --exe out/build/q16-session-tile-native/sub0llm-bench-q16-tile \
  --out out/perf/q16-tile-new.json --rounds 5 --settle 0
```

Build the dense-token target with the preserved real dense profile before replaying the speed manifest.
The measured standalone tool was linked from current source against the preserved frontend static
library and generated headers; its hash and capture command are retained in build.json. Do not replace
that profile with a tiny test configuration. The source tree must be clean for the phase harness.

Next is [#107](https://github.com/CraigHutchinson/Sub0Llm/issues/107): extract immutable model access
and independent session state, then coordinate one row-worker team across two sessions. Never share
mutable Layer activation/capture/profile state between callers. Validate unequal prompts and 2→1/1→2
admission against serial Q16 before comparing concurrent model throughput with the existing llama.cpp
concurrency baseline. Defaults remain gated by #63; broader concurrent admission remains #90.
