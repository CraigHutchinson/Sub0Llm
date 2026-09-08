# Muon CPU optimization — 2026-09-08

The CPU orthogonalization/update-buffer microbenchmark is **1.9–3.5x faster** across six measured
shapes. Outputs are bit-identical to the previous implementation, and the measured call path allocates
zero times, including its first invocation with prepared storage. This is a kernel measurement, not
an end-to-end training throughput or convergence claim.

## Changes

- `newton_schulz5` takes caller-owned scratch instead of growing thread-local vectors.
  `scratch_floats` supplies the capacity for its four working buffers.
- The CPU `AdamW` constructor prepares one buffer per optimizer OpenMP team slot when Muon is
  enabled, including the update vector previously allocated on every matrix/step. Existing
  `MUON_MAX_MN` and `MUON_MAX_MM` constants size the storage. Different matrix shapes and OS-thread
  reuse require no hot-path allocation.
- The symmetric Gram matrix is computed one triangle at a time and mirrored. Its square is computed
  with contiguous row-row dot products instead of strided column reads. Double accumulation, float
  intermediates, normalization division, the smaller-dimension orientation, and five iterations remain.
- The CUDA parity test's CPU-reference call now supplies scratch. The existing reuse test also checks
  exact equality against fresh scratch. `sub0_muon_bench` is a new target behind the existing benchmark
  option, without a generated-config dependency.

No parameter layout, checkpoint format, architecture fingerprint, optimizer setting, or CUDA kernel
changed. Qwen forward math and its external validation harness were not edited.

## Reference and conventions

The [upstream Muon implementation](https://github.com/KellerJordan/Muon/blob/master/muon.py) was fetched
before implementation on 2026-09-07. Its Gram computation is `A = X @ X.mT`. Coefficients remain
`(3.4445, -4.7750, 2.0315)`, followed by `B = bA + cAA` and `X = aX + BX`.

The reference's linear weights use `[out,in]`; this engine uses `[in,out]`, so its fan-ratio scale remains
`sqrt(max(1, cols/rows))` here. The working orientation has `min(rows,cols)` rows, making both Gram
matrices square at that smaller dimension. Mirroring and contiguous row reads exploit symmetry without
changing that mapping. Momentum EMA, Nesterov lookahead, clipping, and decoupled decay are unchanged.
Upstream uses BF16 for iteration; this change preserves the engine's existing CPU precision.

## Measurements

Host: Intel Core Ultra 9 275HX, Windows. Compiler: Clang 22.1.6. The comparison used
`-std=c++20 -O3 -march=native -fopenmp-simd`, compiling the original header from
`7a188e88e5694716839664281f5aea2b8f2781f5` and the optimized header into one process under separate
namespaces. Nine timed samples per version alternated order after warmup. No project build/test/profiler
was running during the comparison. These are medians, not confidence-bounded training estimates.

The old call includes its update-vector allocation and old orthogonalization; the new call uses
prepared update/scratch storage. Both copy the same deterministic random input and expose the result
for comparison. Momentum construction and applying the update to weights are outside this benchmark.

| Matrix | Before, ms | After, ms | Speedup | Relative L2 error | Allocations before / after |
|---|---:|---:|---:|---:|---:|
| 96 x 96 | 2.0562 | 0.8436 | 2.44x | 0 | 1 / 0 |
| 96 x 384 | 4.6071 | 2.3664 | 1.95x | 0 | 1 / 0 |
| 384 x 96 | 4.3600 | 2.1960 | 1.99x | 0 | 1 / 0 |
| 448 x 448 | 328.9050 | 94.5752 | 3.48x | 0 | 1 / 0 |
| 448 x 1792 | 496.7809 | 230.4527 | 2.16x | 0 | 1 / 0 |
| 1792 x 448 | 557.2108 | 233.0594 | 2.39x | 0 | 1 / 0 |

A scoped `operator new` counter measured allocations in the standalone comparison. The old kernel's
internal buffers were warm; its remaining allocation was the update vector. The new kernel was counted
on its first invocation after scratch preparation. This does not count OpenMP runtime initialization
inside the engine DLL; Muon-owned engine storage is prepared by construction and does not grow in step.
The six-shape comparison also passed AddressSanitizer without errors.

Local comparison sources/logs are under `out/muon_review/`: `compare.cpp`, `muon_before.hpp`,
`compare_release.log`, and `asan_check.log`. The checked-in benchmark retains no duplicate old algorithm.

## Memory

Prepared capacity per optimizer team slot is
`(3 * MUON_MAX_MN + 2 * MUON_MAX_MM) * sizeof(float)`, accounting for update, X, BX, A, and AA.
This matches the five buffers' maximum float storage in the old implementation, but reserves the
capacity for every configured slot up front instead of growing buffers as shapes reach OS threads.

At a largest matrix of 96 x 384, this is 504 KiB per slot, or 1.969 MiB for the four-thread validation
build. At 448 x 1792, it is 10.719 MiB per slot, or 257.25 MiB for 24 slots. These are exact scratch
capacity calculations, not measured whole-process peak RSS.

Storage follows the existing process-wide CPU parameter ownership and survives optimizer-object
destruction for reuse until DLL/process teardown. AdamW-only and forward-only builds allocate none of
this scratch. Creating a Muon-enabled host `AdamW` allocates it even when that object subsequently serves
only as a settings holder in GPU training; separating that recipe role from CPU execution ownership is
a distinct follow-up. This allocation does not occur on Qwen's forward-only validation path.

## Validation

Both explicit Release builds used the same isolated neutral CPU configuration: d96, L8, H2, FFN384,
sequence length 64, vocabulary 16824, four threads, optional architecture features disabled. The existing
TinyStories tokenizer/corpus artifacts were only read. No configurator was run against a live corpus
or sidecar, and no pre-existing user build was reconfigured.

| Unfiltered suite | Before | After |
|---|---:|---:|
| Engine | 21,415,930 assertions / 147 cases | 21,415,930 assertions / 147 cases |
| Frontend | 117,431 assertions / 226 cases | 117,432 assertions / 226 cases |
| Eval seam | 119 assertions / 7 cases | 119 assertions / 7 cases |

All passed with Catch seed 12345. The extra frontend assertion is the intentional fresh-versus-reused
scratch comparison. Neutral engine hashes are unchanged: forward `aff13b9234fbd30f`, gradient
`28689b9980d99a39`, decode `c38448bd5bb3306e`.

A separate driver exercised the real CPU `AdamW(..., true).step()` three times with deterministic
gradients across the complete parameter layout. Parameter, momentum, and second-moment array hashes
matched before/after in both Debug and Release, covering actual dispatch of different matrix shapes to
prepared team slots. Release parameter hashes were `5f17079ea1d6444e`, `46907f7a5f69913a`, and
`45d0e89c0f1615ce` after successive steps. Driver and logs: `out/muon_review/optimizer_probe.cpp` and
`optimizer_{before,after}_release.log`. Their timings are not used for a throughput claim.

Existing independent Gram-property tests cover orthogonality, bounded diagonals, zero input, in-place
output, wide/tall/square shapes, and buffer reuse, including d448-scale matrices. The CUDA test translation
unit passes syntax-only compilation with its device seam enabled; the shared header compiles as C++20.
CUDA runtime tests and a training convergence A/B were not run.

The opt-in Catch benchmark built and ran all six shapes with 10 samples each. Repeat in a CPU build:

```powershell
cmake -S . -B out/build/muon_after -DCMAKE_BUILD_TYPE=Release -DSUB0_BUILD_BENCHMARKS=ON
cmake --build out/build/muon_after --target sub0_muon_bench
out/build/muon_after/benchmarks/sub0_muon_bench.exe --benchmark-samples 10
```

Initial isolated builds implicitly selected Debug; those results are not used for the speedup table.
Explicit Release runs followed, as recorded above. The scoped C++ review found no remaining numerical,
scratch-ownership, or consumer-wiring defect; the GPU settings-holder memory tradeoff is recorded above.
