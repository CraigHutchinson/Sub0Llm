# Intel groundwork checkpoint — 2026-09-08

Worktree: `out/worktrees/intel-groundwork`, branch `research/intel-groundwork`, based on `90721bc`.
All source, runner, generated fixtures and results live here; main Qwen sources/builds are unchanged.
This is partial I00/S0/S1 plus early I05/I19 measurement, not completed native-backend qualification.

## Findings and scope

- The installed compiler is DPC++ 2025.3.3; oneDNN reports 3.9.1. The compiler, Level Zero loader and
  libraries were already installed. Explicit MSVC initialization, Intel `bin`/`lib`, and oneDNN `bin`
  paths make the standalone build reproducible without changing the main engine build.
- PCI `8086:7D67` executes through SYCL's Level Zero backend, runtime driver `1.15.39183+3`.
  Do not confuse that runtime version with the previously recorded Windows display driver version.
- Runtime reports 64 compute units, 64 KiB local memory and subgroups 8/16/32; FP16/FP64 aspects true.
  These are reported capabilities, not emitted-ISA evidence or a throughput specification.
- Host/shared/device USM aspects are true; system USM is false. A runtime-managed shared pointer
  is therefore not evidence that an arbitrary `malloc` or file-mapping pointer is GPU-accessible.
- Matrix-combination query succeeds with zero entries. No matrix path is qualified in this runtime;
  DPAS/XMX hardware presence and integer-dot instructions still require I18 code-generation evidence.
- Reported global memory: 36,022,943,744 bytes (33.549 GiB); maximum allocation: 4,294,959,104 bytes.
  We did not allocate these amounts. They are neither free memory nor a sustainable model budget.

## Correctness and interoperability

At 257 and 65,536 floats, a generated read-only Windows file mapping is copied into shared USM, staged
into device USM, transformed twice, and copied back. Every element matches the exact CPU formula.
Explicit USM is at most 512 KiB. The mapping is never dereferenced directly by a kernel.

A custom-kernel → oneDNN ReLU → custom-kernel chain passes on the same context and queue, borrowing
the same USM pointers and checking every output. Explicit completion boundaries are retained: this
proves a conservative interoperability route, not low-overhead asynchronous interop or direct Level
Zero command-list/module loading. The latter remains open in S0/I20.

Five negative checks fail for the intended reasons: missing argument, missing file, truncated fixture,
incorrect fixture contents and Intel GPU hidden by a CPU-only selector. No CPU fallback is accepted.
An initial runner revision mishandled scalar argument splatting; explicit string arrays fixed it and
both positive sizes plus the five negative cases passed afterward. Earlier failed logs remain under out.

## Early dense projection benchmarks

FP32 synthetic values at real projection/FFN plane shapes. Weights are `[K,N]`, activations `[M,K]`,
outputs `[M,N]`. The custom implementation is deliberately a simple coalesced-output dot kernel;
oneDNN uses strict f32 math. All output elements agree within 1e-4 and 64 selected outputs per shape
are independently checked with double CPU dots. Inputs are exact small dyadic values, so this does
not qualify arbitrary real-weight precision or activation quantization.

Five alternating paired warmed trials per shape in one process; timings include host submission and
completion, but exclude weight/activation upload and output download. Those transfers and first-call/
primitive preparation are recorded separately in the raw log. No project-owned allocation is inside
the sampling loop; runtime/library allocation counts have not been instrumented.

| M × K × N | Simple SYCL median, ms | oneDNN median, ms |
|---|---:|---:|
| 1 × 96 × 384 | 0.0402 | 0.0158 |
| 1 × 2560 × 640 | 1.0268 | 0.1182 |
| 32 × 2560 × 640 | 2.7903 | 0.1972 |
| 128 × 2560 × 640 | 7.5912 | 0.4529 |
| 1 × 640 × 2560 | 0.2396 | 0.0923 |
| 32 × 640 × 2560 | 2.2237 | 0.1376 |
| 128 × 640 × 2560 | 9.3175 | 0.3331 |
| 1 × 2560 × 6144 | 1.5077 | 1.1451 |
| 32 × 2560 × 6144 | 33.1421 | 1.1680 |
| 128 × 2560 × 6144 | 124.7710 | 2.8747 |

These results establish useful baselines and show why a naive kernel is insufficient for prefill.
They do not establish a compiler/runtime winner, tuned-kernel limit, CPU-relative speedup or Qwen token
latency. Decode/prefill inference state, GR/GDN/QSA, router, real IQ experts and full vocabulary head
remain absent. In particular the f32 FFN-plane timing is not quantized MoE timing.

## Early USM access comparison

The memory probe uses 4 MiB input plus 4 MiB device output, sequential affine transformation, and five
warmed repetitions. Input modes are host/shared/device USM; the mode order reverses in a second round.
All elements are checked before timing, and a changed CPU input is verified after the handoff run.
Host/shared preparation uses CPU copy; device preparation uses an explicit completed upload.

Warmed medians range approximately 0.120–0.154 ms in this run. The first host-mode call takes ~35 ms,
including cold kernel startup; it must not be attributed to host-memory placement. Handoff observations
are noisy (device mode 0.485/1.329 ms). No allocation mode is promoted from this small experiment.
Output is always device-owned, so this does not yet compare shared versus device write-only scratch.
No arbitrary file-map import, page-size control, sparse expert-cache trace or large working set ran.

## Reproduction and evidence

Run from this worktree in a fresh PowerShell process:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -WithOneDnn -Elements 257
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -WithOneDnn -CheckFailures
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -DenseBenchmark
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -MemoryBenchmark
```

The script builds only one standalone translation unit, generates its own tiny fixture and captures
source/header/runner hashes, executable hash, compiler flags, OS/power configuration, observed competing
processes and exit status. Archived [raw records](intel-groundwork/2026-09-08/dense-probe.txt) and
[manifests](intel-groundwork/2026-09-08/dense-manifest.json) accompany this report. Large binaries and
intermediate build products remain in the manifest's local output directory. No driver/SDK install.

Before trials, the main coordination log and process inventory were checked; no matching active model
or compiler process was observed. This is not proof of a quiet system for the entire run. Temperature,
package energy, background display interference, actual residency, fault counters and confidence
intervals are not recorded yet. Five in-process component repeats are exploratory, not independent
request trials or a p95 claim. Compilation immediately preceded execution. Reserve a measurement window
and extend I21 controls before promoting any performance result.

## Plan feedback / next bounded work

The [Windows USM review](INTEL_IGPU_WINDOWS_USM.md) adds official-source and installed-header findings
plus ordered S1/I19 probes. On 2026-09-09, the ordinary USM capability probe, its separately gated
prepared-copy API variant and the prepared-copy benchmark all compiled and linked with DPC++ 2025.3.3.
No new binary executed, so these are build-availability facts rather than runtime capability or speed.

R0: installed toolchain and conservative library interop are usable; runtime-managed shared memory and
explicit staging deserve comparison. System-pointer access and matrix execution are unqualified.
The user's [USM paper](INTEL_IGPU_ISA_MEMORY_RESEARCH.md#new-paper-and-the-local-probe-distinction)
is integrated into I19's follow-ups, without extrapolating its Linux/discrete results to this machine.

1. Finish S0 direct Level Zero module/argument ABI and library interoperability; do not count SYCL over
   Level Zero as a direct-submission comparison.
2. Extend I19 to cold/warm selected-range access, changed CPU handoffs, write-only scratch, bounded
   staging versus shared allocation and representative expert-cache traces; explicitly measure setup,
   faults/budget where available. Only then increase the working-set envelope.
3. S2: real IQ1_S/IQ2_XXS/IQ4_NL planes, code-generation proof and precision-controlled alternatives.
   Keep quantization/packing crossed with allocation/submission variants rather than attributing all
   differences to the API. Existing f32 numbers are controls, not encoded-path forecasts.
4. S3/S4: dynamic command replay and representative GDN/GR/QSA/MoE chains. I06 remains pending.
5. WP5a/b/c and B20's decode/memory work are merged on `main`; freeze their tokenizer, artifact census,
   generation inputs, measured CPU baseline and updated memory findings in I01. Recheck the active log
   for successor ownership before shared CPU edits; mechanism probes remain isolated regardless.

The scoped C++ review checked the private shared selector/USM deleter, queue/resource lifetimes,
benchmark consumers, failure paths and fixed allocations. No remaining MUST finding was identified.
Default engine tests were not rebuilt because these standalone sources change no engine or CMake path.
