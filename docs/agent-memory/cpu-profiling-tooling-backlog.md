---
name: cpu-profiling-tooling-backlog
description: "Backlog - need a proper CPU profiler (Intel VTune / Windows Performance Analyzer) to find remaining CPU-utilization dips; found two real single-threaded bottlenecks (evaluate(), AdamW::step()) by code inspection alone, but that approach doesn't scale"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-09-03T18:03:49.035Z
---

Investigating a CPU-utilization dip pattern during CPU-forced training (`--content-embed`, d448, 2026-07-16), two real, previously-unparallelized single-threaded bottlenecks were found and fixed purely by **reading the code and reasoning about it** — no profiler was used:
1. `evaluate()` (`src/train_stage.cpp`) — validation NELBO computed via a plain serial loop over up to 128 windows, one `forward()` call at a time, instead of spreading them across worker threads the way `train_batch` already does. Fixed: wrapped in the same `#pragma omp parallel`/`#pragma omp for reduction` shape `train_batch` uses.
2. `AdamW::step()` (`src/backend_cpu.cpp`) — the optimizer update (including Muon's `newton_schulz5` per weight matrix) ran on a single core, once per training step, for every parameter tensor sequentially. At d448 with ~77 Muon-eligible matrices this is likely the dominant serial cost (matches the observed ~1-core-of-24 utilization during the "idle" stretch). Fixed: parallelized across `PARAM_LAYOUT` entries (each owns a disjoint slice of the param/grad/moment arrays, so no synchronization needed).

Both fixes were verified against the full engine test suite (12,320,801 assertions, unchanged) and empirically against a live run (throughput ~2-4 win/s -> ~7-8 win/s). **But after both fixes, the user still observes periodic CPU-utilization drops on the live meter** — meaning there's at least one more serial bottleneck (or several smaller ones) not yet found this way.

**The gap**: this whole investigation happened via manual code reading + reasoning from an empirical "4-5% ≈ 1/24 cores" observation, which is slow and easy to miss things with (there could be other single-threaded hot spots — the CPU per-window materialization loop in `train_stage.cpp` before `train_batch` is called, checkpoint I/O, `opt.zero_grad()`, etc. — none individually confirmed as a dip cause, just not yet ruled out either).

**How to apply**: before the next round of CPU perf work, get a real profiler set up — something that shows per-core timeline utilization + call-stack attribution, the CPU equivalent of what Nsight Systems/Compute already give this project for CUDA (see [[ncu-depth-vs-nsys-busy-time]] and [[hardware-efficiency-iteration-standing-task]] for the established CUDA-side methodology this should mirror). Candidates for this Windows + Intel Core Ultra 9 275HX machine (see [[host-cpu-arrow-lake-hx]]):
- **Intel VTune Profiler** — the closest real analog to Nsight: deep microarchitectural profiling (per-core timeline, thread utilization, stalls, cache misses), free for this kind of use. Primary recommendation.
- **Windows Performance Analyzer (WPR/WPA)** — lighter, Windows-native, no install beyond the SDK; less deep than VTune but zero-friction if VTune setup is a blocker.
- **Intel Advisor** — complementary, more about vectorization/threading design than live profiling; worth knowing about but secondary.

Once a profiler is in place, re-run the same `--content-embed` d448 CPU training scenario and get an actual timeline instead of guessing from throughput numbers and OpenMP-pragma code review.

**UPDATE (2026-07-16): VTune is installed and works without elevation for basic hotspot sampling.**
Found at `C:\Program Files (x86)\Intel\oneAPI\vtune\2026.3\` (source `vtune-vars.bat` to get the CLI on
PATH; the IDE also has the `intel-corporation.oneapi-analysis-configurator`/`oneapi-environment-configurator`
VS Code Insiders extensions, UI-driven, not scripted from here). Confirmed live:
`vtune -collect hotspots -target-pid <pid> -duration 20 -result-dir <dir>` attaches to an ALREADY-RUNNING
process and produces a real hotspot report with zero setup friction (only *microarchitecture* drill-down
needs admin — "Failed to enable the policy to profile system performance" is what plain `wpr -start CPU`
hits without elevation; VTune's user-mode sampling sidesteps that entirely).

Real result against the live `--content-embed` d448 CPU run (post evaluate()/AdamW::step() parallelization
fixes): 20.03s elapsed, 313.0s total CPU time across 24 threads (~65% average core utilization over the
window) -- `sub0::backward` is the single largest named hotspot at 50.9%, matching train_batch's expected
dominance; the rest is split across a few *unresolved* `func@0xADDRESS` entries VTune couldn't name.

**Real gap found, then FIXED (2026-07-16):** the Release build shipped no `.pdb`, so VTune couldn't name
the remaining hotspots. `CMakeLists.txt`'s `CMAKE_CXX_FLAGS_RELEASE` now includes `-g` alongside `-O3`
(RelWithDebInfo-style — full optimization, but every Release build also emits CodeView debug info / a
`.pdb` via lld-link). Verified: reconfigured+rebuilt the `gsm8k` dir, `sub0_core.pdb` (7.4 MB) produced,
no compile errors. Not yet applied to `scratch-ce` (avoided reconfiguring while its own live process had
the DLLs open) — takes effect automatically on that dir's next reconfigure.

**UPDATE (2026-07-16): elevated `uarch-exploration` run against the live `--content-embed` d448 CPU
training process (post both parallelization fixes).** User ran the elevated collection themselves (30s,
`vtune -collect uarch-exploration -target-pid <pid> -duration 30 -result-dir <dir>`); finalization/report
also had to happen from the SAME elevated context (a read-only/non-elevated session can open but not
re-finalize a result captured under elevation — `vtune: Error: 0x40000003 ... Cannot re-finalize a
read-only result`).

Real numbers, and they change the picture from the plain-hotspots run:
- **Effective CPU Utilization: 34.7%** (average 8.318 of 24 logical cores active) — confirms a real,
  additional utilization gap survives both parallelization fixes; matches the user's live-meter
  observation of continuing (much milder) dips.
- **E-cores did far more total clockwork than P-cores** (600.8B vs 67.4B clockticks) despite E-cores
  being individually weaker (CPI 1.099 vs P-core's 0.661, i.e. P-cores run MORE efficiently when used).
  **Root cause confirmed by code search**: this codebase has ZERO P-core/E-core-aware thread affinity or
  pinning anywhere (`backend_cpu.cpp`'s `#pragma omp parallel num_threads(DEFAULT_THREADS)` — used by
  `train_batch`, the new `evaluate()` parallelization, and the new `AdamW::step()` parallelization — is
  entirely default OS/OpenMP scheduling, no `OMP_PROC_BIND`/`OMP_PLACES`/affinity API calls). The
  diffusion codebase's OWN P-core-first pinning module ([[cpu-threading-model]]) does NOT exist here —
  that memory was previously misfiled under Sub0Llm's own section; corrected in MEMORY.md.
- **Back-End Bound dominates both core types** (P-core 75.9%, E-core 92.3% of pipeline slots) — real
  execution/memory stalls, not front-end/decode-bound.
- **P-core DRAM Memory Bandwidth: 38.5% of clockticks** — "a significant fraction of cycles was stalled
  due to approaching bandwidth limits of the main memory." Strong candidate: `train_batch`'s gradient
  reduction (`backend_cpu.cpp`, the `for (i) { for (t) s += g_workers[t]->grad[i]; }` loop) reads
  `nthreads` full copies of the gradient array every step — exactly the "read many redundant large
  arrays" pattern that saturates DRAM bandwidth. Not yet fixed or even isolated in isolation — a
  candidate, not a confirmed root cause.
- **E-core Memory Subsystem Bound 57.7%** (L2 Miss specifically 43.2%).
- **Vector Capacity Usage (FPU): 48.8%** on P-core — SIMD/vector units only ~half-utilized when active,
  real headroom (matches [[single-thread-optimization-and-cycle-bench]]'s existing SIMD-tuning precedent).

**Concrete next-step candidates this surfaced** (none started): (a) add P-core-first thread affinity to
the OpenMP regions (`train_batch`/`evaluate`/`AdamW::step`) — the single biggest lever given the
E-core/P-core clocktick imbalance; (b) investigate whether the gradient-reduction pass can avoid reading
every thread's full grad array (tree reduction, or incremental accumulation as each thread finishes,
instead of a separate full linear pass) given the DRAM-bandwidth-bound finding; (c) revisit vectorization
in the hot loops now that Vector Capacity Usage is measured, not assumed.

**UPDATE (2026-09-03): reusable pattern for profiling a math-core function in isolation, not just a live
training run.** For [[pivot-to-existing-model-support-qwen4-preview]]'s GR/MoE/QSA math cores — no
training loop to attach to, just `*_math.hpp` free functions — built a small standalone `.cpp` (outside
the repo's own CMake build) that `#include`s the real headers directly and loops the hot calls at real
model dims for tens of seconds, compiled with `clang++ -O3 -g -march=native` (the `-g` is required for
VTune to resolve symbols, same lesson as the 2026-07-16 update above). Launch it with PowerShell's
`Start-Process -PassThru` (NOT bash `cmd &`-backgrounding, which was unreliable here — the target process
finished before `vtune -collect hotspots -target-pid <pid>` could attach, twice) and immediately follow
with the `vtune -collect hotspots -target-pid $p.Id -duration <n>` call in the SAME command block (a short
`Start-Sleep` in between is enough) so the attach happens well within the target's own run window. This
found a real defect (MoE's `router_topk_row`, 17.1% of CPU time, an output-major strided loop the earlier
code-reading-only review had missed) that a pure code read did not catch — profiling a realistic call
pattern is a genuinely different, complementary check to code-reading, worth doing routinely after any
"reviewed X for performance" claim, not only when explicitly asked.
