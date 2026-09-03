# CPU performance backlog — fixed defects, open hotspots, and how to pick up either

Living document, not a one-shot report (same spirit as `docs/GATED_DELTANET.md`'s staged-status doc).
Two review passes have happened so far — a targeted code-reading sweep (2026-09-03, WP1-3 post-merge)
and a VTune-guided follow-on the same day — and each found something the other missed. Read `AGENTS.md`
first if you haven't; the standing three-pillar policy (correctness + performance + memory on every A/B)
and the compile-time-over-runtime preference both apply to everything below.

## 1. What's already fixed (closed, for reference — don't re-derive these)

All four are the SAME defect class: an output-major GEMV loop (outer over the output dimension, inner
striding the weight matrix by the output width) instead of this codebase's own established input-major/
contiguous convention (`linear_row`'s idiom in `qsa_math.hpp` — outer over the input/contraction
dimension, inner contiguous over the output, matching every `[rows=in, cols=out]` weight layout in this
project). All four were pure summation reorders, verified bitwise-or-within-fixture-tolerance identical
before/after, no algorithmic or numerical change.

| Function | File | Found by | Measured speedup | Commit |
|---|---|---|---|---|
| `gr::mix()` up-projection | `gated_residual_math.hpp` | code reading | ~5.0-5.1x (106ms → 20-21ms/call, real dims) | `b980638` |
| `moe::expert_ffn_row()` | `moe_math.hpp` | code reading | ~14.3-14.9x (6.2-6.6ms → 0.44ms/call, real dims) | `b980638` |
| `moe::router_topk_row()` router logits | `moe_math.hpp` | **VTune hotspots** (17.1% of CPU time — code reading missed it) | MoE per-token cost ~650-790ms → ~452ms/iter (T=64, real dims) | `46c54b1` |
| `qsa::indexer_select_row()` block-key recompute | `qsa_math.hpp` | code reading | **algorithmic**, not just cache-locality: O(T²/ratio) → O(T/ratio); `pool_block_key()` calls 45→5 at T=20/ratio=4 | `c6db692` |

Full technical writeups: `docs/QSA.md` §11, `docs/MOE.md`/`docs/GATED_RESIDUAL.md` (loop-order fix noted
inline), and memory `pivot-to-existing-model-support-qwen4-preview.md`'s 2026-09-03 entries for the full
before/after numbers and verification detail.

**Lesson driving this doc's existence**: the code-reading review that fixed `gr::mix()`/`expert_ffn_row()`
did NOT look at `router_topk_row()` — same file, same defect class, just not in the two functions someone
happened to read closely. VTune caught it in one profiling run. Neither method alone is sufficient; do
both, and don't declare a "reviewed X for performance" pass complete on code-reading alone.

## 2. Open items — checked, real, NOT yet fixed (start here)

### 2a. `arena_alloc`'s unconditional grad-scratch zero-fill (backend_cpu.cpp)

Found during the QSA performance review (2026-09-03), flagged but not fixed — cross-cutting, not specific
to QSA. `arena_alloc` (the bump allocator over `Worker::act_data`/`act_grad`) always zero-fills BOTH the
returned activation span AND the grad span unconditionally, even for Stage-1 forward-only ops (GR, MoE,
QSA — none has a backward pass yet, so their grad scratch is allocated and zeroed but never read). At
real-model scale, with every layer materializing GR/MoE/QSA activations, this is a real, currently-unpaid
cost: one `memset`-equivalent pass over a buffer nothing will ever consult.

**How to fix**: thread whether backward is actually needed for a given call site down to `arena_alloc`
(a bool parameter, or a separate `arena_alloc_fwd_only()` that skips the grad span's zero-fill), gated so
existing backward-capable ops (attention, FFN, GDN Stage 2+) are completely unaffected. Verify with the
existing neutral-hash + fixture-test battery (same pattern as every fix above) — this should be a
zero-behavior-change, pure-savings fix, easiest to verify of everything in this doc.

### 2b. Training-side findings from the 2026-07-16 VTune session (`AdamW::step`/`train_batch`, live process)

Recorded in memory `cpu-profiling-tooling-backlog.md`; older (predates the Qwen4-preview pivot mechanisms
above) but still real and unaddressed. From an elevated `uarch-exploration` run against a live
`--content-embed` d448 CPU training process, post the `evaluate()`/`AdamW::step()` parallelization fixes:

- **Effective CPU Utilization 34.7%** (8.3 of 24 logical cores average) — a real utilization gap survives
  both parallelization fixes.
- **No P-core/E-core-aware thread affinity anywhere** in this codebase (`#pragma omp parallel
  num_threads(DEFAULT_THREADS)` is plain OS/OpenMP default scheduling). E-cores did 600.8B clockticks vs
  P-core's 67.4B despite P-cores running more efficiently per cycle (CPI 0.661 vs 1.099) — **candidate
  (a): add P-core-first affinity to the OpenMP regions** (`train_batch`/`evaluate`/`AdamW::step`), the
  single biggest lever this run surfaced.
- **P-core DRAM Memory Bandwidth 38.5% of clockticks** — candidate root cause: `train_batch`'s gradient
  reduction (`for (i) { for (t) s += g_workers[t]->grad[i]; }`) reads `nthreads` full copies of the
  gradient array every step. **Candidate (b): tree reduction or incremental accumulation** instead of a
  separate full linear read-every-thread's-array pass. Not yet isolated as confirmed root cause — a
  candidate, not proven.
- **Vector Capacity Usage (FPU) 48.8%** on P-core when active — real SIMD headroom. **Candidate (c):
  revisit vectorization in the hot training loops** now that this is measured, not assumed.
- E-core Memory Subsystem Bound 57.7% (L2 Miss 43.2%) — likely downstream of (a)/(b), re-measure after
  those land rather than chasing independently.

None of (a)/(b)/(c) started. This predates the GR/MoE/QSA math cores above and is about the TRAINING hot
path (`train_batch`, `AdamW::step`, gradient reduction), not the Qwen4-preview mechanisms' forward math —
a different subsystem, worth its own dedicated pass rather than folding into the WP4 forward-pass work.

### 2c. `qsa::linear_row` — genuine compute, not a defect, but a real future target

VTune's 2026-09-03 hotspot run showed `qsa::linear_row` at ~40% combined across 3 call sites (q/gate/k/v/
o projections at real dims: `hidden_size=2560` against `q_width=6144`, `kv_width=512`, etc.) — checked and
confirmed ALREADY input-major/contiguous, i.e. correctly implemented. This is real, unavoidable GEMV work
at real Qwen4 attention dims, not a bug. The only further headroom here is a genuinely different kind of
work: SIMD intrinsics / a BLAS call / explicit threading for these specific GEMVs, which is a bigger
undertaking than any fix in §1 (those were all same-file, same-pattern, zero-risk reorders; this would be
new code with its own correctness surface). **Do not attempt this opportunistically** — scope it
deliberately (own design doc, own fixture-gated correctness pass) if and when the forward-pass throughput
at real Qwen4 scale (WP4) makes it the priority, rather than folding it into an unrelated task.

## 3. Methodology — how to reproduce or extend this profiling

Two complementary techniques, use both, not either:

**Code reading**: grep for GEMV-shaped loops (`for (int o ...) { for (int i ...) out[o] += ... w[i*out_n+o]`
or similar) and check the outer/inner loop order against the weight's own declared `[rows=in, cols=out]`
layout comment. Cheap, but only as complete as the reviewer's own attention — it missed `router_topk_row`
even in the same file/pass that fixed its neighbors.

**VTune, attached to a standalone benchmark** (not the full engine — no training loop or engine build
needed for a math-core-level question): write a small `.cpp` that `#include`s the real `*_math.hpp`
headers directly, calls the hot functions in a loop at REAL model dims for tens of seconds, compile with
`clang++ -O3 -g -march=native` (the `-g` is required — VTune cannot resolve symbols in a Release build
without debug info — the same lesson `CMakeLists.txt`'s `CMAKE_CXX_FLAGS_RELEASE` already applies to the
real engine build, see §1's referenced memory). Launch it via
PowerShell `Start-Process -PassThru` (bash `&`-backgrounding was unreliable here — the process finished
before the VTune attach happened, twice) and attach in the SAME command block with a short
`Start-Sleep` in between:

```powershell
$env:Path += ";C:\Program Files (x86)\Intel\oneAPI\vtune\2026.3\bin64"
$p = Start-Process -FilePath ".\my_bench.exe" -ArgumentList "60" -PassThru -NoNewWindow `
     -RedirectStandardOutput out.log -RedirectStandardError out.err
Start-Sleep -Seconds 2
vtune -collect hotspots -target-pid $p.Id -duration 30 -result-dir .\vtune_result
```

This attaches without elevation (user-mode sampling, `-collect hotspots`) — the same
`vtune -collect uarch-exploration -target-pid <pid> -duration <n> -result-dir <dir>` command gives the
deeper microarchitecture numbers used for the training-side §2b findings, but needs an ELEVATED shell for
BOTH the collection and its finalization/report step (a non-elevated session can open but not re-finalize
a result captured elevated — `Cannot re-finalize a read-only result`). VTune install: `C:\Program Files
(x86)\Intel\oneAPI\vtune\2026.3\` on this machine (source `vtune-vars.bat`, or add `\bin64` to `PATH`
directly as above).

## 4. Priority guidance for whoever picks this up next

1. **§2a (`arena_alloc` zero-fill)** — smallest, safest, well-understood fix; do this first if picking up
   idle cycles before WP4.
2. **§2b(a) P-core affinity** — biggest single lever from the training-side data, but scoped to the
   TRAINING hot path, not the Qwen4-preview forward-math work; do this as its own dedicated pass.
3. **§2c (QSA GEMV SIMD/BLAS/threading)** — real headroom, but deliberately scope it (own design doc) when
   WP4's real-scale throughput numbers make it the actual bottleneck, not before — don't guess ahead of
   the real-scale measurement this doc's own §6c-of-`QWEN4_MEMORY_ORCHESTRATION.md` gate calls for.
4. Re-run the VTune methodology in §3 after ANY of the above lands, and after WP4's real-scale build
   exists — new hotspots emerge as old ones close, same lesson this pass already taught twice.
