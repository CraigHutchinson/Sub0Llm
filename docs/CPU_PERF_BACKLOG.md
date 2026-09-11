# CPU performance backlog — fixed defects, open hotspots, and how to pick up either

Living document, not a one-shot report (same spirit as `docs/GATED_DELTANET.md`'s staged-status doc).

**2026-09-08 Muon follow-up:** CPU optimizer scratch is now prepared before the step loop, and the
symmetric Gram products avoid duplicate/strided work. The six-shape comparison measured 1.9–3.5x
speedup with bit-identical results; full neutral engine assertions/hashes are unchanged. See
[MUON_CPU_OPTIMIZATION.md](MUON_CPU_OPTIMIZATION.md) for measurements, memory cost, and validation.

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

### 2a. `arena_alloc`'s unconditional grad-scratch zero-fill (backend_cpu.cpp) — **DONE (2026-09-09)**

**Closed by a stronger fix than the one proposed below, and by a different route.** The memory audit that
accounted for the real 48-layer decode run's ~41 GiB peak found that the buffer this item wanted to stop
zero-filling should not exist at all in these builds: `Worker::act_grad` was sized `ACT_CAP`
unconditionally, while `Worker::grad` (the per-PARAMETER accumulator, a different array) had had exactly
this treatment since WP4d. `src/backends/cpu/internal.hpp` now derives `ACT_GRAD_FLOATS = FORWARD_ONLY ?
1 : ACT_CAP` from the same three `USE_*` constants `backward_node`'s own refusals are written against, and
`arena_alloc` hands out an EMPTY grad span in that build — mirroring what `mk_param` already did for a
parameter leaf. So the `memset`-equivalent pass this item names is gone, and so is the 7.02 GiB (at the
real Qwen4 axes) of arena it was writing into. No parameter threading was needed: the decision is a
compile-time property of the build, not of the call site, which is why the "bool parameter or a separate
`arena_alloc_fwd_only()`" plan below was not the shape the fix took.

Measured, on the real 48-layer artifact: peak working set 40.98 → 33.96 GiB (−7.02, −17.1%), the WP5c
determinism fixture reproducing all 30 ids and the full continuation byte-for-byte, and the neutral d196
suites unchanged at 28,875,042/147 and 120,889/244 (baseline re-taken on the same tree by stashing the
diff, not cited from an earlier session). The throughput effect is within noise (5.61 → 5.53 s/token), as
B20's own disk-bound finding predicts — the value is that those 7 GiB are now available to the OS file
cache the 37.11 GiB S0Q1 sidecar competes for: at a comparable point in the same run, the sidecar's
resident share rose 4.81 → 7.63 GiB.

The original entry follows, unedited, because its reasoning is still the record of how the cost was found.

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

### 2d. `gguf::dequantize_iq2_xxs` decoded ~6x slower per element than its two neighbours — FIXED (B28)

Found 2026-09-09 by `benchmarks/moe_expert_bench.cpp` (see `docs/WP4_SCOPE.md` §6 "WP6b" for the full
table and method). Isolated on real encoded planes from the real 48-layer sidecar, per expert's worth of
elements (3 x 1,638,400):

| decoder | ms / expert (before) | Melem/s (before) |
|---|---:|---:|
| `dequantize_iq1_s` | 2.21 | ~2,200 |
| `dequantize_iq4_nl` | 2.39 | ~2,000 |
| **`dequantize_iq2_xxs`** | **13.73** | **~360** |

This was not a corner case: **14,336 of this file's 73,728 routed-expert planes are IQ2_XXS**, and an
IQ2_XXS-heavy expert cost more to *unpack* than every other stage of its resolve put together (page-in +
transpose + FFN ≈ 5.8 ms). It had never been visible before because every previous measurement reported
the three decoders as one combined figure (B20's own VTune report: "22.7% combined").

**FIXED 2026-09-11 (B28, `docs/INDEPENDENT_REVIEW_BACKLOG.md`'s own B28 entry).** The hypothesis
recorded here originally — the shared `for (int j = 0; j < 8 && w < n; ++j, ++w)` tail-bound guard
alone stopping the compiler unrolling `KMASK_IQ2XS[j]` away — turned out to be only PART of the real
cause, found by actually reading the two functions side by side rather than trusting the hypothesis: the
inner loop's `(signs & KMASK_IQ2XS[j]) ? -1.f : 1.f` is a BRANCH on an effectively-random per-element bit
(the sign pattern), where `dequantize_iq1_s`'s inner loop is branch-free arithmetic throughout. A
data-dependent branch on near-random bits, executed 8x per group over 6,400 groups/plane, is exactly the
shape of cost a hoisted tail-bound check alone would not have removed.

**The fix, `include/sub0/gguf.hpp`'s `dequantize_iq2_xxs`**: two changes, both proven bit-exact before
merging (see below), not just plausible:
1. Hoist the tail-bound check as originally proposed: a fixed, unrollable 8-iteration loop for the whole-group
   case (every group except possibly the tensor's very last), falling back to the original guarded loop
   only for a genuine partial tail block.
2. Replace the per-element branch with branchless arithmetic: `KMASK_IQ2XS[j] == 1 << j` (confirmed by
   reading `gguf_quant_tables.hpp`, not assumed), so "bit `j` of `signs`" is `(signs >> j) & 1`, and
   `sign = 1.f - 2.f * bit` is EXACT for `bit` in {0, 1} — the identical float the prior ternary produced,
   not an approximation.

**Measured, real 48-layer sidecar, same host, same `moe_expert_bench` sampling (24 experts/5 reps)**:
IQ2_XXS dequant **16.07 ms/expert (306 Melem/s) -> 3.66 ms/expert (1,344 Melem/s), ~4.4x**. (The isolated
benchmark's own baseline reads slightly higher than the original 13.73 ms/expert recorded above — thermal/
scheduling variance between sessions, not a different fix; both numbers describe the same code.)

**Bit-exactness, verified two independent ways, not assumed**: `tests/gguf_tests.cpp` stayed green
(1,478 assertions), AND a from-scratch throwaway tool hashed (FNV-1a over the output float bytes) every
one of the file's real 14,336 IQ2_XXS planes (23,488,102,400 elements) decoded through the OLD code and
through the NEW code — identical hash (`c03a4f35deff2d9c`) both times.

**Real end-to-end decode impact**, `sub0llm-qwen4-forward --tokens 6` (the `forward_one` loop, real
48-layer BF16 artifact): **4.835 s/token before -> 3.85-3.98 s/token after (~18-20% faster)**, with
`forward` vs `forward_one` parity staying exactly 0 and every row's logits byte-identical before/after
(same means/rms printed both times). This is a real, measured, whole-decode-loop improvement, not a
projection from the isolated per-format number — consistent with IQ2_XXS's share of the "where a cold
resolve's time goes" breakdown dropping from 74.2% to 43.3% of a single expert's dequant cost.

**CPU-cache prefetch hints (this backlog's Task 2, dispatched alongside the fix above) were tried and
did NOT measurably help on this host** — software-prefetching the NEXT plane's raw bytes in
`moeq::ExpertCache::resolve()` while the current plane's dequant/transpose ran left the warm full-pipeline
arm unchanged (~4.79-4.88 ms/expert with vs. without, fully overlapping across repeated interleaved runs).
Reverted rather than merged disabled-by-default, per this project's preference against unproven
complexity (AGENTS.md §8). Likely explanation: at these plane sizes (320 KiB-900 KiB encoded), the
sequential read pattern `gguf::to_f32`'s decoders already walk is well served by the hardware prefetcher,
so a software hint adds instruction overhead without buying anything the hardware wasn't already doing.

### 2e. Resolve pipeline's redundant DRAM round-trips (dequant->transpose->FFN) — FUSED (B31)

`docs/INDEPENDENT_REVIEW_BACKLOG.md` B31 has the full writeup; summarized here per this file's own
cross-reference convention. The resolve pipeline (`moeq::ExpertCache::resolve` -> `gguf::to_f32` ->
`transplant::transpose_out_in` -> `moe::expert_ffn_row`) touched each expert's ~19.66 MB plane-set up to
3 times (dequant write, transpose read, transpose write) before the FFN's own read. Fixed by ELIMINATING
the transpose stage for decode's hot path, not by tiling all three stages separately: `expert_ffn_row`'s
existing per-output accumulation order turns out to be reproducible term-for-term as a direct dot product
against the plane's UNTRANSPOSED GGUF source order (`moe::expert_ffn_row_source`, new), so
`moeq::dequantize_expert_source` (new) writes the dequantized plane straight into the resolve pool's own
persistent buffer with no transpose call and no separate scratch buffer at all. Bit-exact (proven in
`expert_ffn_row_source`'s own comment; checked directly by a new `tests/moe_quant_tests.cpp` case, and by
`forward`/`forward_one` parity staying exactly 0 on the real 48-layer artifact). Real measured
before/after (`sub0llm-qwen4-forward --tokens 6`, thermal-confound-aware interleaved A/B, 4 runs each):
**3.675 s/token before -> 3.385 s/token after, ~7.9% faster** — real and reproducible, but notably smaller
than the ~40-50% a pure bandwidth-utilization estimate implied, most likely because MoE resolve is only
part of decode's own per-token cost (backbone, attention, GDN, QSA, router and shared expert are
unaffected) and because the remaining dequant+FFN-read traffic, not the now-eliminated transpose traffic,
appears to dominate single-thread bandwidth demand. A `constexpr`, L2-cache-derived row tile
(`row_tile()`) was also added for the new dot-product loops, per this task's own cache-aware brief, but
measured to make no reliable difference over the plain unblocked loop — same shape of honest null result
as 2d's own prefetch finding above, and for the same underlying reason: the access pattern here (stream a
contiguous row once, reuse a small L1-resident vector) already has no strided axis to block away.

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
