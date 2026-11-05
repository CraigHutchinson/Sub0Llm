---
name: single-thread-optimization-and-cycle-bench
description: "cycle-accurate `sub0llm bench` (rdtsc min-of-N, pinned) is the control measure; op_linear dX/dW fixes gave 2.05x single-thread; next lever = batch-parallel"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

`sub0llm bench [--iters N --threads N]` is the optimization control: single-thread, pinned to core 0, OpenMP forced to N threads, timed in **rdtsc cycles reported as the min over many iters** (the un-throttled iteration), so thermal drift / scheduling don't move the number. Reports forward/backward/optimizer/full-step cycles/window. Use this — not wall-clock win/s — to judge serial optimizations on this thermally-drifting host.

Single-thread baseline was 162M cyc/step (fwd 31M, **bwd 122M**, opt 7M = 19 win/s). Two op_linear fixes (commit f8dfaea) → 79M (**2.05x**, 39 win/s); multi-thread 24-core ~96→128 win/s (~1.3x only — Amdahl, leaner serial code makes per-op fork/join overhead relatively bigger):
1. **dX dot-product** `s += dY[o]*W[o]` is a float reduction clang WON'T vectorize without reassociation permission — add `#pragma omp simd reduction(+:s)` (NOT global -ffast-math). This was the big one (bwd 97→37M); lm_head made it a 2048-wide scalar loop.
2. **dW reorder** `(p,o,t)→(p,t,o)` so the inner loop is a contiguous axpy (vectorizes); kept parallel-over-p (race-free).
3. `__restrict` the pointer trios in forward/backward.

Parity held: train loss identical to 4 dp (2.4747/2.0278 @ steps 1960/3920) vs the pre-opt sweep; tests 9/9; resume bit-exact.

**Fast transcendentals (ad3c9d2)**: fast_exp (vectorizable, ~1e-6) + tanh-form GELU replace scalar libm exp/erf in softmax + GELU, behind `set_fast_math()` / `SUB0_EXACT_MATH=1` (default fast; print_config shows math:). VERDICT — modest: **the model is GEMM-bound, not transcendental-bound** (clang std::exp/erf are only ~10-15 cyc, ~2M of 31M forward). ~5% single-thread (39→41 win/s), ~10% multi-thread (130→145). Training holds: fast vs exact loss within ~0.002 nats. Grad check passes both modes (fast fwd/bwd mutually consistent: tanh-GELU + its exact derivative; softmax bwd reuses stored fast-exp probs).

**Regression net (90bf21e)**: tests/engine_tests.cpp — finite-difference GRADIENT CHECK (directional == ||g||, + top-magnitude per-param), attention causality, forward determinism, init-loss≈ln(VOCAB), AdamW overfit, save/load round-trip. <1s, training-free — run before/after any hot-loop change.

**Batch-level data-parallel DONE (d344dfc) + autotuner (f96cf10)** — the big multi-thread win. `train_batch()` runs each window's whole fwd+bwd on its own thread into a private grad accumulator, then sums into the shared grad; removed the per-op `#pragma omp parallel for` (which left attn/GELU/rmsnorm/softmax serial → Amdahl-capped at ~3.5x). Per-thread state is one `Worker` struct in a static BSS pool indexed by omp thread id (NOT thread_local — a multi-MB thread_local in a DLL overruns Windows' static-TLS and faults; BSS is demand-paged). Results: single-thread **131 win/s** (full step ~23M cyc after FTZ/DAZ cut the optimizer 7M→1.3M — denormals, see [[denormal-stalls-in-training-need-ftz]]); data-parallel **1550 win/s @ batch 384** (11.8x). Autotuner persists best DEFAULT_THREADS/WINDOWS_PER_THREAD, baked into config by sub0-configure (tune.hpp).

**TWO HARD LESSONS:**
- **Removing per-op `#pragma omp parallel for` made single-thread ~2.4x faster** (41→97 win/s): libomp pays fork/join overhead per region *even at 1 thread*, so the earlier "single-thread" bench numbers were contaminated. Per-op parallelism over T=64 rows of tiny matmuls is the wrong granularity; batch-level is right.
- **`pos_ids` use-after-scope** crashed every backward as a Heisenbug: it was changed `static`→forward-local, but op_embed stores the pointer and backward reads it AFTER forward returns → dangling stack. Fix = `static thread_local`. Lesson: any pointer a Node stores (ids, spans) must outlive forward→backward; a stray fprintf "fixed" it by shifting the stale stack (classic UAS Heisenbug).

**fast-math is now COMPILE-TIME** (f96cf10): `constexpr bool FAST_MATH` from `#ifdef SUB0_EXACT_MATH` (CMake option SUB0_EXACT_MATH, default OFF=fast), all dispatch via `if constexpr` (dead branch eliminated); set_fast_math removed.

**Optimization is essentially DONE — next focus is MODEL QUALITY/coherence** (samples still word-salad). Likely capacity-bound: 0.74M params (D=96/L=3/H=4/head_dim=24/seq=64) is below the TinyStories coherence threshold; best val NELBO 1.6231. Throughput now affords a bigger founded model (D=256/L=6/H=8/seq=128) — see [[ch29-model-quality-founded-dims]]. Tests now 27/27.
