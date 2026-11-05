---
name: denoiser-toolbox-and-memory-milestone
description: keep flat/hier/MERA as a perf-vs-memory TOOLBOX (pick by target context length); LATER milestone after core engine stabilises = VRAM accounting + quantized types (Q8/f16)
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

User direction (2026-06): treat the ch32 denoiser variants as a **toolbox of performance-vs-memory
alternatives** for scaling the final engine, not one-winner. Measured tradeoff (ch32_hier_ceiling,
2E_RESULTS):
- **flat** `O(N²)` — best accuracy / simplest, but dies ~N=1024 on 8 GB. Use for short contexts.
- **hier** (single-level, `O((N/c)²)`) — lowest overhead, FAST at short/medium N (beats MERA below
  ~N=4096), but walls ~8192 (its coarse pass is still N²). "Fast but more memory-bound at length."
- **MERA** (recursive, `O(N·w)`) — more per-step overhead at small N, but constant throughput to
  ~8192 (22× hier at N=8192) and scales furthest. "Slower at small N but scales." Pick by target
  context length: flat (short) → hier (medium) → MERA (long).

**Accuracy CONFIRMED (3 seeds, ch32_hier_ab N=256): MERA BEATS flat on NLL** — MERA overall −1.7 / −3.7
/ −3.5% vs flat in 3/3 seeds (content −3…−4.4%), ~2.1× faster, with FEWER blocks than flat (3 vs 4 → not
capacity). Single-level hier was +28–41% WORSE; the recursive multi-level MERA is BETTER — M3 multi-scale
hypothesis vindicated in accuracy (log-depth resolutions beat flat single-resolution attention as an
inductive bias for power-law language).

**Verdict (user's "flat never wins" — CONFIRMED):** flat is **strictly dominated on every metric**
(accuracy, compute, context). Toolbox simplifies: **MERA = the DEFAULT** (most accurate + scales +
~2× faster than flat); **hier** keeps a NARROW niche only where small-N raw speed matters more than
accuracy (lower per-step overhead than MERA below ~N=4096, but +28–41% worse NLL); **flat = reference /
accuracy oracle only** (educational baseline; "is the hierarchy accurate?" only means anything measured
against it) — NOT a deployment option.

**Fixed/preallocated memory is a DESIGN PRINCIPLE (user note 2026-06).** Observed VRAM fluctuating
constantly during training → heavy over-dependence on DYNAMIC allocation; memory management becomes the
bottleneck at scale. Grounded in `src/backends/cuda/backend.cpp` (CORRECTED — the earlier "per-kernel scratch malloc/free
every call" was a misread: those `*_bench` functions are MICROBENCHMARKS, not the training path). Real
hot-path churn: (a) all autograd-graph tensors go through `CudaPool` which caches freed blocks **≤16 MB**
and reuses them (good — repeated small shapes → steady-state cudaMalloc≈0), but **tensors >16 MB BYPASS
the pool** → cudaMalloc/cudaFree (each SYNCHRONISES the device) every step — this is the dominant
fluctuation source AT LARGE N (e.g. B·N·D activations exceed 16 MB by N≈8192); (b) `sum_squares` did a
raw cudaMalloc+cudaFree of a 4-byte scalar EVERY step via clip_grad_norm — FIXED (persistent static
scalar, allocated once); (c) under VRAM pressure (free < 2 GB) the pool stops caching → frees → extra
fluctuation exactly when scaling; (d) inherent intra-step occupancy: reverse-mode autograd builds the
whole forward graph (rises) then tears down in backward (falls) — only gradient checkpointing reduces
that PEAK. Training shapes are FIXED (B, N, D constant) so the working set is identical each step → it
SHOULD preallocate once. Staged fix, cheap→foundational: (1) DONE — sum_squares persistent scalar; (2)
make the pool size-class aware / cache repeated LARGE (>16 MB) activations under the VRAM guard (removes
the large-N malloc/free); (3) a per-step ARENA allocator (one preallocated slab, bump-allocate during
fwd+bwd, reset pointer after opt.step — zero per-step alloc, stable VRAM, predictable peak → removes the
ceiling's thrash-then-OOM, makes max-N sizing safe). Same lever as [[ch29-training-throughput-findings]]
(zero-alloc); underpins the memory milestone — predictable fixed memory lets the toolbox pick
variant+N+precision safely.

**LATER milestone (gated on the core engine stabilising first — do NOT start now):**
1. **VRAM accounting** — properly understand/measure how each variant consumes the 8 GB (the ceiling
   walls are MEMORY: hier = quadratic coarse activations; MERA = linear-but-large autograd graph). Add
   real peak-memory instrumentation (cudaMemGetInfo) so the toolbox choice is data-driven, and add
   **gradient checkpointing** (recompute, don't store) to lift MERA's training ceiling past ~16384.
2. **Quantized type usage** — Q8 int8 / f16 weight+activation reps in the diffusion engine, reusing the
   Ch27 findings ([[ch27-specialized-build-plan]]: Q8 matmul 1.4–2.5×, f16 near-lossless;
   [[ch27-optimization-shortlist]]). Memory is the binding constraint at long context, so quantization
   directly extends the usable-N ceiling for every variant.

Goal: a final engine that **picks the right denoiser + precision for the target context length and the
hardware's VRAM** — performance/memory alternatives in the toolbox. Builds on
[[gist-is-compute-context-primitive]], [[cuda-first-class-iteration-time]],
[[optimization-target-is-hardware-ceiling]].
