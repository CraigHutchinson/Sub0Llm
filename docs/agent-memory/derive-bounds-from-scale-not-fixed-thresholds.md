---
name: derive-bounds-from-scale-not-fixed-thresholds
description: "Caps/limits/thresholds must be derived from the known scale (D/L/B/T), never fixed magic numbers that silently regress when the scale changes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

User principle (2026-06-17): "we must always be vigilant to not have fixed threshold/boundaries where the scale/change is known/relative." A buffer/cache/limit whose right value is a function of the workload's known dimensions must be DERIVED from those dimensions (or self-tuned from observed behaviour), not hard-coded.

**Why:** a fixed constant is correct only at the scale it was tuned for and silently regresses otherwise. Concrete case that prompted this: the TensorPool per-class cache cap was a fixed 16 MB (tuned for D=128/L4); at the FOUNDED default D=256/L6 a step's live buffers exceeded it → 242 free+realloc thrash per per-worker step (8 workers × ~100 step/s ≈ 190K alloc + 190K free/s). My first fix bumped it to a fixed 256 MB — same anti-pattern, would re-thrash at D=512.

**How to apply:** prefer self-scaling mechanisms. For the pool: an adaptive high-water-mark cap (track peak simultaneously-live buffers per size class; cache up to that peak) sizes the cache to the actual working set and auto-scales with D/L/B/T — no magic number, and memory-free because the cap only bounds eviction (cached = min(peak-live, cap), and reclaim only caches an actually-freed buffer). Keep any remaining fixed number only as a generous SAFETY backstop against leaks, never as the operating point. Applies to TensorPool + BlockPool (`src/core/pool.hpp`, `include/sub0llm/core/block_pool.hpp`). Ties into the constexpr-specialized-engine vision where dims become compile-time. See [[ch29-training-throughput-findings]].
