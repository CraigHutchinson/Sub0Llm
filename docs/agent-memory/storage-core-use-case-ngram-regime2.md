---
name: storage-core-use-case-ngram-regime2
description: MemPage/TieredCache's core use case is the 102 GB n-gram table (never fits in RAM); MoE experts are the second consumer. Design owned-cache features for n-gram rows first.
metadata:
  type: project
---

User direction 2026-09-29: the storage stack has two regimes. (1) Everything fits in RAM: OS page
cache + reactive mmap is right and an owned cache isn't needed. (2) Working set exceeds RAM: MemPage
owns the cache (bypassing OS paging) and manages it for the workload. MemPage targets regime 2.

**The core consumer is the real n-gram embedding table (E1)**, a key design element of the
TieredCache architecture: 102.4 GB bf16 vs 63 GB RAM, 16 hash-scattered 320-byte rows per token. It is
regime 2 by construction and the worst case for OS paging (4 KB fault per 320 B row). Routed-MoE
experts (E2, S1b) are the SECOND consumer of the same substrate.

**Why:** the user noted this was not well documented. S1b's MoE work drifted toward optimizing the
regime-1 warm case (a 6-token rerun of a cached prompt), which is not what MemPage is for.

**How to apply:** design owned-cache features (row/extent-keyed slots, unbuffered aligned fills,
access-aware admission/prefetch) for n-gram rows first, then reuse for MoE; never a MoE-only special
case. Benchmark them in regime 2 (long decodes / tables larger than RAM). Priority after S1b is S1c
(real n-gram table wiring). Recorded in `docs/STORAGE_STACK_PLAN.md` "Core use case". See
[[sub0mempage-project-init]], [[memory-architecture-before-mechanisms]].
