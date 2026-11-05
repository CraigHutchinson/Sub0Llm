---
name: self-managed-memory-is-the-direction
description: User direction 2026-10-07 — this host is RAM-constrained, so self-managed (owned, pinned) memory over mmap is the long-term path now, ahead of the n-gram integration; measure decode optimizations with the owned cache under memory pressure too
metadata:
  type: feedback
---
The owned expert cache (Sub0MemPage/TieredCache) already pays before S1c: 7.85 tok/s vs reactive mmap 7.14
in deep regime 2 (2026-10-06). The user: "this system is RAM constrained so under load we shall see more
benefit longer term with a self-managed memory over mmap". Do not frame the cache as a regime-2 special case
or reactive mmap as the normal path.

**Why:** model files are 49.5 GiB on a 63 GiB host with ~20 GiB already committed; reactive's speed follows
whatever RAM the OS spares that day, which is also a likely source of the decode track's long-run noise.

**How to apply:** treat regime 2 + owned cache as the reference configuration. Measure every decode
optimization there as well as idle: `run_perf_suite.py --stage ppl --ballast-room 10 --arm "reactive:" --arm
"cache17:--moe-io-mode cache --moe-cache-gib 17"` (any build dir builds the cache when an arm asks for it;
O1-O12 were only ever measured on reactive). Prefer owned+pinned for memory read every token and
free-to-drop mappings for cold memory. Ranked shared work: Sub0Llm
`docs/optimization/opportunities/O13_storage_convergence.md`. See [[owned-expert-cache-state]],
[[storage-core-use-case-ngram-regime2]], [[defaults-are-the-best-options]], [[memory-architecture-before-mechanisms]].
