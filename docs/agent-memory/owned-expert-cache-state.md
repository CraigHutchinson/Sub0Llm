---
name: owned-expert-cache-state
description: "Owned expert cache (--moe-io-mode cache) status 2026-10-06 — uncached fills are the default in cache mode and put it ~10% ahead of reactive mmap in deep regime 2; the levers, the Windows cached-reader trap, and what is next"
metadata:
  node_type: memory
  type: project
  originSessionId: 4ad183c8-10cc-4e45-8b13-dd85f39fc3f3
  modified: 2026-10-06T23:27:39.771Z
---

As of 2026-10-07 everything is published (Sub0Llm `c13d0c0`; pins TieredCache `1fa4700`, MemPage
`f74b65e`; both libraries' CI green on every job). The owned
routed-expert cache fills its pinned, exact-size slots with non-cached reads. Deep regime 2 (locked RAM
ballast leaving ~10 GiB of room, 17 GiB cache, 2,000 G-PPL tokens, 2 rotated rounds): cache uncached
7.85 tok/s, cache buffered 7.60, reactive mmap 7.14; perplexity identical. Blocking waits 274 -> 189 us.
Absolute tok/s moves with free memory on the day, so compare arms within one run only.

**Why:** the levers, in the order they paid: resident-first waves, the shared expert run first, chunked
256 KiB fills, a matched budget, exact-size slots (`SizeClassedTable`, +9% resident experts), then
uncached fills (`--moe-cache-io`, default uncached in cache mode). In deep regime 2 every fill is a
real disk read, so buffered meant disk -> OS cache -> copy. Pinning prevents a regression but adds no
speed.

**How to apply:**
- READ FIRST, the Windows trap: non-cached reads of a file queue behind each other while that file has,
  or recently (seconds, more than two) had, a cached reader or mapping. An uncached source needs a
  direct-only lifetime: read its header and metadata non-cached too (`sub0/uncached_file.hpp`,
  `Store::Payload::descriptors_uncached`). A benchmark that probes or checksums through the cache
  between uncached runs creates the slowdown it measures.
- Measure first in the sandboxes, then the engine: Sub0MemPage `tools/read_shootout` (12+ strategies,
  seconds; `--cold --direct-session` for uncached arms; `diskspd_reference.ps1` for the drive's bar),
  TieredCache trace-replay (`--size-classed --uncached --compute-us 2800`, under a ballast), then
  scratchpad `deep_regime3.py SP 10 17`.
- Open: whether another process's cached read of the sidecar slows fills mid-run; an IOCP issue path in
  MemPage (~10% on a lone miss); next-layer prefetch (later, per the user).
- Then S1c n-gram integration on the same substrate ([[storage-core-use-case-ngram-regime2]]).
- `--moe-io-mode` still defaults to `reactive` on purpose. Promoting `cache` needs a budget taken from
  available memory at startup (it is a configure-time constant, half of physical RAM), a working
  default under Linux's 64 MiB lock limit, a regime-1 A/B, and the full suite with the cache on. The
  checklist and the other follow-ups are in Sub0Llm `docs/STORAGE_STACK_PLAN.md` "Follow-ups".
- Measurement drivers live in Sub0Llm `scripts/storage_ab/` (README gives the order and the rules).
