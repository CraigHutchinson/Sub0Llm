# Storage A/B drivers

How the owned expert cache (`--moe-io-mode cache`) is measured against reactive mmap. Results and their
reading are in [`docs/STORAGE_STACK_PLAN.md`](../../docs/STORAGE_STACK_PLAN.md). Windows only (the ballast,
`typeperf` and the eviction trick are Windows mechanisms).

Measure in this order; each step is cheaper than the next and has caught a mistake the next would have hidden.

| Step | Tool | Needs | Time |
|---|---|---|---|
| 1. One read, every strategy | Sub0MemPage `tools/read_shootout` (`-DSUB0MEMPAGE_BUILD_DIAGNOSTICS=ON`) | nothing: it generates a 256 MiB file | seconds |
| 2. The cache without the engine | `replay_ab.py` + Sub0TieredCache `sub0tieredcache-trace-replay` (`-DSUB0TIEREDCACHE_BUILD_BENCHMARKS=ON`) | the sidecar and a recorded trace | ~10 min |
| 3. The engine | `deep_regime.py` | the model, sidecar, tokenizer, a configured `out/build/s1b` | ~35 min |

`cold_fraction.py` answers a separate question: how much of the cache's fill traffic really comes from disk.
`trace_predictability.py` needs only a recorded trace: it reports the working set, the hit rate each budget
buys, and whether recent tokens would have predicted the misses
([`O13_storage_convergence.md`](../../docs/optimization/opportunities/O13_storage_convergence.md)).

## Setup

```
clang++ -O2 scripts/storage_ab/ballast.cpp -o ballast.exe
```

`ballast.exe <gib>` holds that much RAM locked until it is killed, so the OS cannot use it for its file
cache: a 64 GB host then behaves like a smaller one (deep regime 2). The drivers size it from the memory
that is free when they start, so close other large programs first.

A trace for step 2 comes from the engine: `sub0llm-qwen4-gen ... --expert-trace PREFIX` writes
`PREFIX.extents` and `PREFIX.trace` (formats: Sub0TieredCache `docs/trace-replay.md`).

## Rules these scripts encode

- **Rotate arms and evict before every sample.** Thermal drift and a warm OS cache both move the result
  more than the effects being measured.
- **Compare arms within one run only.** Absolute tok/s moves with how much memory is free on the day
  (2026-10-01: reactive 6.49 with a 17 GiB ballast; 2026-10-06: 7.14 with 10 GiB).
- **Perplexity must be identical across arms.** It is the parity check: the cache changes where bytes come
  from, never what is computed.
- **Never read the sidecar through the OS cache around an uncached run.** On Windows a cached reader of a
  file makes non-cached reads of it queue behind each other for seconds afterwards (Sub0MemPage
  `docs/investigations/unbuffered-read-ceiling.md`). `replay_ab.py` waits 15 s after evicting for that
  reason.
- **Build the configurator before staging arms.** A stale one does not know newer flags and silently
  stages the wrong build; `deep_regime.py` builds it first, and prints each arm's generated-config hash.

## Status

These are the session scratch drivers behind the 2026-10-01 and 2026-10-06 results, tidied and given
arguments. Their imports and argument parsing are checked; the tidied versions have not themselves been
run end to end.
