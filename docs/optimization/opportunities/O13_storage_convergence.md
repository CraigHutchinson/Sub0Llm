# O13 -- where the storage/cache work and the decode optimization work converge

Status: **analysis, 2026-10-07.** No engine change yet. Evidence is an offline analysis of the recorded
routed-expert trace plus the two tracks' own records (`O10_subphase_roofline.md`,
`../../STORAGE_STACK_PLAN.md`). Section 4 ranks what to do.

## 1. Why the two tracks are one problem

The decode track (O1-O12) raises tok/s when everything fits in RAM (regime 1). The storage track (S1b,
the owned expert cache) raises tok/s when it does not (regime 2). They meet in three places:

- **The same phase.** Routed experts are ~28 ms of a ~107 ms warm token, the largest phase after GDN,
  and the only one whose bytes the cache owns.
- **The same memory.** In regime 2, tok/s is a function of the expert hit rate, and the hit rate is a
  function of how many GiB the engine does not hold for anything else.
- **The same predictor.** Prefetching experts and speculative decoding both need to know what a layer
  will want before it asks.

## 1a. Operating assumption: this host is RAM-constrained (user direction, 2026-10-07)

Self-managed memory is the long-term path here, ahead of the n-gram integration and not only because of
it. The model's files are 49.5 GiB (9.16 + 3.26 + 37.11) on a 63 GiB host that already commits about
20 GiB before the model loads, so under any real load the expert sidecar cannot sit in the OS cache.
The owned cache already wins there: 7.85 tok/s against reactive mmap's 7.14 at a 17 GiB budget, with
identical perplexity (storage plan, 2026-10-06). Consequences for the decode track:

- **Regime 2 with the owned cache is the reference configuration**, not a special case. An optimization
  is measured there as well as on an idle host.
- **Every decode optimization to date (O1-O12) was measured on reactive mmap only.** The optimization
  build dir (`out/build/wp5c_full48`) is not even configured with `SUB0_STORAGE_TIEREDCACHE`; only
  `out/build/s1b` is.
- **Part of the decode track's run-to-run noise is probably reactive mmap itself.** Reactive's speed
  follows how much RAM the OS spares that day (4.93 to 7.14 tok/s across sessions in the storage plan).
  The decode track has recorded an unexplained "slow host state" and day-to-day drift in long runs
  (5.73 against 6.98 tok/s for the same build on 2026-09-29). A 2,000-token run touches 32.3 GiB of
  experts (section 2), more than an idle host has free, so long runs are partly I/O-bound even when the
  host looks idle. That link is an inference from those numbers and has not been tested. A fixed, owned
  budget would remove the variable from A/Bs.
- **The same principle extends past the experts.** Memory decode needs every token should be owned and
  pinned; memory it rarely needs should be droppable for free. Today the engine does the reverse in two
  places: the cold 9 GiB parameter arena is anonymous heap (C1), and the 3.26 GiB `.bbq`, read almost in
  full every token, is a file mapping the OS may trim (C2).

## 2. Evidence: the expert access trace

`out/build/s1b/qwen4-g2000.{trace,extents}` (written by `sub0llm-qwen4-gen --expert-trace`, 2026-10-01):
2,000 G-PPL tokens x 48 layers x 10 experts = 960,000 accesses over 24,576 rows of 1.49-1.68 MiB.
Reproduce with `python scripts/storage_ab/trace_predictability.py out/build/s1b/qwen4-g2000` (pure Python,
a minute or two).

**The working set is wide.** 21,391 of 24,576 experts (32.3 of 37.1 GiB) are touched in 2,000 tokens.

**Hit rate against budget** (LRU over exact-size rows, steady state = second half of the trace):

| Cache budget | Hit rate | Misses per token | Layers with at least one miss |
|---:|---:|---:|---:|
| 10 GiB | 86.3% | 66.0 | 66% |
| 14 GiB | 92.3% | 36.8 | 46% |
| 17 GiB | 95.2% | 23.1 | 33% |
| 20 GiB | 96.7% | 15.9 | 24% |
| 23 GiB | 97.7% | 11.2 | 18% |
| 26 GiB | 98.5% | 7.4 | 12% |
| 30 GiB | 99.2% | 4.0 | 7% |

The 17 GiB row matches the engine's measured 95.3% hit rate, so the replay is a usable price list: each
GiB between 17 and 26 removes about 1.7 misses per token.

**Recency does not predict misses.** Predicting a layer's experts before its router runs:

| Predictor | Recall of all 10 experts | Recall of the experts that MISS (10 / 17 / 26 GiB) |
|---|---:|---:|
| same layer, previous token | 36.4% | 0.0% / 0.0% / 0.0% |
| same layer, previous 4 tokens (27 experts) | 54.9% | 0.0% / 0.0% / 0.0% |
| same layer, previous 8 tokens (43 experts) | 64.2% | 0.0% / 0.0% / 0.0% |
| layer's 40 most popular experts | 39.3% | not computed |

Anything used in the last eight tokens is still resident at every budget measured, so a recency predictor
can only name experts that would have hit anyway. Next-layer prefetch (storage plan, lever 2) therefore
pays nothing unless its predictor is content-based.

## 3. The convergence points

### C1. One memory budget: stop holding the cold parameter arena

- **What is held.** `load_model` reads the whole 9.16 GiB bf16 file into a heap arena
  (`g_param_data`). With the native backbone, decode reads 0.18 GiB of it (2.0%; storage plan,
  "Residency intent and guards"). The other 9 GiB is anonymous memory: under pressure the OS must write it
  to the pagefile before it can reuse the RAM, and until then it competes with expert bytes.
- **What it is worth.** 9 GiB moves a 17 GiB cache to 26 GiB at the same footprint: misses per token
  23.1 -> 7.4, layers that wait 33% -> 12%. A blocking wait is ~190-520 us and about ten of a token's 48
  layers stop waiting, so this is roughly 2-5 ms per token in deep regime 2. A smaller host gains more
  (10 -> 19 GiB removes about 48 misses per token).
  Reactive mmap gains the same 9 GiB of standby.
- **How.** The file is `Header` + the arena verbatim + trailers, so when the file's dtype equals the
  build's, the arena can be a read-only `FileMap` view of the file. Mapped pages are clean: the OS drops
  them for free and decode re-faults the 2% it uses. Load time falls as well (no 9 GiB read). Training
  and any f32-master path keep the heap arena.
- **Open before building.** Arena base alignment (map base + `sizeof(Header)`), every writer to the arena
  in an inference build, and the batched `forward()` path, which reads all of it.
- **Wider audit.** A cache run peaks ~17.5 GiB above its budget; arena + `.bbq` explain 12.4 GiB. The
  remaining ~5 GiB (activation arenas, per-thread scratch sized for training) has not been itemised.

### C2. The cache owns the expert bytes' layout: page size and TLB reach

- Routed experts read about 720 MiB per token from ~480 randomly placed 1.5 MiB rows. At 4 KiB pages that
  is ~184,000 distinct pages per token, and the phase achieves about 25 GB/s, the furthest below the
  ~79 GB/s roof of any phase (O10 has the byte count as unverified).
- Reactive mmap cannot change this: file-backed pages are 4 KiB. The owned pool is `VirtualAlloc` +
  `VirtualLock` memory, so it can be 2 MiB pages (`MEM_LARGE_PAGES`; needs the "Lock pages in memory"
  privilege, which the pool's lock quota already goes most of the way to requiring).
- This is the one route by which the cache could beat reactive when everything fits in RAM, where it
  otherwise only adds misses (storage follow-up 3).
- The same applies to the 3.26 GiB `.bbq`, which is mapped and always hot. The residency table in the
  storage plan gives it no guard ("none needed"): it stays resident by use, which nothing checks under
  load or across an idle gap between prompts. An owned, pinned copy gives it the expert pool's guarantee. O10 lever 4 (K-quant GEMVs
  plateau at ~53 GB/s) lists TLB as a suspect; a pinned large-page copy made through `residency.hpp` at
  load tests it directly.
- **First step is a microbenchmark, not an engine change:** the existing expert and backbone kernel
  benches over 4 KiB and 2 MiB allocations of the same real bytes. Unknown size; could be nothing.

### C3. Prefetch needs the optimization track's predictor

- A layer takes ~2.2 ms and a miss costs 0.19-0.52 ms, so one layer of lead hides a miss completely.
- Section 2 rules out recency. Candidates, both from `../../SPECULATION_NGRAM_MOE_DESIGN.md`:
  1. **Early router.** Run layer L+1's router on layer L's input residual while layer L computes, and
     prefetch its top-k. The residual stream changes slowly between layers; how well this predicts is
     unmeasured.
  2. **An n-gram prior**: context hash -> likely experts per layer, built from route logs.
- `--expert-trace` is the route log that design's experiment E1 asked for, so one artifact serves both
  tracks. Extending it with the early router's top-20 per layer lets recall-on-misses be measured
  offline, in the trace replay, before any prefetch code exists.

### C4. The n-gram table is both the cache's core consumer and the missing quality term

- Both G-PPL arms omit the 102 GB n-gram (PLE) table today, so its effect on quality, and the true gap
  to llama.cpp, are unmeasured. S1c is the storage track's next slice and delivers exactly those rows.
- A token's 16 rows are known the moment the token is chosen and are needed at decoder layer 1, about
  2 ms later. Sixteen parallel uncached reads (p50 ~0.5 ms each) fit in that window if issued at token
  start. Speculative decoding (O6) can issue them earlier still for drafted tokens.
- Its row cache must come out of the same budget as C1.

### C5. One harness, one phase table

- `run_perf_suite.py --stage ppl --ppl-speed-rounds N` (added with this doc) and
  `scripts/storage_ab/deep_regime.py` now follow the same protocol: arms built once, rotated, long runs.
  A ballast option on the suite would let one command measure any optimization in both regimes.
- The phase profiler has no row for time blocked on an expert miss. Until it does, a regime-2 profile
  cannot separate stall from compute, and the O10 roofline cannot be redone under memory pressure.
- `resolve_decode_defaults` is where `--moe-io-mode` would be promoted. The budget it needs is the free
  memory at load, a runtime quantity, so it cannot follow the configure-time pattern unchanged.

### C6. Cores: readers on E-cores

Decode keeps eight P-core workers spinning between regions (`DECODE_OMP_SPIN`). The cache's reader
threads are unpinned; sixteen E-cores are idle. Pinning readers to E-cores keeps a fill from displacing
a spinning worker. Untested.

## 4. Ranked next steps

Ordered for a RAM-constrained host (section 1a): first make the owned cache the measured default, then
give it more memory, then make its misses rarer.

| # | Step | Cost | Evidence it rests on |
|---|---|---|---|
| 1 | Build the optimization dir with the cache; give `run_perf_suite.py` a ballast option, so every A/B runs in both regimes | small | O1-O12 were measured on reactive only |
| 2 | C5: the miss-wait phase row, and the phase table in the long-run tool | small | a regime-2 profile cannot separate stall from compute today |
| 3 | Promote `--moe-io-mode cache` to the auto default: the storage plan's checklist (budget from free memory at load, regime-1 A/B, full suite with the cache on, Linux fallback) | medium | cache 7.85 against reactive 7.14 tok/s in deep regime 2 |
| 4 | C1: map the bf16 arena for inference builds | medium; shared files | 2% of the arena is read; 9 GiB = -68% misses at 17 GiB |
| 5 | C2: own and pin the `.bbq`; 4 KiB against 2 MiB pages for it and for the expert pool | small to measure | experts at ~25 GB/s against a 79 GB/s roof; `.bbq` residency under load is unguarded |
| 6 | C3: log the early router in the trace; offline recall on misses | small | recency recall on misses is 0.0% |
| 7 | C4 / S1c: n-gram rows, issued at token start | large | the storage track's next slice |
| 8 | C6: reader affinity | small | none yet |

## 5. What does not converge

- **Recency-based prefetch**: 0.0% recall on misses (section 2).
- **Pinning for speed**: the pinned pool prevents a regression but measured no gain (storage plan).
- **The cache on a host with RAM to spare**: reactive has no misses when the whole sidecar fits in the OS
  cache, so the cache can only match it there unless C2 pays. Section 1a is why that case is not this
  host's normal one.
