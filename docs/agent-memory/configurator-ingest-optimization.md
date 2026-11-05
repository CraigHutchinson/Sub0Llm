---
name: configurator-ingest-optimization
description: "Configurator ingest at scale — DONE: incremental BPE + scan-state cache + parallel streaming passes (5x); BPE (sequential greedy) is now the dominant ingest cost"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Corpus ingest (the configurator's tokenization) is the cost that grows with corpus size. Built a throughput baseline (committed 436a59c): per-phase timers + "MB/s corpus" in the configurator report. On 22MB tinystories **BPE is 77%** of the 13.8s — the naive merge loop rebuilds the FULL pair-count map every iteration (O(merges × total_symbols)).

**Decisive finding: BPE does NOT scale.** A configure of the 43GB FineWeb-Edu corpus (`--corpus-tok 0`, on-demand) ran **3 hours single-threaded, 3GB RAM, produced nothing** — passes 1-2 are only ~38min, so it was stuck ~2.5hr in BPE over FineWeb's millions of unique letter-run words. KILLED it. So **incremental BPE is the critical blocker** for any large corpus (keep per-pair counts in a heap/index, update only words touched by each merge — the classic linear-ish BPE). The scan cache does NOT help this (BPE runs after the cached scan, every time).

**Scan-state CACHE (committed 2026873):** snapshot the bounded output of passes 1-2 (name-detection COUNTS, unique-word freq table, alphabet usage, stats) to `<corpus>.words` (validated by size+mtime+version, gitignored). A re-run with a different vocab / after a crash / to re-tokenize SKIPS the scan; BPE still runs (vocab-dependent). Keeps the name COUNTS (not just derived `attested`) so the state is MERGEABLE → future multi-corpus / incremental-across-sessions tokenizer. Binary (millions of words; JSON would bloat ~4x). Made tokenizer.bin byte-deterministic (sort `attested`, since the cache load path reshuffles the set order). Validated byte-identical (miss then hit → same corpus.tok as pre-cache reference).

**RESOLVED — incremental BPE + single-thread opts (committed 7fb945c, cc058be, 0877838, 9f04fa1, e77eee0):** incremental BPE (heap + pair->words index) replaced the rebuild-every-merge naive (3hr-stuck → finishes). Then BPE dedup-push (push each changed pair once per merge, not per touch) + position-delta (multiset diff: touch only the pairs a merge changes) → BPE 36→8.7s on a 256MB FineWeb slice (~4.2x; full FineWeb ~55→~13min). Pass2: 1-byte word keys + allocate the word vector only on a cache MISS → 6.7→5s. All byte-identical (validated via SHA on the slice + scan cache to isolate phases). Also fixed a 64-bit overflow: `long` is 32-bit on Windows so counts wrapped (FineWeb reported "-1184634348" alpha words) → TokStats/word_freq/pc/name-counts now `long long`. Validated on the FULL 46GB run: incremental BPE = 55min (vs naive never-finishing); total first-scan ~86min, re-scans cached (skip ~30min scan → ~13min BPE only).

**RESOLVED — streaming passes parallelized (committed 0643d19):** passes 1-2 now run one `std::thread` per newline-aligned segment (`segment_bounds` + `for_each_chunk_range` + `parallel_segments`), thread-local tables merged after join. NO OpenMP (the configurator links none — plain std::thread). Correctness rests on two invariants: a newline never splits a word unit / glyph / truecaser look-back (per-segment == whole-file), AND every output is invariant to word-table ORDER (base alphabet is byte-ordered; BPE picks merges by (count,pair) = order-independent sums + byte-derived ids) — so the merged tables, hence tokenizer.bin/corpus.tok, are byte-identical to the single-thread scan. Validated on 256MB FineWeb (16 segments): tokenizer.bin SHA identical across seq/parallel/cache-reload; pass1 4.0→1.0s, pass2 10.1→1.8s (~5x). Segment count = min(hw_cores, fsz/16MB), so ≥384MB uses all 24 cores. **BPE (greedy, inherently sequential — each merge depends on the last) is now the dominant ingest cost** (~25s/256MB, ~33s/1GB); it stays single-threaded but is already fast post-incremental.

Methodology held: single-thread optimized FIRST as the control (per user + [[single-thread-optimization-and-cycle-bench]]); validated on a head-slice of FineWeb for fast cycle times, isolated from any concurrent training run by driving sub0-configure directly into a scratch output dir.

**DONE — tokenize IO-overlap (reader/writer split), but MEASURED NEGLIGIBLE gain (2026-07-02):**
implemented as planned — a dedicated reader thread streams newline-aligned chunks into batches on a
bounded `BoundedQueue` (cap 2, mutex+condvar, plain std::thread — no sub0pipeline, per user), while the
main thread pops a ready batch, fans it to `nth` encode workers, then writes sequentially (unchanged
logic, just fed from the queue instead of inline `for_each_chunk`). Verified byte-identical corpus.tok
+ tokenizer.bin vs the pre-change binary (git-stash A/B on fineweb_smoke, 1GB).

**But the measured throughput is a wash: 84→85 MB/s, pass3 12.9s→12.6s (within noise).** Root cause:
disk read was NEVER the bottleneck on this hardware. Measured raw sequential read of fineweb_smoke.txt
(1GB): **0.166s = ~6.5 GB/s** (NVMe + OS page cache — 63GB RAM comfortably caches even the 46GB
fineweb_edu corpus after one pass). Against that, Pass 3's throughput ceiling is ~85 MB/s — the
CPU-bound parallel Viterbi unigram ENCODE (already using all 24 cores, from the earlier "8→85 MB/s"
parallelization win in [[configurator-ingest-optimization]]'s own history). Read is ~1-2% of the old
per-batch cycle time, so removing that serialization point saves ~1-2%, not the assumed ~2x. The
earlier "IO-overlap → ~190 MB/s" projection was WRONG — it assumed IO was a comparable-magnitude cost
to encode, which the actual measurement contradicts on this SSD+RAM configuration.

**Kept the change anyway** (correct, harmless, ~80 lines, verified byte-identical) as defensive
infrastructure for a genuinely slower storage path (network mount, spinning disk, a corpus far
exceeding RAM) where it would matter more. **The real remaining lever for Pass 3 throughput is the
Viterbi ENCODE speed itself** (single-word CPU cost), not IO — a different, CPU-bound optimization
target if tokenize-at-scale needs to go faster again.

See [[out-of-core-corpus-pipeline]], [[model-sizing-levers-from-corpus-stats]] (the JOIN/space-fold re-tokenization that would reuse the cache), [[simdjson-migration-backlog]] (portability considered but binary kept for the bulk word table).
