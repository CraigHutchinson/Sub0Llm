---
name: out-of-core-corpus-pipeline
description: Out-of-core corpus path for FineWeb-scale data — training mmaps corpus.tok (TokMap); configurator streams in newline-aligned chunks; both validated
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Goal: train on a corpus that exceeds system RAM (target = FineWeb-Edu; see [[ch29-model-quality-founded-dims]] for why bigger model+data is the lever — the L5 model trains with train≈val so it's still capacity/data-limited, not overfitting).

Two halves, both done:
- **Training (committed 4c964fa, mine):** `include/sub0/tokmap.hpp` = cross-platform read-only `mmap` over `corpus.tok` (S0TK: u32 magic/vocab/ntok, then int32 ids at offset 12). Training/eval consumers take `std::span<const int>` instead of `std::vector<int>&`; the OS pages the stream on demand, so it need never be fully resident and there's no giant heap vector (fits the no-heap rule better than the old vector). `train_stage.cpp` loads via `sub0::TokMap`.
- **Configurator (`tools/configurator.cpp`):** rewritten to STREAM the corpus in newline-aligned chunks (`for_each_chunk`, CHUNK_BYTES=32MB) across 3 passes — (1) name-detection stats, (2) build the bounded unique-word table + base alphabet, (3) emit corpus.tok incrementally with ntok back-patched. Only bounded tables stay resident (unique words saturate by Heaps' law). Newlines are safe split points (never split a glyph/word/the truecaser look-back, which already treats newline as a line start), so it is **byte-identical** to whole-corpus processing.

**Validation method (reuse this):** capture SHA256 of corpus.tok + tokenizer.bin from the old build, run the new configurator into a scratch dir, compare. Tested byte-identical at 32MB (1 chunk) AND 256KB (~85 chunks) → boundary carry-over correct. Base-id order (bytes 0..255, then cap, then up) is preserved so BPE tie-breaking is unchanged.

- **On-demand tokenization (committed 184c766):** corpus.tok is now OPTIONAL — `sub0-configure --corpus-tok 0` skips Pass 3 (corpus.tok ~2x text on disk, so keeping both >2x; this avoids it). Training auto-falls-back when corpus.tok is absent: a bounded in-memory SHUFFLE BUFFER (`TextCorpus`, OD_TRAIN_BUF_TOK≈8M) filled by tokenizing random contiguous regions of the raw text via `sub0::encode` (reproduces configurator truecasing+BPE), refilled each eval interval (separate RNG, never perturbs the resume-critical window RNG) so the run traverses the whole corpus; a fixed val buffer (OD_VAL_BUF_TOK≈512K) keeps eval comparable. Train/eval loop is path-agnostic via a `train_span`/`val_span` pair (mmap subspans OR the buffers). Force on-demand for testing by passing a non-existent `.tok` arg. Validated: both paths descend identically (train 4.0→3.1 / 120 steps).

- **CMake wiring (992218b + e83fba1):** `SUB0_CORPUS_TOK` is tri-state ON/OFF/**AUTO** (default AUTO) → configurator `--corpus-tok 0|1`. AUTO decides by SCALE (not a fixed threshold): emit corpus.tok while the ~2x-text token copy fits in half of physical RAM (`cmake_host_system_information` TOTAL_PHYSICAL_MEMORY + `file(SIZE)`), else on-demand — so a small corpus pre-tokenizes and a >~16GB-text corpus streams automatically (host ~64GB). OFF also removes any stale corpus.tok so training reliably goes on-demand; header + tokenizer.bin emitted either way. So the full out-of-core path (mmap / streaming configurator / on-demand shuffle buffer / AUTO CMake toggle) is DONE.

**Still pending:** (a) FineWeb-Edu not downloaded yet (hundreds of GB — user's call), so end-to-end out-of-core is unproven on real >RAM data; passes 1-2 still do 2 streaming reads of the whole corpus on the first configure (unavoidable: VOCAB is derived), only Pass 3 is now skippable. (b) the OOB validation scan in the mmap train path faults in the whole mapping once (one-time, acceptable). (c) capacity push: L5 model still trains with train≈val (not data-limited on the 22MB corpus yet), so more layers/width likely still gains — user trains these interactively.
