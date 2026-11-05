---
name: tokenizer-redesign-implementation-complete
description: "The full tokenizer redesign plan (Stage 1 WS1-4, Stage 2 WS5/WS5b, orthogonal WS6/WS7) is implemented, tested (162 tests green), and documented in docs/TOKENIZER_REVIEW.md §5.1-5.10 -- still uncommitted as of 2026-07-10"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Plan: `C:\Users\craig\.claude\plans\smooth-noodling-kurzweil.md`. Fully executed across this and the
prior session. All work is on `include/sub0/casing.hpp`, `include/sub0/tokenizer.hpp`,
`src/tokenizer.cpp`, plus new `include/sub0/tok_interchange.hpp` / `src/tok_interchange.cpp` /
`tools/sub0llm-tokenizer.cpp`. Verified on an isolated CPU-only build dir (`out/build/tok_cpu`),
never touching the GPU training build. **Not committed yet** — see [[boy-scout-rule-always-resolve-inconsistencies]]
for why docs were kept in sync as this landed, not left to rot.

**What shipped:**
- Stage 1 (WS1-4): `TokenId` enum replacing 14 `constexpr int`s, `byte_base`/14 `*_id` fields
  deleted from `Tokenizer` (redundant with the enum), `std::span` for read-only view params, legacy
  BPE split out of the runtime hot path into `BpeAnalysisVocab`/`learn_bpe_analysis` (analysis-only).
- Stage 2 WS5: `TOK_TURN_START`/`TOK_TURN_END` (ChatML literals `<|im_start|>`/`<|im_end|>`, verified
  against Qwen AND Gemma prior art — neither mints per-role token ids), 16 reserved marker slots,
  `kSchemeVersion` constant, magic bump `"S0TE"`→`"S0TF"` (`n_base` 270→288).
- Stage 2 WS5b: 6 bracket-glue markers (`TOK_GLUE_OPAREN` etc.) collapsing the JOIN tax on glued
  brackets (`f(x)`: 7 tokens → 4) — consumed 6 of the 16 reserved slots as a RENAME (no second magic
  bump, only `kSchemeVersion` 1→2).
- WS7: `sub0::tok::to_interchange()` (dense id→text/score/kind export shape) +
  `sub0llm-tokenizer export --format json|tsv`, engine-free tool.
- WS6: `normalize_text` SIMD rewrite shipped (~20% real win). `truecase_tokenize` SIMD was
  investigated and deliberately NOT changed (see below).

**Real bugs this session's own testing discipline caught (not hypothetical value — actually fired):**
1. Directional-quote line-initial-open case: measured on 100MB real TinyStories dialogue (not
   assumed) — 89.07% fire rate, 87.8% of the miss was a quote right after a bare `\n`. Fixed
   (→98.67%); the smaller British-style `"Hello", she said` case (0.94% of quotes) was left as
   documented known behavior, not fixed — not worth the broader trailing-punctuation design it'd need.
2. Bracket-glue's first implementation inverted the open/close `dps` assignment
   (`!is_close_bracket` instead of `is_close_bracket`) — broke round-trips specifically for a bracket
   immediately followed by a DIFFERENT bracket family (`"{code}[index]"`). Caught by the existing
   full-source-dogfood + 8000-case fuzz suite going red, not by the hand-picked worked examples
   (which happened to avoid that exact adjacency). Direct, concrete validation of keeping generative
   fuzz alongside curated pinned examples.

**WS6 methodology lesson, worth remembering for ANY future perf work on this machine:**
`sub0_frontend` (casing.hpp/tokenizer.cpp's actual compile target) never gets an OpenMP flag —
only `sub0_core` does (`cmake/OpenMP.cmake`). Every `#pragma omp simd` in casing.hpp was silently
INERT until `-fopenmp-simd` was added to `sub0_frontend` specifically in `CMakeLists.txt`. Separately,
and more generally: a first (wrong) sequential-process before/after measurement showed
`truecase_tokenize`'s SIMD rewrite as a ~13% REGRESSION; re-measured INTERLEAVED (same process,
back-to-back calls, alternating old/new) it was actually within ±3% (noise). A d512 GPU training run
was active in the background throughout, adding load variance sequential runs are vulnerable to. This
directly reinforces [[thermal-confounds-ab-wallclock-testing]] — don't trust a single sequential
before/after number on this machine, interleave same-process instead. Ended up reverting
`truecase_tokenize` to the exact original scalar form (kept as a documented non-result in its own
comment, not silently dropped) since the dominant cost there (per-word case classification: a
lowercase copy + an `attested` hash lookup) is inherently not a SIMD target anyway.

**Not done, explicitly future work per docs/WORKFLOW_ARCHITECTURE.md's own staged sequencing**:
`sub0llm-tokenizer` absorbing `vocab`/`encode`/`decode`/`roundtrip` off the engine (WS7 only built
`export`, on purpose — narrowest scoped item).

See [[tokenizer-special-token-extension-review]] for the pre-implementation design research this
built on, and `docs/TOKENIZER_REVIEW.md` §5.1-5.10 for the full session-by-session record (§5.2 is
marked superseded in place rather than rewritten, so the design's reasoning trajectory — including
the caught 4-per-role-marker contradiction — stays legible).
