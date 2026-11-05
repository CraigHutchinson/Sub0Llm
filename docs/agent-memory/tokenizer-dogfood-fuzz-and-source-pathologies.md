---
name: tokenizer-dogfood-fuzz-and-source-pathologies
description: dogfood+fuzz tok test framework (tests/tok_roundtrip_fuzz.cpp) is GREEN (0 RT fails / 8000 fuzz + full source tree); measured source-code tokenization pathologies rank the next quick wins
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Built the dogfood+fuzz tokenizer test framework the user prioritized: `tests/tok_roundtrip_fuzz.cpp` (links only `sub0_tok`; `SUB0_SOURCE_DIR` baked via CMake). Layers: worked examples (one per construct), **dogfood** (learn a JOIN tokenizer on the project's OWN C++ source, round-trip every file), **fuzz** (4000 mutated seeds + 4000 random byte blobs, fixed-seed reproducible, expected-vs-actual diff on fail), plus non-failing **pattern census** + **symbolic token-breakdown renderer** (`describe_tokens`). Contract: `detokenize(encode(x)) == normalize_text(x)`.

**Result: 0 round-trip failures** across 8000 fuzz cases + the whole source tree → the JOIN decode FSM is correct after the [[join-tokenizer-validated-result]] NASA's fix. So the framework's value is not bug-finding but **measuring tokenization pathologies that rank the next quick wins**.

Measured on the project's own C++ source (~537KB): **bytes/token 1.49** (worse than char-level in places; prose was ~2.9), **JOIN = 28.5% of all tokens** (glued code punctuation), **64% of lines indented** (a `\n`+indent can't be NEWLINE → verbatim `\n`+space bytes; ~60k space-byte tokens, one per space, never BPE-merged). Breakdowns: `std::vector<int>` = 24 tok, `ThisIsMyAwesomeFunction` = 20 tok (≈1/letter), `save_scan_state` = 9 tok (each `_` = 2 JOINs), `sunday` = s+u+n+day (NOT sun+day). **CamelCase shatters** because interior capitals are distinct bytes (`C`≠`c`) so lowercase merges never apply mid-word.

**Ranked quick wins (data-justified):** (1) interior-CAP CamelCase split — emit `Non <CAP> commercial`, reuses existing CAP marker, lets lowercase merges apply, round-trips free; (2) snake_case/hyphen as one bound-connector token (not 2 JOINs) — the [[tokenizer-normalization-and-factored-tokens]] separable-affix idea; (3) indentation/`SPACE_N` run-length (the SPACE_N/TAB item in [[join-tokenizer-strategy]], now justified for code); (4) vocab/word-boundary BPE (sun+day). Start with #1 (highest impact, reuses markers). Still pending: implement the wins, then the tokenizer code-quality review (refactor for optimal op, flag SIMD/CUDA TODOs, remove legacy path). Framework NOT yet committed (keep backend_cuda.cu WIP out).
