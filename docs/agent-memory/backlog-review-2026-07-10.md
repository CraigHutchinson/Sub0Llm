---
name: backlog-review-2026-07-10
description: "Consolidated snapshot of Sub0Llm's open backlog/TODOs as reviewed 2026-07-10 (code TODOs via grep + project-level memory), saved for follow-up per explicit user request. Index only -- each item's real detail lives in its own memory or in the code comment cited."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User asked to save this list for follow-up after a full backlog review (code TODOs via grep across
`include/`, `src/`, `tools/`, `tests/`, `CMakeLists.txt` + project memory, cross-checked against
`docs/TOKENIZER_REVIEW.md`'s own priority list). Snapshot, not live state -- re-grep/re-check before
trusting any specific claim below as still current.

## Code TODOs (persisted in the code itself, not duplicated here -- grep to see current text)
- `include/sub0/muon.hpp`: hot-loop allocation (violates the project's own no-alloc-in-hot-loop
  rule), 2x unnecessary-transpose-route notes, division-vs-reciprocal-multiply perf note.
- `src/backend_cuda.cu`: ternary-gpu (BitNet quant CPU-only for now), cublasLt fused bias+GELU
  epilogue, 2x backward-scratch-size-should-be-precomputed, BF16 activation storage (~half scratch
  at sm>=80), saved-activation checkpointing at large batch, SUB0_CUDA_CHECK early-return VRAM
  leak (robustness), per-AdamW-step QKV network rebuild cost.
- `src/train_stage.cpp`: remove getenv-based config (SUB0_TRAIN_CPU/SUB0_FIXED_SEQ) -> CLI/compile-
  time, `tune-train` (wire an auto-tuner for training hyperparams), `dynamic-training-mode`
  (extend SUB0_TUNING into a dynamic-training build mode).
- `include/sub0/core.hpp`: `use_muon` is runtime but arguably wants to be compile-time -- directly
  adjacent to [[central-run-config-json]]'s fix, worth revisiting together.
- `CMakeLists.txt`: CLI migration to `clap` blocked on Clang C++26 reflection (not actionable yet);
  `tune` escalation to `biteopt` only if search space grows (explicitly deferred).

## Tokenizer doc backlog (`docs/TOKENIZER_REVIEW.md`'s own priority list, still-open items only --
## confirmed §1 legacy-path-removal and the markers-struct item are DONE; §3's old
## `bpe_encode_word` memoization item is now MOOT, WS2 deleted that code path entirely)
- CUDA tokenizer (one block per chunk) -- "large, later" per the doc's own priority ordering.
- FSM declarative dispatch table (§2) -- deliberately deferred a second time during WS5/WS5b this
  session (evaluated and judged the branch bodies too heterogeneous to clearly benefit); see
  [[tokenizer-redesign-implementation-complete]].
- `sub0llm-tokenizer` absorbing `vocab`/`encode`/`decode`/`roundtrip` off the engine -- WS7 only
  built `export`, on purpose, per `docs/WORKFLOW_ARCHITECTURE.md`'s own staged sequencing.

## Project-level backlog (each has its own dedicated memory -- this is just the index)
- **GPU-fault auto-retry resilience** -- user's own ask from the d512 incident, still open. See
  [[d512-fault-resume-optimizer-switch-bug]] (backlog item #1 there; item #2, the optimizer-persist
  bug, is now FIXED -- see [[central-run-config-json]]).
- **`memplan` VRAM-prediction gap at d768 scale** -- NEW 2026-07-10 finding, not yet investigated.
  See [[memplan-vram-prediction-gap-at-scale]].
- **`tune --backend gpu` vs. production throughput discrepancy** -- ~half real throughput measured
  at the same shape, root cause open. See [[gpu-tune-benchmark-vs-production-discrepancy]].
- ~~**`engine_tests` gradient-check fails at production dims** (d448/L11/H7) but passes at tiny dims~~
  **OBSOLETE 2026-07-27**: re-measured, d448 PASSES (full suite too); the claim was inverted and the
  real cause was a toy vocabulary, not dims. See [[engine-tests-gradient-check-dims-dependent]].
- **GGUF import** -- reader/SwiGLU-FFN/tensor+dequant done (steps 1-3a); actual weight transplant
  (3b), vocab + e2e parity (4-5), and a CUDA kernel still open. See
  [[gguf-import-feasibility-review]].
- **Scratch tokens (chars/fragments -> word, context/KV-cache efficiency) + spelling-via-training
  (word -> chars, reasoning-harness spike)** -- confirmed 2026-07-10 to work IN TANDEM (spelling is
  plausibly a prerequisite for scratch tokens, not just thematically related). BOTH memories
  substantially reworked this session with the tandem relationship + concrete efficiency framing
  ("aabbcc" collapsing 6 tokens of context/KV-cache to 1). See
  [[scratch-tokens-context-translation-layer]] and [[spelling-via-injected-char-word-training]] --
  read both together, not separately, when picking this up.
- **Attention register-pressure redesign** -- deferred, folded into the standing
  [[hardware-efficiency-iteration-standing-task]].

## Memory-hygiene finding (aside, not part of the backlog itself)
Three memories indexed under "Sub0Llm (current project)" in MEMORY.md turned out, on
session-lineage inspection, to actually belong to the separate diffusion/chapter-education track:
`simdjson-migration-backlog`, `tokenizer-normalization-and-factored-tokens`,
`heterogeneous-worker-split-followup` (all `originSessionId: f6f73404-...`, the diffusion track's
session, vs Sub0Llm's own `2a6b481b-...` lineage; content confirms it too -- e.g. file paths under
`diffusion/src/`, `ch29`/`ch30` chapter references). Not fixed yet (flagged to the user, not asked
to fix) -- if picked up later, move their MEMORY.md index lines to the diffusion section rather
than deleting them, they're still valid memories, just mis-filed.
