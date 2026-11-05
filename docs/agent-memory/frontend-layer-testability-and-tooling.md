---
name: frontend-layer-testability-and-tooling
description: "Frontend = engine-free pre-model layer (config_util/memplan/registry/casing/tokenizer/unigram); separation's payoff is engine-free testability (memplan+registry now covered, 224 assertions) + an engine-free sub0llm-tokenizer tool for diagnostics/interchange; doc WORKFLOW_ARCHITECTURE.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Groundwork + stance for the longer cleanup (the [[staged-workflow-config-architecture]] Stage-3
`sub0_frontend` extraction). Full write-up in **docs/WORKFLOW_ARCHITECTURE.md** ("Frontend layer" section).

**The frontend layer** = the engine-free, pre-model logic the configurator + stage tools share but the
engine doesn't define: `config_util`, `memplan`, `registry`, `casing`, `tokenizer`, `unigram` (all
header-only + std-only today).

**Payoff #1 — testability (the real win; first slice DONE).** Because it's header-only it runs in the
fast engine-free `sub0_tok_tests` target (no engine build, no GPU). Added `tests/frontend_tests.cpp`
covering `memplan` (VRAM clamp/cap math) + `registry` (`compatible()` = the `models --prune` rule) — the
two areas this session changed, previously tested only via the slow engine-linked target (memplan) or
NOT AT ALL (registry). 200→**224 assertions**. Still-open gaps: `casing` edge cases (only indirect via
tokenizer), registry meta read/write + `scan` (needs a temp-dir fixture), the configurator's header emit.

**Payoff #2 — `sub0_frontend` lib (Stage 3).** Extract a static lib (those headers) as ONE named home
the configurator + stage tools link, plus a single `sub0_frontend_tests` target. Low-risk CMake/include
reshuffle; the engine-free test target already proves the value. `frontend_cuda` (runtime device probe)
stays deferred — CMake bakes device facts today.

**Tokenizer/vocab tool — JUDGEMENT: YES, build `sub0llm-tokenizer` (engine-free, links sub0_frontend).**
For diagnostics + interchange, NOT preprocessing. `encode "text"` (debug CamelCase-shatter/indented-code
pathologies), `decode`, `roundtrip <file>`, `vocab [--limit]` (MOVE it off the engine — today `sub0llm
vocab` wrongly lives in the train stage lib, forcing a full engine build to print a table), `export
--format json|tsv` (token↔id/piece/logp for external/cross-tokenizer analysis + the diffusion-viz
scrubber). Operates on a BUILT tokenizer.bin + sample input. Do NOT duplicate corpus→corpus.tok or
`--dump-vocab` — those stay the configurator's (it owns the scan/out-of-core/.words + the BPE-vs-Unigram
A/B + vocab curve). Sequencing: sub0_frontend lib → move `vocab` off the engine → add encode/decode/
roundtrip/export.
