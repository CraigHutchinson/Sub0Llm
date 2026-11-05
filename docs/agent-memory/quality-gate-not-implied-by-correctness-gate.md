---
name: quality-gate-not-implied-by-correctness-gate
description: "Correctness+perf gates do not catch code-quality defects -- run cpp-review before merge, load cpp-write before authoring (AGENTS.md §12)"
metadata:
  node_type: memory
  type: feedback
---

User directive, 2026-09-21: "ensure all work is optimized first class code."

**Why:** Sub0Llm's established discipline gates every work package on correctness (parity/fixtures) and
a real measured throughput number. Both necessary; neither says anything about maintainability. Ran
`cpp-review` over ~1,650 lines merged across six packages in one session (all green, all honestly
measured) and it still found: a direct AGENTS.md §1 violation (staging buffers sized inside a
per-layer-per-token hot path — first call heap-allocating *inside* decode); `[[nodiscard]]` absent from
every new failure-signal API (`submit`/`wait`/`dequantize_*` returning droppable success flags) despite
12 existing headers using it; and verbatim copy-pasted loop scaffolding in a brand-new header.

**The sharpest signal**: the new header modelled on an existing well-formed one (`fp8.hpp` ← `bf16.hpp`)
inherited every convention correctly. The two written *fresh* under time pressure had zero. Quality
follows the template you hand the author — so hand them one.

**How to apply:**
- Delegating C++ authoring to a subagent → tell it to invoke the `cpp-write` skill BEFORE writing, and
  to self-review against `cpp-review` before reporting. Relay this mid-flight if a package is already
  running ([[fable-subagent-for-big-risky-tasks]] — always review the diff regardless).
- Before merging any subagent's C++ → run `cpp-review` on the diff yourself, not just the suites.
- Generic cpp references assume `.h`/PascalCase; this repo is `.hpp`/snake_case. Incumbent convention
  wins — raising those is an anti-finding (the skill's own legacy-irregularity rule).
- Recorded permanently as AGENTS.md §12 + its pre-ship checklist line, so it survives this session.
