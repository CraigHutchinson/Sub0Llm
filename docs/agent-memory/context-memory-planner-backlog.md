---
name: context-memory-planner-backlog
description: User backlog lever 2026-10-08 — size KV/context memory per chat (prompt + max response + thinking limit) via a memory planner module, so a 32k-configured window does not hold 32k of memory and more model weights stay resident
metadata:
  type: project
---
Context-length memory (K/V caches, QSA indexer keys, per-position scratch) is sized by the configured
`SEQ_LEN`. The user wants it sized at chat start to what the chat can reach (input prompt + maximum
response length + thinking limit), with the configured window only as the upper bound, so the freed RAM
keeps more weights (expert cache / OS cache) in memory. "This needs a planner module to make the decisions
or be fixed based on config."

**Why:** the host is RAM-constrained ([[self-managed-memory-is-the-direction]]); context memory competes
with expert bytes.
**How to apply:** design first. One planner should own every runtime memory split: context size, the
expert cache budget (storage plan follow-up 1: budget from free memory at load), headroom. Extend
`include/sub0/memplan.hpp` from reporting to deciding. Tension: AGENTS.md S1/S2 (compile-time dims, no
per-token allocation) — keep capacity baked, choose the committed size once per chat. Unmeasured: how much
context memory a real-axes build holds at 32k (all measurements so far use a 128-token window). Recorded
as C7 in Sub0Llm `docs/optimization/opportunities/O13_storage_convergence.md`.
