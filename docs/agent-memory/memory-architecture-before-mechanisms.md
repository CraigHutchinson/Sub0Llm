---
name: memory-architecture-before-mechanisms
description: "When planning a multi-mechanism implementation effort that has to run under a real memory/VRAM/disk budget, design the cross-tier memory/data orchestration FIRST as its own phase -- not as a deferred \"integration\" work package planned after each mechanism already has its own ad hoc small-scale layout"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5cbd8e2b-5223-4a4c-ad2b-a3a3e9b3d19d
  modified: 2026-09-02T10:03:59.123Z
---

**What happened**: planning the "Sub0Llm runs a real Qwen3.8-Flash-Next forward pass" effort
(see [[pivot-to-existing-model-support-qwen4-preview]]), my first draft plan sequenced three missing
mechanisms (Gated Residual, MoE, QSA) as WP1-3, each with its own "storage design" question decided
locally, and pushed the actual memory/disk/VRAM budgeting to "WP4, scoped later." The user corrected this
directly: the primary phase should be an in-depth memory/data orchestration plan (disk/CPU/GPU
handoff/layout, budgeted AND flexible, expected to need iteration) — done BEFORE any mechanism
implementation, not after. Also flagged two specifics: Sub0Firn's placeholder role for the n-gram tier
should be considered in that architecture even though it isn't built yet, and the n-gram table itself
isn't actually necessary for an initial inference test (so it can be explicitly excluded from the first
budget rather than blocking on it).

**Why**: the hardest single design question in a multi-mechanism, memory-constrained effort like this
(here: MoE's expert-storage layout, which has no existing precedent in the codebase) has a strictly
better answer if the surrounding placement architecture is already decided — otherwise each mechanism's
own team/session either re-derives a placement policy in isolation (drifting from what its neighbors
assumed) or defers the real question, and by the time "WP4" arrives to reconcile everything, at least one
already-merged mechanism's storage shape has to be revisited/redesigned rather than just used.

**How to apply**: when a plan involves multiple new capabilities that all have to compose under one real
resource budget (memory, VRAM, disk, or similar), put the cross-cutting resource-architecture design as
the FIRST phase, not a later integration step — even though it will necessarily be incomplete/approximate
until the individual mechanisms report real measured costs back into it (this is fine, and should be
stated explicitly as an iteration loop, not a reason to defer starting it). Each mechanism's own
work-package brief should then be written to explicitly build against that architecture's placement
policy, not decide storage/placement locally. This generalizes beyond Qwen4 — it's the same shape of
lesson as [[compile-time-decisions-preferred]] (decide the cross-cutting structural constraint before the
local implementations that must honor it), just applied to memory hierarchy instead of compile-time
constants.
