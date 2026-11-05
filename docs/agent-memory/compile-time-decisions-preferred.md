---
name: compile-time-decisions-preferred
description: Standing preference -- bake every decision that's knowable upfront as constexpr/template/compile-time (zero runtime cost), matching this project's existing config-baking philosophy; runtime flags are only a TEMPORARY dev-workflow convenience and must carry a TODO to convert
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's explicit standing instruction (2026-07-04): "we make all decisions that are known upfront as
constexpr/template/compiled-in decisions for zero cost as possible. But for dev we can make temporary
workflow exceptions but these must be noted as todo to resolve."

**Why**: this is already this whole project's architecture -- every model dimension, precision choice,
positional encoding scheme, ternary/gated-FFN toggle is a `constexpr` baked by `sub0llm-configure` into
the generated header, specifically so the hot paths (`if constexpr`) compile away the unused branch
entirely rather than paying a runtime check. Introducing a NEW runtime-toggle knob for something that
is actually known upfront (the user decides once, doesn't change per-step or even per-run typically)
works against that design, even where the runtime cost of the branch itself is small.

**How to apply**: when adding a new feature/knob, default to compile-time (a new configurator CLI flag
emitting a `constexpr bool USE_X`, gated with `if constexpr` in the engine, following the
USE_TERNARY/USE_GATED_FFN/POS_ENCODING pattern already established). If a runtime flag is chosen
instead for DEV-ITERATION SPEED (e.g. avoiding a rebuild to A/B two options quickly, as with
`--optimizer adamw|muon` in [[muon-optimizer]] -- switching optimizers needs no rebuild since it
doesn't change PARAM_LAYOUT/checkpoint shape, unlike a dims/architecture choice), that is an accepted
TEMPORARY exception -- but it must be flagged with an explicit TODO (in code AND in memory) to convert
it to a compile-time decision later, not left as a permanent, undocumented deviation from the pattern.

**Open TODO from this instance**: `--optimizer`/`AdamW`'s `use_muon` constructor bool
(`include/sub0/core.hpp`, `src/train_stage.cpp`) is currently runtime-only. Convert to a
configure-time `constexpr bool USE_MUON` (configurator `--optimizer adamw|muon` flag emitting it into
`sub0_system.hpp`, next to `DEFAULT_GPU_BATCH`/tuned runtime defaults -- it's a MACHINE/training-run
choice, not a model-identity one, so it likely belongs in the system header not the corpus header) once
the A/B evidence justifies committing to it, so `AdamW::step()`'s per-param `if (muon_eligible)` branch
can become `if constexpr` and the unused path (Muon math when off, or the plain-AdamW path when Muon is
the committed default) compiles away entirely.
