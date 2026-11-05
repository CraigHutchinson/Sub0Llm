---
name: slot-encoder-checkpoint-persistence-design
description: "BACKLOG design (scoped 2026-07-16, not implemented): promote CharEncoder/ConvPool's enc_w to a real PARAM_LAYOUT entry, gated by a NEW compile-time flag (mirrors USE_QK_NORM etc.). Fixes checkpoint persistence AND removes the single-threaded-training-only constraint in one change -- the actual production blocker for every learned scratch-slot encoder."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Context: [[scratch-content-embedding-meanpool-vs-charencoder]], [[meanpool-alternatives-prior-art-and-math]]
(`include/sub0/layout.hpp`'s `PARAM_LAYOUT`, `include/sub0/scratch_slots.hpp`'s `ScratchBindings::enc_w`/
`enc_w_grad`). User (2026-07-16) prioritized this work AHEAD of the parked bigger-scale spike test, as pure
CPU-side production-code progress that doesn't depend on CUDA/hybrid-execution timing.

**UPDATE (2026-07-16, same day, later): this no longer blocks the CURRENT leading candidate.** `HRR`
(circular-convolution binding) decisively won the 3-way A/B against Hash and ConvPool (mean held-out 0.744
vs chance 0.333 -- see [[meanpool-alternatives-prior-art-and-math]]'s RESULT section) and, like Hash, needs
NO learned parameters at all -- so it does not depend on this checkpoint-persistence work to be wired into
production. This design remains valuable (CharEncoder still needs it for the presence/contains axis, and
any future learned encoder will too) but is no longer the blocking dependency for shipping the STRONGEST
currently-known order-sensitive encoder. Re-prioritize accordingly: wiring HRR into `train_stage.cpp` can
proceed without waiting on this.

## The problem, precisely

`ScratchBindings::enc_w`/`enc_w_grad` (CharEncoder needs `[C,C]`, ConvPool needs `[2,C,C]`) are CALLER-OWNED
buffers today (a `std::vector<float>` the test/spike harness allocates itself), not part of the model's
`PARAM_LAYOUT`. Two real consequences:
1. **No checkpoint persistence**: `save_model`/`load_model` only round-trip `PARAM_LAYOUT`'s flat blob. A
   model trained with CharEncoder/ConvPool loses those learned weights on reload -- unusable in production
   (this is the standing `TODO(charencoder-production)` in `scratch_slots.hpp`).
2. **Single-threaded training only**: `enc_w_grad` has no per-thread reduction arena (every OTHER param's
   grad does, via the standard `PARAM_LAYOUT`-driven mechanism + `reduce_gradients()`). This is WHY
   `train_ce_steps` (the CharEncoder/ConvPool A/B spike harness) had to be a bespoke single-threaded loop
   instead of using the normal (possibly multi-threaded) `train_batch` path every other encoding uses.

Promoting `enc_w` to a real `PARAM_LAYOUT` entry fixes BOTH at once -- it's the same underlying gap.

## Design

### 1. A new compile-time flag, not a runtime one
`content_embed` is a RUNTIME `RunConfig` flag (decided per training invocation via `--content-embed`), but
`PARAM_LAYOUT` is entirely COMPILE-TIME (`consteval`, driven by `USE_*` macros from `sub0_config.hpp` --
`USE_QK_NORM`, `USE_GATED_FFN`, `USE_TIED_EMBEDDINGS` are the existing precedent). These are different
axes and must stay different: mirror the EXISTING pattern exactly rather than inventing a new one -- add a
new compile-time flag (e.g. `USE_CONTENT_ENC`, wired through `sub0_config.hpp` / a new CMake option the
same way `SUB0_QK_NORM` etc. already work) that gates whether the param tensor exists in the layout AT ALL.
A model built WITHOUT this flag cannot use CharEncoder/ConvPool (parameter-free options -- MeanPool, Hash,
HRR -- remain available regardless, since they need no `PARAM_LAYOUT` entry). This avoids wasting
`[2,C,C]` floats (~1.6MB at d448) on every model that never touches content-embed, consistent with this
project's "new capability, off by default, no cost when unused" convention (see
[[arch-features-off-by-default-in-production]] for the same shape of tradeoff already made elsewhere).

### 2. Size the tensor for the LARGER shape (ConvPool's `[2,C,C]`), not per-encoder
Rather than adding a SECOND compile-time axis to distinguish "which learned encoder" (more complexity, two
new flags interacting), size ONE new `PARAM_LAYOUT` entry at `[2,C,C]` -- ConvPool's shape, which strictly
covers CharEncoder's `[C,C]` as its first half. The RUNTIME `SlotEncoding` selection (already how
CharEncoder vs ConvPool vs Hash vs HRR vs MeanPool is chosen today, via `ScratchBindings::encoding`) decides
which portion is actually read/written; the unused second half for CharEncoder is a minor, acceptable waste
compared to the complexity of a second compile-time selector. New `PKind::ScratchEnc` (or similar name) in
the `PKind` enum; `add(2, C*C, PKind::ScratchEnc, /*decay=*/?, /*ternary=*/false)` in `make_param_layout()`
-- `decay` and Muon-eligibility (`is_muon_kind`) are open questions, not yet decided (see below).

### 3. Open questions to resolve, not default silently
- **Muon eligibility**: shape-wise this is a `[C,C]`-like projection matrix, similar to `Wq`/`Wk`/etc., so
  it's PLAUSIBLE Muon (`--optimizer muon`) would suit it -- but it's a small, rarely-active, encoder-specific
  projection, not a core transformer layer, and there's no evidence either way yet. Default to AdamW-only
  (`is_muon_kind` returns false for it) for the FIRST cut -- simplest, avoids entangling this change with
  Muon-dispatch correctness -- and treat "should this route through Muon" as a separate follow-on empirical
  question once the basic wiring works.
- **Weight decay**: probably `true` (it's a projection weight, matching `Wq`/`W1`/etc.'s own `decay=true`
  convention), but not yet confirmed against any evidence specific to this param.
- **Init distribution**: the spike harnesses use `N(0, 1/sqrt(C))` (matching `op_linear`'s usual scale) --
  reuse that for `init_weights`'s treatment of the new tensor.

### 4. Wiring changes needed (not yet done)
- `include/sub0/layout.hpp`: new `PKind` value, `make_param_layout()` entry (conditional on the new flag),
  `NUM_PARAMS` formula updated.
- `Model::build_layout` (backend_cpu.cpp) and `init_weights`: bind/initialize the new tensor like every
  other param.
- `ScratchBindings` construction sites (`train_stage.cpp`'s per-window fill loop, `gen_stage.cpp`): point
  `enc_w`/`enc_w_grad` at the REAL param-arena offset (`sub0::params_ptr() + PARAM_LAYOUT[i].off` and the
  matching grad-arena offset) instead of a caller-owned buffer -- once this lands, `enc_w`'s gradient flows
  through the SAME `reduce_gradients()`/`AdamW::step()` path every other param already uses, so
  `train_ce_steps`'s bespoke single-threaded harness becomes UNNECESSARY (the spike tests could switch to
  the normal `train_steps`/`train_batch` path, same as Hash/HRR/MeanPool already use).
- `save_model`/`load_model`: likely automatic (if they iterate `PARAM_LAYOUT` generically, which the
  DRY architecture this project has generally maintains) -- CONFIRM this by reading them, don't assume.
- CUDA: `PARAM_LAYOUT`-driven code paths in `backend_cuda.cu` (upload/download, Muon/AdamW dispatch) would
  pick this up automatically IF genuinely driven by the same table -- but content-embed stays CPU-only
  regardless (see [[hybrid-cpu-gpu-execution-design]]), so this is a "shouldn't break GPU builds" check,
  not a "must work on GPU" requirement for this change.

### 5. Verification bar (matching this project's own standing discipline)
- `tests/layout_tests.cpp`-style offset/count consistency tests (this file ALREADY exhaustively tests
  `PARAM_LAYOUT`/`DECAY_RANGES`/`MUON_RANGES` -- extend it, don't bypass it).
- Re-run the CharEncoder and ConvPool gradient-check + A/B spike tests against the NEW wiring (real param
  arena, not a caller-owned buffer) to confirm identical behavior -- a regression here would be a real,
  subtle correctness bug (e.g. an offset mistake silently aliasing another param's memory).
- A real save/load round-trip test: train briefly with ConvPool active, save, reload, confirm `enc_w`'s
  values survive exactly (not just that other params do).
- Multi-threaded `train_batch` now exercising content-embed with a learned encoder for the first time --
  worth a dedicated test proving no data race (the exact hazard `enc_w_grad`'s current caller-owned buffer
  was previously not safe for).

## Sequencing

This is a real architectural change (new compile flag + `PARAM_LAYOUT` entry + several call-site rewires),
bigger than any single encoder-candidate spike, per AGENTS.md's "checkpoint-format caution" standing
concern -- deserves its own careful, dedicated implementation pass, not a rushed addition alongside other
work. Recorded here fully scoped so it can be picked up and implemented directly without re-deriving the
design. Blocks: any production wiring of CharEncoder or ConvPool into `train_stage.cpp`'s content-embed
path (Hash and HRR, being parameter-free, do NOT need this and could be wired sooner if either wins the
A/B round).
