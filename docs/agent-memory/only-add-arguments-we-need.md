---
name: only-add-arguments-we-need
description: "don't add speculative CLI args / CMake knobs; only expose interface for things actually used"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Don't add CLI arguments, CMake cache variables, or other config knobs unless something actually consumes them. When I added `SUB0_VOCAB`/`SUB0_MIN_MERGE` CMake knobs + passthrough that nothing set, the user told me to revert: "We should only add arguments for things we need."

**Why:** speculative knobs are interface surface and maintenance cost with no caller; they imply a flexibility that isn't wired up.

**How to apply:** before exposing a new option, confirm a real consumer exists (a build step, a script, a documented workflow). If the value is fixed in practice, leave it as the existing default/constant rather than plumbing a new lever. Existing options with a genuine purpose (e.g. configurator `--vocab` caps model size — VOCAB sizes the embedding/LM-head) are fine to keep. Relates to [[architecture-solidification-plan]] (thin tools).
