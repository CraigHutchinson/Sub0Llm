---
name: repeatspike-harness-collapse-result
description: "PROVEN - harness-driven (not model-requested) collapse of a repeated multi-token mention beats fuzzy repeat-copy 0.95 vs 0.13 held-out, and mid-stream binding works as well as pre-arranged binding"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

New spike (`include/sub0/repeatspike.hpp` + `tests/repeatspike_engine_tests.cpp`, tag `[.repeatspike]`), committed as part of the content-embed session work. Unifies two threads: [[deterministic-mechanisms-thesis]]'s proven "harness binds, model resolves" pattern (scratchspike's `nth_char_task`/`select_task`, pre-arranged before the episode) with the "live retroactive collapse" idea (a multi-token span the model writes out normally gets silently swapped for its scratch slot on recurrence, no model request) that came up discussing a persistent compound-word cache. Sidesteps the real unbuilt engineering gap (splicing a live KV-cache mid-generation, which RoPE position-baking makes non-trivial) by testing the mechanism over baked training traces — the same full-forward path every other spike in this codebase uses.

**Task**: an OOV word appears 3x in a passage; query asks for its Nth character.
- **FUZZY** (all 3 mentions spelled out in full, no scratch mechanism, plain in-context lookup): held-out accuracy stays low/noisy, 0.02 → peaks ~0.32 → ends **0.13** at step 3000.
- **COLLAPSE** (mention 1 spelled normally — no marker, no request, exactly as a live harness would observe and bind; mentions 2/3 replaced by the bound slot, masked; query resolves via UNCOMBINE): climbs to **0.95** by step 3000.

**Why it matters**: (1) confirms the copy/localization wall bites even when the answer is plainly visible in fed context (not just for arithmetic — for entity/spelling tracking too); (2) proves harness-driven binding works when formed **mid-stream** from a natural first mention, not just when pre-arranged before the episode starts (closing the gap toward genuine live compaction); (3) independently reconfirms scratchspike.hpp's own documented finding that `define_task` (#1, model-driven binding via emitting `combine`) is "copy-bottlenecked" and excluded from production — harness-driven isn't just *an* option, it's the one that actually works at this scale, now shown for a second trigger (repeat-recognition, not just pre-arrangement).

**How to apply**: this was the mechanism-level proof, not yet the production feature, as of the session that wrote it. **UPDATE (2026-07-18): the live version is now built** — see [[wordspike-natural-prose-collapse]]. The KV-cache splice turned out to need no new engine primitive at all (the CPU KVCache is a flat, position-indexed buffer with no separate length counter, so a retroactive `ctx.resize()` + re-`feed()` is just a normal overwrite). Full writeup + the RoPE/causality analysis is in docs/DETERMINISTIC_MECHANISMS.md's "Repeated mentions" paragraph (now with a 2026-07-18 update note) and docs/SCRATCH_TOKENS.md's "Natural-prose word collapse" section.
