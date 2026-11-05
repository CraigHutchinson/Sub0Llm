---
name: agents-md-established
description: AGENTS.md created at the repo root (2026-07-04) -- a pre-flight engineering checklist derived from this session's real near-misses/lessons (heap-alloc-per-call, checkpoint-format risk, constexpr-vs-runtime, reference-verification, multi-scale testing). Read it, and keep it updated when a new lesson is learned the same way.
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's instruction (2026-07-04, right after the Muon A/B and the "pause for sleep" request): "we
should never allow per heap alloc on every call - we should setup project rules based on how the rest
of the engine has been developed to be optimal. Setup an agents.md based on our recent works to ensure
we always strive to get things right and ship shaped first time around."

**What was created**: `AGENTS.md` at the repo root, NOT this ~/.claude memory directory -- a
repo-committed (once the user commits it), codebase-wide checklist any agent/tool working on this
project should read, distinct in purpose from this memory system (memory = my own cross-session
continuity notes; AGENTS.md = durable, shared engineering rules). 9 numbered rules, each grounded in a
concrete case from this session, not generic advice:
1. No heap allocation in per-step/per-call hot paths (the motivating case: `newton_schulz5` allocates
   fresh `std::vector`s every call, measured ~11x slower than AdamW at d128 -- see [[muon-optimizer]]).
2. Bake known-upfront decisions as constexpr/`if constexpr`, not runtime branches (see
   [[compile-time-decisions-preferred]] for the full standing-preference writeup this rule summarizes).
3. Checkpoint/binary-format changes are the highest-blast-radius category here -- check whether an
   EXISTING field already discriminates before adding a new one (the gated-FFN near-miss: almost added
   a `gated_ffn` Header/.ckpt field, would have broken resuming the live production checkpoint; found
   `param_floats`/`trainable_floats()` already sufficed).
4. New capability defaults off; verify the DEFAULT build's test-suite assertion COUNTS are identical
   before/after, not just "no new failures."
5. Verify precise ported algorithms against their actual reference source (fetched, not recalled) --
   the Muon momentum-EMA-vs-raw-accumulation and axis-convention traps, see [[muon-optimizer]].
6. New math needs an independent numerical correctness check before it's "done", not just "compiles
   and the loss drops."
7. Test at more than one model scale -- see [[engine-tests-gradient-check-dims-dependent]] (same test,
   same code, dims-dependent pass/fail).
8. Only add CLI/config surface that's actually consumed (references the SUB0_VOCAB/SUB0_MIN_MERGE
   revert, see [[only-add-arguments-we-need]]).
9. Validate import/interop features against a REAL file/corpus, not synthetic fixtures alone (the GGUF
   real-download validation, the Muon tinystories A/B).

Ends with a literal "before you ship" checklist recap.

**How to use this going forward**: read AGENTS.md at the start of substantive engine-level work in
this repo (it is short, meant to be actually read, not skimmed once and forgotten). When a NEW lesson
gets learned the same way these 9 were (a real near-miss caught, or a real bug found post-hoc) — add
it to AGENTS.md with the same "concrete case, then the rule" structure, don't let the file drift into
generic advice disconnected from what actually happened in this codebase.

**Not yet done**: `newton_schulz5`'s actual heap-alloc fix (rule #1's motivating case) is still QUEUED,
not applied -- the user asked for the RULE to be set up first, not necessarily the fix executed in the
same breath (they'd just asked to pause for PC sleep). See the architecture-ideas-backlog / the
open "optimize newton_schulz5" todo for the actual fix.
