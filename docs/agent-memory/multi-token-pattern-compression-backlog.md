---
name: multi-token-pattern-compression-backlog
description: "BACKLOG idea (user, 2026-07-16): extend the persistent (unbounded id>=VOCAB) slot substrate beyond single OOV words to recurring MULTI-token/multi-word patterns -- especially valuable for code's repetitive idioms. No new engine work needed at the substrate level; the gap is mining + substitution."
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's idea, given while a Hash/RoPE positional-encoding A/B spike was running (see
[[meanpool-alternatives-prior-art-and-math]]): the [[persistent-slot-range-engine-substrate]] (unbounded
id>=VOCAB scratch range, engine-safe, tested, committed) was designed and described around a single
motivating example -- a recurring multi-token OOV WORD (e.g. "Rumpelstiltskin") getting one stable id.
The user's point: the same substrate generalizes to recurring multi-token / multi-WORD PATTERNS, not just
single words -- **and this is especially powerful for code, where repetitive idioms are extremely common**
(boilerplate loops, common guard clauses, import blocks, license headers, standard formatting). Collapsing
a whole recurring line/idiom to ONE token would save context length and KV-cache substantially on
code-heavy corpora, likely a bigger win than the single-OOV-word case the substrate was originally framed
around.

**Why this needs no new engine work (the substrate already generalizes)**: `PersistentBindings`/
`encode_slot`/`encode_slot_bwd` take a fragment SPAN of token ids and a `tok_emb` pointer -- nothing in
that path assumes the fragments are byte-level characters of one word. A fragment span could equally be
several already-tokenized WORDS or pieces (e.g. the token ids for `"for (int i = 0; i < "` verbatim). The
gap is entirely upstream of the engine: (1) a mining strategy to find frequent multi-token spans in a
corpus, (2) an encode-time substitution step (recognize a known span, substitute its persistent id), (3)
maybe a live decode-time collapse (mirrors [[repeatspike-harness-collapse-result]]'s proven harness-driven
mid-stream binding, but for the unbounded range instead of the bounded 6-slot pool -- explicitly listed as
"not yet connected" in the persistent-slot-substrate memory).

**A real design distinction worth flagging before assuming this is simple**: code idioms are often
STRUCTURALLY repetitive but not BYTE-IDENTICAL -- `for (int i = 0; i < N; i++)` recurs as a shape, but the
variable name and bound differ per occurrence (`i` vs `j`, `N` vs some other variable). A literal-span
frequency count (exact byte-sequence match) only catches the fully-identical cases -- still valuable
(license headers, standard imports, common exact guard clauses, whitespace/formatting runs), but does NOT
by itself compress "same shape, different identifier" recurrences. Solving THAT would need something more
like a parameterized/templated slot (a pattern with one or more "holes" filled in per occurrence) -- a
materially harder, separate mechanism, not a natural byproduct of frequency-mining literal spans. Keep
these as two distinct scopes: (a) literal-span compression (cheap, substrate-ready, do this first) vs.
(b) templated/parameterized pattern compression (a real research question, not yet designed at all).

**How to apply**: when [[persistent-slot-range-engine-substrate]]'s "next real step" (encode-time
substitution or live decode-time collapse) gets picked up, do NOT scope it to single OOV words only --
design the mining/substitution step to handle arbitrary-length multi-token spans from the start (the
substrate already supports it), and explicitly decide (a)-first / (b)-later rather than accidentally
under-scoping to single words or over-scoping to templated patterns in the first cut. A code corpus
(this project doesn't currently train on one, per [[capability-roadmap-discrete-experts]]'s corpus catalog)
would be the natural test case if/when this is picked up, since that's where the payoff is claimed to be
largest.
