---
name: persistent-slot-selection-problem-backlog
description: "BACKLOG with a DESIGNATED direction (not started): a model can never PREDICT/emit a persistent (id>=VOCAB) slot id (no logit column), but raw-id emission was an artifact anyway (op_curriculum's graded g(S0) is a synthetic-template choice) -- the path is plain-vocab HANDLES (ordinal bytes) dereferenced by the harness/node. Work item: reimplement op-chain references as handles + A/B vs S0-emission; unlocks full 6-in-vocab-pool deprecation."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Found 2026-07-16 while scoping how far the persistent (unbounded, `id >= VOCAB`) scratch-slot substrate
([[persistent-slot-range-engine-substrate]]) could generalize the proven scratchspike mechanism
([[scratch-tokens-context-translation-layer]], [[repeatspike-harness-collapse-result]]).

## The constraint

`PersistentBindings`' ids live entirely outside `[0, VOCAB)` — `include/sub0/scratch_slots.hpp:294-308`.
`sample_token`/cross-entropy targets are hard-bounded to `[0, VOCAB)` (`src/engine_core.cpp`,
`persistent_slots_engine_tests.cpp:117-120`'s own comment: "a persistent id can never legally be
[a target]: it has no logit column"). So a persistent id can be **consumed** (attended to, its composed
vector reasoned over) but never **produced** — the model cannot be trained to literally emit/choose one
as a graded prediction, unlike the ephemeral 6-slot pool (which sits inside real vocab ids and DOES have
logit columns, letting e.g. `select_task`/`content_select_task` train the model to output
`k.answer_byte = scratch_slot(query_idx)` directly).

## What this rules out (for now)

Reading `include/sub0/scratchspike.hpp`'s actual task set: `nth_char_task`/`roundtrip_task` are
ATTEND-ONLY (the query already NAMES which slot to resolve; the model only ever copies/resolves
already-injected content) -- these transfer cleanly to the persistent range. But `select_task`,
`content_select_task`, `content_contains_task`, and `content_contains_reason_task` all require the model
to CHOOSE which slot to reference by literally predicting its id as the graded answer -- the associative
"reason over several bound entities and pick the right one" capability. None of that transfers to
persistent ids as designed. This is the harder, more valuable half of what scratchspike proved (multi-slot
disambiguation), and it's exactly what's blocked.

## PHASE 2 FINAL VERDICT (2026-07-17): SELECTION AT K=12 SOLVED via Route-A -- 0.903 held-out, NO
## engine override needed

Full pairspike result set (all 3-seed, EpochWalk, d128):
- ONE-HOP select-via-pair (extract content from the override embedding at selection time): NULL at K=12
  (0.092) AND K=3 (0.347 vs chance 0.333) with or without override, while dedicated-id one-hop + HRR had
  hit 0.744 at K=3 -- the pair OUTPUT interface (handle letters' static lm_head rows serve double duty)
  is the one-hop wall. Sentinel DISPATCH proven correct independently (4 engine differentials,
  tests/sentinel_pair_engine_tests.cpp, default suite).
- ROUTE-A (user-directed: "first letter queries should track through the uncombine" -- resolve each pair
  live, restate query locally, verdict, copy the matching pair): **no-override 0.903 held-out at K=12**
  (seeds 0.850/0.950/0.908; drilled ~0.95); HRR-override 0.775 (0.792/0.800/0.733) -- the override arm
  is WORSE (~-0.13): the content-carrying embedding at pair positions adds interference, not help, once
  content enters via injected bytes. Timing (three-pillar): ~700s train + ~137s eval per seed,
  multi-threaded (the stopped single-core run projected ~7h total; MT did 6 seeds in ~85min).

CONSEQUENCES: (1) selection over unbounded pair-addressed entities is a SOLVED capability via the
proven local-CoT scaffold + plain 2-token pairs -- no logit columns, no embedding override, no engine
mechanism beyond the existing interceptor; (2) the SentinelBindings embedding override is RE-SCOPED:
not load-bearing for selection (and mildly harmful there); its remaining candidate value is one-hop
compaction/consumption contexts -- keep the engine mechanism (tested, inert-by-default) but don't build
on it until a use case measures a win; (3) the 6-in-vocab-pool deprecation path is now fully de-risked:
handles work for op-chaining (Phase 1, 1.000) AND selection (Phase 2, 0.903).

## PHASE 1 RESULT (2026-07-17): handle references PROVEN -- 1.000 every seed, == reserved-slot arm

The chaincapstone 3-arm x 3-seed A/B/C (commit 005968c): DIGIT 0.055/0.000/0.000 finals (known wall),
SLOT 1.000x3, **HANDLE ($k plain-byte pair, bind::make_handle_collapse_callback pre-substitution deref)
1.000x3 -- exactly matching SLOT on every seed**. The association skill fully survives the reserved-id-free
2-token interface, zero seed variance when the mechanism is right. Phase 2's gate (sentinel-pair spike) is
PASSED; the 6-in-vocab-pool deprecation's reference-interface blocker is dissolved with data, not just
argument. Also noteworthy as contrast: the K=30 persistent RESOLVE task shows wild seed variance at the
same budget -- variance lives in capability-edge tasks, not in the reference mechanism.

## The FULL design (2026-07-16, latest): the `<|scratch>` sentinel PAIR -- user-proposed, sanity-checked

The handle idea below evolved into its complete form the same day: a single reserved in-vocab SENTINEL
token followed by an ordinary index token (`<|scratch> a`), where the sentinel changes how the NEXT token
embeds -- position t gets the bound entity's composed content vector (encode_slot/HRR) instead of letter
a's static row. Sanity-checked against the engine: (1) both tokens emittable (logit columns) -- selection
becomes ordinary next-token prediction in a sentinel-created namespace; (2) the embedding override slots
into the exact 3 dispatch sites the slot mechanism already uses (add an `ids[t-1]==TOK_SCRATCH` check);
(3) casing.hpp markers ALREADY have documented after-effects (TOK_ODQUOTE/TOK_SPELL_START) -- established
pattern, applied at the embedding layer; (4) the index can dereference into the PERSISTENT table, making
the pair the model-facing addressing mode over the unbounded store (bare persistent ids stay for
harness-injected consumption-only refs, 1 token vs 2). TWO HONEST UNKNOWNS gating it: output-side handle
CHOICE scores against static lm_head rows (tied embeddings) -- an in-context association skill, proven
only at K<=6 (select_task), needs its own spike at scale (fallback: pointer-style head, engine work,
held in reserve); and reserved-id headroom is ZERO -- the permanent sentinel id comes from the 6-slot
pool's deprecation (a SPIKE can commandeer TOK_RESERVED_9 with no format bump, leaving K<=5 for the
comparison arm). Full phased plan (0: finish shootout/saturation; 1: handle A/B on op chains; 2:
sentinel-pair spike; 3: deprecate pool + reclaim ids -- ALSO unblocks [[op-marker-dedicated-token-backlog]];
4: word-level CODE context, the motivating domain -- long compound identifiers, heavy repetition) is
written up in docs/SCRATCH_TOKENS.md "The end-state: word-level context via sentinel-pair addressing".

## The direction (DESIGNATED 2026-07-16, earlier same day -- elevated from speculation by a user insight)

How could a model ever "choose" an unpredictable token? The user's op-collapse observation settled the
direction: the graded raw-slot-emission in op chains (op_curriculum.hpp:127's `g(S0)`) is a
synthetic-template ARTIFACT, not a necessary interface -- the op executes in the scalar node outside the
model, so the reference convention is a harness-side choice. **Designated path: plain-vocab HANDLES** --
the model emits a short in-vocab reference (ordinal byte '0'/'1' per-context, or an implicit
"previous result" default for linear chains) and the harness/node dereferences handle -> actual binding
(ephemeral or persistent, the handle scheme is range-agnostic). This is the pointer-network idea in its
cheapest form: the prediction target is an ordinary byte token, no logit-column problem, likely NO engine
change at all -- curriculum + node-side deref only. NOT yet built or A/B'd; the concrete work item is:
reimplement the op-collapse chain curriculum with handle references, A/B against the current S0-emission
form, and if it holds, the same handle scheme unlocks selection-style tasks over the persistent range
(the model answers "which entity?" with a handle, never a raw id). See
[[persistent-slot-range-engine-substrate]]'s deprecation-plan section for how this feeds full deprecation
of the 6-in-vocab pool.

## Why this is separate from the current narrowed spike

2026-07-16: user explicitly narrowed the immediate next step to attend-only resolve tasks only
(delegated to a Fable agent, see the spike this produced), and asked to keep this selection problem as an
explicit backlog item for later rather than lose the framing. Do not conflate the two: the attend-only
spike does NOT attempt or need this; this is follow-on work only worth picking up if/when the
"resolve at scale" spike itself proves out and the harder "reason over many entities, pick one" capability
is wanted on the persistent range specifically (as opposed to just keeping that capability on the existing
capped-at-6 ephemeral pool, which already has it and works).
