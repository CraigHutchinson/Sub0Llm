---
name: corpus-scale-word-collapse
description: "DONE+committed (32d1a1d) - scales wordspike's proven word-collapse mechanism to a sampled subset of REAL corpus documents at train time. IMPORTANT: the original TinyStories validation numbers below were measured against a mistargeted tokenizer bug -- see [[tokenizer-schemev3-spell-unify]] for the fix and corrected numbers"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Follow-on to [[wordspike-natural-prose-collapse]]: the deferred idea from that work ("apply the same
repeat-collapse to the REAL base corpus, teach the pattern at full corpus scale") got picked up in the
same session. New `"corpus_collapse"` blend-schedule generator (`include/sub0/corpus_collapse.hpp`, wired
into `train_stage.cpp`/`gen_stage.cpp` exactly like `wordspike`) samples a subset of real documents from
whatever corpus.tok a training run is already using and applies wordspike's exact `ScratchTable::
combine_recurrence`/`detail::word_span` walk to them, per-document `ScratchTable` reset.

**A real scope correction made during planning, worth remembering**: the original deferred note (written
when wordspike shipped) sketched this as a *configure-time* pass baking collapse directly into
`corpus.tok` itself. Investigating found `corpus.tok` carries **no mask array at all** — every token
loaded from it is always fully graded. Baking collapse in directly would force either a breaking on-disk
format change, or silently making the model GRADED on predicting scratch-slot emission from real corpus
exposure — a fundamentally different, unproven capability claim wordspike's own capstone never tested
(its slots are always mask=0, harness-injected, never a prediction target). Shipped the lower-risk version
instead: a train-time sampled curriculum (in-memory Dataset, same architecture as wordspike/scratchspike/
op_curriculum), no format change, no new capability claim — just scales the ALREADY-PROVEN mechanism to
more diverse real text. The configure-time/format-changing version stays recorded as a further, deliberately
deferred promotion (`docs/CORPUS_COLLAPSE.md`'s "Gaps, risks, and extensibility" section).

**Masking discipline chosen deliberately differently from wordspike's own**: mask=1 for all real prose
(including a word's own first mention) except the exact substituted-slot position (mask=0 there) —
matches `op_curriculum.hpp`'s own result-collapse precedent, NOT wordspike's/scratchspike's more
aggressive whole-mention masking (which exists only because THEIR content is synthetic/unlearnable
either way; real corpus prose is normal, learnable language and masking it out would waste real signal).

**A real bug caught by a design-review subagent before any code was written** (worth the pattern, not
just this instance): a proposed `doc_index.empty()` → "warn and skip" behavior would have registered a
zero-size `BlendSource`, which can make the WFQ scheduler's `pick_source_staged` pick it forever (ties
resolve to a `size==0` source's always-`0.0` normalized progress) and then hand `sample_window` a
`train_tok` of 0 — `train_tok - full` underflows to a huge `size_t`, an out-of-bounds crash. Changed to a
hard `return 1` error instead. Caught by explicitly asking a Plan-mode subagent to stress-test the design
against the actual scheduler code before implementing, not discovered by running it.

**Validated on TinyStories** (`out/build/d196check`, D_MODEL=196), matched-budget A/B via the REAL
production scheduler (`sample_blend_staged`), equal-weight base:collapse blend (deliberately aggressive —
a real production schedule would likely weight this much lower).

**SUPERSEDED FINDING, kept for history only**: the first validation pass measured held-out NELBO moving
+0.014 nats and 91% of documents hitting a collapse candidate. That result looked like a mundane "no harm"
finding but was actually measuring a badly mistargeted mechanism — asking to SEE real examples (not just
trust the aggregate number) revealed the tokenizer's 2-piece-word encoding was ambiguous with ordinary
punctuation-glue, so almost all "collapses" were words like `room.`/`said,`, not rare entities. Fixed in
[[tokenizer-schemev3-spell-unify]] (committed be1cf3f). **Corrected numbers after the fix**: collapse rate
dropped to 24% of documents, NELBO delta flipped to -0.0075 (slightly helped), `build_dataset` got 9x
faster (235ms→25ms). See that memory and `docs/CORPUS_COLLAPSE.md`'s status log for the full story —
**don't cite the +0.014/91% numbers as current**, they describe the bug, not the mechanism.

**How to apply**: if picking this up further — (1) the "if this works" natural next step is the
configure-time/format-changing promotion, now informed by the CORRECTED real result; (2) a genuine, named,
un-solved gap: prefix-truncation only ever exercises recurrence in a document's OPENING portion, since
`corpus_collapse`'s documents must fit one training window — real prose often introduces an entity once
near the start and references it much later, well past the truncation point (see the doc's own "Gaps"
section for why the obvious fix — collapse-then-crop — doesn't cleanly solve this either); (3) don't
reuse the `[.corpus_collapse]` capstone's own non-TinyStories runs as meaningful results — its skip-guard
only catches "no corpus.tok at all," not "wrong corpus" (any correctly-configured build's own corpus.tok
trivially passes its own vocab check), so it will happily train against whatever corpus a build has;
(4) the standing lesson this whole episode reinforces: a clean aggregate metric is not proof of correct
behavior — spot-check real qualitative output before trusting a new mechanism's numbers.

Full design record + gaps/extensibility review + the real (corrected) result: `docs/CORPUS_COLLAPSE.md`.
