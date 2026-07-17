# Scratch tokens — a per-context translation layer for out-of-vocabulary words

Scope note: this is a **feature explainer + status** for the scratch-token capability — the mechanism,
what works and what doesn't as worked examples, and the two distinct routes to "reasoning about a
slot's content." It complements [ROADMAP.md](ROADMAP.md) (where the corpus/architecture is headed) and
the project memory `scratch-tokens-context-translation-layer`. The spike code lives in
`include/sub0/scratchspike.hpp` (curriculum), `include/sub0/scratch.hpp` (the production binding table),
`include/sub0/scratch_slots.hpp` (the engine-facing foundation), and the content-embedding scaffold in
`src/backend_cpu.cpp`.

---

## TL;DR — what works, and the one thing that doesn't (plain terms)

Think of a scratch slot as a **temporary nickname** for a word the model doesn't have a token for. Almost
everything about the nickname works:

1. ✅ **Give an unknown word a nickname** (bind a slot) — works.
2. ✅ **Use the nickname as a stand-in** — carry it through the text, refer back to it, pick it out by
   association ("the one tagged ★") — works cleanly.
3. ✅ **Spell the nickname back out on demand** (resolve / "uncombine") — works, *even for words the model
   never saw during training* — it reads the spelling from the context, it isn't memorizing.

The last mental step — **read several spelled-out words and answer which one contains a given letter** —
splits into two sub-skills:

- ✅ **Checking** ("is the letter in *any* / *all* of them?") — works nearly perfectly.
- ⚠️ **Localizing** ("*which* word contains it?") — the hard one, but **now solved** (read on).

For a while localizing looked like a hard wall: the model could tell the letter was *there* but not *which*
word — it pointed at random, and that survived a bigger model, more attention heads, and a "show your work"
scaffold. The breakthrough was realizing **why**: the model couldn't hold the queried letter in mind while
scanning words listed far earlier in the text. It's a *distance* problem, not a reasoning one.

The fix is almost embarrassingly simple: **repeat the letter right next to each word as you check it**
("looking for `z`… `cstk`? no. `tzi`? yes → that one"). With the question restated locally beside the data,
the tiny d128 model gets it **100% right on held-out words at 3 candidates**. So the capability is *there*;
it just needs the reasoning **grounded locally** — put the question next to the data, not paragraphs away.
(How we cornered this: [What isn't working → what fixed it](#what-isnt-working-the-decomposition).)

---

## The problem

A model only knows a fixed vocabulary of tokens. A word that isn't on the list — a rare name, an OOV
construction — has to be **spelled out in fragments every time it appears**. If `Zblorptax` tokenizes as
`Z + bl + or + pt + ax` (5 tokens) and a story mentions it 20 times, that is **100 tokens** spent
re-spelling one name — burning context window and KV-cache — and the model never gets to treat the word
as a *single thing* it can reason about.

## The core trick: sticky-note nicknames

Reserve a few spare token ids — **scratch slots** (blank sticky notes / variables). The first time an
unknown word appears, **bind** a slot to its fragments and keep that in a per-document **binding table**:

```
Slot #4  →  [Z, bl, or, pt, ax]      "#4 means Zblorptax, for now"
```

Every later mention is then **1 token** (`#4`) instead of 5. Two operations:

- **combine**   — fragments → `#4`   (compress: give it a nickname)
- **uncombine** — `#4` → fragments   (expand: spell it back out when needed)

The table is **per-context** — it resets between documents. In the next story `#4` might mean `Xyzzy`.
It is a scratchpad, not permanent memory. Net effect: an OOV costs **1 token per mention instead of N**,
and becomes a single symbol the model can carry and reason over.

The binding table is minted on the fly by an interceptor during generation (`ScratchTable` in
`scratch.hpp`); during training the curriculum (`scratchspike.hpp`) pre-binds slots to model
"a definition that happened earlier in the context."

---

## What works — all "in-context symbol" reasoning ✅

These work because the slot only has to be a **consistent symbol**; the model never needs to look
*inside* it.

**Resolution (round-trip).** Expand a slot back to its spelling.
```
Prompt:  ~ #4
Answer:  Z bl or pt ax          (partial at d128, ~0.31–0.46 cold; rises with training)
```

**Associative reasoning over slots (works cleanly, ~1.000).**
```
Context:  The wizard is #4 . #4 owns a cat named #7 .
Question: Who owns the cat?  →  #4
```
`#4` is carried like a pronoun. What it *spells* is irrelevant — it is a stable token.

**Disambiguation / selection.** Several slots bound; pick the right one from an in-context tag pairing
(`select_task`). Works — matching symbols, not reading contents.

## What is harder — reasoning about the *content inside* a slot

Ask a question that needs the **letters inside** a slot:

```
Context:   #4 = [Kryptonite],  #7 = [Zebra]
Question:  Which one contains the letter K ?
```

A bare scratch slot is a **generic reserved id**: `#4`'s embedding vector is identical whether it is
bound to `Kryptonite` or `Zebra`. The letters are on the *back* of the sticky note; the model only sees
the nickname on the front. There are **two different ways** to get at the content — and they trade off.

---

## The two routes to content (this is the subtle part)

### Route A — uncombine + loss-masking (spell it back out)

The model **invokes resolution**: it emits `UNCOMBINE`, the binding table injects the fragments into the
stream, and the loss is **masked** over those injected bytes — so the model is rewarded for *choosing to
resolve* and for the final answer, but never for copying the fragments (that closes the "just read it off
the stream" shortcut). This is the exact mechanism the in-vocabulary spelling spike used. The trace for
`nth_char_task` (`scratchspike.hpp`):

```
#  2  #4   [ UNCOMBINE  z b l o r  UNCOMBINE_END ]   =  l  EOS
└── prompt ──┘          └─ injected, MASKED ──┘        └ graded ┘
```

**Status: the mechanism works; the reasoning it enables is capacity-bound.** The `nth_char` / `roundtrip`
tasks pass, and — tested directly on the content question (`content_contains_reason_task`: resolve *every*
slot, then select the one whose spelling contains the queried char) — the resolve **protocol is learned
flawlessly**: the model reliably emits `<slot> UNCOMBINE` and the interceptor injects every spelling into
the stream, even for held-out OOVs (verified by dumping the generated traces). But the *downstream*
char-match-and-select is model-capacity-bound at d128/4-layer:

- **K=2** (chance 0.50): held-out **~0.72** — **works and generalizes**; the exact, learn-nothing route
  delivers. Sample: `% b → S0=ezd S1=bvk(has b) → answer S1` ✓.
- **K=3** (chance 0.33): **collapses to a degenerate constant slot** (~chance) — the 3-way match exceeds
  this model's capacity, so it takes the always-`S0` shortcut even with all three spellings in the stream.

So Route A is **not mechanism-limited** (resolution is perfect at both K); its ceiling is the model's
capacity for the multi-way match — the **same wall Route B hits below**.

- **Cost:** O(K × word-length) tokens — you must expand *every* slot you want to inspect.
- **Signal:** full, exact spelling.
- **Caveat:** on *scratch bindings* (read the spelling from context) even single-slot resolution is only
  partially solved cold at d128 (~0.31–0.46), weaker than the in-vocab spelling spike (0.97, productionised)
  because that one recalls a spelling baked into weights rather than reading it from context.

### Route B — content-derived slot embeddings (bake content into the vector)

Instead of `#4`'s **fixed** embedding row, **build its embedding from its fragments** so the content
leaks into the vector and the model can answer in **one hop, no resolution**. Two encoders tried:

- **mean-pool** — average the fragment vectors. Averaging blends everything to grey mush; the
  "contains-K" signal is diluted away. → **0.325** (chance ≈ 0.333).
- **CharEncoder** — a *learned* per-fragment transform `relu(W·row)`, summed. Keeps each character's
  *presence* separable. → **0.392** — above chance and above plain. Trained with AdamW; SGD failed only
  because it could not keep pace with the model's own AdamW (fair-test lesson).

- **Cost:** O(1) — one token already carries the signal.
- **Signal:** compressed / lossy (presence, first-letter) — not a full spelling.

**The `content_contains` / `content_select` tasks deliberately omit `append_resolve`** — they force a
single-hop answer straight off the embedding, precisely to isolate the question "can the slot's embedding
*by itself* carry content?" That is why those tasks sit at chance without Route B, and why the mean-pool
→ CharEncoder lift is the meaningful result there. It is **not** that masking "stopped working"; it is
that this probe was built to deny the model the uncombine step on purpose.

### Which route when

They are complementary, not substitutes — and they share one ceiling:

| | Route A: uncombine + mask | Route B: content-derived embedding |
|---|---|---|
| How | model spells the slot back out, then reads it | slot embedding built from its fragments; answer directly |
| Inference cost | O(K × word-length) tokens | **O(1)** token |
| Signal | full, exact spelling | compressed (presence / first-letter), lossy |
| Mechanism | ✅ flawless (resolves every slot, held-out too) | ✅ wired + FD-verified |
| Content reasoning @ d128 | K=2 ✅ 0.72 · K=3 ✗ collapses (~chance) | K=3 mean-pool ✗ · CharEncoder ✓ 0.392 |

The distinction that **is** real is **inference cost** (O(1) glance vs O(K·len) spell-out) and **signal
fidelity** (exact vs lossy). The distinction that turned out **not** to separate them is the reasoning
ceiling: **both routes hit the same capacity wall** on the multi-way (K=3) content match at d128 — Route A
by collapsing to a constant slot, Route B by barely clearing chance. The spellings being *present* (Route A)
is no easier for the model than the content being *embedded* (Route B); the bottleneck is the composed
match-and-select, not how the content is delivered. **Scale is therefore the deciding experiment for both.**

Want a cheap content-*glance* → Route B. Want the exact spelling → Route A. Want K≥3 content reasoning to
clear chance → a bigger model (the open question).

---

## Scorecard (d128 spike, cold)

| Capability | Task | Result |
|---|---|---|
| Carry a slot as a symbol | associative `select_task` | ✅ ~1.000 |
| Disambiguate among slots | multi-binding | ✅ works |
| Expand a slot (Route A) | `roundtrip` / `nth_char` | ◻ partial (~0.31–0.46, rising) |
| Model drives the binding | `define_task` (emit combine) | ✗ weak (~0.16) |
| Content, plain slot | `content_contains` (no Route A/B) | ✗ chance |
| Content, mean-pool (Route B) | `content_contains` | ✗ chance (dilution) |
| Content, CharEncoder (Route B) | `content_contains` | ✅ above chance (0.392, modest) |
| Route A, presence check | `content_contains_reason` NONE/ALL | ✅ **~1.00** (K=2 and K=3) — genuinely checks every word |
| Route A, localization (plain) | `content_contains_reason` ONE (which slot) | ✗ **~random** (survives 2× scale, 8 heads, ONE-heavy, plain CoT) |
| Route A, localization (**local grounding**) | CoT + local query restatement | ✅ **1.000 held-out K=3** — the fix |

**Read the two content rows together:** Route A's resolve protocol is perfect at both K (the spellings are
in the stream). The reasoning on top **decomposes**: the model learns the *presence* check but not the
*localization* — see the next section. Route B fails from the other side (a lossy embedding). The lever is
scale/architecture on the pointer step, not the delivery mechanism.

## What isn't working — the decomposition

The single-hop scorecard rows (`content_contains`, ✗ chance) and the first Route-A runs looked like "the
model can't reason about which slot." Adding the **ALL** and **NONE** cases (`content_contains_reason` now
mixes ONE / NONE / ALL) cornered *exactly* which step fails.

Why those cases matter: the original task guaranteed *exactly one* word matched, so K=2 was solvable by
**elimination** — "not in word 0 ⇒ it's word 1" — a single check that never looks at word 1 and does **not**
compose to K=3. (The old K=2 "0.72" was this shortcut, not real reasoning.) Once NONE/ALL are possible, "not
in word 0" no longer determines the answer, so the model is forced to *actually test every word*.

Result (d128), split by outcome:

| Outcome | What it needs | Held-out result |
|---|---|---|
| **NONE** — no word has the letter | verify absence in *all* words | ✅ **~1.00** |
| **ALL** — every word has it | verify presence in *all* words | ✅ **~1.00** |
| **ONE** — one word has it → name *which* | verify + **localize + point to that slot** | ✗ **~random** (~0.45 K=2, ~0.25 K=3) |

So the model **now genuinely checks every word** (NONE/ALL are perfect — the point of adding them) — but it
still can't **localize**: point to *which* slot. Told to (an ONE-heavy 60/20/20 mix), it outputs a slot but
picks almost at random; given an equal mix it doesn't even try, falling back to a "present→ALL, absent→NONE"
global-OR heuristic (~0.67 overall, but ONE-rate ~0.05).

### Two probes that pin the wall down further

**Scale doesn't buy it.** Re-running the ONE-heavy regime at **d256** (4L/8H, LR-scaled) leaves the ONE-rate
at ~random (0.48 K=2 / 0.30 K=3) — unchanged from d128. 2× width and 8 heads don't help.

**A chain-of-thought scaffold doesn't buy it either.** Idea: have the model emit a per-slot verdict (`+`/`−`,
"does *this* slot contain the char") right after resolving each slot, so the final answer is just "copy the
slot that got a `+`". The final copy *does* work when the verdicts are right — but the verdicts themselves
come out wrong (`% t → uwqcos:+ tzi:−`, both inverted). Why: emitting slot *i*'s verdict requires isolating
*that* slot's just-injected bytes from everything else — the **same** localization it can't do. So ONE-rate
stays ~random (0.44 K=2 / 0.25 K=3).

### The diagnosis, and what fixed it: local grounding

The wall looked like **binding a content-match to a specific slot's byte-region**: the model does a *global*
scan ("is the char anywhere?" → NONE/ALL perfect) but not a *segment-local* one ("is the char in *this*
word's bytes"). It survived 2× scale, more heads, ONE-heavy weighting, and the CoT verdict scaffold — so it
seemed to need much more model.

It didn't. The real cause was narrower: the model couldn't **hold the queried char in mind while scanning
segments listed far earlier** — a *non-local binding* problem, not segment isolation. The test: **restate the
queried char immediately before each verdict** (`… <bytes_i> <c> <verdict_i>`), so the check is over adjacent
tokens, and drop the now-redundant front slot-list. Result at d128:

| Variant | K=2 `one`-rate | K=3 `one`-rate (overall) |
|---|---|---|
| plain CoT | ~0.44 | ~0.25 |
| **local-query CoT** | **0.91** | **1.00** (held-out **1.000**) |
| control (drop front list, *no* restatement) | — | ~0.35 (rules out the confound) |

The traces are flawless and generalize to held-out OOVs:
`want S2 | % z … cstk z− … uwqcos z− … tzi z+ = S2` ✓. The control — dropping the front list *without*
restating — stays at ~random, so **the restatement (locality), not the shorter context, is the driver**.

**Conclusion:** localization is *not* a capacity wall — the d128 model can do it perfectly. It just needs the
reasoning **grounded locally**: put the query next to the data it's compared against. Scale/heads/CoT-alone
don't help; local grounding does.

## Status — the scale run

All the scorecard numbers are at **d128** (4 layers, 4 heads, head-dim 32) — the original spike size.
Because the K=3 wall is a capacity limit, the live question is whether a bigger model clears it. Same
`content_contains_reason` curriculum (Route A, no engine changes), K=2 and K=3, 3000 steps each.

**Run 1 — d256, auto-sized to 8 layers × 2 heads (same lr/steps recipe): the naive scale-up REGRESSED.**

| | d128 (4L/4H) | d256 (8L/2H) |
|---|---|---|
| K=2 held-out | ✅ 0.72 | ✗ **0.43** (collapsed to a constant slot) |
| K=3 held-out | ✗ 0.33 (chance) | ✗ 0.33 (chance) |

Making the model bigger *the naive way* not only failed to clear K=3 — it **lost the K=2 win**. This is
almost certainly a **training/config confound, not a ceiling**: the auto-sizer traded width for depth
(8 layers) but gave only **2 attention heads**, and it ran at the d128-tuned recipe. Two suspects: (1) 2
heads is too few **parallel comparison lanes** for a K-way match — the very thing this task stresses; (2) an
8-layer model is much deeper and likely **under-trained / needs warmup** at 3000 steps, so it fell into the
degenerate always-`S0` basin the shallower model escaped. Lesson: bigger ≠ better *for free* — shape and
recipe matter, and this task is head-count-sensitive.

**Run 2 — d256, pinned to 4 layers × 8 heads (head-dim 32 — the *scaled working shape*): ALSO collapsed.**
K=2 held-out flat at ~0.48 (never rises above chance), always-`S0`; K=3 chance. So the *scaled version of
the exact shape that worked at d128* fails too — which **rules out head-count/shape** as the cause. Both
d256 runs collapse to the degenerate basin at K=2, and the drilled curve is stuck from step 0 (no learning
at all, vs d128 which climbed to 0.79). The common factor is **2× width at the same recipe** → the
d128-tuned learning rate does not transfer.

| | d128 (4L/4H) | d256 (8L/2H) | d256 (4L/8H) |
|---|---|---|---|
| K=2 held-out | ✅ 0.72 | ✗ 0.43 | ✗ 0.48 |
| K=3 held-out | ✗ 0.33 | ✗ 0.33 | ✗ 0.33 |

This is the well-known **hyperparameters-don't-transfer-across-scale** problem (cf. muP): a 2×-wider model
generally wants roughly **½ the learning rate**.

**Run 3 — d256 (4L/8H) at half the learning rate (0.003 → 0.0015): recipe confound CONFIRMED and FIXED.**
One variable changed, and it cleanly separates recipe from capacity:

| | d128 (4L/4H, lr .003) | d256 (4L/8H, lr .003) | d256 (4L/8H, lr .0015) |
|---|---|---|---|
| K=2 held-out (peak) | ✅ 0.72 | ✗ 0.48 (collapsed) | ✅ **0.83** (beats d128) |
| K=3 held-out (peak) | ✗ 0.33 | ✗ 0.33 | ✗ **0.33** (still collapsed) |

- **K=2 recovered** to a peak of **0.83** — *better* than d128. So the earlier d256 collapses were a mistuned
  optimizer, not a ceiling. (It peaks around step 1500 then overfits/decays, so the test now reports **best**
  held-out, not last.)
- **K=3 still collapses** even with the corrected LR — always-`S0`, stuck from step 0.

**Verdict.** Doubling width to d256 (done right) **recovers and slightly improves K=2 but does not reach
K=3**. Two lessons are now baked into the test (`scratchspike_engine_tests.cpp`): `kLr` **auto-scales
~1/width** (`0.003·128/D_MODEL`, validated at d128/d256) so a scale run isn't silently sabotaged by a stale
LR; and the report tracks **peak** held-out (the capability can peak then overfit).

> **Note — these three runs predate the ALL/NONE decomposition above.** They were on the *old* one-match
> regime, where "K=3 collapses" actually meant *localization* collapses (the K=2 "win" was the elimination
> shortcut). Follow-up settled it: scale does **not** buy localization (d256 ONE-heavy left the `one`-rate at
> random), and neither does a plain CoT scaffold — but **local query restatement does** (held-out 1.000 at
> K=3, d128). See [What isn't working → what fixed it](#what-isnt-working-the-decomposition). So the lever was
> never scale; it was grounding the reasoning locally.

## Where this is going

The content-derived embedding is wired as a **spike with no model-format change**: the learned `enc_w`
`[C,C]` projection rides in `ScratchBindings` (not the model param arena), threaded through `op_embed` /
the Embed backward / `forward_one`. The single caveat (documented in `scratch_slots.hpp`): `enc_w_grad`
has no per-thread reduction, so a multi-threaded `train_batch` would race — training the encoder is
single-threaded for now.

Next steps, in order:
1. ~~Crack localization~~ **— done.** Local query restatement solves it (held-out 1.000 at K=3, d128; the
   confound control rules out context-length). The takeaway generalizes: **ground the reasoning locally** —
   put the query next to the data. Follow-ups: confirm at higher K (4–6 candidates), and fold the local-CoT
   trace shape into the production `scratchspike` blend-schedule curriculum so real models learn to answer
   content queries over scratch slots by resolving + locally checking. (No engine changes — all
   masked-training curriculum.)
2. **Promote `enc_w` to a model param** (`PARAM_LAYOUT` bump + per-thread grad reduction) — ends both the
   single-thread limit and the race, unlocking multi-threaded training + real `scratchspike` integration
   for Route B, and lets Route B ride the same scale run.
3. CUDA embed branch (parity-gated), gen wiring (`ScratchTable::to_bindings`), reserved-range growth, and
   an **order-aware** encoder (the first-letter task is missed by both mean-pool and the order-agnostic
   CharEncoder).

## Status update (2026-07-16) — what the list above got right, and what has since shipped

The next-steps list above is left as written (it was the honest plan at the time); here is what actually
happened, so a reader doesn't chase items that are already done or superseded:

- **The order-aware encoder (item 3's last clause) is DONE and is now the production default.** Three
  order-sensitive candidates were spiked the same day (`Hash` RoPE-style positional rotation, `ConvPool`
  char-CNN, `HRR` circular-convolution role binding — all in `scratch_slots.hpp`'s `SlotEncoding`); **HRR
  won decisively** (mean held-out 0.744 vs MeanPool's 0.356 on the content-select probe, chance 0.333,
  consistent across seeds) and — because it needs **no learned params** — it skipped item 2's `enc_w`
  promotion prerequisite entirely and went straight into production: it is the `content_embed` default a
  blend-schedule config selects (`"content_embed": "hrr"`). Route B is live, without the param-layout work
  item 2 anticipated (that work is still real, but only for the learned encoders, CharEncoder/ConvPool,
  which remain spike-only). Gen wiring (item 3) also shipped: `gen_stage` reads the pinned
  `blend_schedule.json`, drives `ScratchTable::to_bindings`, and composes bound slots at decode time.
- **"Reserved-range growth" (item 3) took a different shape than a marker-block bump: an unbounded
  PERSISTENT range at ids `>= VOCAB`** (`PersistentBindings`, `scratch_slots.hpp`) — no tokenizer-format
  change, engine-dispatch-tested, plus a first trained attend-only resolve spike at K=6/30 concurrent
  slots (`persistent_scratchspike.hpp` / `[.persistent_scratchspike]`) showing real but unsaturated
  held-out generalization well past the ephemeral pool's 6-slot cap. One hard constraint discovered and
  worth internalizing: a persistent id has **no logit column**, so the model can *consume* one (attend to
  it, resolve it) but can never *emit/choose* one — the associative SELECTION half of this doc's task
  family does not transfer to that range (open research question, tracked in project memory as
  `persistent-slot-selection-problem-backlog`). Full record + measured numbers:
  `docs/DETERMINISTIC_MECHANISMS.md`'s persistent-slot entries.
- The CUDA embed branch (item 3) remains NOT done — the scratch/content-embed path is still CPU-only, now
  bridged in production by the hybrid CPU/GPU source-routed training split rather than a device kernel.

## The end-state: word-level context via sentinel-pair addressing (design, 2026-07-16)

**The goal, stated plainly**: a system that works on *words* (whole semantic units) instead of multi-token
fragments. The motivating domain is **code**, where identifiers are long compounds
(`getUserAccountBalanceById`, `parse_scan_state`) that (a) cost many tokens per mention, (b) repeat
heavily within a file, and (c) are exactly the order-sensitive compounds a permutation-invariant composer
would confuse (`overtake`/`takeover` — see the order-sensitive encoding section of
DETERMINISTIC_MECHANISMS.md). Everything above (the 6-slot pool, content-derived embeddings, the
persistent range) is infrastructure toward this; this section is the coherent end-to-end shape.

**The mechanism: a `<|scratch>` sentinel PAIR.** A single reserved, in-vocab sentinel token followed by an
ordinary index token: `<|scratch> a`. In context, the sentinel changes how the NEXT token embeds — instead
of letter-a's static embedding row, position t gets the **composed content vector** (HRR over fragments,
`encode_slot` — the machinery already in production) of whatever entity handle `a` is bound to in this
context's binding table. This is not a new kind of trick for this codebase: `casing.hpp`'s own markers
already carry documented **after-effects** (`TOK_ODQUOTE`/`TOK_SPELL_START`: the content immediately
afterward is interpreted differently) — the pair applies the established pattern at the embedding layer.
See also `docs/TOKENIZER_REVIEW.md` §5 (the prior-art-grounded special-token extension design this
revives).

Why the pair beats both existing ranges at what they can't do:
- vs the **6-slot in-vocab pool**: one reserved id + the whole index-token space = unbounded addressable
  entities (26 letters alone beat 6 slots; any-token indexing scales to thousands), and `a`-after-sentinel
  never collides with literal `a`.
- vs the **persistent (`>= VOCAB`) range**: the pair is EMITTABLE — both tokens have logit columns, so the
  model can *choose* to reference entity `a` by ordinary next-token prediction. The persistent range stays
  as the harness-side store; the pair's index dereferences INTO it (handle -> persistent binding), making
  the pair the model-facing addressing mode over the same unbounded table. Harness-injected,
  consumption-only references can keep using bare persistent ids (1 token instead of 2).
- The engine change is small and local: the 3 embedding-dispatch sites (`op_embed` forward/backward,
  `forward_one`) already substitute composed vectors per position; the pair adds an `ids[t-1] ==
  TOK_SCRATCH` condition alongside the existing `is_scratch_slot`/`is_persistent_slot` checks.

**Two honest unknowns, and the plan gates on both:**
1. *Output-side selection is an association skill, not free pointer semantics.* Choosing a handle scores
   against static lm_head rows (tied embeddings: literal letter rows) — the input-side override doesn't
   help the output side. `select_task` proved the association skill at K<=6 with dedicated ids; pairs at
   larger K need their own spike. Fallback if association walls: a pointer-style head (score handle
   logits against the bound entities' composed vectors) — real engine work, held in reserve, not assumed.
2. *Reserved-id headroom is currently ZERO* (`TOK_RESERVED_4..9` == the 6-slot pool; a marker-count bump
   invalidates every checkpoint). The pool's deprecation (below) reclaims exactly what's needed — and a
   SPIKE needs no format change at all: commandeer `TOK_RESERVED_9` as the sentinel, leaving a K<=5 pool
   for the comparison arm.

**The phased plan (each phase gated on the previous one's evidence):**
- **Phase 0 — complete the compaction capability (in flight).** The 4-encoder shootout + a longer-budget
  saturation run on the persistent attend-only resolve spike. The current approach compacts context and
  needs finishing regardless of what follows.
- **Phase 1 — handle-reference A/B on op chains (no vocab change).** Reimplement `op_curriculum`'s chain
  references as plain-vocab ordinal handles (the graded raw-`S0` emission is a synthetic-template
  artifact — see DETERMINISTIC_MECHANISMS.md's "interface revision slated" note) and A/B against the
  S0-emission form. This proves the handle-association skill in the cheapest possible setting.
- **Phase 2 — sentinel-pair spike (TOK_RESERVED_9, no format bump).** Embedding-override engine change +
  a bind-K-entities/select-and-resolve-via-pairs curriculum; A/B against dedicated-id `select_task` at
  K<=5, then scale K well past it. This is the novel step — pairs give BOTH unbounded addressing AND
  model-driven selection, which neither existing range has.
- **Phase 3 — deprecate the 6-slot pool; reclaim the ids.** Gated on phases 0-2 proving out. Frees
  `TOK_RESERVED_4..9`: the sentinel gets a permanent id, and the remaining ids unblock the op-marker
  dedicated-token backlog (`TOK_OP_START`/`END`) that has been waiting on this exact headroom.
- **Phase 4 — word-level code context.** Apply to a code corpus: harness binds identifiers at first
  mention (repeatspike proved harness-driven mid-stream binding wins decisively), later mentions become
  pairs; ties into the multi-token pattern-compression backlog (code idioms as bound spans, not just
  single identifiers). This is where "the model works on words" pays off: common vocabulary stays
  ordinary tokens (the JOIN tokenizer already compresses the static half), per-context compounds become
  single semantic units with content-derived, order-sensitive vectors, and `uncombine` remains the exact
  bridge down to characters when spelling matters.
