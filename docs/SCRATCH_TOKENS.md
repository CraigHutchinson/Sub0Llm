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

The last mental step — **read several spelled-out words and answer a question about which one contains a
given letter** — splits into two sub-skills, and we now know exactly which one is the wall:

- ✅ **Checking** ("is the letter in *any* of them? in *all* of them?") — **works, nearly perfectly.** Ask
  the model "does any word contain a Z?" and it reliably says yes/no, having genuinely looked at every word.
- ❌ **Localizing** ("*which* word contains it?") — **this is the wall.** The model can tell the letter is
  *there*, but can't reliably say *which* word it's in — it points almost at random.

And the crucial detail: this is **not** a plumbing failure. The model *does* correctly spell every word out
into the stream — the letters are right in front of it — and it can even do the yes/no presence check. What
it can't do is the **pointer** step: match the letter to a word and then name *that word's* nickname. That's
a lookup-and-point operation ("find the row that matches, report its label"), and a small model can't nail
it. **It's a brain-size limit on one specific operation, not a plumbing limit** — so the open question is
whether *scale*, *more attention heads* (parallel pointer lanes), or a *pointer-focused curriculum* buys it.
(How we cornered this: see [What isn't working](#what-isnt-working-the-decomposition) and
[Status](#status--the-scale-run).)

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
| Route A, localization | `content_contains_reason` ONE (which slot) | ✗ **~random** slot-guess (the real wall) |

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
global-OR heuristic (~0.67 overall, but ONE-rate ~0.05). The wall is a **pointer / argmax-over-words**
operation — *find the row that matches and report its label* — not checking, and not specific to K=3.

Modest numbers are expected: d128 is capacity-starved. Scale/architecture is the amplifier.

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
> shortcut). The refined open question is therefore **not** "does K=3 work at scale" but **"does *localization*
> (pointing to the right slot) clear** with scale, more heads (parallel pointer lanes), or a pointer-focused
> curriculum?" The immediate next probe is the ONE-heavy regime at d256 (LR-scaled) — does the per-kind
> `one`-rate rise above random?

## Where this is going

The content-derived embedding is wired as a **spike with no model-format change**: the learned `enc_w`
`[C,C]` projection rides in `ScratchBindings` (not the model param arena), threaded through `op_embed` /
the Embed backward / `forward_one`. The single caveat (documented in `scratch_slots.hpp`): `enc_w_grad`
has no per-thread reduction, so a multi-threaded `train_batch` would race — training the encoder is
single-threaded for now.

Next steps, in order:
1. **Crack localization** — the one open wall. The model checks every word (NONE/ALL solved) but can't point
   to *which* slot. Probes, cheapest first: (a) the ONE-heavy regime at **d256** (LR-scaled) — does the
   `one`-rate rise above random? (b) **more heads** (parallel pointer lanes) — the pointer/argmax step may be
   head-count-bound; (c) a **pointer-focused curriculum** or an easier localize-only task to bootstrap the
   skill. All need **no engine changes** (plain masked training on `build_dataset_contains_reason`).
2. **Promote `enc_w` to a model param** (`PARAM_LAYOUT` bump + per-thread grad reduction) — ends both the
   single-thread limit and the race, unlocking multi-threaded training + real `--scratch-mix` integration
   for Route B, and lets Route B ride the same scale run.
3. CUDA embed branch (parity-gated), gen wiring (`ScratchTable::to_bindings`), reserved-range growth, and
   an **order-aware** encoder (the first-letter task is missed by both mean-pool and the order-agnostic
   CharEncoder).
