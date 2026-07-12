# Scratch tokens — a per-context translation layer for out-of-vocabulary words

Scope note: this is a **feature explainer + status** for the scratch-token capability — the mechanism,
what works and what doesn't as worked examples, and the two distinct routes to "reasoning about a
slot's content." It complements [ROADMAP.md](ROADMAP.md) (where the corpus/architecture is headed) and
the project memory `scratch-tokens-context-translation-layer`. The spike code lives in
`include/sub0/scratchspike.hpp` (curriculum), `include/sub0/scratch.hpp` (the production binding table),
`include/sub0/scratch_slots.hpp` (the engine-facing foundation), and the content-embedding scaffold in
`src/backend_cpu.cpp`.

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
| Content, resolve-then-select (Route A) | `content_contains_reason` K=2 | ✅ **0.72** held-out (generalizes) |
| Content, resolve-then-select (Route A) | `content_contains_reason` K=3 | ✗ collapses to a constant slot (~chance) |

**Read the two content rows together:** Route A's resolve protocol is perfect at both K (the spellings are
in the stream); K=3 fails on the *match-and-select* step, not the resolution. Route B fails the same step
from the other side. The shared bottleneck is model capacity for the K-way content match — the lever is
scale, not the delivery mechanism.

Modest numbers are expected: d128 is capacity-starved. Scale is the amplifier for both routes.

## Where this is going

The content-derived embedding is wired as a **spike with no model-format change**: the learned `enc_w`
`[C,C]` projection rides in `ScratchBindings` (not the model param arena), threaded through `op_embed` /
the Embed backward / `forward_one`. The single caveat (documented in `scratch_slots.hpp`): `enc_w_grad`
has no per-thread reduction, so a multi-threaded `train_batch` would race — training the encoder is
single-threaded for now.

Next steps, in order:
1. **The deciding experiment — scale.** Both routes hit the same K-way-match wall at d128, so the open
   question for *both* is whether a bigger model clears K≥3 content reasoning. **Route A can be scale-tested
   immediately** — it needs **no engine changes** (plain masked training on `build_dataset_contains_reason`
   at a larger model); if K=3 clears chance there, the resolve-then-select route is production-viable as-is.
2. **Promote `enc_w` to a model param** (`PARAM_LAYOUT` bump + per-thread grad reduction) — ends both the
   single-thread limit and the race, unlocking multi-threaded training + real `--scratch-mix` integration
   for Route B, and lets Route B ride the same scale run.
3. CUDA embed branch (parity-gated), gen wiring (`ScratchTable::to_bindings`), reserved-range growth, and
   an **order-aware** encoder (the first-letter task is missed by both mean-pool and the order-agnostic
   CharEncoder).
