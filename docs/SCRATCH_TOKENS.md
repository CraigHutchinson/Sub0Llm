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

**Status: works.** The `nth_char` / `roundtrip` tasks pass — this route is real and merged. It is the
"spell it out when you need it" path.

- **Cost:** O(K × word-length) tokens — you must expand *every* slot you want to inspect.
- **Signal:** full, exact spelling.
- **Caveat:** on *scratch bindings* (read the spelling from context) it is only partially solved cold at
  d128 (~0.31–0.46), weaker than the in-vocab spelling spike (0.97, productionised) because that one
  recalls a spelling baked into weights rather than reading it from context.

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

They are complementary, not substitutes:

| | Route A: uncombine + mask | Route B: content-derived embedding |
|---|---|---|
| How | model spells the slot back out, then reads it | slot embedding built from its fragments; answer directly |
| Inference cost | O(K × word-length) tokens | **O(1)** token |
| Signal | full, exact spelling | compressed (presence / first-letter), lossy |
| Status | **works** (nth-char / round-trip) | mean-pool ✗, learned CharEncoder ✓ (modest at d128) |

Want a cheap content-*glance* → Route B. Want the exact spelling → Route A.

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

Modest numbers are expected: d128 is capacity-starved. Scale is the amplifier for both routes.

## Where this is going

The content-derived embedding is wired as a **spike with no model-format change**: the learned `enc_w`
`[C,C]` projection rides in `ScratchBindings` (not the model param arena), threaded through `op_embed` /
the Embed backward / `forward_one`. The single caveat (documented in `scratch_slots.hpp`): `enc_w_grad`
has no per-thread reduction, so a multi-threaded `train_batch` would race — training the encoder is
single-threaded for now.

Next steps, in order:
1. **Promote `enc_w` to a model param** (`PARAM_LAYOUT` bump + per-thread grad reduction) — ends both the
   single-thread limit and the race, unlocking multi-threaded training + real `--scratch-mix` integration.
2. **A d448+ scale run** — d128 caps the content-match gain.
3. CUDA embed branch (parity-gated), gen wiring (`ScratchTable::to_bindings`), reserved-range growth, and
   an **order-aware** encoder (the first-letter task is missed by both mean-pool and the order-agnostic
   CharEncoder).
