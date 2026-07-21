# Scratch tokens: framing, criteria, and the open mathematics of packing/unpacking

Scope note: this is a **requirements-and-open-questions** document, not a status report. It exists so
that anyone picking this line of work back up — including a future session with no memory of how we got
here — has the full criteria set a scratch/packing mechanism needs to satisfy, and the actual mathematical
question this work has converged on, stated precisely. It complements
[SCRATCH_TOKENS.md](SCRATCH_TOKENS.md) (the mechanism's implementation history and what's been built/
proven) and [FACTSPIKE.md](FACTSPIKE.md) (the factual-retrieval experiment thread that surfaced the
attention-capacity finding this document centers on). Written 2026-07-21, after parking the training-
schedule ("Pack-Aware Training") axis and finding the first quantitatively clean signal on a different
axis entirely.

---

## The core problem, in one image

If the training vocabulary happened to contain **"Rumplestiltskin"** as a single token, the model would
attend to it and use it correctly without any special machinery — it's just one embedding row, looked up
and processed like any other token. Nothing about that row is "special"; everything the model knows about
that word lives in how that one row interacts with the layer weights during an ordinary forward pass.

But normally it doesn't get a token. It tokenizes as something like `Rum` + `ple` + `stilt` + `skin` — and
critically, **the "knowledge" about this word is not resident in any one of those four rows**. It only
exists as the *process* of running those four pieces through the model's layers together: each layer, the
later pieces attend back over the earlier ones (each already refined by the layer before), and only by the
last piece has anything resembling "the model's understanding of this word" actually formed. The knowledge
is genuinely **sparse and distributed** — spread across the interaction between four embedding rows and
every layer's attention/FFN weights, not stored anywhere as a single thing.

**A scratch/packing mechanism's actual job, stated precisely**: find where that sparse, distributed
knowledge lives, and represent it — cached, compactly, re-insertable into a fresh context — well enough
that a later mention costs close to one token instead of paying the full multi-piece attention process
over again. Every axis below is really a different angle on how faithfully that job gets done, and the
open mathematics section is about whether it can be done close to losslessly instead of by approximation.

---

## The many axes a scratch-token mechanism needs to satisfy

Each row: what it means, current status, where it was tested. Status current as of 2026-07-21.

| # | Axis | What it means | Status | Evidence |
|---|---|---|---|---|
| 1 | **Symbolic carrying** | Hold a stable identity across mentions; be referred back to like a pronoun | ✅ solved | `SCRATCH_TOKENS.md` associative `select_task`, ~1.000 |
| 2 | **Disambiguation** | Multiple bound slots coexist in one context without collision | ✅ solved | `blended_capstone` (wordspike + op_curriculum share the pool, collision-free) |
| 3 | **Exact resolution (Route A)** | Spell the slot back out on demand, byte-perfect | ◻ partial, capacity-bound not mechanism-bound | `SCRATCH_TOKENS.md` — resolve protocol is flawless even for held-out OOVs; downstream K-way match is what caps out |
| 4 | **Single-hop content access (Route B)** | Answer a question about the packed content with zero resolution step | ◻ partial | `SCRATCH_TOKENS.md` — MeanPool fails (dilution), CharEncoder clears chance modestly |
| 5 | **Order-sensitivity** | `overtake` vs `takeover` must compose differently | ✅ solved | HRR (circular convolution, position-bound roles) beats MeanPool decisively; production default |
| 6 | **Factual/world-knowledge retrieval through the packed form** | A fact learned via ordinary text about a subject must still be retrievable when that subject is introduced *only* via a packed slot | ◻ partial, real gap to baseline | `FACTSPIKE.md` Phase C: peak 0.56 vs 0.125 chance (real signal) vs. baseline's 1.00 (real gap) |
| 7 | **Zero-shot generalization** | Works for a subject/word never itself seen packed during training | ◻ ambiguous at small n | `FACTSPIKE.md` held-out arm — peak 0.33 (n=3), statistically indistinguishable from noise at this sample size |
| 8 | **No shortcut / leakage** | The above isn't secretly memorization or an in-document/in-training tell | checked, ongoing discipline | `FACTSPIKE.md` Phase 0 piece-balance check; each new curriculum re-applies this discipline |
| 9 | **Attention-capacity preservation** | The packed form preserves enough of the multi-hop computation a real multi-piece span gets | 🔴 **ACTIVE, the current live question** | `FACTSPIKE.md` hidden-state diagnostic: Pearson r(piece_count, cos_sim) = **-0.614** |
| 10 | **Cost efficiency** | O(1) token / O(1) KV-cache slot per mention, not O(n) | THE reason this mechanism exists — must not be silently given up by any fix | `SCRATCH_TOKENS.md`'s own framing: 20 mentions of a 5-piece OOV = 100 tokens without packing |
| 11 | **Model-emittable addressing** | The model itself can *choose* to reference a packed handle, not just consume harness-injected ones | open, gated | `SCRATCH_TOKENS.md` sentinel-pair Phase 2 — selection through the pair embedding is a wall; solved instead via Route A (resolve + local-CoT), not embedding override |
| 12 | **Training-exposure requirements** | The model needs *some* exposure to reading packed forms to use them at all — how much, shaped how | investigated, one sub-question now parked | `FACTSPIKE.md` Phase B (no exposure = no packed retrieval) through Phase F (PAT axis, parked — see below) |
| 13 | **Engine/format compatibility** | No breaking checkpoint changes; minimal, additive engine surface | a standing constraint on every design considered | Every mechanism shipped so far (HRR, persistent range, sentinel-pair) has honored this |
| 14 | **Scale-behavior** | Does a bigger model compose better for free, or does the recipe need to change too | open, and non-trivial | `SCRATCH_TOKENS.md`'s own d128→d256 runs: naive scale-up *regressed* K=3 localization; recipe (LR) mattered more than width |
| 15 | **Generalizes past this experiment's own subjects** | A fix can't just work for the specific words a training run happened to drill on | the reason PAT was parked | see below |

Axis 9 is where the sharpest, cleanest signal in this whole line of work currently sits, and the rest of
this document is about it.

---

## Why the training-schedule axis (Pack-Aware Training) was parked

Phases D/E/F of `FACTSPIKE.md` chased whether a better-shaped training signal (task-contingent gradient,
QAT-style) could close the gap on axis 6. It produced a real, monotonically improving trend on the scratch
arm (0.00 → 0.11 → 0.22 across three fixes) but also produced a result — baseline degrading as drilled
subjects got *more* plain-text exposure, the opposite of what the working hypothesis predicted — that
doesn't resolve into a clean causal story at single-seed scale. More importantly, even a fully successful
result on that axis would only prove the fix works **for the specific subjects one training run happened
to drill on**. It says nothing about whether an arbitrary word sharing these same piece tokens, encountered
for the first time, would pack correctly — and that generalization is the actual requirement (axis 15).
Training harder against a fixed axis 6 test doesn't move axis 9 at all; it just makes the model more
confident inside a fixed I/O shape. Parked, not abandoned — worth revisiting once axis 9 has a real answer,
since a better packing mechanism might make training pressure on it worthwhile again.

---

## Two fundamentally different mechanisms for "packing"

### (A) Embed-level composition — what's built

`encode_slot()` (`scratch_slots.hpp`) computes a fixed, parameter-free (for HRR/MeanPool) formula —
average, or circular-convolution-bind-by-position — over the word's **raw, pre-transformer embedding
rows**, and injects the result as a single position's *input* embedding. Everything after that is
`N_LAYERS` of ordinary processing on one position, with no sibling positions of "itself" to attend back
over at all.

This is a **crude, one-shot approximation** to the real process. It tries to guess, with a fixed formula,
what the outcome of `N_LAYERS × (n−1)` real sibling-attention hops would have been — computed *before* any
of those hops actually happen. Axis 9's finding (r=-0.614 between piece count and final-representation
similarity) is exactly consistent with this: the more real computation the approximation is standing in
for, the worse it does.

### (B) KV-trace memoization — the mathematically well-founded alternative, not yet built

Look at what downstream attention actually reads. In this engine's own `KVCache` (`backend_cpu.cpp`):

```cpp
struct KVCache {
    std::vector<float> k, v;   // [N_LAYERS][SEQ_LEN][D_MODEL], flat
    float* krow(int l, int pos) { return k.data() + ((l * SEQ_LEN) + pos) * D_MODEL; }
    float* vrow(int l, int pos) { return v.data() + ((l * SEQ_LEN) + pos) * D_MODEL; }
};
```

A later position doesn't attend to "the word." At every layer `l`, it attends to **each earlier piece
position's own `(K_l, V_l)` pair independently** — a real multi-piece word offers `N_LAYERS × n` distinct
`(K, V)` targets, each shaped by everything before it in that specific context. *This* is where the sparse
knowledge actually resides: not in a vector, but in this per-layer, per-piece trace.

The mathematically exact answer to "how do I reuse this cheaply" is **memoization, not approximation**:
run the real forward pass over the word's pieces once (whenever it's first bound, or precomputed), capture
the resulting `(K_l, V_l)` trace at every layer, and splice that trace directly into `KVCache` wherever the
word is mentioned again — reusing the *actual* computation instead of guessing at its outcome. This is not
speculative engineering: this engine already has the primitive one step away from it. `SCRATCH_TOKENS.md`
documents that `KVCache` is already a flat, position-indexed buffer with no separate length counter, and
`wordspike`'s live generation splice already does a retroactive `KVCache` overwrite (`ctx.resize` back,
re-`feed()`) — a same-context splice. What's missing is *cross-context* reuse: caching a trace once and
splicing it into a *different* context later.

---

## The honest tension: fidelity vs. compactness

- **Full n-position KV splice (mechanism B, literally)**: exact, zero approximation. But it costs
  `N_LAYERS × n × D_MODEL` floats and `n` attention targets per mention — the *same* asymptotic cost as
  just leaving the word unpacked. It solves axis 9 perfectly by giving up axis 10 (the entire reason this
  mechanism exists) entirely.
- **Current 1-position embed composition (mechanism A)**: fully compact (axis 10 ✅), but loses the
  multi-hop structure (axis 9's r=-0.614).
- **k intermediate positions** (already considered, and explicitly the wrong answer per your own read):
  a linear interpolation between the two above. Marginal recovery, real residual loss — not a principled
  fix, just a knob between two extremes that are each principled on their own terms.

The real question is whether there's a middle ground that isn't just "cut the loss in half arbitrarily,"
but is instead mathematically motivated by *where* the trace's information actually concentrates.

---

## Two candidate directions, in order of how soon they're buildable

### 1. Post-hoc per-layer KV pooling (nearer-term)

Let the real forward pass over the `n` pieces happen once — however triggered (first mention, harness
pre-encode) — and capture the **actual** per-layer `(K_l, V_l)` pairs, not the raw input embeddings. *Then*
compress `n → 1` **after** each layer's real computation (an attention-weighted pool, a small learned
compressor, or an HRR-style bind — but applied to genuine per-layer K/V outputs, not to pre-transformer
rows) for cache storage and later splicing.

This directly targets the mechanism-A vs mechanism-B distinction: the real multi-hop computation *has
already happened* by the time compression occurs, so compression only has to lose whatever's genuinely
redundant across the `n` pieces at each layer, not approximate the entire process from nothing. Testable,
concrete hypothesis: does this compress-after-computing approach show a **weaker** (closer to zero)
correlation between piece count and fidelity than mechanism A's r=-0.614, since the thing being lost is
narrower?

### 2. Landmark-style transparent expansion (more ambitious, most faithful)

Keep the full `n`-way per-layer trace cached out-of-band entirely — don't compress it at all. The visible
"scratch token" in the sequence is a **pointer**: when any layer's attention actually needs to read this
word, it transparently expands its attention window to include the cached `n`-way trace, rather than being
limited to one collapsed `(K, V)` pair. This doesn't compress the multi-hop structure at all; it compacts
the *visible sequence length* and avoids re-deriving the trace on every mention, while leaving attention's
actual reach into the real computation untouched.

This is not a novel idea in isolation — it's structurally what **Memorizing Transformers** (Wu et al.,
external kNN-retrieved memory over cached KV from earlier context) and **landmark attention** (a landmark
token triggers expanded attention into a cached block) already do in the wider literature. The distinctive
part here is that the cache entry is *per-word*, addressed by the scratch-slot binding table rather than
by document position — closer to a content-addressable memory than a positional one.

---

## Open mathematical questions, not yet answered

- **How context-sensitive is a word's own internal `(K, V)` trace, really?** `K_l`/`V_l` at position `p`
  are computed from that position's own residual-stream state, which itself depends on everything before
  `p` *in that specific context* via earlier-layer attention. If a word's own internal piece-to-piece
  composition is largely context-*insensitive* (plausible — a name's own spelling-composition process
  probably doesn't depend much on the surrounding sentence), precomputing its trace once and splicing it
  into arbitrary later contexts is a good approximation. If it's meaningfully context-sensitive, naive
  reuse introduces a real, currently-unmeasured error term. This is directly testable with the existing
  `hidden_state_cosine()`/`last_hidden_ptr()` infrastructure, extended to intermediate layers: compute the
  same word's trace under two or three *different* surrounding contexts and measure how much it moves.
- **What's the right canonical context to precompute a trace in** — isolation, a fixed carrier sentence, or
  the specific document it's about to be spliced into? Each has a different cost/fidelity trade.
- **Does the trace's information concentrate in particular layers?** If later layers matter far more than
  early ones for what downstream attention actually needs, a cheaper partial-depth cache might capture
  most of the value at a fraction of the storage. Testable by extending `last_hidden_ptr()`-style capture
  to every layer boundary, not just the final one — a small, additive engine change in the same shape as
  the one already built for the final-hidden-state diagnostic.
- **Is there an analytical bound to check the empirical r=-0.614 against?** HRR's own literature (Plate)
  gives a known result: retrieval fidelity from a bundle of `n` bound pairs degrades roughly like
  `1/sqrt(n)` in a `D`-dimensional space, with `D` providing the noise-averaging headroom — and that
  literature typically assumes `D` in the many hundreds to low thousands, well above this project's toy
  `D_MODEL=96`. Fitting the existing 9-point `(n, cos_sim)` dataset against a `1/sqrt(n)` curve (or against
  `D_MODEL` directly) is a cheap, purely analytical next step that doesn't require a single new training
  run — it would say whether the observed degradation is "textbook HRR crosstalk, nothing more" or whether
  something beyond that is also going on.

---

## What not to lose sight of while pursuing this

- **Cost efficiency (axis 10) is the entire reason this mechanism exists.** Any fix that reduces to "just
  keep the word unpacked" has quietly abandoned the point, however good its fidelity numbers look.
- **A fix must generalize past this experiment's own drilled subjects** (axis 15) — this is precisely why
  the training-schedule axis was parked in favor of this one.
- **Leakage discipline (axis 8) carries forward** into any new mechanism — a KV-trace cache in particular
  introduces a new place a shortcut could hide (e.g., if the "canonical context" used to precompute a
  trace ever overlaps with eval material), and would need its own version of Phase 0's check.
