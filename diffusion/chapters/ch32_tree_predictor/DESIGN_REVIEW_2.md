# Ch32 — Design Review II: Out-of-the-Box Lenses

> Sequel to [`DESIGN_REVIEW.md`](DESIGN_REVIEW.md), which converged on the *conventional* synthesis
> (latent-word diffusion + char codec + gist). This pass deliberately leaves that comfort zone:
> domains not yet used, and framings that could **change** the design, not decorate it. Each idea is
> scored **[ACTIONABLE]** / **[REFRAMES]** / **[SPECULATIVE]**. Still PARKED behind the flat-diffusion
> training research — this widens the option space before we commit.

The bar: the conventional design treats the three levels as *separate modules* (codec / denoiser /
conditioner). The interesting question is whether some other field already solved **"many levels, one
object"** more elegantly. Several have.

---

## 1. Physics of language — criticality & MERA  **[REFRAMES — the big one]**
Lin & Tegmark (2017), *Critical Behavior in… Formal Languages*: natural-language mutual information
between two symbols decays as a **power law** in their separation, not exponentially. Text is
**"critical"** (scale-free long-range correlations). This is *exactly our long-range problem* stated in
physics terms — and flat models (RNNs, fixed-window transformers, our denoiser) have **exponential**
correlation decay, so they structurally **cannot** match a power law. That's why coherence keeps slipping
at distance; it isn't a bug, it's a representational mismatch.

Physics has a purpose-built answer: **MERA** (Multi-scale Entanglement Renormalization Ansatz, Vidal) —
a tensor network of alternating **disentanglers** (remove short-range redundancy) and **coarse-grainers**
(pool), stacked in **log-depth** layers. MERA represents power-law/critical correlations *efficiently*
because a correlation at distance `d` is carried by a path of only `~log d` coarse-graining steps. Map to
us: a token pair `d` words apart is connected through `~log d` levels of the hierarchy — so the gist
level reaches across the whole passage in log-depth. **This reframes the entire ch32 motivation:** the
hierarchy isn't a nice-to-have structural prior — it's the *known* way to get power-law (critical)
correlations that flat diffusion provably can't. → Architecture cue: alternate **disentangle (local
attention) + coarsen (pool)** with skip-state, log-depth — a "MERA transformer."

## 2. Holographic / Vector-Symbolic representation  **[ACTIONABLE — the literal "one object"]**
The README's hard requirement — *all three levels in one coherent representation* — is solved cleanly by
**Holographic Reduced Representations / Vector Symbolic Architectures** (Plate; Kanerva; Gayler). A
single fixed-width hypervector holds many role–filler pairs by **binding** (circular convolution `⊛`,
i.e. elementwise multiply in the Fourier domain) and **bundling** (addition):

```
s = (r_gist ⊛ v_gist)  +  (r_word ⊛ v_word)  +  (r_char ⊛ v_char)
v_word ≈ s ⊛ r_word⁻¹      # unbind by correlation; a "cleanup memory" denoises the recovered vector
```

One vector, three levels, **fixed width regardless of depth**, and *holographic*: every component is
smeared across all dimensions, so partial/noisy `s` still recovers a degraded answer (graceful
degradation built in — the property char-level had, now at the representation layer). Binding/bundling
are **cheap** (an FFT or elementwise mul). The wild-but-apt combination: **diffuse in hypervector space** —
the denoiser cleans a superposed `s`, and the iterative refinement *is* a cleanup-memory step that
progressively sharpens each bound level. → Add a `HyperVector` type (bind/bundle/unbind) as the
token-state representation feeding the existing transformer; the "residual stream" becomes a hypervector
stream. Risk: superposition capacity (≈ `dim / items`), needs a learned cleanup.

## 3. Renormalization Group — principled coarsening  **[ACTIONABLE — fixes the pooling objective]**
Our weakest spot is *how* `up`/pool should coarsen. Physics' **RG** says: coarse-grain by **integrating
out the fast/irrelevant modes, keeping the slow/relevant ones**, where "relevant" = *what governs
long-distance behaviour*. Operationally that's the **information bottleneck**: the word→gist pooling
should keep exactly the information **predictive of the rest of the passage** and discard spelling/local
detail. Mehta–Schwab showed stacked RBMs ≈ variational RG, so this is learnable. → Replace mean-pool /
naive summary-regression with a **predictive-information bottleneck** pooling objective: `g = Pool(words)`
trained to maximize `I(g ; future words)` while minimizing `I(g ; exact tokens)`. This is the *principled*
version of "gist = content words" and structurally prevents collapse.

## 4. Morphogenesis — grow text from a seed  **[REFRAMES generation order]**
Neural Cellular Automata (Mordvintsev, *Growing NCA*) and Turing **reaction–diffusion** generate complex
structure from a seed via **local update rules** — the global form **emerges**, never centrally planned.
Reframe generation: seed the canvas with the **gist** as a coarse field/gradient; apply **local**
update rules iteratively; prose and spelling **grow** to satisfy the local field. This is our iterative
denoising but with (a) explicit **locality + persistence** (a committed token *stays* and shapes
neighbours, like our §13.8c bootstrap, but as a first-class rule) and (b) **emergent** rather than imposed
hierarchy. Thematically resonant — we already "diffuse." → A concrete variant: condition each position's
denoising on a **gist gradient** interpolated across the canvas (so far-apart regions share a slowly
varying plan), and make commitment a CA-style local rule. Cheap to try on top of the current sampler.

## 5. Hierarchical control / Feudal RL — gist as a MANAGER  **[ACTIONABLE — trains level A]**
We had no training signal for the gist *conditioner* beyond reconstruction. **Feudal Networks**
(Dayan; Vezhnevets) give one: a **Manager** emits a goal vector at a coarse timescale; a **Worker** is
rewarded for moving the state toward that goal. Map: the **gist planner = Manager** (sets a goal per
span, wide context), the **word denoiser = Worker** (rewarded when conditioning on the goal *improves*
its masked-token prediction). → Train the gist vector by the **improvement it causes** in the worker's
loss (a goal-conditioning advantage), not by reconstructing it. This makes level A earn its keep
functionally rather than be a passive summary — directly attacks topic-drift.

## 6. Cognitive science — gist as a *semantic graph* (Language of Thought)  **[SPECULATIVE→ACTIONABLE-lite]**
Fodor's **Language of Thought** / **AMR** (Abstract Meaning Representation): the gist of "a wise girl
wanted to walk" is the predicate-argument graph `want(girl[wise], walk)` — language-independent,
**structured** (who-did-what-to-whom), richer than a content-word bag. → A lightweight realization without
a parser: represent level A as a short sequence of **(predicate, args) triples** induced from
content-word co-occurrence, and diffuse *that* first (a 2-level cascade where the coarse level is
relational, not lexical). Harder; flagged for later, but it's the principled ceiling for "gist."

## 7. Algorithmic information — gist as a PROGRAM  **[SPECULATIVE — a useful constraint]**
Kolmogorov/MDL view: the best gist is the **shortest program** that, when run, elaborates to the text;
prose/surface are its **execution trace**. Practical takeaway even without program synthesis: **keep the
gist tiny and generative** (a few bits/tokens that *expand*), and prefer the coarsening that **compresses
most** (ties back to RG §3 and successive-refinement). It's a design constraint — "gist must be a small
generator, not a summary" — more than an architecture.

## 8. Music — elaboration grammar (Schenker / GTTM)  **[ACTIONABLE-lite]**
Schenkerian analysis: a piece is a deep skeleton (*Ursatz*) **elaborated** to the surface by formal
**prolongation** rules (insert neighbours/passing tones). GTTM formalizes the reduction. Language analogue:
gist words are elaborated by **inserting** modifiers/function words between them — a concrete
**insertion/elaboration** operator (the README's "head-first insertion," now with a real grammar of
elaboration moves). → If we go non-diffusion for level B→C, this is the rule-set: expand each gist word
into a phrase by learned insertion.

---

## 9. What actually changes the design

Three of these are strong enough to **evolve Approach C**, and one is a genuinely new candidate:

- **Upgrade α — MERA framing (§1) is the real motivation.** Re-anchor ch32 on it: flat diffusion has
  exponential correlation decay; language is power-law/critical; a **log-depth disentangle+coarsen
  hierarchy** is the *known* fix. This is a stronger "why" than "syntax is a tree," and it predicts the
  *shape* (alternating local-attention + pooling, with skips). **Adopt as the headline rationale.**
- **Upgrade β — RG/info-bottleneck pooling (§3) + Feudal gist training (§5).** Together they give the two
  missing *objectives*: pooling keeps predictive info (not spelling), and the gist is trained by the
  improvement it causes downstream. These drop straight into Approach C's training without new modules.
- **New candidate D — Holographic diffusion (§2).** Represent token state as a **hypervector** binding
  gist/word/char; diffuse and clean up in that space. This is the most literal answer to "one coherent
  expression" and the most novel — but the highest risk (capacity, cleanup). Worth a **separate small
  probe** against Approach C, not a replacement yet.
- **Optional flavour — morphogenesis/CA commitment (§4)** as a sampler-level experiment (gist-gradient
  conditioning + local persistent commit) we can try *today* on the existing model — cheapest of all.

**Revised recommendation:** keep **Approach C** as the build, but (1) re-motivate it with **MERA/criticality**
and let that reshape the pooling into **log-depth disentangle+coarsen**; (2) adopt **info-bottleneck
pooling + Feudal gist training** as the objectives; (3) spin a **small holographic-diffusion probe** (§2)
as the high-upside side bet; (4) try the **CA/gist-gradient sampler tweak** (§4) on the current model now,
since it costs nothing.

---

## 10. Modern C++ notes for the new pieces
- **HyperVector (§2):** a `struct HyperVector { Tensor data; }` with free functions `bind`/`unbind`
  (FFT-based circular convolution — reuse a small FFT kernel) and `bundle` (add). Keep it a value type with
  `[[nodiscard]]` ops; it slots in as the residual-stream element without touching the transformer.
- **MERA stack (§1):** the `ResolutionLevel` concept from Review I already fits — a MERA layer is
  `disentangle()` then `coarsen()`; compose `std::array<MeraLayer, K>` with `K = log2(seq_len)` as a
  BuildTime config field (reflected, in `config_sha`). No virtuals, log-depth known at compile time.
- **Info-bottleneck pooling (§3):** a loss term, not a type — a `pool_ib(words) -> gist` plus an auxiliary
  predictor head; pure autograd, no new infrastructure.
- **Feudal gist (§5):** train-time only — compute the worker-loss delta with/without `g` and use it as the
  manager's signal; a few lines in the trainer, no architecture change.
- Everything stays the repo's ethos: value types, concepts over inheritance, BuildTime shape in the
  reflected config, hot path (the word denoiser) untouched.
