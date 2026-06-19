# Chapter 32 — Tree Predictor: Hierarchical / Structural Generation (DESIGN)

> **Status: PARKED (design skeleton, not implemented).** Proposes a *third* generation
> paradigm alongside autoregression (Ch08) and masked diffusion (Ch28–31), motivated by the
> local-interpolator finding in [`../../TRAINING_DESIGN.md` §13.8](../../TRAINING_DESIGN.md)
> (flat parallel denoising provides no cross-span coordination scaffold). Sections marked
> **[OPEN]** are unresolved; **[PHASE n]** tags sequence an eventual build.
>
> **Why parked (2026-06-19):**
> 1. **Not necessary — flat diffusion demonstrably works at scale.** DiffusionGemma-26B
>    (`[[diffusiongemma-reference-run]]`) is a coherent flat masked-diffusion LM. So our §13.8
>    local-interpolator limit is a property of *our small-scale setup/recipe*, not of flat
>    diffusion per se. The tree is a *possible novel improvement*, not a required fix.
> 2. **Research prerequisite.** Before building a novel architecture, find public info on **how
>    DiffusionGemma (and LLaDA / MDLM / block-diffusion) are actually trained** — their recipe
>    (scale, noise schedule, block/semi-AR masking, objective weighting) may already supply the
>    long-range coordination we're missing, far cheaper than a new paradigm. **[RESEARCH TASK]**
> 3. **The balanced-binary-pyramid prior (§5-A) is theoretically weak for language.** Syntax is
>    *unbalanced* and *n-ary* (a sentence is not a balanced binary tree); a fixed balanced
>    pyramid imposes structure that doesn't match real constituency. This pushes weight toward
>    the §4-B/§4-C (real / induced trees) or §5-C (soft-structure) variants and is itself an
>    argument to validate the *idea* against the literature before committing. **[OPEN — see §4]**
>
> **Unpark when:** the DiffusionGemma/LLaDA training research is done AND (a) their recipe
> doesn't already close the §13.8 gap on our setup, and (b) we still judge a structural prior a
> promising novel lever. Until then this stands as a recorded idea, not active work.

---

## 1. Motivation — generation *order* is a design axis

Every generator commits tokens in some order. That order is the real design choice:

| paradigm | commit order | coordination mechanism | chapter |
|---|---|---|---|
| **Autoregressive** | left → right (position) | left context conditions each next token | Ch08 |
| **Masked diffusion** | high → low confidence | none — positions predicted from independent marginals | Ch28–31 |
| **Tree predictor** *(this)* | root → leaves (structure) | the **parent node** conditions its children | — |

§13.7 found *why* flat diffusion produces salad: it fills masked positions from their
**marginals with no left→right (or any) constraint**, so locally-probable pieces fail to
coordinate into globally-valid units (`con`+`ick`+`ers` → `conickers`). AR avoids this
because left-context coordinates each next token — but pays a strict sequential bottleneck.

**The thesis of this chapter:** language is *hierarchical* (a sentence is a tree of
phrases, not a flat list), so the natural coordination scaffold is the **tree itself**.
Generate a sentence by expanding its structure top-down: each parent node supplies the
shared context its children need, so siblings can be produced **in parallel** (no AR
bottleneck) **without** the uncoordinated-marginals problem (the parent *is* the
coordination). Tree generation is "outside-in / coarse-structure-first," a third point
distinct from AR's "front-to-back" and diffusion's "confident-first."

---

## 2. Core idea — generation as recursive expansion

```
            [ROOT]                         level 0: one node summarising the whole utterance
           /      \
       [Aːsummary] [Bːsummary]            level 1: ROOT expands to 2 child summaries
        /   \        /    \
      ...   ...    ...    ...             level k: each node expands to its children
      |      |      |      |
     the   king   said   "..."           leaves: the actual tokens (chars or words)
```

- **Top-down.** Start from one root embedding; predict its expansion; recurse until leaves
  are concrete tokens.
- **Parallel within a level, conditioned across levels.** All nodes at level *k* are
  predicted together (bidirectional attention over the level), each conditioned on its
  parent at level *k−1*. This is the key: **breadth-parallel, depth-sequential.**
- **The parent is the scaffold.** A child is never predicted from a bare marginal — it is
  predicted from its parent's summary plus its siblings, which is exactly the coordination
  flat diffusion lacks.

---

## 3. Prior art (survey — to be expanded)

- **Recurrent Neural Network Grammars** (Dyer et al. 2016) — joint model of words + phrase
  structure via shift/reduce actions; explicitly tree-structured but strictly sequential.
- **Insertion Transformer** (Stern et al. 2019) & **Levenshtein Transformer** (Gu et al.
  2019) — non-monotonic / edit-based decoding (insert between existing tokens); "outside-in"
  cousins without an explicit tree.
- **Tree Transformers / constituency-attention** (Wang et al. 2019; Nguyen et al. 2020) —
  bias attention toward induced constituents.
- **Syntactic / latent-tree VAEs** (Yin et al.; Kim et al.) — induce trees as latent
  variables.
- **Hourglass / U-Net Transformers & hierarchical diffusion** (Nawrot et al.; multi-scale
  diffusion) — the *multi-resolution* mechanics we borrow for the pyramid in §5-A.
- **SUNDAE / step-unrolled denoising, Diffusion-LM** — flat-diffusion baselines to beat.
- **[OPEN]** position this chapter precisely against insertion-based decoding — our claim is
  that an explicit **level/parent scaffold** gives stronger coordination than flat insertion.

---

## 4. Where does the tree come from? (the central design fork) **[OPEN]**

Three sources, in increasing ambition:

- **A — Induced by construction (no parser).** Impose a *fixed* structural prior: a balanced
  binary pyramid over the token sequence (§5-A). The "tree" is just multi-resolution; the
  model learns what each coarse node should mean. **Cheapest, codebase-aligned, recommended
  for [PHASE 1].**
- **B — Supervised parses.** Use a constituency/dependency parser to produce gold trees;
  train the expander to reproduce them. Linguistically grounded, but adds a parser
  dependency and a Python tool step (against the no-Python-core rule — parser runs in
  `tools/`, emits trees to disk). **[PHASE 2].**
- **C — Latent / induced trees.** Learn the tree as a latent variable (grammar induction).
  Most powerful, hardest to train (discrete latent structure). **[PHASE 3 / research].**

The pyramid (A) is the pragmatic start: it tests the *mechanism* (does parent-conditioned
breadth-parallel expansion beat flat diffusion?) without the parser/latent machinery.

---

## 5. Proposed architectures

### 5-A. Recursive pyramid expansion — **recommended first build [PHASE 1]**

A balanced binary pyramid over a length-`T` canvas (`T` a power of two). Level `k` has
`2^k` nodes; the leaf level `K = log2(T)` is the token sequence.

**Representations.** Each node holds a `D`-dim embedding. Leaves embed real tokens; internal
nodes are *summaries*. Build training targets by **pooling**: a parent's target summary is a
(learned or mean) pool of its two children — so the pyramid is a learned encoder downward
and a generator upward.

**Training (teacher-forced, all levels at once).**
1. Encode a real token window → leaf embeddings.
2. Pool up the pyramid to get every node's "true" summary (the encoder pass).
3. For each level `k`, run a **transformer over that level's nodes** (bidirectional, with
   *tree-positional* encodings: depth + index-within-level + parent pointer) conditioned on
   the parent summaries from level `k−1`, and train it to predict the level-`k` summaries
   (and, at the leaf level, the actual token logits).
4. Loss = Σ_levels (summary regression / distillation) + leaf cross-entropy. **[OPEN]**:
   regress summaries vs. quantize them to a codebook (VQ) so each level is a classification —
   VQ likely trains more stably and gives a clean "expansion vocabulary."

**Generation (top-down).** Predict ROOT summary from a learned BOS → expand to level 1 →
… → leaf logits → sample tokens. `K = log2(T)` sequential steps (e.g. **7 for T=128**),
each fully parallel across its level — far fewer sequential steps than AR's `T`, with a
real coordination scaffold unlike flat diffusion.

**Why this is the right first cut:** it reuses our `Denoiser` transformer block almost
verbatim (it's "a transformer over a set of nodes with positional encodings"), needs no
parser, and isolates the core hypothesis.

### 5-B. Head-first insertion decoder — **linguistic variant [PHASE 2]**

Generate the dependency *heads* (content skeleton) first, then recursively **insert**
dependents (modifiers, function words) between them, each insertion conditioned on its head.
Closer to real syntax and to Insertion/Levenshtein Transformers; needs a head ordering
(source B/C). Stronger linguistic prior, more moving parts.

### 5-C. Soft-structure prior — **fallback if explicit trees underperform [OPEN]**

Keep a flat canvas but add a learned **span/constituent bias** to attention + a coarse-to-fine
*schedule* (commit head words before modifiers) — captures some hierarchy benefit with
minimal departure from Ch29. A cheap ablation to attribute gains to "structure" vs "the tree
machinery."

---

## 6. Fit with the existing codebase

**Reuse as-is:** the transformer block & RMSNorm/RoPE/attention (`sub0diff/nn/denoiser.hpp`),
the char/word tokenizers (Ch03 — leaves are tokens), the raw corpus reader and token cache
(Ch29), checkpointing (Ch24), and the autograd engine.

**New components (small surface):**
- `TreePositionalEncoding` — (depth, index-in-level, parent-index) instead of linear position.
- a **pool-up encoder** (mean or learned/VQ) to build per-level targets.
- a per-level **expansion head** (parent summary + level transformer → child summaries / leaf
  logits).
- a top-down **generation loop** (the analogue of `refine_canvas`).

**Minimal first experiment [PHASE 1]:** `T=128`, char-level, balanced pyramid (`K=7`),
mean-pool summaries, summary-regression + leaf-CE, on **TinyStories** (see §8). Compare leaf
perplexity + free-generation coherence head-to-head with the Ch29 char-diffusion baseline at
matched params.

---

## 7. Why it could beat AR and diffusion (hypotheses to test)

- **vs diffusion:** the parent scaffold supplies cross-position coordination, so children
  shouldn't mis-assemble the way flat marginals do (§13.7). *Prediction:* fewer incoherent
  local transitions at equal token granularity.
- **vs AR:** `log2(T)` sequential steps instead of `T`, with structural lookahead (the root
  "plans" before leaves commit). *Prediction:* faster decode, more globally-consistent long
  outputs.
- **Predicted failure modes [OPEN]:** (a) summary regression collapses (all nodes alike) →
  needs VQ / contrastive targets; (b) a *fixed* balanced pyramid mismatches real linguistic
  structure → motivates B/C; (c) error compounding down levels (a wrong root dooms the
  subtree) → the diffusion-style weak-model precondition reappears, mitigated by per-level
  refinement passes.

---

## 8. Evaluation plan

- **Corpus:** **TinyStories** (small-model-coherent; see the corpus discussion in the session
  notes / `[[ch29-char-level-run-results]]`) as the primary; Shakespeare retained as the
  hard, low-data stress test.
- **Baselines (matched params/corpus):** Ch29 char-diffusion, Ch08/Ch24 AR-GPT.
- **Metrics:** leaf perplexity; free-generation coherence (human-read + a coherence proxy);
  **non-word rate** (the §13.7 lens — does structure further reduce it?); decode
  sequential-step count; long-range consistency.
- **Ablations:** pyramid vs soft-structure (5-C) vs flat diffusion; mean-pool vs VQ summaries;
  per-level refinement on/off.

---

## 9. Roadmap

- **[PHASE 0]** this design doc + a tiny numpy/host sketch of the pyramid pool/expand on one
  sentence to sanity-check shapes and the training signal.
- **[PHASE 1]** pyramid expander (5-A), char-level, TinyStories; beat-or-match char-diffusion
  on coherence at matched params. **Go/no-go gate.**
- **[PHASE 2]** VQ summaries + per-level refinement; word-level; head-first variant (5-B).
- **[PHASE 3]** induced/latent trees (4-C); scale corpus (WikiText).

---

## 10. Open questions

1. Summary targets: **regression vs VQ codebook vs contrastive** — which gives a stable,
   informative per-level signal? *(leading: VQ.)*
2. Is a **fixed balanced pyramid** enough, or does the win require *real* linguistic trees
   (B/C)? The 5-A vs 5-C ablation answers this.
3. **Decoding:** single top-down pass vs. iterative per-level refinement (diffusion-style)
   vs. beam over expansions — and where the weak-model error-compounding precondition bites.
4. **Length handling:** powers-of-two padding vs. n-ary / unbalanced trees for natural
   sentence lengths.
5. Does the structural scaffold measurably **lower the §13.7 non-word rate** at subword
   granularity — i.e. can it rehabilitate BPE for parallel generation?
