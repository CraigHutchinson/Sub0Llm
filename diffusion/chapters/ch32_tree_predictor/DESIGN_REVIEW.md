# Ch32 — Multidisciplinary Design Review: Unified Multi-Resolution Language

> Companion to [`README.md`](README.md). Goal: pull **one concrete, usable idea** from each of
> several disciplines, then converge them onto **a few buildable approaches** and a **modern-C++
> expression**, so the three-level concept (gist A / prose B / surface C) becomes implementable.
> Still PARKED behind the flat-diffusion training research, but this sharpens *what* we'd build.

The recurring problem statement (from the README): the three levels **coexist and need ONE coherent
means to express** — not three bolted-together models. Each lens below is judged by whether it helps
*unify*, not just decorate.

---

## 1. Lenses — one actionable idea each

### 1.1 Linguistics — *double articulation* + the content/function split → an OPERATIONAL gist
- Martinet's **double articulation**: language is meaningful units (morphemes/words, level B) built from
  meaningless ones (phonemes/chars, level C). Our B↔C is textbook; the novelty is the user's **third**
  level A above words.
- **Defining A without a parser:** the **content vs function** distinction. Content words (nouns, verbs,
  adjectives — open class, low-frequency, high-information) carry the *gist*; function words (the, of,
  was, and — closed class, ~150 types, high-frequency, low-information) carry *grammar*. So:
  **level A ≈ the content-word subsequence; level B adds the function words and order.**
- **This is implementable today, no POS tagger:** content-ness ≈ low corpus frequency / not in the
  top-K function-word list. A frequency threshold *is* a content/function classifier (Zipf guarantees
  the split is sharp). → **the gist level is just "the rare/informative tokens," computable from the
  tokenizer counts we already build.**

### 1.2 Information theory — *successive refinement* / progressive coding → the TRAINING OBJECTIVE
- Frame the whole model as a **scalable/progressive codec** (like progressive JPEG or scalable video):
  gist = **base layer** (coarse, decodable alone), prose = **enhancement**, surface = **refinement**.
- **Rate–distortion "successive refinement"**: a source is successively refinable if coarse-then-fine
  encoding costs no more than joint. Language is *approximately* so (gist→prose loses little). → train
  each level to be the **optimal coarsening** of the next (MDL: the hierarchy that compresses best).
- **Free bonus:** a progressive codec lets you **stop at any level** — emit only the gist for
  summarization/planning, or only through prose for fast drafts. Generation and summarization become the
  same mechanism at different depths.

### 1.3 Signal processing — the *Laplacian pyramid* → a PRINCIPLED per-level residual signal
- Burt–Adelson **Laplacian pyramid** / Mallat **multiresolution analysis**: store at each level only the
  **detail the coarser level cannot predict**. The earlier README §5-A balanced pyramid was a crude
  version; the principled version is: **level B models only the residual gist→prose (function words,
  exact lexical choice); level C models only word→spelling (mostly relevant for OOV).**
- This gives a clean, non-degenerate **per-level loss** (the failure mode of naive summary-regression was
  "all summaries collapse"; a residual target can't collapse — it's defined as what's left over).

### 1.4 Computer graphics — *cascaded diffusion / LOD / subdivision* → the GENERATION shape
- **Cascaded diffusion** (Imagen, Ho et al.) and **latent diffusion** (Stable Diffusion) are the proven
  multi-resolution generative template: diffuse a coarse signal, then a finer model **conditioned on the
  coarse output** super-resolves it. Maps 1:1: **gist-diffusion → word-diffusion(|gist) → char-decode.**
- **Latent diffusion specifically:** diffuse in a *learned latent* (our word level — semantic, cheap,
  where we measured grammar is learned fast) and use a separate **decoder** to the surface (chars). This
  is the single most practical template for our engine.
- **Subdivision surfaces / LOD:** a coarse control structure + a refinement rule → smooth output, and you
  render only the detail you need. Same coarse-to-fine, with **early-out** (cf. 1.2).

### 1.5 Neuroscience — *predictive coding* + *hierarchical timescales* → FLOW and CONTEXT
- **Predictive coding** (Rao–Ballard, Friston): each level predicts the level below; only the **prediction
  error** flows upward. This *is* the Laplacian pyramid as a network — levels communicate via residuals,
  which is cheap and matches 1.3.
- **Hierarchical timescales** (Hasson): higher areas integrate over longer windows (chars↔ms,
  words↔seconds, narrative↔minutes). → **different context lengths per level**: the gist level attends
  over the whole passage; the char level only needs a local window. Concrete, and it makes the long-range
  problem a *property of level A's wide context*, not something the flat char model must do.

### 1.6 Modern ML — *U-Net skip connections* → why the pyramid needs lateral wiring
- U-Net/hourglass: pool down, process, pool up, **with skip connections** carrying fine detail across.
  Without skips, upsampling hallucinates surface detail; the skips ARE the Laplacian residual (1.3). →
  any pooled design must keep **per-word (and per-char) skip state**, or OOV spelling/detail is lost.

---

## 2. Three candidate approaches (converging the lenses)

| | A. Cascaded (3 models) | B. Laplacian pyramid (1 stack) | C. Latent-word diffusion + char codec + gist (synthesis) |
|---|---|---|---|
| shape | 3 stacked diffusions, coarse→fine | 1 network, residual per level, predictive-coding flow | word-level diffusion; char-composition embed/decode; gist conditioner |
| unification | weak (bolted) | strong (one residual stream) | **strong & simple** (one diffusion, codec below, conditioner above) |
| OOV | char model at bottom | level C residual | **char-composition embedding (no OOV by construction)** |
| gist (level A) | a gist diffusion | top pyramid level | content-word plan: conditioning vector OR short pre-diffused content seq |
| fit to our engine | medium (3 trainers) | low (new pyramid plumbing) | **high (reuses Denoiser + tokenizers + sampler)** |
| proven prior art | Imagen cascade | wavelets/predictive coding | latent diffusion |
| risk | 3× training, interface drift | collapse / plumbing complexity | gist definition + char-codec quality |

**Recommendation: Approach C** — it is where every lens converges and it sits directly on our stack.
A and B are valuable *framings* (A = the mental model for generation order; B = the principled training
signal), but C is the **coherent thing to build first**.

---

## 3. The recommended coherent design (Approach C)

One word-level masked-diffusion model, wrapped by a char codec (below) and a gist conditioner (above):

```
                 ┌─────────────── level A: GIST ───────────────┐
content-word     │  pooled content-word plan  g = Pool(W_content) │  (wide context: whole passage)
selection (freq) └───────────────────┬─────────────────────────┘
                                     │ conditions (AdaLN / cross-attn)
   level B: PROSE   ┌────────────────▼─────────────────┐
   word canvas      │  Denoiser (existing): masked-diff │  (medium context: the window)
   x_word           │  over word tokens, conditioned g  │
                    └───────┬──────────────────┬────────┘
                       embed │                  │ decode
   level C: SURFACE  ┌───────▼──────┐    ┌──────▼───────────┐
   chars (no OOV)    │ CharComposer │    │   CharDecoder    │  (local context: the word)
                     │ chars→w_emb  │    │ w_state→char log │
                     └──────────────┘    └──────────────────┘
```

- **Level C as a codec, not a third diffusion.** `CharComposer`: a small Conv1d/attention over a word's
  character embeddings → that word's vector — so **any** word (seen or OOV) gets a representation from its
  spelling (1.1 morphology, CharCNN/fastText). `CharDecoder`: word state → character logits, so output is
  spelled char-by-char (no OOV on the way out either). In-vocab words **blend** a lookup embedding with the
  composed one (a learned gate), getting word-level efficiency where it helps and composition everywhere.
- **Level B is our existing Denoiser**, unchanged in spirit, operating on word embeddings.
- **Level A as conditioning.** Compute the content-word subsequence by frequency (1.1), pool it to a gist
  vector `g` (or keep it as a short sequence), and condition the denoiser via AdaLN/cross-attention. `g`
  has **wide context** (1.5) so it carries long-range topic — directly attacking the topic-drift we still
  see. Optionally `g` is itself produced by a tiny gist-diffusion first (then it's a 2-level cascade, 2.A).
- **Training (Laplacian/successive-refinement, 1.2/1.3):** word-level diffusion loss (the masked-CE we
  have) + a char-reconstruction loss through the codec + a gist-consistency loss (the pooled content
  words should predict / agree with `g`). Each is a residual the others can't trivially satisfy.
- **Generation = coarse-to-fine with early-out (1.4):** plan `g` → denoise the word canvas conditioned on
  `g` → spell each word through the decoder (OOV words included). Stop at `g` for a summary; stop at words
  for fast text.

This is the "**simple coherent means to express**" the user asked for: **one diffusion at the word level**,
with the surface folded into the *embedding/decoding* (not a separate model) and the gist folded into
*conditioning* (not a separate model). Three levels, one network.

---

## 4. Modern C++ (C++23) expression on our stack

The unification should show up in the **types**: every level is a module with the same shape, composed in
one pipeline; the residual stream is one tensor type threaded through.

```cpp
// A "resolution level" maps a finer representation up and a coarser one down. Concept, not inheritance —
// matches the repo's template<ComputeScalar T> / concepts-over-SFINAE style.
template <class L>
concept ResolutionLevel = requires(L lvl, Variable fine, Variable coarse) {
    { lvl.up(fine)     } -> std::same_as<Variable>;   // encode finer → this level (pool)
    { lvl.down(coarse) } -> std::same_as<Variable>;   // decode this level → finer (unpool, + skip)
};

struct CharComposer {                 // level C → B embedding (OOV-proof)
    Conv1d  char_conv;                // over per-word char embeddings
    Embedding word_lookup;            // in-vocab fast path
    Variable gate;                    // learned blend lookup vs composed
    [[nodiscard]] Variable up(const WordChars& w) const;   // chars → word embedding
};
struct CharDecoder  { [[nodiscard]] Variable down(const Variable& word_state) const; }; // word → char logits
struct GistPlanner  {                 // level A: content-word pool (freq-defined) → conditioner
    std::span<const std::uint8_t> is_content;   // from tokenizer counts (Zipf split), like is_word_start
    [[nodiscard]] Variable pool(std::span<const TokenId> words) const;
};

// The whole model is the existing Denoiser at level B, plus a codec below and a planner above.
struct MultiResDenoiser {
    GistPlanner   gist;     // A
    dn::Denoiser  prose;    // B  (reused as-is, + conditioning input)
    CharComposer  embed;    // C↑
    CharDecoder   spell;    // C↓
    Variable forward(const WordCanvas& x, const GistCond& g) const;  // one coherent forward
};
```

Why this is "coherent" in code, not three programs:
- **One residual-stream type** (`Variable`) flows through every level; `up`/`down` are the only cross-level
  verbs — the same uniform interface the README asked for.
- **Reuse, not rewrite:** `prose` is the current `Denoiser`; `is_content` mirrors the existing
  `is_word_start` table (built from tokenizer counts in ch29) — so level A costs ~one array.
- **C++23 leverage:** `concept ResolutionLevel` enforces the up/down contract at compile time; the level
  stack is a fixed-size aggregate (no virtual, matches the config module's value-type ethos); pooling/
  selection use `std::span`/ranges over the token stream; the level shapes are BuildTime config fields
  (reflected, part of `config_sha`) so a specialized engine bakes them in.
- The **char codec is a leaf module** — it never participates in the diffusion loop, only in embed/decode,
  so the hot path (the word-level denoiser) is unchanged and stays fast.

---

## 5. First experiment & open questions (when unparked)

- **Smallest test:** add only the **CharComposer/CharDecoder** around the existing word-level denoiser on
  TinyStories — i.e. word-level diffusion that can read/write OOV via chars, **no gist yet**. Success = the
  word model's fast grammar (measured: ~step 2–4k) *plus* zero OOV cliff on held-out rare words. Then add
  the gist conditioner and measure **topic-drift** reduction.
- **[OPEN] gist representation:** single pooled vector vs short content-word sequence vs tiny gist-diffusion
  (2-level cascade). Start with a vector (cheapest).
- **[OPEN] content/function threshold:** which frequency rank splits gist from grammar — sweep it; Zipf
  says it's not sharp-critical.
- **[OPEN] does char-composition rescue OOV meaning at our scale**, or only with more data? The OOV-cliff
  test answers it directly.
- **Unpark gate (unchanged):** only after the DiffusionGemma/LLaDA/MDLM training research, and only if a
  well-trained flat subword diffusion still shows the semantic/topic-drift gap this targets.
