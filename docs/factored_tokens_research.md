# Factored Tokens — research dive & design for sub0llm

> Research findings (2026-06-23). Question raised: our word-level vocab holds separate ids for what are
> really the SAME word under different *attributes* — case (`NEED`/`Need`/`need`), possession (`Billy's`),
> number (`cats`), quote style (`Billy's` vs `Billy's`), and even typos (`quix`→`quick`). Can we collapse
> these axes **elegantly, without increasing memory/compute when balanced against the token-count
> reduction**? Short answer: yes — done right, factoring *reduces* compute. This doc grounds that in prior
> art and proposes a staged plan that fits our constraints (C++ core, weight-tied head, diffusion LM).

## 1. The duplication axes (what we're collapsing)

| Axis | Example dupes | Carries meaning? | Predictable from context/position? |
|------|---------------|------------------|------------------------------------|
| **Sentence-initial case** | `The`/`the`, `Need`/`need` (at `.`) | No — purely orthographic | **Yes** — from position after `.`/start |
| **Emphasis case (all-caps)** | `NEED` vs `need` | **Yes** — pragmatic stress | No |
| **Quote glyph** | `Billy's` (') vs `Billy's` (’) | No — Unicode noise | n/a (normalize) |
| **Trailing/closing quote** | `Monster'` (close-quote weld) | No | partly (from surrounding quotes) |
| **Number** | `cat`/`cats` | **Yes** — agreement/semantics | sometimes |
| **Possession** | `Billy`/`Billy's` | **Yes** | sometimes |
| **Tense/aspect** | `walk`/`walked`/`walking` | **Yes** | sometimes |
| **Typo** | `quix`→`quick`, `teh`→`the` | No (usually) | **Yes** — from context proximity |

The crucial split is the last two columns: an axis that is **predictable/redundant** can be *stripped at
input and restored at output for free*; an axis that **carries meaning** must remain visible to the model
during processing. This directly answers "are some attribute axes necessary during processing?" — **yes,
the meaningful ones** (emphasis, number, possession, tense); the redundant ones (sentence-case, quote
glyph) are pure surface and cost us vocab for nothing.

## 2. Prior art (what's been tried, and the reusable mechanism)

- **Factored NMT — Sennrich & Haddow 2016.** Generalize the embedding layer to sum/concat a *lemma*
  embedding with small *factor* embeddings (case, number, gender, POS…). "Using lemmas beside tokens
  reduces data sparsity and lets inflectional forms share representation." Improves perplexity/BLEU — i.e.
  the factors are **useful signal**, not just decoration.
- **Sparsely Factored NMT — 2021.** Word vector = **sum** of lemma + factor embeddings. **Output is
  factored too**: predict the **lemma first over a small vocab**, then small conditional softmaxes for each
  factor. Only lemmas + a handful of factor values need parameters — large vocab/param savings vs dense
  surface-form vocab. Finding: *linguistically structured* factors beat arbitrary subword splits.
- **LOVE — Imputing OOV Embeddings (ACL 2022).** A **tiny** char+subword model (~6.5M params, ~100× smaller
  than fastText) trained by **contrastive learning** to **mimic** a pretrained embedding: augment each word
  (char swap/del/ins, keyboard-distance, synonym), pull the corrupted form to the original, push hard
  negatives (orthographically-close, semantically-different words) apart. **Plug-and-play, OOV-only,
  no base retrain** — in-vocab words use the original table unchanged. Recovers most accuracy under heavy
  typo corruption (e.g. SST2 90% typos: fastText 60.1%→67.2%) at ~zero clean-data cost.
- **Position-invariant truecasing — 2021.** Treat case as a **separable per-token class**
  (lower / UPPER / Capitalized / miXed) predicted by a small head; "position-invariant" because
  sentence-initial caps are trivially predictable from position and shouldn't dominate the signal.
- **RoVe / Char2Subword / fastText.** Robustness via char n-grams / B-M-E (prefix-stem-suffix)
  decomposition; typos share subword pieces so OOV degrades gracefully. (We already have a `CharComposer`
  — see the caveat in §5.)

## 3. The efficiency case (why factoring is a *net win* for us)

Our setup: word vocab **V≈11,938**, **D=256**, **weight-tied** head (`matmul_bt(x, tok_emb)`). The LM head
(and the tied embedding) is the single biggest matrix: **V×D ≈ 3.05M params**, and the per-position output
softmax over V is the dominant output FLOP — especially in diffusion, where we score **every** masked
position in parallel.

Factoring replaces that with:
- a **lemma table** `V_lemma×D` where `V_lemma` ≈ unique lemmas (case+number+possession collapse many
  surface forms → empirically 30–50% smaller on English word vocabs), plus
- a few **tiny factor tables**: case (4) + number (2) + possession (2) + tense (~4) ≈ **12 rows × D**, and
- a **decomposed output**: one softmax over `V_lemma` + a handful of ≤4-way softmaxes.

Net: the embedding/head shrink ~roughly in proportion to the surface→lemma collapse, and the extra factor
heads are negligible (≤14 logits vs ~12k). So **balanced against the token reduction, compute goes DOWN,
not up** — the opposite of the usual "add features → add cost" worry. Input-side, summing K≤4 factor
vectors is K adds per token (free).

## 4. The "strip at input, re-embellish at output" path (your low/no-cost idea, made precise)

Two tiers, by the §1 split:

**Tier A — strip & restore (zero meaning loss, pure vocab win):**
- **Sentence-initial case**: tokenize on a *lowercased* stream; the case is restored at output from
  position (after sentence boundary → capitalize). Collapses `The`/`the`, `Need`/`need`@start.
- **Quote glyph / closing-quote welds**: normalize `’→'`, split closing quotes. (Breaks the tokenizer's
  lossless round-trip invariant — a deliberate design change, see the bpe.cpp TODO.)
These need **no axis fed to the model** — they're deterministic surface restoration.

**Tier B — keep as a cheap factor the model sees AND predicts (meaning-bearing):**
- **Emphasis (all-caps)**: 1 bit. `NEED` = lemma `need` + emphasis=1. The model must see it (stress changes
  meaning) and predict it.
- **Number / possession / tense**: small factors. Require a (rule-based for English: `-s`,`'s`,`-ed`,`-ing`
  + irregular table; or Morfessor for unsupervised) **lemmatizer** at tokenize-time.

So the answer to "can we strip everything?" is **no** — Tier A strips the redundant axes for free, but Tier
B axes are part of the language's meaning and earn their keep as factors (Sennrich's perplexity gains are
the evidence). The elegant system is the **hybrid**.

## 5. Typos — the LOVE-style "proximity" path (and our specific caveat)

Your framing ("`quix` → `quick` because context says so") is exactly **input-side OOV imputation**:
- For an OOV/typo at **input**, don't mint a new id — **map it to the nearest in-vocab token** by a small
  char model (LOVE-style, contrastively trained so `quix`≈`quick`), and let the bidirectional model +
  context disambiguate (`the ___ brown fox` ⇒ `quick`, not `quit`). In-vocab tokens pay nothing; this runs
  only on the **serve/CLI input path**, not training.
- **Project caveat (important):** our `ch32-char-composition-oov-falsified` finding showed char-*composition*
  for OOV **hurts under a weight-tied head** (spelling geometry ≠ prediction geometry). LOVE differs (it
  *mimics the embedding* via contrastive learning, not spelling reconstruction) — but with our tied head, a
  synthesized input embedding is implicitly an output-head row too, so the same tension can bite. **The
  clean sidestep:** treat typos as **retrieval to an existing id** (input normalization), NOT as a new
  embedding/head row. Both input and the tied head stay on the real vocab; nothing novel pollutes the head.
  This also matches "typos should be mapped to the relevant token."

## 5.5. Sequential subwords vs parallel factors (compound/morpheme decomposition)

A separate way to exploit morphology is **subword decomposition** — split words into reusable chunks:
inflectional (`zoom`+`ed`, `play`+`ers`), compositional (`zoo`+`keep`+`ers`, `mon`+`day` sharing `day`).
This is exactly BPE/WordPiece (we already have `BPETokenizer::train`) and, linguistically-aware,
**Morfessor**. Two mechanisms for the same morphology — and the choice matters for **diffusion**:

| | Sequential subwords (BPE/Morfessor) | Parallel factors (§4) |
|---|---|---|
| `zoomed` → | `zoom` + `ed` (2 **positions**) | lemma `zoom` + tense factor (1 token) |
| Vocab | ↓ (shared chunks) | ↓ (shared lemmas) |
| **Token count / seq length** | **↑ (one word → 2–3 tokens)** | unchanged |
| Composition | model composes across positions via attention | summed at the embedding |
| Carries semantics? | **yes** — stems (`zoo`,`keep`,`day`) are meaning units | only grammatical features |
| Needs | merge table (have) / Morfessor | morphological analyzer |

**The diffusion catch:** subwords DON'T reduce the token count — they increase it (only the *vocab*
shrinks). At a fixed `seq_len`, `zoo-keep-ers` consumes 3 of 128 slots, so **fewer words fit per window**
(less context) and MERA's O(N·w) cost rises. Word-level packs the most *meaning per position*. So the
elegant split is by morpheme TYPE:
- **Inflectional** suffixes (`-ed/-s/-ing/-ers`) → better as **parallel factors** (keep seq length).
- **Compositional** stems (`zoo`+`keep`, `mon`+`day`) → genuine semantic units → **sequential subwords**
  earn their position (the model gets `day`'s shared meaning across all weekdays for free).
A hybrid — Morfessor-style stem segmentation for content morphemes + factored inflection — is the target.

## 6. Diffusion-specific angle (a genuinely novel fit)

- Our denoiser already predicts **all positions in parallel** with a full-vocab softmax each. Adding
  per-position **factor heads** (case/number/…) is cheap and parallel, and the lemma softmax is *smaller* —
  so factoring is even more favorable here than in autoregressive decoding.
- **Lemma ↔ gist alignment:** factoring separates *meaning* (lemma) from *surface* (case/number/tense) — the
  same separation the MERA **gist** is reaching for (`gist-is-compute-context-primitive`,
  `verbosity-time-slider-from-gist`). Natural design: **coarse MERA levels carry lemmas (plan/meaning);
  the finest level realizes surface factors.** That makes the verbosity/terseness dial fall out, and gives
  the coarse levels a smaller, denser vocab to reason over.

## 7. Recommended staged plan (highest ROI first; all gated behind the core engine stabilising)

1. **Truecasing factor (Tier A+emphasis).** Tokenize lowercased; emit a 4-way case class per token
   (lower/Cap/UPPER/miXed); restore at output (Cap is position-predictable, UPPER is the emphasis bit).
   Highest ROI: case is the most frequent duplication axis, classes are well-defined (no lemmatizer), head
   decomposition is trivial. **Quick win, principled, retrain-required.**
2. **Quote/Unicode normalization** (`’→'`, closing-quote split). Cheap; accept the round-trip-invariant
   change; bundle with (1) since both need a retrain.
3. **English morphology factors** (`-s` number, `'s` possession, `-ed/-ing` tense + irregular table; or
   Morfessor unsupervised). Bigger lift (needs a lemmatizer + factored output heads); validate the
   compute win and quality (perplexity) before committing.
4. **Typo retrieval at input** (LOVE-style nearest-id, serve/CLI only). Independent of 1–3; no base retrain.

**Risks / gates:** (a) lemmatizer errors inject noise — measure net perplexity, not vocab size; (b) the
weight-tied-head falsification (§5) — keep factoring on input + *decomposed real-vocab* heads, don't
synthesize head rows; (c) round-trip invariant is intentionally given up (fine for a generative LM); (d)
the corpus has genuine typos (`Needles`?) we can't blindly "fix" — only normalize *user input*, never the
training corpus's real tokens.

## Sources
- [Factored Neural MT — Sennrich & Haddow 2016 (IWSLT)](https://aclanthology.org/2016.iwslt-1.3.pdf)
- [Sparsely Factored NMT — 2021](https://arxiv.org/pdf/2102.08934)
- [Neural MT by Generating Multiple Linguistic Factors](https://arxiv.org/pdf/1712.01821)
- [LOVE: Imputing OOV Embeddings with Little Cost — ACL 2022](https://arxiv.org/pdf/2203.07860)
- [Position-Invariant Truecasing (word+char hierarchical RNN) — 2021](https://arxiv.org/abs/2108.11943)
- [Capitalization & Punctuation Restoration: a Survey](https://arxiv.org/pdf/2111.10746)
- [Char2Subword: robust character compositionality](https://arxiv.org/pdf/2010.12730)
- [Computational morphology survey — 2024](https://arxiv.org/html/2406.05424v1)
