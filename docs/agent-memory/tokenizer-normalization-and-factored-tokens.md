---
name: tokenizer-normalization-and-factored-tokens
description: "Backlog — input normalization (curly→straight quotes, trailing-quote welds) is a quick win but breaks lossless round-trip; deeper idea = factored tokens (lemma + separable case/number/possession/tense axes)"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Backlog (raised 2026-06-23, observing the TinyStories word-level vocab). Two layers:

**Quick win — input normalization.** The vocab carries orthographic-noise duplicates that fragment a
word's statistics: curly vs straight apostrophe (`Billy’s` U+2019 vs `Billy's` U+0027 = two ids), and
TRAILING apostrophes from closing quotes (`Luxury'`, `Monster'`, `Nibbles'`) that `split_words` keeps
welded (it intentionally keeps `th'`/`dogs'`). A normalization pass (NFKC-style ’→', collapse ambiguous
trailing quotes) merges them. **Catch:** it BREAKS the lossless `decode∘encode == identity` invariant the
tokenizer guarantees (and a test asserts) — fine for a generative LM, but a deliberate design change, not
free. It also CHANGES the vocab ⇒ needs a retrain, so DON'T fold into a model mid-training. Already done:
LEADING quotes no longer weld (`'And`/`''What's` → separate quote + word). See [[char-level-tokenization-lever]].

**Deep idea — factored / morphological tokens (RESEARCHED 2026-06-23, full write-up
`docs/factored_tokens_research.md`).** One id per lemma + small SEPARABLE attribute axes (case/number/
possession/tense), `e = lemma_emb + case_emb + number_emb + …`. Key conclusions:
- **It REDUCES compute, doesn't add it** (the user's worry, inverted): collapsing surface→lemma shrinks the
  tied embedding/LM head (V≈11938×D256≈3M params, our biggest matrix + dominant output softmax) ~30-50%;
  the extra factor heads are ≤14 logits (negligible). Factored OUTPUT = predict lemma over small vocab +
  tiny per-axis softmaxes (Sparsely Factored NMT 2021). Even more favorable in diffusion (parallel heads).
- **Strip-vs-keep split** (answers "are some axes needed during processing?" → YES the meaningful ones):
  Tier A = strip & restore for free (sentence-initial case is position-predictable; quote glyph noise);
  Tier B = keep as a cheap factor the model sees+predicts (all-caps EMPHASIS=1bit, number, possession,
  tense — Sennrich&Haddow show these IMPROVE perplexity, so they're signal not decoration).
- **Typos = input-side nearest-id RETRIEVAL** (LOVE ACL2022: tiny ~6.5M char model, contrastive mimic,
  OOV-only, no base retrain). For us: map typo→existing in-vocab id at serve/CLI input ONLY — do NOT
  synthesize new embeddings/head rows (sidesteps [[ch32-char-composition-oov-falsified]]: spelling geom ≠
  prediction geom under our weight-tied head). "quix"→"quick" via char-proximity + context.
- **Lemma↔gist alignment** (novel fit): coarse MERA levels carry lemmas (meaning/plan), finest level
  realizes surface factors → makes the verbosity dial fall out. See [[gist-is-compute-context-primitive]],
  [[verbosity-time-slider-from-gist]].
- **Sequential subwords vs parallel factors** (user, "zoom+ed", "zoo-keep-ers", "mon-day"=shared 'day'):
  morphology can be SEQUENTIAL subword tokens (BPE — we HAVE `BPETokenizer::train` — or Morfessor) OR
  PARALLEL factors (§4). Diffusion catch: subwords INCREASE token count (one word→2-3 tokens), shrinking
  only VOCAB not corpus length — at fixed seq_len fewer words fit per window (less context) + MERA O(N·w)
  rises. So split by morpheme TYPE: inflectional (-ed/-s/-ing/-ers) → parallel FACTORS (keep seq len);
  compositional STEMS (zoo+keep, day) → sequential SUBWORDS (carry semantics, earn the position). Hybrid =
  Morfessor stems + factored inflection. Coarse MERA levels plan over the smaller stem/lemma vocab.

Staged plan: (1) truecasing factor [quick, no lemmatizer, highest ROI] (2) quote normalization (3) English
morphology: factored inflection + optional Morfessor stem segmentation [needs analyzer] (4) typo retrieval
at input. Gate (1)-(3) behind a retrain; judge by perplexity not vocab size.
