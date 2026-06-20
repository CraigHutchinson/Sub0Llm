# Chapter 32 — Unified Multi-Resolution Language (DESIGN)

> **Status: PARKED design vision (not implemented).** This supersedes the earlier
> "tree predictor / balanced-binary-pyramid" sketch — that was one (weak, syntax-flavoured)
> expression of a deeper idea. The real idea, distilled from this volume's empirical findings
> (§13.6–§13.9) and the char-vs-word experiments: **language exists at several coexisting levels
> of abstraction at once, and a good model should represent all of them in one coherent system.**
> Sections marked **[OPEN]** are unresolved; **[PHASE n]** sequence an eventual build.
>
> **Parked pending** (unchanged): research into how flat text-diffusion (DiffusionGemma, LLaDA,
> MDLM, block-diffusion) is actually trained — a known recipe may close the long-range gap on our
> setup far cheaper than new architecture. Don't build before that research. See
> [[tree-predictor-parked-research-diffusion-training]].

---

## 1. The observation — three levels coexist, and we've seen all three

We kept finding the "same" model behaving at different granularities. Step back and they are
**three coexisting levels of linguistic abstraction**, each one we have *empirically produced* in
this project:

| level | unit | what it carries | what we observed |
|---|---|---|---|
| **A — gist / keyword** | content words | *what happens* — the semantic skeleton | the word model's content spine: *"girl. Want walk. Saw ball. Wanted…"* — coherent above the sentence |
| **B — prose / word** | whole words | grammar, function words, syntax | word-level: grammatical sentences fast (~step 2k on TinyStories), **but** closed vocab / OOV-blind |
| **C — surface / sub-token** | chars (→ bytes) | spelling, morphology, **no OOV** | char-level: spells anything, graceful degradation, **but** must *also* learn to spell → slower to grammar |

These are not competing tokenizer choices — they are **layers that are all true at the same time**.
"Once upon a time, there was a wise girl who wanted to walk" *is* simultaneously a gist (girl,
walk), a grammatical word sequence, and a character string. The granularity experiments
(§13.6–§13.9) were really probing **one layer at a time**; the design goal is to hold **all three at
once**.

## 2. Meaning resolution — the in-vocab / OOV split (the crux)

The layers differ in *how a unit acquires meaning*:

- **In-vocabulary word → meaning is KNOWN.** It has a learned representation (the word-level
  efficiency we measured: grammar+semantics learned per-word, no spelling overhead).
- **Out-of-vocabulary word → meaning is FOUND**, two ways that reinforce each other:
  1. **Composition (level C ↑):** build the word's representation from its *characters* — morphology
     generalises ("happily" lands near "happy" it has never seen; cf. CharCNN word embeddings,
     fastText). This is the answer to "train the character-level relation to a known word meaning."
  2. **Context / proximity (level B):** the surrounding known words constrain the unknown one — exactly
     how a human reads a word they've never met.

So **level C is the OOV safety net for level B**, and **level A is the semantic plan that keeps level B
on topic.** A unified model gets word-level efficiency *without* the closed vocabulary, *and* a coarse
plan that fights the topic-drift we still see (the wise-girl→"He saw" slips).

## 3. The hard part — ONE coherent expression, not three models

The user's framing is the whole challenge: *the layers technically coexist and need a simple coherent
means to express.* It is easy to bolt three models together; the goal is a **single representation /
single objective** in which a token simultaneously carries its surface (chars), its lexical identity
(word), and its semantic role (gist) — and generation can move between resolutions cleanly. The "tree"
was a clumsy attempt at this; the levels above are the real structure, and they happen to align with
**tokenization granularity**, which is concrete and buildable (no parser, no syntax labels).

## 4. Candidate unifying mechanisms **[OPEN — this is the research]**

- **4-A. Multi-resolution residual stream (hourglass / U-Net).** One network, one residual stream, three
  resolutions: **chars → pool to words → pool to gist → process → unpool to words → unpool to chars.**
  Pooling boundaries are induced by whitespace (words) and by content-word/saliency (gist). This is the
  "simple coherent means": the *same* stack is all three levels at different depths. (Cf. CANINE, ByT5,
  Charformer, Megabyte/MambaByte, hourglass transformers — char/byte I/O, word-ish processing.)
- **4-B. Multi-scale per-position embedding.** Each char position's vector = char-emb ⊕ word-emb
  (lookup if in-vocab, CharCNN-composed if OOV) ⊕ gist-emb (pooled). The model *reads* all three levels
  at every position; one stream, three views.
- **4-C. Diffusion across resolutions (fits our engine best).** Keep the masked-diffusion canvas at the
  **word** level (we measured this is where grammar is learned cheaply), give each word its embedding via
  **char-composition** (no OOV), and condition the denoiser on a **gist vector** (a pooled content-word
  plan, itself optionally diffused first). Generation = plan gist → denoise word canvas → spell any OOV
  words from chars. Reuses our `Denoiser`, char/word tokenizers, and the iterative sampler.

## 5. What this volume already proved that grounds each level

- Level C works and is OOV-proof but slow to grammar (char-level, §13.6/§13.8).
- Level B learns grammar fast but is OOV-blind (word-level TinyStories: grammatical by step ~2k).
- Subword (B↔C bridge) is *coherent once decoded right* (§13.9) — so the surface/word coupling is not
  inherently broken; our earlier "salad" was the decode bug, not the representation.
- Coherence is **recipe/decode-bound**, and the remaining gap everywhere is **semantics / topic drift** —
  precisely what **level A (gist plan)** is meant to supply.

So the empirical case for this design is: the pieces each work in isolation; the open question is the
**unifying mechanism** of §4, and whether it beats a well-trained flat subword diffusion model (which we
must understand first — see parked criteria).

## 6. Minimal first build (when unparked) **[PHASE 1]**

§4-C on our stack: char→word pooling encoder (CNN over a word's chars → word embedding; lookup blended
in for frequent words), word-level masked-diffusion denoiser (existing block), char upsampling head for
the loss + OOV output, and a single pooled **gist conditioning vector**. TinyStories first (small vocab,
known-coherent), compared against the char- and word-level baselines we now have. Go/no-go: does
char-composition remove the OOV cliff *and* keep word-level's fast grammar, and does the gist vector cut
topic drift?

## 7. Open questions & unpark criteria

1. **The unifying mechanism (§4)** — residual-stream pooling vs multi-scale embedding vs
   diffusion-across-resolutions. Which gives "one coherent system" rather than three bolted models?
2. **Where do the pooling boundaries come from?** Whitespace gives words for free; the **gist** level is
   the hard one — saliency/content-word selection, learned, or a second diffusion over a content-word
   subset. **[OPEN]**
3. **Does char-composition actually rescue OOV meaning** at our scale, or only at large scale?
4. **Unpark only when:** the DiffusionGemma/LLaDA training research is done AND a well-trained flat
   subword diffusion model still shows the semantic/topic-drift gap this design targets.
