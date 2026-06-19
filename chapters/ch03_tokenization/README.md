# Chapter 03 — Tokenization

Before a transformer sees any text, that text becomes integers (token IDs):

```
raw text  →  pre-tokenization  →  BPE merges  →  token IDs  →  embeddings
```

`main.cpp` builds **Byte-Pair Encoding (BPE)** end to end: count adjacent pairs →
merge the most frequent → repeat to a target vocab size; the GPT-2 space-marker
convention (`Ġ` = leading space); encode/decode round-trip; save/load
(`vocab.json` + `merges.txt`). Run it:

```bash
cmake --build --preset debug --target ch03_tokenization
./build-debug/bin/ch03_tokenization
```

## The thing BPE hides: granularity is a *choice*, and it interacts with the model

BPE gives you a **dial** — `vocab_size` — between two extremes. It's easy to treat
that dial as "bigger = better compression" and move on. It isn't that simple: the
granularity you pick changes *what a single prediction error looks like*, and that
matters enormously for some model families.

`BPETokenizer` now exposes the whole spectrum (the same class, three constructions):

| granularity | factory | vocab (Shakespeare) | a token is… | a wrong token is… |
|---|---|---|---|---|
| **char** | `char_level()` | ~101 | one code point | a 1-char blemish (`nighthe`) — still a near-word |
| **BPE-N** (small) | `train(corpus, 512)` | 512 | a **fragment** (`con`,`ick`,`ers`) or, for top words, a whole word | a non-word chunk once fragments are mis-assembled |
| **word** | `word_level()` | ~33k | a whole word | a *different real word* (never a non-word) |
| **large subword** | (e.g. GPT-2/Gemma) | 50k–262k | mostly whole/frequent units | usually still a real unit |

Two ends of the dial (`char`, `word`) share a property: **a token error stays a real
unit** — a real letter, or a real word. The small-BPE middle is the awkward zone: most
of its vocabulary is *fragments* that only mean something once glued to their
neighbours.

### The consumer matters: autoregressive vs diffusion

Here is the part that surprised us, and the reason this section exists.

A small-BPE vocabulary is **fine for an autoregressive model**. AR generation is
left-to-right: when it emits the fragment `con`, the *next* step is conditioned on
`con`, so it tends to continue with `tinue`/`quer`/`demn` — a real word. Left-context
keeps fragments coordinated.

The **same** vocabulary is *toxic for a masked-diffusion model*. Diffusion fills many
masked positions in parallel, each from its own marginal distribution, with no
left→right constraint. So it can confidently commit `con`, then `ick`, then `ers` —
each locally probable, the concatenation a non-word (`conickers`). Char-level sidesteps
this because its atoms are single characters (a wrong one is a small blemish); word-level
sidesteps it because every token is already a whole word.

So **"which tokenizer?" is not answerable without asking "which model consumes it?"**
On the diffusion side of this repo, exactly this bit us: a 512-merge BPE produced
fluent-looking *non-word salad* (`itome`, `WIACHER`, `conickeders`) at every sampling
temperature, while char-level on the *same corpus and engine* produced real English with
verse structure. Full write-up and the controlled experiments:
[`diffusion/TRAINING_DESIGN.md` §13.6–§13.7](../../diffusion/TRAINING_DESIGN.md), and the
diffusion trainer's `--char-level` / `--word-level` flags
([`diffusion/chapters/ch29_diffusion_training`](../../diffusion/chapters/ch29_diffusion_training)).

### Scale rehabilitates subwords

If small BPE is bad, why does every frontier model use ~50k–262k subword vocabularies?
Because **at scale the dial moves back toward "real units."** A 262k vocabulary (e.g.
DiffusionGemma) is mostly whole words and frequent morphemes, trained on trillions of
tokens — the fragments that survive are common and well-modelled. The rule of thumb that
falls out:

> Tokenizer granularity should track data/model scale — **char/byte at tiny scale,
> large subword at large scale.** A 512-merge BPE on ~5M characters is the wrong point
> on that curve.

### Takeaways

- BPE's `vocab_size` is not just a compression knob — it sets the **granularity of
  errors**.
- The two ends of the spectrum (char, word) keep errors as real units; small-BPE
  emits fragments.
- Whether fragments are safe depends on the **consumer**: AR coordinates them via
  left-context; parallel diffusion does not.
- Match granularity to scale: tiny data → char/byte; large data → large subword.
