# Tokenizer Design

Forward-looking design for the next tokenizer — the **JOIN / implicit-space** scheme that
supersedes the current "space-as-token" BPE. Not yet implemented; gated on incremental BPE
(done) + ingest single-thread optimisation. Where it differs from today's behaviour, the
current state is noted. Concepts we do **not** yet employ are captured as one-liners in §8 so
the intent is not lost.

## Guiding principle

**Collapse the common case to implicit/free; specialise the rare case with an explicit
token; and DERIVE which cases to specialise from the corpus distribution** (not guesswork —
the same data-driven method used for model sizing). Information-theoretically: make frequent
patterns cost ~0 and mark structured/rare patterns explicitly.

The lossless contract is unchanged: `decode(encode(x)) == normalize_text(x)` (lossy only on
typographic glyph identity, which `normalize_text` already folds to ASCII). Every pattern
below must round-trip; the configurator already verifies this per chunk.

## 1. Base alphabet — complete, not corpus-only  *(change vs today)*

Today the base alphabet is the **distinct byte values used in the corpus** + markers, so a
user prompt containing a byte the corpus never had has no token and is **silently dropped in
`encode()`**. The tokenizer must be *total*: include the **full 256-byte base set** so any
input is encodable (byte-level in the worst case), even out-of-corpus characters and
arbitrary UTF-8. Cost: 256 vs ~82 base tokens out of 2048+ — negligible. Unused bytes get
weakly-trained embeddings but remain encodable (and §8 misspelling-training improves them).
This shifts base ids → a new vocabulary, so it lands with this scheme as a clean break.

## 2. Whitespace — implicit single space, specialise the rest

| Pattern | Frequency (tinystories) | Encoding |
|---|---|---|
| Single space between content | 40.7% | **implicit** (default boundary) — free |
| No space at a boundary | ~16% | **JOIN** sentinel (also the 2-token compose op) |
| Single newline | 1.4% | `NEWLINE` token |
| Paragraph break `\n\n` | common | `PARA` token (1 token, not 2 newlines) |
| Indentation / repeated spaces | code/structured | **run-length** `SPACE2/3/4/8`, `TAB`, `TAB2…` |

Run-length widths are chosen from the **measured** whitespace-run histogram, not assumed.
After a `NEWLINE`, an 8-space indent is one `SPACE8`, not eight tokens.

## 3. Punctuation — directional pairs

Bare `"` is ambiguous (open == close). Mint **directional** tokens classified by spacing
context at encode time:

* `OPEN_DQUOTE` = ` "` (space-before, glue-after) ; `CLOSE_DQUOTE` = `" ` (glue-before, space-after)
* same for `'` where it is a quote (not a contraction apostrophe, which `normalize_text`/casing already handle).

This (a) disambiguates open/close, (b) gives the model a **matched pair to attend**
(start↔end of a quoted region, like brackets), and (c) bundles the common spacing so no
separate JOIN is needed. Both decode back to the same byte → lossless. Mismatched/line-initial
quotes (messy web text) fall back to a bare quote + JOIN. Brackets/parens are already
directional characters, so they need only JOIN for spacing.

## 4. Word encoding & the SPELL region

A word-unit is BPE-encoded to N sub-tokens; the encoding of the *region* depends on N:

| N sub-tokens | Encoding | Rationale |
|---|---|---|
| 1 (common word) | bare token | nothing needed |
| 2 (`sun`+`day`) | **JOIN** between | 1 token < 2 delimiters |
| ≥3 (OOV / misspelled / `DoThisFooBar`) | **`SPELL_START` … `SPELL_END`** | 2 delimiters ≤ N−1 JOINs, and signals a spelled-out region |

Within a multi-token word, implicit-space would wrongly insert spaces between sub-tokens, so
we suppress with JOINs (N=2) or **encapsulate** (N≥3). The SPELL region is a *spaceless
group* — implicit space applies before `SPELL_START` and after `SPELL_END`, never inside.
One encapsulation mechanism therefore covers **OOV/rare words, misspellings, acronyms, and
CamelCase / function-like names** (`DoThisFooBar`) as compound spaceless groups, with internal
case markers as usual. The delimiters are a matched pair the model can attend (start↔end),
the same pattern as the directional quotes; whether that attention earns the extra token is
an open empirical question (§8). The `sun+day` case deliberately uses JOIN, not
encapsulation — encapsulating two known tokens is strictly worse.

## 5. Decode FSM & round-trip

Reconstruct left-to-right with two state bits: `pending_space` (emit a space before the next
content) and `pending_case` (cap/up the next content). Transitions:

* **content** token: if `pending_space` emit ' '; emit text (apply `pending_case`); set `pending_space = true`.
* **JOIN**: clear `pending_space` (the next content glues).
* **whitespace / directional-quote / SPELL_* tokens**: emit their literal spacing; set `pending_space` explicitly (false inside a SPELL region, true after a quote's trailing space, etc.).
* **CAP/UP markers**: set `pending_case`; do not change spacing.

Start state: `pending_space = false` (no leading space). The exact transition table is the
correctness-critical part and is gated by the per-chunk round-trip check.

## 6. Special-token inventory

`JOIN`, `NEWLINE`, `PARA`, `SPACE2/3/4/8`, `TAB`(+levels), `OPEN_DQUOTE`/`CLOSE_DQUOTE`,
`OPEN_SQUOTE`/`CLOSE_SQUOTE`, `SPELL_START`/`SPELL_END`, `CAP`/`UP` (existing). ~18–24 tokens
out of 2048+. **Only mint a special token if it clears a measured frequency floor** — rare
specials dilute training signal.

## 7. Data-driven specialisation (the step before implementing)

Extend the corpus analysis to histogram (a) whitespace-run lengths and (b) punctuation
spacing contexts, then pick the specialised token set from the head of those distributions.
Validate the scheme by **tokens/doc reduction** *and* an **ablation** that the model actually
uses the structure (perplexity/coherence improves, not just compression).

## 8. Concepts we don't yet employ (one-liners)

* **Misspelling robustness** — train on corrupted spellings → correct word; the SPELL region is the denoising unit (`quix`→`quick`), a natural fit for the diffusion paradigm.
* **CamelCase / acronym compounds** — `DoThisFooBar`, `NASA` as SPELL-encapsulated spaceless groups with internal case markers.
* **Attention-based pairing** — open/close quote and SPELL_START/END as matched pairs the model learns to bind; net benefit is empirical.
* **Cross-JOIN BPE merges** — optionally let BPE absorb very frequent compounds into one token (a knob), trading composition for compactness.
* **Digit-runs as word-units** — numbers BPE-merge like words (today digits are standalone, which implicit-space would mis-split).
* **Factored morphology** — lemma + separable case/number/possession/tense axes (a small morphological analyser) layered above BPE.
* **Verbosity/latency slider** — meaning-preserving terseness dial once a gist/coarsening generator exists.
* **Mergeable vocabularies** — sum two corpora's scan-states (counts kept) for a joint or incremental tokenizer across sessions.

## 9. Dependencies & sequencing

1. Incremental BPE — **done** (re-tokenization must be cheap to iterate schemes).
2. Ingest single-thread optimisation (then parallel passes) — fast cycle times on a partial corpus.
3. Measure the ws/punct distribution (§7) → fix the special-token set from data.
4. Complete base alphabet (§1) + implement the decode FSM (§5) + round-trip tests.
5. Retrain (clean version break).
