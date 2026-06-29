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

**Measured on FineWeb** (see [TOKENIZER_WS_PUNCT_STATS.md](TOKENIZER_WS_PUNCT_STATS.md) — % of
whitespace *runs*):

| Pattern | FineWeb (runs) | Encoding |
|---|---:|---|
| Single space between content | **97.2 %** | **implicit** (default boundary) — free |
| No space at a boundary | 92 % of word→punct | **JOIN** sentinel (also the 2-token compose op) |
| Single newline | 2.6 % | `NEWLINE` token |
| Paragraph break `\n\n` | 0.13 % | `PARA` token (1 token, not 2 newlines) |
| Indentation / repeated spaces | **0.004 %** (prose) | **run-length** `SPACE2/3/4/8`, `TAB`, `TAB2…` — *corpus-conditional* |

Run-length widths are chosen from the **measured** whitespace-run histogram, not assumed. After a
`NEWLINE`, an 8-space indent is one `SPACE8`, not eight tokens. **For clean prose (FineWeb)
multi-space runs and tabs are statistically nonexistent, so `SPACE_N`/`TAB` are NOT minted** — they
stay specified and are minted only when a corpus's measured frequency clears a floor (code/markdown
/TSV). This is the "only mint a special if it clears a measured floor" rule (§6) applied per corpus.

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

**Measured (FineWeb, [stats](TOKENIZER_WS_PUNCT_STATS.md) §C):** `"` is **48.8 % open / 41.1 %
close** (90 % cleanly directional) → `OPEN_DQUOTE`/`CLOSE_DQUOTE` are decisively justified. `'`
is **81.6 % glue/glue = contraction apostrophes** (already kept inside the word-unit), only ~18 %
true quotes → `OPEN_SQUOTE`/`CLOSE_SQUOTE` **deferred** (low floor; bare `'` + JOIN fallback).
`(`/`)`/`[`/`]` are ~94 %/≈99 % directional already → char + JOIN, no new token.

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

Full specified set: `JOIN`, `NEWLINE`, `PARA`, `SPACE2/3/4/8`, `TAB`(+levels),
`OPEN_DQUOTE`/`CLOSE_DQUOTE`, `OPEN_SQUOTE`/`CLOSE_SQUOTE`, `SPELL_START`/`SPELL_END`,
`CAP`/`UP` (existing). **Only mint a special token if it clears a measured frequency floor** —
rare specials dilute training signal.

**Minted for FineWeb** ([measured](TOKENIZER_WS_PUNCT_STATS.md)) — the active set is small:

| Minted (prose) | Corpus-conditional (mint when measured) | Deferred |
|---|---|---|
| `JOIN`, `NEWLINE`, `PARA`, `OPEN_DQUOTE`, `CLOSE_DQUOTE`, (`CAP`/`UP`) | `SPACE2/3/4/8`, `TAB`(+levels) | `OPEN_SQUOTE`/`CLOSE_SQUOTE` |
| `SPELL_START`/`SPELL_END` pending the §4 word-`N` measurement | — | — |

That is **5 new tokens** for clean prose, not ~18–24. The whitespace/tab specials are not deleted
— the configurator histograms each corpus at configure time and mints the ones that clear the floor
(code/markdown would add `SPACE_N`/`TAB`). See [stats](TOKENIZER_WS_PUNCT_STATS.md).

## 7. Data-driven specialisation (the step before implementing)

Extend the corpus analysis to histogram (a) whitespace-run lengths and (b) punctuation
spacing contexts, then pick the specialised token set from the head of those distributions.
Validate the scheme by **tokens/doc reduction** *and* an **ablation** that the model actually
uses the structure (perplexity/coherence improves, not just compression).

**DONE for FineWeb** → [TOKENIZER_WS_PUNCT_STATS.md](TOKENIZER_WS_PUNCT_STATS.md) (the measured
ws-run + punct-spacing histograms and the token decisions they imply; §2/§3/§6 above are now
grounded in it). Still to measure: the word sub-token `N` histogram (§4, needs BPE → in the
configurator). The ablation runs after the scheme is implemented.

## 8. Concepts we don't yet employ (one-liners)

The unifying bet: several of these are the **same shape** — a *paired-delimiter region* (open + close
markers the model attends as a bracket) and/or a *separable affix* token. If enough cases reduce to
one or two mechanisms, the system becomes elegant: `OPEN_DQUOTE`/`CLOSE_DQUOTE` and
`SPELL_START`/`SPELL_END` are the first two instances; the rest below should reuse them.

* **Extensible paired-delimiter regions** — generalise the directional-quote mechanism into a *family*
  of OPEN/CLOSE pairs driven by a table: `()` `[]` `{}` `<>`, back-/single-quotes, etc. (today only `"`).
  One classify-by-spacing-context encoder + one `in_region` decoder state covers them all. Brackets are
  already directional glyphs (need only spacing), so this is mostly a data-driven token table + the
  generic FSM. Source code (lots of `(){}[]<>`) is the natural stress corpus — **dogfood our own C++**.
* **CamelCase / snake_case structural splitting** — `ThisIsMyAwesomeFunction`, `my_snake_case`,
  `kDoThisFooBar` are compound identifiers. Today they fall into one SPELL group (N≥3) as opaque bytes.
  Better: split on the *internal boundaries* (capital-camel-joins; `_` snake-joins) into real sub-word
  tokens joined by a marker the model can attend — so `Function`/`Awesome` reuse their normal word
  tokens. A region/JOIN concept, not opaque spelling. (`save_scan_state` etc. are snake-join examples.)
* **`'s` possessive (separable affix)** — make the possessive clitic `'s` its own token rather than
  `'` + `s`, so it attends BOTH the owner (name) and the owned object — a common, learnable relation.
  A concrete instance of factored morphology; if several affixes (`'s`, `n't`, `'re`, `-ing`, `-ed`,
  plural `-s`) clear a frequency floor, a small **separable-affix** mechanism becomes worth it.
* **Acronym generation / spelling-awareness** — a word *token* hides its spelling, so the model can't
  form/expand acronyms (NASA ↔ National Aeronautics…) from a token alone. Needs a path to the verbatim
  letters: a `SPELL` token that forces a word to be emitted letter-by-letter, reachable by a
  reasoning/thinking loop. Likely a future investigation unless an elegant in-line form appears.
* **Misspelling robustness** — train on corrupted spellings → correct word; the SPELL region is the denoising unit (`quix`→`quick`), a natural fit for the diffusion paradigm.
* **Vocab size + word-boundary-aligned BPE** — when a word must split, prefer sub-tokens that are
  themselves *whole shorter words* (`sun`+`day`) over arbitrary fragments (`su`+`nd`+`ay`), and consider
  a larger vocab so common words stay N=1. A merge-scoring / vocab-budget lever to investigate (today
  plain frequency-greedy BPE can pick mid-word fragments). Measure with the word-`N` histogram.
* **Attention-based pairing** — open/close quote and SPELL_START/END as matched pairs the model learns to bind; net benefit is empirical.
* **Cross-JOIN BPE merges** — optionally let BPE absorb very frequent compounds into one token (a knob), trading composition for compactness.
* **Digit-runs as word-units** — numbers BPE-merge like words (today digits are standalone, which implicit-space would mis-split).
* **Factored morphology** — lemma + separable case/number/possession/tense axes (a small morphological analyser) layered above BPE.
* **Verbosity/latency slider** — meaning-preserving terseness dial once a gist/coarsening generator exists.
* **Mergeable vocabularies** — sum two corpora's scan-states (counts kept) for a joint or incremental tokenizer across sessions.
* **Testing methodology** — round-trip property tests over (a) worked examples per construct, (b) a
  **dogfooded** corpus of the project's own source (exercises brackets/braces/CamelCase/snake_case),
  and (c) **malformed/random-mutation stress** (fuzz: random byte flips/insertions must still satisfy
  `detokenize(encode(x)) == normalize_text(x)`). Capture each real failure (e.g. `NASA's`) as a pinned case.

## 9. Dependencies & sequencing

1. Incremental BPE — **done** (re-tokenization must be cheap to iterate schemes).
2. Ingest single-thread optimisation, then **parallel passes — done** (`0643d19`, ~5×) — fast
   cycle times on a partial corpus.
3. Measure the ws/punct distribution (§7) → fix the special-token set from data —
   **done** ([stats](TOKENIZER_WS_PUNCT_STATS.md)); word-`N` histogram (§4) still pending in the configurator.
4. Complete base alphabet (§1) + decode FSM (§5) + round-trip tests — **DONE** (`sub0::tok`,
   `LearnOptions::join_scheme`, default off). Complete 256-byte base + always-on
   `CAP/UP/JOIN/NEWLINE/PARA/OPEN_DQUOTE/CLOSE_DQUOTE/SPELL_START/SPELL_END` (n_base 265). The
   encode/decode mirror a pending-space state `dps` (+ `in_spell`): implicit single space, JOIN
   for glue, `NEWLINE`/`PARA`, verbatim-whitespace fallback, **directional double quotes (§3)**,
   and **`SPELL` encapsulation for N≥3 words (§4)** with `CAP`/`UP` carried across a word's
   sub-tokens. Validated by `sub0_tok_tests` (67 assertions, both schemes). Single-quote
   directional tokens stay **deferred** (data: `'` is 82% contractions, already in the word-unit).
5. **Wire into the pipeline + measure — DONE** (`SUB0_JOIN_TOKENIZER` build flag → `sub0-configure
   --join` → `corpus.tok` via `sub0::tok::encode`; engine deserialises + uses the FSM). tinystories
   A/B (same d160 model): **−29.0% tokens, bits/byte −4.2% (matched GPU) to −8.7%**, lossless,
   coherent generation. The configurator reports the word-`N` histogram (tinystories: N1 92.9% /
   N2 4.4% / N≥3 2.7% SPELL). Flag defaults off, so existing models/`corpus.tok` stay valid; a
   retrain under the flag is the deliberate clean break.
