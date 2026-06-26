# Tokenizer ws/punct distribution — measured (the data behind the JOIN token set)

This is step §7/§9.3 of [TOKENIZER_DESIGN.md](TOKENIZER_DESIGN.md): **derive which cases to
specialise from the corpus distribution**, not guesswork. Every special-token decision below
is justified by a measured frequency, and any token that does *not* clear a frequency floor is
explicitly **dropped or made corpus-conditional** (per the design's "only mint a special token
if it clears a measured frequency floor" rule).

## Method

- **Corpus:** FineWeb-Edu, first **534,586,828 bytes** of `data/fineweb_smoke.txt` (a 1 GB head
  slice of the 43 GB corpus). A stationary distribution — 0.5 GB is representative.
- Typographic glyphs folded to ASCII first (curly quotes/dashes/ellipsis → ASCII), mirroring
  `normalize_text`, so curly quotes count as `"`/`'`. Whitespace is untouched by normalization.
- Measurement is a regex histogram over the raw glyph-folded bytes (scratch tool, not committed;
  the keeper logic is reimplemented in C++ in the configurator). It approximates token boundaries
  by character class — distribution-accurate for *deciding* the token set; exact per-word-unit
  counts will come from the configurator's truecase pass when the scheme is implemented.
- **Caveat:** numbers are FineWeb (clean web prose). Code/markdown/TSV corpora have very
  different whitespace — see *Corpus-conditional* below. The configurator must measure per-corpus.

## A. Whitespace runs — implicit single space wins overwhelmingly

86,819,828 maximal whitespace runs:

| Run type | Count | % of runs | Decision |
|---|---:|---:|---|
| **single space** | 84,418,776 | **97.23 %** | **implicit default boundary — free** |
| spaces ≥ 2 | 3,218 | 0.004 % | **drop SPACE2/3/4/8 for prose** (corpus-conditional) |
| newline (any) | 2,397,834 | 2.76 % | **mint `NEWLINE`** |
| → exactly 1 `\n` | 2,285,952 | — | `NEWLINE` |
| → exactly 2 `\n` (paragraph) | 111,882 | 0.13 % | **mint `PARA`** (1 token vs 2 `NEWLINE`) |
| → ≥ 3 `\n` | 0 | 0 % | none (corpus has no 3+ runs) |
| tab run | 0 | 0 % | **drop `TAB`/`TAB`-levels for prose** (corpus-conditional) |

Pure-space run lengths: len 1 = 100.00 % of space-only runs; len 2 = 3,153, len 3 = 49, len 4 = 12,
len ≥5 = 4 total. **Multi-space runs are statistically nonexistent in prose** — the `SPACE_N`
run-length machinery from the design §2 buys nothing here. Keep it specified but **gate minting on
a measured floor** (it pays off only on indented/code corpora).

## B. Word↔punctuation glue — the JOIN workhorse

Implicit-space inserts a space before every content token, so any *glued* punctuation needs a
`JOIN` to suppress it.

| Boundary | Glued (no space) | Spaced | % glued |
|---|---:|---:|---:|
| word → punct (`word,` `word.` `word!`) | 12,861,235 | 1,118,071 | **92.0 %** |
| punct → word (`(word` `,word` `.Next`) | 3,544,855 | 9,153,199 | 27.9 % |

**`JOIN` is essential and high-frequency.** Trailing punctuation alone is ~12.9 M glues in 0.5 GB
(≈ 24 M/GB, ≈ 1 B over the full corpus). After-punct glue is rarer (28 %) because most punct→word
is `, ` / `. ` (comma/period then space); the glued remainder is dominated by openers (`(`, quotes)
handled by the directional tokens below, leaving the rest to `JOIN`.

## C. Quotes & brackets — directional split is decisive for `"`, marginal for `'`

`space` = whitespace neighbour, `glue` = non-whitespace neighbour. **OPEN** = space-before/glue-after,
**CLOSE** = glue-before/space-after.

| Char | Total | OPEN (sp/glue) | CLOSE (glue/sp) | glue/glue | Decision |
|---|---:|---:|---:|---:|---|
| `"` | 776,671 | **48.8 %** | **41.1 %** | 9.6 % | **mint `OPEN_DQUOTE` / `CLOSE_DQUOTE`** — 90 % cleanly directional |
| `'` | 832,244 | 8.2 % | 10.1 % | **81.6 %** | **defer** — 81.6 % glue/glue is contractions (handled by word-unit); quotes are only ~18 % |
| `(` | 625,114 | **94.2 %** | 0.1 % | 5.0 % | char + `JOIN` (already directional; glues after 94 %) |
| `)` | 636,794 | 0.3 % | **52.7 %** | 46.5 % | char + `JOIN` (glues before ≈ 99 %) |
| `[` | 39,457 | 81.6 % | 0.4 % | 15.5 % | char + `JOIN` |
| `]` | 39,336 | 0.9 % | 65.0 % | 31.9 % | char + `JOIN` |

`"` is strongly bimodal (open vs close) → directional tokens disambiguate it *and* hand the model
a matched pair to attend. `'` is dominated by contraction apostrophes (`don't`, `it's`) which the
existing word-unit logic already keeps inside the word, so `OPEN_SQUOTE`/`CLOSE_SQUOTE` clear only a
low floor (~70 K each) — **deferred**; messy/quote `'` falls back to bare `'` + `JOIN`. Brackets are
already distinct open/close glyphs, so they need only `JOIN` for the glue side, no new token.

## Decision summary — FineWeb special-token set

Derived from the data above (clean web prose):

| Token | Status | Frequency basis |
|---|---|---|
| implicit single space | **default (free)** | 97.2 % of ws runs |
| `JOIN` | **mint — essential** | 92 % of word→punct boundaries glue |
| `NEWLINE` | **mint** | 2.6 % of ws runs |
| `PARA` (`\n\n`) | **mint** | 111,882 (0.13 %); 1 token vs 2 |
| `OPEN_DQUOTE` / `CLOSE_DQUOTE` | **mint** | `"` 49 % open / 41 % close |
| `CAP` / `UP` | keep (existing) | corpus-aware truecasing |
| `SPACE2/3/4/8` | **drop (corpus-conditional)** | multi-space ≈ 0.004 % in prose |
| `TAB` / `TAB`-levels | **drop (corpus-conditional)** | 0 tabs in prose |
| `OPEN_SQUOTE` / `CLOSE_SQUOTE` | **defer (optional)** | quote-`'` only ~18 %; rest are contractions |

**Net: 5 new tokens** (`JOIN`, `NEWLINE`, `PARA`, `OPEN_DQUOTE`, `CLOSE_DQUOTE`) + existing
`CAP`/`UP` — far below the speculative ~18–24 in design §6. Fewer rare specials = less diluted
training signal, exactly the §6 caution realised with data.

## Corpus-conditional minting (the general rule)

The whitespace/tab specials are **dropped for FineWeb but not deleted from the design** — they
exist to be minted *when the corpus warrants it*. The configurator should, at configure time,
histogram ws-runs + punct spacing for the *actual* corpus and mint a special only when its
frequency clears a floor (e.g. ≥ 0.1 % of runs). For a code/markdown/TSV corpus, `SPACE2/4/8`,
`TAB`, and `TAB`-levels would clear it and `PARA`/indent tokens would shift; for prose they don't.
This makes the token set **data-driven per corpus**, consistent with the model-sizing methodology.

## Open measurement — word sub-token N (JOIN vs SPELL threshold)

Design §4 routes a word-unit by its post-BPE sub-token count `N` (1 = bare, 2 = `JOIN`,
≥3 = `SPELL_START…SPELL_END`). That needs the BPE output, so it is measured **in the configurator**
(C++), where the word table + merges live — not in this byte-level scan. Pending: histogram `N`
over the word-frequency table to confirm the `≥3 → SPELL` cut-over and size the SPELL floor.

## Reproduce

Scratch tool (not committed): `scratchpad/ws_punct_stats.py <corpus> [byte_cap]`. Keeper logic →
configurator histogram (C++) when the scheme is implemented.
