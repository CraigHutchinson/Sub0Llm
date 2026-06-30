# Tokenizer Vocabulary Analysis

A `sub0-configure --dump-vocab <prefix>` mode writes three readable files for reviewing how the
tokenizer compacts a corpus (scan + learn, then exit — no `corpus.tok`/header):
- `<prefix>.corpus_vocab.txt` — every unique word-unit (truecased) + occurrence count, freq-desc.
- `<prefix>.token_vocab.txt` — every learned token (byte / marker / merge), its text, and the corpus
  occurrences it covers (over/under-used tokens + tail waste visible at a glance).
- `<prefix>.ngrams.txt` — k-char substring frequencies (k=2..16) over the corpus vocab, weighted by
  word count — the **order-independent** view of the common sub-chunks (BPE merges depend on greedy
  order; this does not).

## Measured on TinyStories (vocab as the knob)

| | vocab 2048 | vocab 8192 |
|---|---:|---:|
| unique corpus word-units | 10,436 | 10,436 |
| merges (total) | 1,779 | 7,923 |
| **merges with ZERO occurrence (dead)** | **24** | **831 (10.5%)** |
| unique words that became a single token | ~2,000 | 7,092 (68%) |
| base bytes with zero occurrence | 199/256 | 199/256 |
| markers with zero occurrence | 13/13 | 13/13 |

Two findings:

**1. The vocabulary is too small.** TinyStories alone has **10,436 unique word-units**; at vocab 2048
only the ~2k most frequent become single tokens and the other ~8k are split. FineWeb has *hundreds of
thousands* of unique words, so 2048 is drastically undersized — heavy splitting, poor compression
(this is the "consider if we have enough vocab" question, answered: **no, raise it**, especially for
FineWeb).

**2. BPE wastes ~10% of the vocabulary on dead intermediate merges.** At vocab 8192, **831 merges
(10.5%) have zero occurrence in the final encoding** — they are intermediate greedy merges (e.g.
`ch`+`en`=`chen`) that a later, longer merge superseded, so they never appear as a final token. That
is 831 slots spent on nothing. The greedy *bottom-up* order is the cause; it cannot retract a merge
that later turns out wasteful.

The unused base bytes (199/256) are the deliberate "total tokenizer" safety net (any byte is
encodable) — justifiable, not waste. The unused markers (SPACE_N/TAB etc.) simply don't occur in
clean prose — they earn their keep on code/markdown corpora.

## The better approach (matches the "minimise tokens by occurrence" intuition)

The proposed direction — take the corpus vocab, and **choose the sub-token set that minimises the
total tokens needed to encode it, weighted by occurrence** — is exactly the **Unigram Language Model**
tokenizer (Kudo 2018; the default in SentencePiece), and it beats BPE precisely because it is
**top-down and global** rather than greedy bottom-up:

1. **Seed** a large candidate set of sub-tokens — every frequent substring (our `ngrams.txt` is exactly
   this: all k-char chunks with counts, the "longest common sub-sections" like `tion`/`ing`/`ience`).
2. **Score** each candidate by how much *total encoding cost* (Σ word_freq × tokens-per-word under a
   Viterbi/optimal segmentation) it saves.
3. **Prune** to the target vocab size by repeatedly dropping the candidates whose removal least
   increases total cost (an EM loop), so the final set is globally optimised for the corpus — **no
   dead intermediate tokens**, and every slot earns its occurrence.

The "tree of open options refining toward the target size" in the brief is the prune-search; the
Viterbi segmentation is the optimal-tokens-per-word objective. This is a **ground-up vocabulariser**
that reuses our existing pieces: the `Scan` word table (corpus vocab + counts) as input, the n-gram
table as the candidate seed, and the existing JOIN encode/decode FSM + the round-trip tests as the
regression guard (the *vocabulary* changes; the spacing/casing scheme does not).

## Reporting the ideal vocab size (the "crux")

`--dump-vocab` also writes `<prefix>.vocab_curve.txt` and prints the **ideal vocab size**. Each BPE
merge's selection-count is *exactly* the corpus tokens it removes, so `total_word_tokens(n_base+k) =
total_word_bytes − Σ_{i<k} merge_count[i]` gives the **whole bytes/token-vs-vocab curve from one
learn**. As vocab collapses toward the base alphabet, bytes/token → 1.0 (character encoding — the
"devolves to char" floor); as it grows the curve flattens. The knee is reported as the vocab
capturing X% of the total achievable token reduction.

TinyStories (learn to vocab 16000):

| vocab | bytes/token | tokens saved by that merge |
|---:|---:|---:|
| 269 | 1.00 | — (char floor) |
| 525 | 2.30 | 7,611 |
| 1293 | 3.17 | 933 |
| 2317 | 3.50 | 267 |
| 4365 | 3.73 | 69 |
| 8461 | 3.80 | 5 |
| 11035 | 3.81 | 2 (BPE exhausts useful merges) |

→ **Ideal vocab (knee): 90% of compression @ ~1008, 95% @ ~1694, 99% @ ~3965.** bytes/token climbs
1.0 → 3.50 by vocab ~2300, then only → 3.81 over the next 5× vocab. So for TinyStories the current
2048 is well-placed; below ~1000 it slides toward char encoding, above ~4000 it pays vocab for <1%.
BPE also hits a hard ceiling at 11035 (no pair occurs ≥ min_merge). A larger/richer corpus (FineWeb)
shifts the whole curve right — run the curve there to size its vocab.

## Next steps

1. Dumps are in place — review `corpus_vocab` / `token_vocab` / `ngrams` for FineWeb (run
   `sub0-configure --corpus data/fineweb_smoke.txt --dump-vocab …`; slower, uses the `.words` cache).
2. Decide the target vocab size from the FineWeb `corpus_vocab` size + a coverage curve (what fraction
   of occurrences single-token coverage buys at 8k / 16k / 32k).
3. Prototype the Unigram vocabulariser (candidate seed → Viterbi cost → prune) as an alternative to
   `learn()`'s BPE, A/B by **bits/byte** (scheme-independent) and by the dead-token count (should be ~0).
   Keep BPE behind a flag until the A/B is decisive.
