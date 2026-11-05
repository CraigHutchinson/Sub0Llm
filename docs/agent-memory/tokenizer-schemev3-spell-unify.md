---
name: tokenizer-schemev3-spell-unify
description: "DONE+committed (be1cf3f) - fixed a 93.9% false-positive rate in compound-word detection by unifying 2-piece and 3+-piece words under TOK_SPELL_START..END; found by asking to SEE real examples, not trusting an aggregate NELBO number"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Follow-on to [[corpus-scale-word-collapse]]: asked "do we have examples of the feature working?", building
a standalone diagnostic against real TinyStories text revealed the corpus_collapse mechanism was badly
mistargeting -- 91% of sampled documents hit a "collapse candidate" that turned out to be ordinary words
like `room.` or `said,` (word + trailing punctuation), not rare/OOV named entities.

**Root cause**: `encode_join`'s old encoding for a word that Viterbi-segments into exactly 2 pieces used a
bare `TOK_JOIN` between the two piece ids (no wrapping markers) — but `TOK_JOIN` is *also* the general
"no space between adjacent content" glue used everywhere else in the tokenizer (punctuation immediately
after a word, closing quotes, bracket adjacency). An ordinary single-piece word immediately followed by
punctuation produces the byte-identical shape to a genuine 2-piece word split. Measured on a 5M-token real
sample: only 6.1% (35,339) of that shape were genuine splits; 93.9% (540,674) were the false positive.

**Fix (`kSchemeVersion` 2→3, `casing.hpp`)**: unify — every multi-piece word (N≥2, not just N≥3) now gets
`TOK_SPELL_START..END` wrapping, one unambiguous "this is a multi-piece word" signal. `TOK_JOIN` narrows
to general glue only, never a word-boundary signal. `sub0::detail::word_span` (`scratch.hpp`) simplified
to match — the ambiguous 2-piece branch is gone entirely.

**Verified safe before touching code** (a Plan-mode design review, not assumed): `detokenize_join`'s
`TOK_JOIN` decode case was always a single generic rule, never distinguishing word-split from general
glue — no dead decode path, round-trip correctness unaffected either way. UP-casing carry across a
multi-piece word is already handled by the SPELL-group mechanism (per-byte `is_alpha` check), not the
JOIN-specific carry check — so N==2 words moving into SPELL groups is safe by the SAME already-proven N≥3
path. No vocab/relearn impact: piece assignment (`viterbi_encode_word`) is unchanged; only the
marker-wrapping decision after it changes. Confirmed empirically: re-tokenizing TinyStories produced the
exact same 16535-token vocab, 21s for 671M tokens.

**A real, separate pre-existing gap this surfaced and fixed alongside**: `configurator.cpp`'s TOKSTAMP
reuse-cache explicitly assumed "the tokenizer scheme is always JOIN, so it is not a variable" — false the
moment `encode_join`'s emitted bytes change, and would have silently kept serving a stale wrong-scheme
`corpus.tok` with no warning if left unfixed (confirmed by testing: re-running `sub0llm-configure` after
the encode_join fix, but BEFORE rebuilding the configurator binary itself, silently reused the OLD
corpus.tok — a real trap, not hypothetical). Fixed by tracking `kSchemeVersion` in the stamp
(`TOKSTAMP_VERSION` 1→2).

**Measured impact on corpus_collapse**: documents hitting a collapse candidate dropped from 91% (1548/1693)
to 24% (401/1693); real examples flipped from punctuation noise (`<~room.~>`) to genuine recurring nouns
(`<~dive~>`, `<~clay~>`, `<~oak~>`) and actual proper names correctly collapsing on their second mention
(`<~Beth~>`, `<~Jase~>`, `<~Mae~>` twice). Held-out NELBO delta flipped from +0.014 (collapse slightly hurt)
to -0.0075 (collapse slightly helped); `build_dataset` got 9x faster (235ms→25ms, far less spurious binding
work). Full result in `docs/CORPUS_COLLAPSE.md`'s status log.

**The lesson worth keeping, standing pattern**: the original +0.014 NELBO delta alone read as a mundane
"no harm" result and gave zero hint the mechanism was mistargeting by this much — an aggregate metric
looked fine while the underlying behavior was badly wrong. Asking to see concrete, real examples (not just
trusting a summary statistic) is what surfaced this. Apply this pattern to future capstones: when a metric
comes back "fine" for a new mechanism, still spot-check real qualitative output before treating it as
validated — see [[three-pillar-shootout-policy]] (correctness+performance+memory, never a single number)
for the adjacent standing discipline this reinforces.

**A real environment gotcha hit during this work**: Visual Studio silently updated mid-session, moving its
install path from `C:\Program Files\Microsoft Visual Studio\2022\Community\...` to
`...\Microsoft Visual Studio\18\Community\...` (a version-number-named install replacing the "2022" naming
scheme) — broke `nvcc`'s ability to find `cl.exe` with no other symptom. If a previously-working
`vcvars64.bat` path suddenly fails only for CUDA (`.cu`) compilation while ordinary C++ still builds, check
whether the VS install path itself moved before assuming a PATH-length or environment corruption issue.
