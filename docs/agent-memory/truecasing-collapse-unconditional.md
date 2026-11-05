---
name: truecasing-collapse-unconditional
description: "MEASURED + SHIPPED 2026-07-30 (schemeV5): case collapse is now UNCONDITIONAL. The `attested` precondition cost 9.7% of the vocabulary on a duplicate cased piece inventory and bought NO information. A threshold could never have fixed it -- 92.4% of withheld forms never appear lowercase at all"
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T20:09:24.121Z
---

The backlog called this "the truecasing **threshold**". Measuring first showed the threshold was the
wrong knob entirely — the decisive result, and the reason not to have just tuned it.

## The rule that was there

`derive_attested` (src/tokenizer.cpp): a lowercase form is eligible for CAP/UP-marker collapse unless
its mid-sentence-capital uses outnumber its lowercase uses (`mid > lc` -> withheld -> the capitalized
word is emitted VERBATIM). Justified as "a name-dominated form like Spot stays a distinct verbatim
token".

## MEASURE ON THE BLEND, NOT ONE CORPUS (user correction, and it mattered)

The first pass decided this from **cosmopedia prose alone** and then "verified" it on **TinyStories**,
which is synthetic and has **no code at all** — so `ALL_CAPS` constants and CamelCase identifiers, the
two populations most likely to break the decision, were absent from every number. The user stopped it:
tokenizer decisions must be made on cosmopedia + minipile (code) + fineweb_edu together.

Re-measured on the blend, the conclusion **held and got stronger** — code raises the cased share, so
the saving grows faster than the token cost:

```
A/B, 95 MB learn / 11 MB held out, vocab_target 16384, all three sources
  verbatim names (old) : pieces=16196  cased=1952 (12.05%)  tokens=3728038
  always-collapse      : pieces=16196  cased= 319 ( 1.97%)  tokens=3778654
prose only (cosmopedia)  9.74% -> 1.59%  (-8.16 pp) for +1.17%
blend with code         12.05% -> 1.97% (-10.08 pp) for +1.36%   <- 1633 slots, 10.1% of vocab
```

A prose-only measurement **understated** the win. Standing lesson: this project's corpus is a blend, so
a tokenizer number from one source is not a result — see [[three-pillar-shootout-policy]].

## The harness silently stopped measuring once the change shipped

Re-run after the change landed, the A/B reported `cased=0 vs cased=0, delta 0%`. Vacuous: it simulates
the old policy by varying `attested`, and schemeV5 made `emit_word` ignore `attested`, so both arms ran
the SAME policy. It now REQUIREs the arms differ, naming the one-line revert needed to re-measure.
Same class as [[caps-bit-nothing-reads-is-not-a-guard]] — a check that cannot fire is not a check.

## Measurement 1 — a threshold cannot reach the problem (pass 1 only; corpus-independent)

Pure names are **92.9% of withheld forms in prose, 91.0% in code, 91.5% on the blend** — the finding
does not depend on corpus. Original cosmopedia detail:

```
COLLAPSED (attested, mid<=lc)   forms=30052  cap-occurrences=1053671
WITHHELD ambiguous (0<lc<mid)   forms= 3258  cap-occurrences= 314156
WITHHELD pure name  (lc==0)     forms=39612  cap-occurrences= 351127
```

**92.4% of withheld FORMS never appear lowercase anywhere.** They are therefore absent from
`lower_count`, `derive_attested` never considers them, and **no value of K in `mid > lc*K` reaches
them.** Sweeping K to infinity moves only the 3258 ambiguous forms — 7.6% of withheld forms. Since BPE
vocab pressure is driven by distinct FORMS, not occurrences, tuning the threshold would have been a
plausible-looking no-op.

## Measurement 2 — the rule change, A/B (64 MB learn / 7 MB held out, vocab_target 16384)

```
verbatim names (today) : pieces=16198  cased=1578 (9.74%)  held-out tokens=1927333
always-collapse        : pieces=16198  cased= 257 (1.59%)  held-out tokens=1949909
delta: cased pieces -8.16 pp,  token count +1.17%
```

9.74% independently reproduces the earlier 9.1% (1469/16220) figure from
[[repetition-is-a-proper-noun-defect]] — two different harnesses, same number.

**1321 vocabulary slots returned (8.2% of the whole vocab) for +1.17% tokens.** The residual 257 cased
pieces are genuinely mixed-case forms (`iPhone`, `McDonald`) that fail `rest_lower` and legitimately
need them.

## Why collapsing loses nothing

`<|cap|>spot` and `spot` are DIFFERENT token sequences. The old rule was not protecting a distinction —
the marker factors case out rather than discarding it. What it actually bought was a second, sparsely
trained cased piece inventory that only capitalized words ever use, which is the representational half
of the proper-noun repetition defect: each name needed several independent predictions from rarely
trained cased embeddings, so the cheapest way to emit a VALID name was to copy one from context.

## As shipped

* `casing.hpp` `emit_word`: `collapse = true`. **kSchemeVersion 4 -> 5** — load-bearing twice over,
  because the configurator's reuse stamp keys on it, so without the bump a `corpus.tok` tokenized under
  the old rule would be silently reused ([[caps-bit-nothing-reads-is-not-a-guard]] shape again).
* `tests/casing_policy_tests.cpp` — the hidden `[.casing]` harness, both measurements above. Re-runnable.
* `tests/tokenizer_tests.cpp` "a name that shadows a noun..." was INVERTED, not deleted: it now pins
  that the name still collapses AND that name != noun and both round-trip. That pair is the
  justification, so it is the thing worth asserting.
* `tests/ws6_simd_differential_tests.cpp` holds a scalar copy of the algorithm; its job is to guard the
  SIMD REWRITE, not the policy, so it tracks the policy change deliberately.

## FOLLOW-UP: TODO(drop-attested) — the bigger prize, not yet done

`attested` is now **unread by `emit_word`**. Corpus **pass 1** (`Scan::add_names` ->
`lower_count`/`midcap_count` -> `derive_attested`) exists for no other purpose, and `attested` is also
threaded through `Scan::add_words`, `Tokenizer`, serialize/deserialize and the configurator scan cache.
Removing it **deletes a whole pass over a 44 GB corpus** and a set from the on-disk format. ~8 files, so
it was deliberately NOT done as a drive-by on a behaviour change. Marked in `casing.hpp`'s
`truecase_tokenize`.

Related: [[repetition-is-a-proper-noun-defect]] (the defect this addresses — REP#1-5 backlog is still
open and the generation-side effect is NOT yet re-measured), [[tokenizer-schemev4-merge-backlog]].
