---
name: repetition-is-a-proper-noun-defect
description: "MEASURED — generated repetition is 6.7x enriched for CAPITALISED words and does NOT respond to top-k width, so it is a proper-noun representation failure, not a sampler or attention bug"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T15:07:08.065Z
---

Measured 2026-07-30 on arm A (d192, 9.66M params, val_nelbo ~2.09). Full write-up in
`docs/REPETITION.md`. Two probes, both cheap, both surprising.

## Probe 1: top-k width has NO effect

10 seeds x 500 tokens per condition, same prompt:

```
topk=20    2507 words   immediate X-X 3   "X and X" 5   (1 per 501 words)
topk=100   2463 words   immediate X-X 5   "X and X" 6   (1 per 410 words)
```

A **5x wider candidate set produced no reduction**. If a narrow cutoff were inflating a mid-probability
repeat via renormalization, admitting 100 candidates would spread the mass and the rate would fall. It did
not. (At counts of 5-6 the Poisson 95% interval is ~[1.6, 11.7], so this does NOT show topk=100 is worse —
what it rules out is the LARGE reduction the sampler hypothesis requires.)

## Probe 2: the repeats are 6.7x enriched for CAPITALISED words

12 seeds x 500 tokens, case preserved:

```
words   3028   capitalised 418  (13.8% base rate)
repeats   41   of which capitalised 38  (92.7%)   -> 6.7x enrichment, ~14 sigma
```

This is a **capitalised-word defect**, not generic repetition.

## Mechanism

`casing.hpp` does **corpus-aware truecasing**: incidental (sentence-initial) capitals collapse to
`<|cap|>` + lowercase, but habitually-capitalised words are *left verbatim* (the `names` counter). So
capitalised text in generation is overwhelmingly **proper nouns**, on the verbatim path.

The vocab shows the cost: **1469 of 16220 merges (9.1%) contain uppercase, and they are TWO-CHARACTER
fragments** (`Al`, `Ma`, `Ch`, `Am`, `Sa`). Proper nouns are not single tokens — they are assembled from
short rare cased pieces. Each name needs several independent predictions from poorly-trained embeddings,
and the cheapest way to emit a *valid* name is to copy one already in context. Hence
`Chief of the Chief of Musher Officer Officer`.

## Consequences

* **NOT an attention/decode bug.** val_nelbo goes through batched `forward()` and descends monotonically;
  generation is bit-deterministic; long-range coherence (names, themes over hundreds of tokens) is intact,
  which a broken KV cache could not produce.
* **A repetition penalty is the wrong instrument.** Forbidding the repeat in a proper-noun slot yields a
  *different arbitrary name* — `Officer Officer` becomes `Officer Bronx`. Samples look better, model is
  identical. This is the concrete form of the user's "the penalty is a plaster" instinct.
* The `''` / `'''` artifact is the SAME defect, not a tokenizer bug: id 39 is a plain apostrophe byte and
  no marker renders as a quote.

## Backlog ordering (revised by these probes)

1. Confirm with the distributional read — `p(repeat)` before top-k, its rank, entropy — via batched
   `forward()`. Blocked while a build dir is busy training.
2. **Unlikelihood or contrastive term in the training objective** — promoted, addresses the cause.
3. **Corpus near-duplicate audit** — cheap, bounds how much is data vs capacity.
4. min-p sampling — still worth doing (top-k's fixed cutoff is wrong in both entropy regimes) but **must
   not be reported as having fixed this**.
5. Repetition penalty — last, stopgap only.

Also a concrete **schemeV4 input**: 9.1% of vocab goes on short cased fragments, so the truecasing "name"
threshold is worth checking — see [[tokenizer-schemev4-merge-backlog]].

Related: [[lr-schedule-wsd-cooldown-backlog]], [[tokenizer-normalization-and-factored-tokens]]
