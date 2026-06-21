# Ch32 Phase-0 M2 — topic-drift metric + baseline (the P2 gate)

**Status: M2 IMPLEMENTED, UNIT-TESTED, and BASELINED on word-TinyStories. The validated coherence
signal is CONTENT RECURRENCE (does the text reuse the entities it introduces?). Corpus recurrence is
5.7× above the unigram-chance floor — real structure. The flat model sits EXACTLY at that floor: it
reuses content words no more than i.i.d. random sampling. That zero-entity-persistence gap is what
P2's gist conditioning must close.**

Metric: [`eval/topic_drift.hpp`](../../include/sub0diff/eval/topic_drift.hpp) /
[`src/topic_drift.cpp`](../../src/topic_drift.cpp). Baseline runner:
[`m2_drift.cpp`](m2_drift.cpp) (`ch32_m2_drift` — trains a flat Denoiser on GPU, generates passages,
scores M2 on corpus vs generations vs a control). Unit test `[diffusion][topic_drift]` validates the
mechanism on constructed sequences.

## What M2 reports (per passage, averaged)
- **distinct-2/3/4** = unique n-grams / total — degenerate-looping detector (↑ = less looping).
- **content_recurrence** = fraction of a passage's DISTINCT content types that occur ≥2× — entity
  persistence (↑ = coherent). *The validated coherence signal.*
- **persistence_near/far** = content-set Jaccard between windows at distance d. *Reported but NOT a
  valid coherence discriminator on TinyStories — see below.*

"Content" words are the frequency-split complement of the `stop_k` most-frequent (function) types —
mirrors `is_word_start` / `rare_type_mask`, the same split P2's `is_content` table will use.

## Baseline (word-TinyStories, 540 train paras → 1721 vocab, flat Denoiser 2000 steps, gen temp 0.9, spread)

stop_k = 100 (content = 1621/1721 types), window 16:

| passage set | content_recurrence | distinct-3 | persist_near |
|-------------|-------------------:|-----------:|-------------:|
| **corpus** (held-out paragraphs) | **0.121** | 0.896 | 0.071 |
| unigram-control (freq-matched, no structure) | 0.021 | 0.902 | 0.002 |
| **flat-model generations** | **0.021** | 0.917 | 0.005 |

- **Validation (the unigram control):** corpus recurrence 0.121 ≫ unigram-chance 0.021 (**5.7×**) ⇒
  entity reuse is REAL structure, not just word frequency. The metric is a genuine coherence signal.
- **The flat-model deficiency:** model recurrence 0.021 = the unigram floor **exactly** — the model
  reuses entities NO MORE than i.i.d. random sampling. Zero entity-persistence structure. (At
  stop_k=600: corpus 0.282 vs unigram 0.000 vs model 0.000 — same verdict, model at the floor.)
- **Not looping:** model distinct-3 (0.917) ≥ corpus (0.896) — the spread sampler + temp 0.9 keep
  generation non-repetitive (consistent with [`4B_RESULTS.md`]). The failure is NOT degenerate
  repetition; it is the *absence of entity reuse*.

## Methodological finding (negative result, recorded)

**Window-rearrangement controls do NOT work on TinyStories**, and neither does window-Jaccard
*persistence* as a coherence metric. A "mixed" control (windows drawn from random paragraphs) gave
recurrence/persistence ≈ corpus at every `stop_k` (100→1000): TinyStories is so lexically homogeneous
(a tiny, formulaic content vocabulary recurs across *all* stories) that random windows reuse the same
common content words as a real story. So a *rearranged-real-text* control is statistically ≈ real.
**The correct control is distribution-matched** (unigram resample): same word frequencies, zero
deliberate structure — which cleanly separates corpus (above chance) from a structureless baseline.
Lesson: validate a coherence metric against a *distribution-matched* null, not a *rearrangement* null.

## The P2 gate

P2 (gist conditioning) must lift the model's **content_recurrence above the unigram-chance floor
toward the corpus level** (0.021 → 0.121 at stop_k=100), WITHOUT regressing distinct-n into looping.
Measure with `ch32_m2_drift` (or wire `--topic_drift` into the P2 model's diagnostics). Use the
unigram control as the floor and the corpus as the ceiling; report recurrence as a fraction of the
(corpus − unigram) gap closed. distinct-n is the no-looping guardrail.
