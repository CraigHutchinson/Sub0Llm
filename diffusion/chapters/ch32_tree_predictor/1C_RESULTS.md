# Ch32 P1 1c — OOV A/B: results (the kill-test)

**Status: the naive char-composition codec FAILS the 1c kill-test. It marginally helps in-vocab
but WORSENS the OOV cliff. P1's *premise* stands (M1: real 6–8× cliff); P1's proposed *fix*, in
this form, does not work. This is a falsification, not a pass.**

Runner: [`oov_ab.cpp`](oov_ab.cpp) (`ch32_oov_ab`). Trains a baseline `Denoiser` and a
`CodecDenoiser` on the SAME word-TinyStories data + step budget, then measures M1 on both via the
generalised `evaluate_oov_cliff`.

## Result

Word-TinyStories, 570 train paragraphs → 1749-word vocab, D=256 L=4 T=64, **2000 steps each**
(step-budget-matched), held-out, rarest 50% of types = OOV proxy:

| model | NLL_common | NLL_rare | cliff | train nelbo |
|-------|-----------:|---------:|------:|------------:|
| **baseline** Denoiser | 1.978 | 12.858 | **6.50×** | 2.35 |
| **CodecDenoiser**     | 1.940 | 15.669 | **8.08×** | 2.24 |

**Verdict:** cliff 6.50× → **8.08× (+24%, WORSE)**; in-vocab NLL −1.9% (slightly better); codec
train-NELBO lower (2.24 vs 2.35). Kill-test bar (BUILD_PLAN §Phase 1) was "cliff → ~1"; it rose.

## Interpretation — why it fails (coherent + falsifiable)

The composed embedding is **shared** (its gradient comes from every word), and the masked-token
loss is **dominated by common words** (61321 common vs 4484 rare masked tokens, ~14:1). So the
composer optimises representations that help the *common* words — which is exactly what we see
(in-vocab NLL improves, train-NELBO drops). Rare words get a composed component that is (a) not
specialised to them and (b) added on top of their already-noisy lookup row (the per-element
additive gate `E = lookup + α⊙composed` keeps the noisy `lookup_rare`). Net: rare-word logits
become *more* confusable, so NLL_rare rises and the cliff widens.

This matches the design's own stated kill condition: **the OOV cliff persists** ⇒ the naive codec
does not buy OOV robustness. The mechanism (compose from spelling → content-addressed word) is not
wrong in principle, but a shared composer trained by a common-dominated objective will not allocate
capacity to rare words on its own.

## What would change the verdict (follow-ups, untested)

Ordered by how directly they attack the failure mechanism above:

1. **Rare-weighted / frequency-balanced loss** — upweight rare-type masked tokens so the composer
   is forced to learn their spellings (counter the 14:1 common dominance). The most direct fix.
2. **Convex blend that DROPS the lookup for rare words** — `E = g·lookup + (1−g)·composed` with `g`
   driven toward 0 for low-frequency types (not the additive form, which retains noisy `lookup_rare`).
   NOTE: only helps if composed_rare is actually *good* — which (1) is needed to ensure.
3. **Composer pretraining** — pretrain the char autoencoder (P1 1a/1b, already validated) so
   composed vectors are meaningful spellings before the LM objective biases them toward common words.
4. **Longer training** — codec is ~10× slower/step (compose_vocab over the whole vocab each step:
   651 s vs 64 s for 2000 steps); 2000 may undertrain the composer. Lower-value (the in-vocab gain
   shows it *is* learning; the rare regression is structural, not just undertraining).

## Bearing on the hierarchy

P1's **premise** is intact — M1 confirmed a real 6–8× OOV cliff ([`M1_RESULTS.md`](M1_RESULTS.md)),
and the codec **components** work in isolation ([`P1_RESULTS.md`](P1_RESULTS.md): round-trip,
order-sensitivity, compose_vocab parity, CodecDenoiser trains). What's falsified is the *naive
integration* (additive gate, unweighted loss). Before P2/P3, the OOV layer needs follow-up (1) — a
rare-aware objective — or P1 should be re-scoped. **Do not build P2 on the assumption that 1c, as
run, closed the cliff: it did not.**
