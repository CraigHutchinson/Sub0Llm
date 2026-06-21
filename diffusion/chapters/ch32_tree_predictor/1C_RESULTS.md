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

---

# Follow-up (1): rare-weighted objective — the 2×2 (tested)

**Status: the rare-aware objective is a REAL cliff lever (−24…−30% on both models), but it does NOT
rescue the codec. Under the *identical* rare-weighted objective a plain baseline still beats the
codec on the cliff. Follow-up (1) thus deepens the falsification of the additive-gate codec and
isolates the culprit: the additive gate retains the noisy rare lookup.**

Wired a per-token-id loss multiplier into `batched_diffusion_loss` (`tok_weight`, indexed by the
clean target id; `weighted_cross_entropy` normalises by the weight sum so only the ratios matter).
`rare_weight` defaults to *auto* = the common:rare token-mass ratio, which equalises the two
buckets' total gradient mass. `oov_ab.cpp` now runs the full 2×2 in one process (identical split +
seeds).

Word-TinyStories, 570 train paragraphs → 1749-word vocab, D=256 L=4 T=64, 2000 steps each,
auto `rare_weight = 38.4` (common 43622 / rare 1137 masked-eligible tokens):

| arm | NLL_common | NLL_rare | cliff |
|-----|-----------:|---------:|------:|
| baseline / unweighted | 2.049 | 13.042 | 6.36× |
| **baseline / rare-weighted** | 2.538 | 12.306 | **4.85×** ← best cliff |
| codec / unweighted | 1.953 | 15.817 | 8.10× |
| codec / rare-weighted | 2.460 | 13.919 | 5.66× |

**Findings:**
1. **Rare-weighting works as a mechanism.** It lowers the cliff on *both* models (baseline
   6.36→4.85, codec 8.10→5.66) by trading common NLL for rare NLL — exactly the intended effect.
   The simple frequency-balanced objective *alone* (baseline/rw 4.85×) gives the lowest cliff of all
   four arms. **If the goal were just "shrink the cliff", rare-weighting beats the codec outright.**
2. **The codec still loses — even under its own prescribed fix.** Under the identical rare-aware
   objective, codec/rw cliff 5.66× is **worse** than baseline/rw 4.85× (+17%). Not a ratio artifact:
   codec's NLL_rare is worse in absolute nats (13.919 vs 12.306). In *both* objectives the
   char-composed component, *added* on top of lookup, makes rare words **more** confusable
   (uw 15.8 vs 13.0; rw 13.9 vs 12.3) while helping common (codec always has the lower NLL_common).

**Sharpened diagnosis.** A shared composer preferentially helps common words (more distinct contexts
to exploit the composed signal) under *any* loss weighting, and the **additive gate**
`E = lookup + α⊙composed` *keeps the noisy `lookup_rare` row* — so adding a non-rare-specialised
composed vector on top only blurs the rare logits. The gate form, not just the loss, is the problem.

**Next test (now the best-motivated):** follow-up (2) — a convex blend that *replaces* lookup for
rare words, `E = g·lookup + (1−g)·composed` with `g→0` for low-frequency types — paired with
follow-up (3) composer *pretraining* (so `composed_rare` is a meaningful spelling vector before the
LM objective biases it toward common). (2) only helps if `composed_rare` is actually good, which is
exactly what (3) guarantees; this 2×2 shows additive blending of an LM-trained composer is not.

**Bearing on the hierarchy unchanged:** the OOV cliff is real and *reducible* (rare-weighting proves
it), but the char-composition layer as designed (additive gate) does not deliver OOV robustness over
a matched baseline. Do not build P2 on it. The cheap, robust lever in hand is the rare-weighted
objective itself.
