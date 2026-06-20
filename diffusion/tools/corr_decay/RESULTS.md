# M3 — Correlation-Decay Results (ch32 Phase 0)

Tests the Lin & Tegmark (2017) criticality claim on **our** corpora: is character-level
mutual information `I(X_i ; X_{i+d})` power-law (critical, long-range) or exponential
(short memory)? The ch32 hierarchy is only justified if real text is power-law **and** a
short-memory control is not. Tool: [`mi_decay.py`](mi_decay.py) (char-level MI, Miller-Madow
corrected; analytic Markov-1 reference from the fitted transition matrix; shuffle bias floor).

## Result — PREMISE VALIDATED (2026-06-20)

| corpus | chars | real fit | α (power-law) | exp R² | Markov-1 decay | real/Markov-1 at d≥50 |
|---|---|---|---|---|---|---|
| **TinyStories** | 8.4M | **power-law R²=0.993** | 1.88 | 0.755 | exponential (→0 by d≈6) | **~412×** |
| **Shakespeare** | 5.6M | **power-law R²=0.946** | 1.79 | 0.637 | exponential (→0 by d≈14) | **~10⁶×** |

- Real character MI follows a **power law** (α≈1.8) far better than an exponential — both corpora,
  two very different registers (simple modern prose vs Elizabethan verse).
- The **Markov-1 surrogate** (same 1-step statistics) decays **exponentially** to ~0 within a handful of
  characters. So the long-range structure is **real**, not an artifact of local letter statistics.
- Shakespeare shows a periodic **bump at d≈45–55** = verse line length — independent evidence the
  estimator is reading genuine structure.

## Why this justifies the hierarchy
Any flat fixed-window / exponential-memory model (RNN, fixed-context transformer, our masked-diffusion
denoiser — empirically a **short-range local interpolator**, §13.8) sits in the Markov family: its
correlation function decays **exponentially**. Power-law data + exponential model ⇒ a **structural
mismatch** that gets worse with distance — exactly the topic-drift we see in long generations
(word-TinyStories: "play in the sun and … play in the sun"). A **MERA-style log-depth hierarchy** carries
a distance-`d` correlation in `~log d` coarse-graining steps, which is the **known** way to represent
power-law correlations efficiently (DESIGN_REVIEW_2 §1). So M3 says: **build the hierarchy — the premise
holds on our data.**

## M3b (model generated-text MI) — CONFOUNDED, do not over-interpret
Generated 549 canvases (~37k chars) from the char TinyStories model (`ch30 --samples`, intra-canvas MI via
`--reset-on-newline`) and compared to the corpus:
- **Short range (d=1–7): model MATCHES the corpus** — it learned local structure well.
- **Long range (d≈18–24): the model's MI stops falling and RISES** (corpus keeps decaying ~power-law).
  This is **not** genuine long-range modeling — it is the **repetition artifact**: the model loops
  ("play in the sun and play in the sun"), and a repeated phrase at distance `d` manufactures spurious
  correlation at `d`. The elevated tail is the *signature of the topic-drift/looping failure*, not structure.
- **Caveat:** only 37k chars → large finite-sample bias (shuffle floor 0.077 vs corpus 0.0008); generated-
  text MI is the **wrong instrument** when the model repeats (looping inflates exactly the probed tail).
  More generation would not fix the repetition confound.

⇒ **M3b is consistent with "the flat model lacks genuine long-range structure" but is not clean evidence.**
The model-side claim rests more cleanly on **§13.8** (recall probe: the model ignores distant context —
contiguous recall flat ~16%, no lift from far context). **The hierarchy-premise verdict stands on M3
(corpus = power-law/critical) + §13.8 (model = short-range local interpolator), not on M3b.**

Reproduce:
```
python diffusion/tools/corr_decay/mi_decay.py data/tinystories_clean.txt --max-d 100 --limit-mb 8
python diffusion/tools/corr_decay/mi_decay.py data/complete_shakespeare.txt --max-d 100 --limit-mb 8
```
