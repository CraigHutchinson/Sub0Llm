# Ch32 M3 — Correlation-Decay Results (criticality / hierarchy premise)

> Phase-0 de-risk from [`BUILD_PLAN.md`](BUILD_PLAN.md). Tests the Lin & Tegmark (2017) claim on
> **our** corpora: is character mutual information `I(X_i ; X_{i+d})` **power-law** (critical,
> long-range) or **exponential** (short memory)? The hierarchy is only justified if real text is
> power-law and a short-memory control is not.
>
> Measured with a throwaway **local** Python script (not committed, per repo C++ policy —
> see `[[python-not-committed-cpp-library]]`): char-level MI with Miller–Madow bias correction, an
> **analytic Markov-1 reference** (same 1-step statistics ⇒ exponential by construction), and a
> shuffle bias floor. **Backlog: reimplement as a C++ `diffusion/tools/corr_decay` binary** (like
> `sub0diff-recall-probe`) if we want it as a permanent, repeatable tool.

## Result — PREMISE VALIDATED (2026-06-20)

| corpus | chars | real fit | α (power-law) | exp R² | Markov-1 | real/Markov-1, d≥50 |
|---|---|---|---|---|---|---|
| **TinyStories** | 8.4M | **power-law R²=0.993** | 1.88 | 0.755 | exponential (→0 by d≈6) | **~412×** |
| **Shakespeare** | 5.6M | **power-law R²=0.946** | 1.79 | 0.637 | exponential (→0 by d≈14) | **~10⁶×** |

- Real character MI follows a **power law** (α≈1.8) far better than exponential — both corpora, two very
  different registers. The **Markov-1 surrogate** (identical local statistics) decays **exponentially** to
  ~0 within a few characters ⇒ the long-range structure is **real, not local**. Shakespeare shows a
  periodic bump at d≈45–55 = verse line length (independent evidence the estimator reads real structure).

## Why this justifies the hierarchy
Any flat fixed-window / exponential-memory model (RNN, fixed-context transformer, our masked-diffusion
denoiser — empirically a **short-range local interpolator**, §13.8) has **exponential** correlation decay.
Power-law data + exponential model ⇒ a **structural mismatch** that worsens with distance — exactly the
topic-drift in long generations. A **MERA-style log-depth hierarchy** carries a distance-`d` correlation in
`~log d` coarse-graining steps — the known way to represent power-law correlations (DESIGN_REVIEW_2 §1).
**Verdict: build the hierarchy — the premise holds on our data.**

## M3b (model generated-text MI) — CONFOUNDED, not load-bearing
Generated 549 canvases (~37k chars) from the char TinyStories model (`ch30 --samples`, intra-canvas MI):
- **Short range (d=1–7): model matches the corpus** (learned local structure).
- **Long range (d≈18–24): model MI stops falling and *rises*** — but that is the **repetition artifact**
  (looping manufactures spurious correlation at the repeat distance), the *signature of the topic-drift
  failure*, not genuine long-range modeling.
- **Caveat:** 37k chars ⇒ large finite-sample bias (floor 0.077 vs corpus 0.0008); generated-text MI is the
  wrong instrument when the model repeats.

⇒ M3b is *consistent with* but not clean evidence for "flat model lacks long-range structure." The
model-side claim rests on **§13.8** (recall probe: distant context ignored). **The premise verdict stands
on M3 (corpus power-law) + §13.8 (model = local interpolator), not M3b** — and M3b's repetition tail is a
*quantified instance* of the drift the P2 gist level targets.
