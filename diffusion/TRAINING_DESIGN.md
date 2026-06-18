# sub0llm Text-Diffusion — Training Design Specification

> A distilled, accurate specification of how we train the text-diffusion model (Ch28–Ch31),
> written both as a durable chapter resource and as the document an independent diffusion-LLM
> expert is asked to review. **Central open question:** held-out NELBO plateaus around **3.5**;
> intuition says a well-fit text model should reach **< 2**. Did we solve the underlying issue or
> just move it? What is the shape/value of *good* convergence for this objective on this data?

Status as of 2026-06-15. Code: `diffusion/` (project `sub0diff`), chapter binary
`ch29_diffusion_training`.

---

## 1. Objective — absorbing-state masked-diffusion NELBO (MDLM / LLaDA)

We train the **negative evidence lower bound** of an absorbing-state discrete diffusion model,
which Rao-Blackwellizes (Sahoo et al. MDLM; Nie et al. LLaDA) to a weighted masked cross-entropy:

```
L = E_{t ~ U(t_min, 1]}  E_{x_t ~ q(·|x_0, t)}  [ (1/t) · Σ_{i : masked} −log p_θ(x_0[i] | x_t) ]
```

- Sample a noise level `t`, corrupt each position to `[MASK]` with probability `t`, predict the
  clean token at the masked positions, weight the summed CE by `1/t`.
- `t ~ U(0.02, 1.0]` (the formal objective; `t_min = 0.02` floor). Every noise level is trained
  every epoch — Ch28's curriculum is a *variant* of reshaping the distribution of `t`.
- **Implementation** (`diffusion/include/sub0diff/train/diffusion_loss.hpp`): `weighted_cross_entropy`
  returns the MEAN over masked positions; the per-sample NELBO term is recovered by scaling with
  `w = n_masked / (t · T)` (≈1 in expectation since `E[n_masked] = t·T`). T = seq_len = 64.

### Variance-reduction levers (default ON, Ch29/Ch31)
- **exact-count masking** (`--exact-noise`): mask EXACTLY `round(t·T)` positions (partial
  Fisher-Yates) instead of an independent Bernoulli per position. Removes the count variance, so
  realised noise = nominal exactly. Measured: raised gradient within-level consistency 0.31 → 0.51.
- **shared-t** (`--shared-t`): one `t` for all B windows in a step, so the B-window average sharpens
  the per-noise-level gradient (consistency ∝ √B) instead of diluting across near-orthogonal levels.

---

## 2. Model — bidirectional denoiser (`diffusion/include/sub0diff/nn/denoiser.hpp`)

A pre-norm transformer that, given a corrupted sequence and its scalar noise level, predicts the
clean token at **every** position with **full (non-causal) attention** (no left-to-right order to
protect). Reuses sub0llm's modern blocks unchanged.

```
x = TokEmb(ids) + NoiseCond(t)                  # (T, D)
repeat L times:  h = x + GQA_noncausal(RMSNorm(x))      # RoPE inside attention
                 x = h + SwiGLU(RMSNorm(h))
logits = RMSNorm(x) · TokEmbᵀ                    # WEIGHT-TIED head → (T, Vm)
```

- **Tokenizer**: BPE, `vocab_size ≈ 512` (+1 absorbing `[MASK]` row ⇒ model vocab `Vm = V+1`).
- **Noise conditioning** (`time_embedding.hpp`): a **parameter-free sinusoidal** embedding of `t`,
  amplitude-matched to the token embedding scale `1/√D` (a unit-amplitude version was ~11× louder
  than content and stalled learning ~10×), **added** to every token row. A *learned* projection of
  this is noted as an unimplemented upgrade.
- **Blocks**: RMSNorm, GQA (RoPE, configurable query/kv heads), SwiGLU FFN. No absolute pos-emb.
- **Head is weight-tied** to the token embedding.
- **Architectures used**:
  - *old default* (most historical results): D=128, L=8, heads=8/4 (GQA), d_ff=384 → head_dim **16**,
    d_ff = 3·D. ~1.6 M params. Flagged mis-proportioned (head_dim < 32, d_ff < 4·D).
  - *founded* (`--founded`): D=256, L=6, heads=8/4, d_ff=1024 → head_dim **32**, d_ff = 4·D. ~6 M params.

---

## 3. Data

- Corpus: `data/complete_shakespeare.txt` (and smaller `shakespeare.txt`). Paragraph-per-line.
- Flat BPE token stream; **random-offset sliding windows** of `seq_len = 64` (train and eval share
  one window distribution); **95/5 train/eval split**.
- Scales used: `--paragraphs 10000` (proxy A/B) up to the full ~146k-paragraph corpus.
- Note: 10k Shakespeare paragraphs × 64 tokens ≈ a *very small* token budget for a 1.6–6 M model.

---

## 4. Training recipe

- **Optimizer**: AdamW (lr 1e-3, decoupled wd 0.01) — the A/B winner (+1.4pt word-level over plain
  Adam). Muon supported but routes embedding/output→AdamW; see §6.
- **Unified (B, W) data-parallel step** (`train/parallel.hpp`): effective **batch-size B** windows
  averaged per optimizer step (gradient consistency ∝ √B, the QUALITY knob) split across **W workers**
  (`--threads`, SPEED only — gradient identical for any W via per-window deterministic seeding).
  Default B = W. Weights shared (one L3-resident copy); grads private then mean-reduced.
- **Gradient clipping**: global L2 norm ≤ 5.0.
- **LR schedule**: constant by default; warmup+cosine is opt-in and *measured to hurt* the known-good
  baseline (decays it into an earlier early-stop).
- **Curriculum** (opt-in): frontier-point noise ceiling that rises only once mastered (Ch28). Default
  off (uniform `t`).
- **Self-termination**: every `eval_every` steps measure held-out NELBO; checkpoint on improvement;
  stop after `patience` evals without one, with a `min_epochs` floor. (`--steps` is a safety bound.)

---

## 5. Evaluation

- **Held-out NELBO** (averaged diffusion loss over fixed-seed eval windows) — the early-stop signal.
- **Recall sweep**: over the held-out stream at noise levels {10, 25, 50, 75}%, fraction of masked
  positions recovered exactly (greedy argmax). Broken down into **word-level / word-START /
  continuation** (a BPE word-piece's first vs later subword). The headline metric.

---

## 6. Current results & what we've learned

| run (full corpus, old arch D128/L8) | best held-out NELBO | overall recall | word-START | @10% noise |
|---|---|---|---|---|
| √4 baseline (W=4, shared-t, exact-noise) | 3.712 | 19.0% | 14.6% | 42.2% |
| **√16 (W=16)** | **3.519** | **21.5%** | **16.9%** | **48.4%** |

Progression of the headline recall over the project: ~15% → 19% (exact-noise + shared-t) → **21.5%**
(√16 consistency). Key findings:

- **Ch31 optimizer sandbox**: on the isolated gradient pathology (orthogonal per-level blocks +
  within-level noise, depth-2 coupling, LR sweeps), **Adam ≈ Muon** at every consistency level — the
  optimizer is *interchangeable*; **gradient within-level consistency is the crux**, and it bounds
  both optimizers identically. The consistency-collapse near convergence (0.50 → 0.07) is reproduced
  and is optimizer-independent.
- **Consistency lever confirmed at scale**: √4 → √16 (more windows averaged at one noise level)
  buys +2.5pt recall and a lower NELBO; it also trains productively far longer. This is the only
  lever that has moved the ceiling recently.
- **Muon canary**: Muon trained on the *old* arch (head_dim 16) FAILS catastrophically (basin-stuck,
  NELBO 5.69, 3.2% recall) while AdamW reaches ~19%. On small/serial configs Muon ≈ AdamW. A
  founded-arch (head_dim 32) Muon vs AdamW 2×2 is in progress to localize whether the failure is the
  narrow head. (Muon amplifies any mis-scaled update direction Adam's per-coordinate norm hides — so
  a Muon failure can be a *canary* for a latent gradient issue that also caps Adam.)

---

## 7. THE OPEN PUZZLE — did we solve it or move it?

The consistency lever raised recall but **held-out NELBO is stuck around 3.5** and recall around 21%.
A well-fit text model "should" reach NELBO well under 2. We have **not** closed that gap — we
mitigated it. The reviewer's central task is to diagnose *why* and propose the highest-leverage fix.

Candidate explanations (to be confirmed/refuted):

1. **Objective-bound, not model-bound.** The NELBO is an *average over all `t ∈ (0.02,1]`*, including
   the near-impossible `t ≈ 1` regime (predict ~all tokens from an almost-empty canvas), whose loss
   is irreducibly high (≈ the data entropy). Is **3.5 actually near-optimal** for this `1/t`-weighted
   objective on this data, such that the "< 2" intuition (an AR per-token cross-entropy number) is a
   category error? What is the **theoretical NELBO floor** for absorbing diffusion on Shakespeare-BPE,
   and how should we *normalize/report* it to make "good" legible (e.g. per-`t` loss curves, or a
   masked-only CE at fixed low `t`)?
2. **Under-data (Chinchilla).** ~10k–146k paragraphs is tiny. Masked diffusion sees each token under
   many maskings (data-augmentation-like), so its *effective* token budget per parameter differs from
   AR. **Do Chinchilla-style compute-optimal scaling laws transfer to masked diffusion LMs?** Is a
   **smaller** model in fact compute-optimal here, and what is the right model-size / token-budget
   ratio for dLLM? (This is the explicit research ask.)
3. **Underfit / mis-proportioned.** Old arch had head_dim 16, d_ff 3·D. Founded fixes both but is
   bigger. Is capacity or training-length the binding constraint at this data scale?
4. **Recipe gaps.** Parameter-free (not learned) noise conditioning; weight-tied head; no LR schedule;
   `t ~ U` vs importance/antithetic `t` sampling; constant LR; AdamW vs better. Which matter?
5. **Metric mismatch.** Greedy exact-match recall at high noise is a harsh metric; is NELBO-vs-recall
   decoupling hiding real progress (or real failure)?

### Specific questions for the reviewer
- Is held-out NELBO ≈ 3.5 consistent with a *correctly-implemented* MDLM/LLaDA objective that is
  **near its achievable floor** on this data — or a clear symptom of under-fitting / a bug? How would
  you tell them apart cheaply?
- What does a healthy NELBO-vs-`t` curve look like, and what should we plot to know "good convergence"?
- For masked diffusion LMs, how do compute-optimal scaling / Chinchilla findings apply? Is smaller
  better here, and what model/data ratio would you recommend for Shakespeare-scale corpora?
- What is the single highest-leverage change to break the ~3.5 NELBO / ~21% recall ceiling?
- Any correctness red flags in §1–§4 (the objective weighting, exact-count masking, tied head,
  noise conditioning, bidirectional attention, the (B,W) gradient estimator)?

---

## 8. Reproduce

```bash
cmake --build --preset native --target ch29_diffusion_training
# best-defaults run (AdamW, shared-t, exact-noise; B=W); pick W for speed, B for gradient quality
./build-native/bin/ch29_diffusion_training --ckpt-dir /tmp/ch29 \
  --corpus data/complete_shakespeare.txt --threads 16        # √16 ≈ 21.5% recall
# raise consistency independent of cores:
./build-native/bin/ch29_diffusion_training ... --batch-size 32 --threads 8   # √32 on 8 cores
```

---

## 9. Independent expert review — synthesis (2026-06-15)

Three independent reviewers (diffusion-LM objective expert, scaling-laws/Chinchilla-for-dLLM
researcher, adversarial recipe critic), each grounded in the literature. They **converge** on the
answer to "did we solve it or move it": **we MOVED it.** The consistency lever is a real *variance*
fix, not a floor-breaker — and the "NELBO < 2" target is the wrong yardstick.

### 9.1 The "< 2" intuition is a category error (high confidence, all three agree)
- The reported **3.5 is in NATS** (= 5.05 bits) and is **`t`-averaged over `t∼U(0.02,1]`**, so it is
  structurally dominated by the **high-`t`, near-empty-canvas regime whose loss is irreducibly ≈ the
  corpus unigram entropy** (~4–5.5 nats). The `1/t` weight *cancels* the mask count in our per-token-
  mean formulation (this is the correct MDLM Rao-Blackwellized form), so the average is NOT upweighted
  toward easy low-`t`.
- **Literature anchor:** MDLM reports **perplexity ≈ 23–31 on LM1B** ⇒ per-token NELBO **≈ 3.1–3.4 nats**
  *on a large clean corpus*. WikiText MDM NELBO ≈ 4.50. **Our 3.5 on tiny Shakespeare-512BPE is within
  ~0.2–0.4 nats of MDLM's large-scale number** — i.e. plausibly **near the objective floor for this
  data**, not catastrophically underfit. The diffusion irreducible-loss floor is structurally higher
  than AR (fitted E≈2.41 vs Chinchilla 1.69).
- **Therefore the real bug is MEASUREMENT**: we report one conflated scalar instead of the **per-`t`
  loss curve**. That single missing artifact is why "solved vs moved" is currently unanswerable.

### 9.2 We are strongly DATA-bound, and diffusion is the right objective for it (scaling reviewer)
- tokens/param ≈ **0.1–6** here (10k-proxy ↔ full corpus, 1.6–6M params) vs AR-Chinchilla optimum ~20
  and **diffusion-optimal ~40–100** — we are **1–2 orders of magnitude below optimal**. Textbook
  data-bound; no model size reaches low loss on one pass.
- **Chinchilla does NOT transfer**: masked diffusion is 2–5× more data-hungry per param, BUT tolerates
  **~100–500 epochs of data reuse** (vs AR's ~4–15) because random masking is built-in MC augmentation
  ("Diffusion Beats AR in Data-Constrained Settings", 2507.15857; "Super Data Learners", 2511.03276).
- **Our self-termination / early-stop likely truncates training before the reuse regime where
  diffusion's advantage lives.** The compute-optimal response to a fixed tiny corpus is **smaller,
  properly-proportioned model + many more epochs**, not a bigger model.

### 9.3 Concrete recipe flaws found in the code (adversarial critic — these are the model-side levers)
- **P0 — noise conditioning is informationally weak (top suspected cause).** Parameter-free sinusoid,
  **added** at `1/√D` amplitude (partly washed by the pre-norm RMSNorm), with a **mis-banded frequency**
  (`kTimeEmbedScale=1000`, base 10000 → no low-frequency channel monotonic across `t∈[0,1]`; aliasing).
  If the model can't cleanly read `t` it collapses toward a `t`-agnostic denoiser → marginal/unigram
  floor (≈3.5). **Fix:** learned **AdaLN/FiLM** time conditioning (MLP over correctly-banded Fourier
  features, per-block scale/shift) — the standard MDLM/LLaDA recipe we omitted.
- **P2 — exact-count + `t_min=0.02` biases the `1/t` estimator at low `t`.** `round(t·T)` clamps to 1
  for a range of `t` while the weight `1/(t·T)` keeps varying → breaks unbiasedness exactly in the
  low-`t` regime where the model should shine. **Fix:** raise `t_min` so `t_min·T ≥ ~3` (≈0.05 at T=64),
  weight by realized `k`. Also: exact-count removes the Binomial mask-count spread the model faces at
  inference (a train/inference mismatch worth an A/B).
- **P3 — weight-tied head may be under-capacity for a denoiser.** Untie (≈130K params at D=256) and
  retest; cheap, decisive.
- **P4 (cleared):** RoPE under bidirectional attention is correct — not a bug, don't spend effort.

### 9.4 The challenge to our own conclusion (take seriously)
The Ch31 sandbox proved optimizer-independence *conditional on a fixed gradient pathology*; it did
**not** prove the pathology is irreducible. Weak `t`-conditioning (P0), the low-`t` estimator bias
(P2), the tied-head cap (P3), and data starvation (§9.2) **each independently produce an optimizer-
independent NELBO floor near the data entropy** — exactly what we see. "Consistency is the crux" is
true as a *variance* statement but may be **masking** these floor-setting causes. The √16 win is real
but is "variance reduction lets us train longer before overfitting a tiny dataset," not a floor break.

### 9.5 Prioritized cheap experiments (do these BEFORE any more optimizer/consistency work)
1. **Per-`t` loss curve** — bin eval windows into ~10–20 `t` buckets, plot mean masked-CE vs `t`.
   *Flat ≈3.5 ⇒ `t`-agnostic (P0 confirmed); monotone rising from <1 nat at low `t` ⇒ near-floor.*
   The single most informative missing artifact (~20 lines). Also switch eval to a fixed `t`-grid
   (kills per-window `t`-sampling noise in checkpoint comparisons).
2. **Single-batch overfit test** — drive train NELBO on ~64 windows toward 0. *Can reach <2 but held-out
   stays 3.5 ⇒ data/generalization-bound; cannot even overfit <2 ⇒ capacity/conditioning bug.* The most
   decisive falsifier of "consistency is the crux." (Note: an earlier tiny-model overfit reached only
   5% train recall / NELBO ~4.6 — a hint it could NOT overfit, pointing at P0/P3 — but redo it properly
   with the founded arch + the per-`t` plot.)
3. **AR baseline on the identical token stream** (reuse `modern_gpt`). *AR ≈ 3.5 too ⇒ it's the data;
   AR ≈ 2–2.5 ⇒ objective/conditioning gap.* The honest reference number.
4. **Measure the unigram-entropy floor** `H₀` over the train stream (10-line pass) — pins the `t→1`
   ceiling and tells you how close the `t`-average already is to the floor.
5. **Report NELBO as perplexity** (`exp(NELBO)` ≈ 33 PPL) for cross-paper legibility, and report the
   **per-`t` recall** (the @10% number, 48.4%, is the meaningful one) rather than the noise-averaged 21%.

### 9.6 Highest-leverage changes (after the diagnostics)
1. **Learned AdaLN time conditioning** (P0) — top model-side fix; the clearest deviation from MDLM/LLaDA.
2. **Train many more epochs** (50–200) on the full corpus with `patience`/`min_epochs` raised far up —
   diffusion monetizes data reuse where AR overfits; directly tests floored-vs-underfit.
3. **Smaller, properly-proportioned model** (~1.5–3M, head_dim 32, d_ff 4·D) rather than the 6M founded —
   bigger overfits earlier at this data scale.
4. **Untie the head** (P3); **raise `t_min`** to ≥3/T (P2).
5. **More / real data** if NELBO substantially below ~3 is the goal — the corpus is a genuine ceiling.

> **Reframe of the project's narrative:** the optimizer/consistency line of work (Ch29–Ch31) correctly
> characterized and fixed the *gradient-variance* axis. The *floor* is set elsewhere — measurement
> (report per-`t`), conditioning (learned AdaLN), and data (epochs/size). The next chapter's work
> should pivot from "make the gradient cleaner" to "is the model `t`-aware and is it data-bound."

**Key sources:** MDLM (2406.07524), LLaDA (2502.09992), Diffusion-Beats-AR-in-Data-Constrained
(2507.15857), Quokka optimal-DLM-scaling (2510.03280), Super-Data-Learners (2511.03276),
MDM-scaling (2410.18514).

---

## 10. Founded long-run baseline + trend extrapolation (2026-06-16)

Best-candidate long run (founded D256/L6, √16, patience 40, full corpus). Per-t diagnostic on an
earlier old-arch checkpoint had shown the model is **t-aware but underfit at low t**; the founded
arch directly attacks that. Result (power-law fit over 162 eval points, `fit_trend.py`):

- **Headline: recall 21.5% → ~33% (+11pt), word-START 16.9% → ~25%, best NELBO 3.519 → ~3.34.**
  Low-t train NELBO fell ~1.94 → ~1.4 — the underfit marker closing (founded arch fits the easy
  regime much better). This confirms the ceiling was **fit/proportions**, not optimizer/conditioning.
- **Trend:** `NELBO = 3.06 + 202·s^-0.55`, `recall = 40.0 − 1710·s^-0.45`. Asymptotes **≈40% recall /
  ≈3.06 NELBO**. Diminishing hard: recall rate +1.62 → +0.62 pt/10k steps.
- **Extrapolation:** +24h ≈ +2.5pt (→35.4%), +48h ≈ 36.3%; **90% of the remaining gain to the 40%
  asymptote needs ~176× current compute.** ⇒ **more-of-the-same training is NOT the lever** — a day
  buys ~2.5pt. The next gains must come from efficiency/data/recipe, not steps.
- **Ch30 sufficiency (iterative refinement):** the prior net-negative (iterative LOSES to one-step
  via error-compounding on the 857K model) has **flipped to break-even** — iterative ≈ one-step
  (25% noise +0.3..+1.0pp, 50% −0.1pp). So the reverse-process mechanism no longer compounds errors,
  but it's not yet a clear win, and **generation is still incoherent** (mask-artifact fragments). Ch30
  *runs* now but won't impress until the model strengthens (data/proportions, not the sampler).

## 11. dLLM training-EFFICIENCY axis — how each window is consumed (next work)

Independent dLLM-efficiency review (grounded in MD4 / MDLM / RADD / Prime / DiffuCoder). The suspected
inefficiency is real: masked-only loss supervises ~t·T positions per forward, and `t∼U` spends ~30% of
forwards at `t>0.7` (near-empty canvas, irreducible loss ≈ unigram entropy, ~no learnable gradient).
These are **efficiency/variance levers** (reach the floor faster/cheaper) — same family as the √16 win;
they shave NELBO and cut compute but the ~3.3 *floor* is still set by data/proportions. Ranked by
gain/effort:

1. **Mid-`t` importance sampling + realized-`k` weighting** (~30 lines, top ratio). Replace `t∼U(0.02,1]`
   with a proposal concentrated on `t∈[~0.05,0.7]`, correct the loss by `p_U(t)/p_prop(t)` (stays
   unbiased), and weight by the **realized** mask count `1/(k/T)` not `1/t`, with `t_min·T≥3` (fixes
   the §9.3-P2 low-`t` bias). Validated to beat loss-reweighting at equal FLOPs (Hang et al. 2407.03297);
   removes the ~30% near-useless high-`t` forwards.
2. **Complementary (antithetic) masking pairs** (low effort). Per window, run the mask AND its complement;
   average the two masked-only NELBO terms → **100% token coverage per pair, ½ masking variance**. Same
   family as √16 but on the masking axis (additive). Used by DiffuCoder coupled-GRPO; antithetic pairs
   are an MD4 default.
3. **Low-discrepancy / stratified-`t` across the batch** (~5 lines). `t^(i)=(u+i)/B` instead of iid or
   `--shared-t` (which collapses all B to one `t`). Cheapest validated change; part of MDLM's +17%
   perplexity-bound gain. **NB:** this is in tension with our `--shared-t` (collapse vs spread) — A/B them.

**De-prioritized:** all-position/copy loss (breaks the valid masked-only bound — MD4); token-frequency
weighting (no validated bound gain); RADD time-removal (conflicts — our per-`t` shows we WANT
`t`-conditioning); Prime partial-masking (large gain, high architectural cost — backlog).

**Sources:** MD4 (2406.04329), MDLM (2406.07524), RADD (2406.03736), Improved-Noise-Schedule
(2407.03297), DiffuCoder coupled-GRPO (2506.20639), Prime partial-masking (2505.18495).

## 12. Minimal-case experiments (2026-06-16) — capacity is cleared; the constraint is GRADIENT QUALITY

Three controlled experiments via the new `--overfit N` harness (train only the first N
non-overlapping windows; stop on TRAIN recall over exactly those windows; identical
ParallelTrainer + diffusion_loss path). The point: stress the real loop on cases whose right
answer we know. Tooling: `ch29 --overfit N`, `overfit_ladder.ps1`, `tmax_ab.ps1`.

### 12.1 Capacity ladder (N=1/4/16/64 × D64-L2 / D128-L4 / D256-L6, all founded-proportioned)
| arch | params | N=1 | N=4 | N=16 | N=64 (batch=1) |
|------|-------:|-----|-----|------|------|
| D64-L2 | 157K | FIT 100% @1800 | FIT 98% @4600 | FAIL 79% | FAIL 3% |
| D128-L4 | 1.05M | FIT 100% @800 | ~FIT 98% @3200 | ~FIT 91% | FAIL 2% |
| D256-L6 | 6.05M | FIT 98% @400 | ~FIT 93% | FIT 98% | FAIL 3% |

- **Capacity is NOT the corpus-scale bottleneck.** Even 157K params memorize N=1/4; 1M fits N≤16.
  ⇒ "do we need 8 layers / 3 heads?" — **no.** The founded +11pt (§10) was **proportions**
  (head_dim 16→32, d_ff 3D→4D), not depth/size. Depth and head-count are not the lever.
- **N=64 collapses to ~3% (the marginal/unigram floor) for ALL archs, including 6M** — impossible
  as a capacity limit (6M params trivially hold 64×64=4096 tokens). A *discrete* cliff, not a
  gradual rolloff ⇒ a training-dynamics failure, not capacity. (At batch=1, 64 windows cycled over
  8000 steps = only ~125 visits/window AND a single-window/single-`t` gradient per step.)

### 12.2 The N=64 collapse is GRADIENT VARIANCE, not capacity (decisive)
Re-ran D128-L4 N=64 with **full-batch (B=64, all 64 windows per step)**: **FIT to 99% by step 1800**
— the *same arch* that failed at 3.4% with batch=1. So the collapse was the high-variance batch=1
gradient sinking into the marginal basin; the consistent √B gradient drives full memorization fast.
**This is the session's through-line:** the binding constraint at this scale is **gradient quality
(variance/consistency)**, not capacity, optimizer, or conditioning — the same mechanism behind the
√16 win (§10) and the `t_max` win below. (Full-batch also raised visits/window; both are
training-dynamics levers, neither is capacity — the representational sufficiency is proven.)

### 12.3 `t_max` 1.0 vs 0.8 A/B — dropping the high-`t` tail wins (matched 30k-step budget, seed-matched)
| `t_max` | overall recall | word-level | word-START | best held-out NELBO |
|--------:|---------------:|-----------:|-----------:|--------------------:|
| 1.0 | 14.1% | 12.4% | 10.4% | 4.129 |
| **0.8** | **15.0%** | **13.0%** | **11.0%** | **4.070** |

`t_max=0.8` wins every metric (+0.9pt overall, +0.6 word-START, lower NELBO) at equal compute —
recall scored on identical fixed noises ≤0.75, so the eval is `t_max`-independent. Confirms §11:
the `t>0.8` tail (near the entropy floor, ~no learnable context) is net-negative per unit compute.
Same variance-reduction family as §12.2. *Caveat:* single seed / D128 scale; the bigger principled
version is **mid-`t` importance sampling** (§11.1, keeps the full range unbiased). Ch30 blank-canvas
generation uses the `t≈1` regime, so don't drop the default to 0.8 blindly — confirm on founded + a
2nd seed first. **Shipped now:** the conservative, always-correct slice — a structural **min-1-VISIBLE
floor** (`k∈[1,T-1]` in `corrupt_into`, both exact-count and Bernoulli paths): every training window
keeps ≥1 visible token (a learnable context) *and* ≥1 masked token (a target). It is the symmetric
twin of the existing min-1-masked floor and removes only the ~0.8% fully-blank windows (zero-signal,
highest-variance), so it's safe with `t_max=1.0` default and Ch30 generation.

**Net for the session:** capacity/depth/heads are NOT the lever (cleared); the levers are
**gradient quality** (batch/consistency, mid-`t` sampling, lower high-`t` mass) and **data**
(the ~3.3/40% floor, §10). Next: confirm `t_max`/importance-sampling on founded + a 2nd seed,
then more/real data.

## 13. Convergence-gated frontier curriculum (2026-06-16) — integer-`k`, decided per-epoch

The §12 finding (a single exact `k` is the lowest-variance gradient) motivates a second
curriculum, distinct from the Ch28 token-gated EMA ceiling (`NoiseCurriculum`). `FrontierCurriculum`
(`--curriculum-converge`): train at **exactly `k` masked tokens** starting at `k=1` (recover one
token from `T-1` visible — the easiest possible task, maximal context), and raise `k → k+k_step`
**only when the held-out NELBO at the current level plateaus**, judged at the **per-epoch eval**
(the one clean read of the model's standing — within-step training loss is far too noisy to gate
on). Like a child mastering 1-token infilling, then 2-token, … before the full noise range. At
`k_max = T-1` (the §12.3 min-1-visible cap) it converges and the trainer switches to the full
formal objective. The rule the user identified — *parameters and the curriculum level only change
on a full-epoch boundary* — falls out naturally: a difficulty step never contaminates the
convergence signal.

- Why this and not the EMA ceiling: gates on a **clean per-epoch held-out** signal (not a noisy
  within-step EMA), moves an **integer `k`** (not a fractional ceiling), and the per-level NELBO is
  measured at *exactly that level* so "mastered?" means mastered-this-difficulty, not global drift.
- Global early-stop is **suspended while the curriculum progresses** (the easy-level global NELBO is
  dominated by the untrained high-`k` tail and would trip patience spuriously); normal early-stop
  resumes, fresh, once it converges to the full objective.
- Validated end-to-end (smoke: D64/L2, patience 1): clean `k=1→2→…` progression, level-NELBO
  descends then plateaus then advances. New unit test (`FrontierCurriculum - per-epoch NELBO
  plateau advances integer k`). Flags: `--curriculum-converge [--curriculum-patience N]
  [--curriculum-k-step K]`. **Open:** a matched-comparison vs the uniform baseline (curricula are
  schedules, so compare *to-convergence/final* recall, not fixed-step) and a sensible `k_step` for
  the low-signal mid-range (step-1 is fine-grained; difficulty barely moves from `k=30` to `31`).
- **Forgetting the easy levels — CHECKED, refuted (the easy levels are the MOST improved).**
  Frontier-POINT trains at exactly `k`, so during the climb the model stops seeing the easy `k=1`
  regime, and `k=1` is *not* a strict subset of `k=20` (the latter conditions on far less context).
  ch29 prints a live **`base(k=1)-NELBO` forgetting watch** each curriculum epoch; per-`t` recall at
  any checkpoint is the retrospective check. **If forgetting were observed,** the fix is a **cap/bias**:
  train the *range* `t∈[t_min, k/T]` up to the frontier (every easier level stays in the mix) instead
  of exactly `k` — a one-line trainer change. Kept available but NOT needed (see §13.1).

### 13.1 Curriculum A/B result (2026-06-16) — WINS by +3.8pt; no forgetting
Final self-terminated recall, matched arch (D128/L4) / corpus (3000 paras) / seed:
| run | overall | word-level | word-START | best NELBO | steps |
|-----|--------:|-----------:|-----------:|-----------:|------:|
| uniform baseline | 15.9% | 14.0% | 11.7% | 4.006 | 33.3k |
| **convergence curriculum** (k_step=2) | **19.7%** | **17.9%** | **14.7%** | **3.819** | 65.9k |

- **+3.8pt overall / +3.0pt word-START / NELBO 4.01→3.82** — the session's biggest controlled win.
  The curriculum converged cleanly (k=63 at step 54.9k, well under the 80k bound), then the full-
  objective phase ran to self-termination. Not "just more steps": the uniform baseline **plateaued
  and early-stopped at 33k** (its eval NELBO had bottomed); the curriculum reaches a genuinely lower
  optimum because the staged difficulty avoids that premature plateau. (Caveat: single seed, D128.)
- **Per-`t` recall, baseline → curriculum:** 0.05: 34.0→42.4 (+8.4), 0.10: 31.8→39.9 (+8.1),
  0.30: 24.7→31.2 (+6.5), 0.50: 17.3→22.1 (+4.8), 0.70: 10.4→12.7 (+2.3), 0.90: 4.8→5.2 (+0.4).
  **The gain is LARGEST at low `t` and decays monotonically to the high-`t` entropy floor** (H0=5.575)
  — the easy regime was *strengthened*, not forgotten, and the foundation transfers upward. Low-`t`
  masked-CE dropped 2.94→2.61, closing the §10 underfit marker further. **Forgetting refuted** for the
  deliverable model; the cap/bias variant is unnecessary here.
- **Next:** confirm on the founded arch (D256/L6) as the new candidate long run; sweep `k_step`
  (2 was fine; the mid-range barely changes difficulty so larger may save epochs); 2nd seed.

### 13.2 WHY the curriculum works — the "polluted global NELBO" hypothesis + an organic follow-up
Leading explanation (to confirm against the founded run, and only worth pursuing if that model
proves *useful*, not just lower-NELBO):

- **The curriculum's real win is the SIGNAL it terminates on, not the easy-first ordering.** Uniform
  training early-stops on the **global** NELBO `E_t[CE]`, which is dominated by the **irreducible
  high-`t` tail** (per-`t` curve rises monotonically to `H0`≈5.6 and barely moves there). The genuinely
  learnable progress lives at low–mid `t` but is a small fraction of that average, so the global signal
  **plateaus early and trips patience prematurely** — the high-noise levels *pollute* the stop metric.
  The curriculum instead gates on **per-level** plateaus, tracking each difficulty's own learnable
  progress, so it never terminates on the drowned-out average. Evidence: uniform baseline stopped at
  33k while the curriculum kept finding real gains to 66k (§13.1); per-`t` gain is largest at low `t`.
- **Transfer DOWN is proven, transfer UP is the open question.** Training higher `k` improved
  `base(k=1)` (3.69→1.54) — denser supervision feeds the easy regime (down). Whether mastering low `k`
  gives high `k` enough prior structure to resolve (up) is read directly from the **climb velocity as
  `k` rises**: steady ~epochs/level ⇒ the foundation feeds up; ballooning epochs/level ⇒ a **wall**
  (high `k` intrinsically under-resourced — the same starvation that hindered uniform training).
  *Metric, free from existing logs:* the advance cadence in the founded run. *Richer, retrospective:*
  `--per-t` per-level curve on any saved checkpoint (no restart) — compare an early vs late checkpoint
  to see whether high-`k` NELBO falls *before* the frontier reaches it.
- **Follow-up idea — the ORGANIC (self-scaling) curriculum.** Don't hand-set a `k` schedule: train the
  **full** noise spread but **per-level-plateau-aware** — sample `t` preferentially from levels still
  learning, and define the *global* stop as "**all** levels plateaued." This unifies three threads —
  it is adaptive **importance sampling** (§11), driven by **per-level mastery** (§13), and it
  structurally fixes the global-NELBO pollution (no single averaged metric to drown the signal). It
  needs only a per-level plateau tracker (generalize `FrontierCurriculum` to all levels) + a sampling
  weight. *Cheap metrics to add when we next restart* (more knowledge is cheap): a fixed mid-`k`
  NELBO probe in the curriculum log alongside `base(k=1)` (forward-transfer-up signal); the full
  per-level curve already exists via `--per-t`. **Gate:** decide after the founded run finishes AND we
  judge whether the end model is actually useful (coherent generation / recall worth the schedule),
  not merely lower-NELBO.

### 13.3 Level-NELBO rises with `k`, and the loss-weighting / per-level-cadence question (2026-06-17)
Observed live (W=16 founded run): the per-level NELBO **rises** as the frontier climbs —
k17 2.36, k18 2.39, k19 2.48, k20 2.57, k22 2.70 — while `base(k=1)` stays ~1.28–1.38.

**First, what is NOT a bug.** `weighted_cross_entropy` returns the **mean over masked positions**,
then the per-window NELBO scales it by `w = n_masked/(t·T)`. With **exact-count** masking (default) at
the frontier, `t·T = k = n_masked` exactly ⇒ **`w = 1.0`**, so the reported level-NELBO **is the mean
per-masked-token CE** (per-token NLL). As `k` rises, each masked token is predicted from less context,
so per-token CE climbs toward the unigram **entropy floor** `H0`≈5.6 — i.e. **level-NELBO rising with `k`
is the per-`t` difficulty curve, expected, not a defect.** Do not chase it as one.

**This sharpens §13.2's early-termination story.** Because `w≈1`, the global NELBO `E_t[CE]` is the
`t`-averaged *per-token* CE; every level contributes its per-token mean **equally per window**, so the
high-`t` near-floor region (most of the `t` range) dominates the average and the learnable low-`t`
progress is a thin slice — exactly "the higher levels are slow to move and the signal is lost in the
noise." Precise restatement of why uniform training trips patience early.

**Two genuinely-new investigables (backlog — see memory `diffusion-loss-weighting-and-level-cadence`):**
1. **Loss-weighting balance (the user's "not weighted evenly for the larger token count").** The NELBO's
   `(1/t)` weight makes each **window** contribute equally to the `t`-integral — which is the unbiased
   estimator, but it means an individual **high-noise token** prediction is weighted *less* per-token than
   a low-noise one (a `k=40` window's 40 predictions share the same window-weight as a `k=4` window's 4).
   That may train high-noise levels too slowly **and** be a fundamental reason mixed-noise gradients
   "conflict" (different effective per-token weights → competing directions; cf. the grad-conflict probe
   and [[ch31-diffusion-optimization-sandbox-plan]] variance findings). A/B to run: NELBO `(1/t)` weighting
   vs **per-token-equal** (drop `1/t`, weight by token count) vs an SNR-style weight — does per-token
   weighting move the high-`t` curve faster and reduce conflict, at what cost to the low-`t` win?
2. **Advance-while-still-learning / per-level cadence.** The curriculum advances on the current level's
   per-epoch plateau (relative-2% bar), but at high `k` the learnable gain is smaller and slower, so the
   plateau test can fire while the level is still inching down ("learning", not converged). This is the
   case **for** the noise-**spread** / organic curriculum (§13.2): train a *band* up to the frontier so
   each level keeps being reinforced as the front advances, rather than a point that leaves it behind.
   Cheap levers to consider: scale patience (or the bar) with `k`; gate advancement on
   learnable-progress-relative-to-floor `(CE−H0)` rather than absolute ΔNELBO.

**Gate / non-disruption:** do NOT perturb the live founded run to test these — they are A/Bs for a fresh
run (ideally the [[ch31-diffusion-optimization-sandbox-plan]] sandbox, where noise/weighting are isolated).
Mechanistically these underpin the organic curriculum (§13.2) and the polluted-global-NELBO story.

**Live evidence — a WITHIN-level NELBO sign-flip at k≈26 / t≈0.41 (2026-06-17).** Parsing the W=16
run's per-epoch level-NELBO triples (≈3 evals/level before advance): below ~k=25 the within-level
Δ (last−first) is mixed/noisy and leans DOWN (the level is still net-improving while trained); at and
above **k=26 it is consistently POSITIVE — 10 consecutive levels (k26–k35) all up, none down.** Ten
same-sign in a row is not 3-sample noise (random would be ~50/50; p≈0.001), so there is a real
**phase transition: frontier-POINT training makes a level's held-out NELBO WORSE once t≳0.41.** Past
~40% masking there is no net-learnable signal for the point objective, so the optimizer drifts in a
direction that worsens held-out high-noise prediction — most plausibly **overfitting** (with little
context the model can only memorize the seen high-noise windows, which don't generalise) and/or
gradient-noise/conflict near the floor. The curriculum's "mastered→advance" there detects
NON-improvement, but it is mild REGRESSION; the best per-level checkpoint is the arrival epoch.
Implications: (a) the net-negative tail starts at **t≈0.4, not t>0.8** as §11/§13.2 assumed — the real
learnable frontier for this model is ~0.4; (b) strong support for the noise-SPREAD/organic curriculum
(don't dwell at high k; keep low-k in the mix to anchor) and an LR-decay-at-high-k lever; (c) the
within-level slope sign-flip is itself a FREE diagnostic for "where training stops helping" — usable to
cap the curriculum frontier or switch to spread. Δdelta-per-level is small (~+0.01–0.02) and the post-
convergence full-objective phase re-anchors the learnable levels, so this is an efficiency/design signal,
not run-ruining — but it sharpens both the loss-weighting and per-level-cadence A/Bs above.

**CONFIRMED — the sign-flip is OVERFITTING (capacity-driven), via a data-matched A/B (2026-06-18).** A
**157K-param** model (D=64/L2, founded proportions; 38× smaller than the 6M, ≈ Chinchilla-matched to the
~2M-token corpus) was trained with the IDENTICAL curriculum/tokenizer/streams/B=16. Within-level slope: the 157K is
flat/scattered through k≈36 (k20–29: 2 up / 4 down) where the 6M was already **10/10 all-up from k=26**;
a **mild** within-level UP only **emerges at k≥37 (t≥0.58)** (+0.008→0.012, last 3 levels). So the onset
**SCALES WITH CAPACITY**: more capacity ⇒ *earlier + stronger* overfitting (6M flips hard at t≈0.41;
the data-matched 157K resists until t≈0.58 and only gently). [Correction: an earlier note here said the
157K was "immune / NO sign-flip" — that was from k≤29 data; it is far more RESISTANT, not immune.] The
mechanism stands and is cleaner for being capacity-monotone: the over-capacity 6M has room to MEMORIZE
high-noise training windows → held-out NELBO rises as it trains them; the data-matched 157K has little
spare capacity, so the effect is delayed to extreme noise and small.

**GENERATION coherence — DATA is the ceiling, not capacity (2026-06-18).** Ch30 iterative sampling at the
SAME curriculum step (k=36) on both models: the 6M and the 157K BOTH produce word-salad ("ROMEO: Fort S
Inightthe one alter…" vs "ROMEO: so a fav) death emacre's a… shall nature, drebans"), comparably
incoherent (the 157K arguably has more word-like fragments). A 38× capacity gap ⇒ the SAME incoherence,
so coherent generation is bottlenecked by DATA quantity/diversity, not model size. Final ledger for "just
train smaller?": the data-matched 157K wins honesty (no/late overfitting), memorization-resistance, and
~25× speed/cost, TIES on generation (both word-salad), and loses only raw held-out NELBO (which was partly
the 6M overfitting) — so for this data regime it is the better practical/educational choice; coherence
needs more data (full corpus via GPT-2 BPE / external text), not param count. (To re-test at the small
model's CONVERGENCE too — pending.) Corollary
results from the same A/B: the 157K is **on par at k=1** but **uniformly weaker on held-out NELBO** as the
task hardens (base(k=1) plateaus ~2.6 vs the 6M's ~1.3; per-level gap ~0.8) — capacity buys real held-out
modeling, not just memorization. **Net:** training a data-matched smaller model FIXES the overfitting/
memorization pathology (no high-k regression, cleaner honesty) at the cost of higher raw NELBO — a trade,
not a free win. The last open axis is GENERATION coherence (Ch30): if both are word-salad, DATA is the
coherence ceiling and the cheaper honest model is the better educational choice. (Run: `ch29_small_100k`.)
