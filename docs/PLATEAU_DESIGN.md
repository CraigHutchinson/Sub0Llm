# Stopping criterion redesign — design note for review

**Status: DESIGN, not agreed.** Written to be argued with. Open questions are marked **[Q]**; several are
load-bearing and should be settled before any code. Nothing here lands mid-sweep.

## 1. Why this is being reopened

Evidence gathered 2026-07-30, all of it from this project's own runs:

* **Every arm in the published token-matched comparison was still descending at every one of its final six
  evals.** Arms B, C and D each set a new best at their last eval. None had plateaued.
* **Arm B stopped by the detector at step 8586 while that very eval set a new best**, against a cap of
  143,070. Clean case of a detector-induced stop.
* **Arm A, resumed, has since gained −0.058** — roughly 10x the entire D−B margin the comparison rested on.
* `DEFAULT_EXPECTED_PLATEAU_EPOCH = 2.0` is derived from a ledger (`d128 2.00, d192 1.70, ... 
  ce256_prod_fixed 2.80`) that, on the above, plausibly records **stopping-rule artifacts rather than
  observed convergence**.
* Arm A reached **5.4 epochs still improving** — roughly double the top of that ledger's range.

The current detector is two coupled tests: a least-squares trend fit over `PLATEAU_WINDOW = 6` evals
against `PLATEAU_MIN_REL`, AND `PLATEAU_PATIENCE = 3` evals with no new best. In practice the trend test
has been trivially satisfied for most of arm A's run (being far past the hint selects the *loosest*
threshold, 2%), so **only the patience guard has been preventing premature stops** — it broke five
separate 2-of-3 approaches with step-downs of −0.0063, −0.0048, −0.0048, −0.0022 and −0.0049.

That is a detector held up by a backstop. It needs rebuilding, not tuning.

## 2. The reframe worth settling first

**[Q1] Should we detect a plateau at all?**

Frontier practice does not. DeepSeek-V4 trained V4-Pro and V4-Flash to **fixed 33T / 32T token budgets**,
then applied a cosine cooldown at the end. The entire reason WSD-family schedules exist is that they are
*horizon-free* — you can decide to cool down at any point without having committed to a schedule shape in
advance. So the modern answer to "when do we stop" is closer to **"pick a budget from scaling laws, cool
down at the end"** than to "watch for flatness".

That suggests two viable designs:

* **(A) Budget + cooldown.** Choose a token budget per config (Chinchilla-informed, or from our own
  ledger once it is trustworthy), train at constant LR, cool down over the final fraction. Simple,
  reproducible, trivially comparable across arms, and it matches the reference implementations. Its weakness
  is that the budget is itself a guess, and a wrong one wastes or truncates.
* **(B) Detect-then-cooldown.** Keep an adaptive stop, but use it to *trigger the cooldown* rather than to
  end the run. Retains the ability to exploit an easy config, at the cost of the machinery below.

These are not exclusive: (B) with a hard budget cap degenerates to (A) when the detector never fires.
**Recommendation: build (B) with (A) as the outer bound**, but only if §4's validation succeeds — otherwise
ship (A), which is defensible on reference practice alone.

## 3. Design principles

Derived from the failures above, in priority order.

### P1 — No arbitrary magnitude threshold; adapt to the data's own scale

`PLATEAU_MIN_REL = 0.005` is a number someone chose. It cannot be right across TinyStories (low entropy,
saturates ~2 epochs) and a fineweb blend (still improving at 5.4). The criterion must derive its scale
from the observed series: its own **noise level** and its own **improvement rate**.

Honest caveat to state up front: **no stopping rule is truly parameter-free.** The goal is to make the
remaining parameter *dimensionless and decision-relevant* (e.g. a confidence level, or "gain per GPU-hour
worth having"), not to pretend it does not exist. A plan claiming threshold-freedom is hiding one.

### P2 — Invariant to the LR schedule's shape

This is the deepest issue and the one that makes the current design ill-posed. Under a decaying LR,
flattening is **confounded with the decay** — measured on arm A, improvement fell to 0.35x while LR fell to
0.66x, so part of the observed flattening was step size, not saturation. Worse, under inverse-sqrt the
cumulative learning distance `∫lr dt ∝ √t` **diverges**: the loss creeps down forever, ever more slowly,
and there is no true flat line to find. Any magnitude threshold is therefore necessarily arbitrary.

**Proposal: measure progress against learning distance, not steps.** Define

```
    L(t) = ∫ lr dt          (cumulative learning distance, a running sum, trivially checkpointable)
    rate = Δ val_nelbo / Δ L(t)
```

Loss-vs-`L` removes the schedule's contribution to the x-axis. A model still learning shows a roughly
stable `rate`; a saturating one shows `rate → 0` *even while LR is constant*. This directly answers "the
detector must not assume a particular LR falloff" — it assumes none, only that LR is known, which it is.

**[Q2]** Is `∫lr dt` the right learning-distance proxy under **Muon**? Muon's orthogonalized updates have
a different magnitude/geometry from AdamW's per-element-normalized ones (the codebase already notes this
where `MUON_LR_BASE` is defined), and this project runs a hybrid. A better proxy might be accumulated
update norm `Σ‖Δθ‖`, which is measurable and optimizer-agnostic — at the cost of a reduction per step.
**This is the single most important open question in the design.**

### P3 — Tolerate the staircase

Arm A's series is not smooth decay: flat stretches punctuated by drops larger than anything in the flat
part (jitter ~0.001 vs steps ~0.005). Any criterion evaluated on a window shorter than the staircase
period will fire in a tread. The window must be long enough to span a plausible tread, which is a
**signal-to-noise requirement**, not an epoch or run-fraction one.

Both previously-considered denominations were wrong: eval-count shrinks as a fraction of long runs;
fraction-of-run **grows without bound** and would eventually span multiple epochs, making the detector
progressively unable to fire. Neither is the invariant.

### P4 — Deterministic and resumable

The stop decision must be reproducible from checkpoint + `state.json`. **The cooldown trigger point must
persist in `state.json`, NOT the checkpoint** — `CKPT_MAGIC`/version is exact-match with no back-compat
readers, so a field addition would invalidate every existing `.ckpt` including all four arms. Precedent:
`tokens_seen` was routed through `state.json` for exactly this reason (2026-07-29).

## 4. Validation — and the problem with our data

**[Q3] This is where the plan is weakest and needs the most scrutiny.**

We have a real corpus of training curves: every arm's `train.log`, plus the historical ledger. But **they
are all truncated before a true plateau** — that is the finding that started this. So:

* they can validate the **false-positive** half: a good criterion must NOT fire anywhere in arm A's 5.4
  epochs, nor before the end of B/C/D. That is a genuine, immediately-usable test.
* they **cannot** validate the **false-negative** half. None of them contains a true plateau, so no
  amount of replay proves a criterion would fire when it should.

Two consequences:

1. **Simulation is required, not optional.** Generate synthetic curves with known ground truth — staircase
   descent with configurable tread length, asymptotic approach with no true flat, genuine plateau after a
   knee, plus realistic eval noise — and measure false-positive and false-negative rates against the known
   answer. This is the only way to test the positive half before collecting data, and it is cheap.
   **[Q4]** How do we make the synthetic generator faithful rather than convenient? A generator tuned to
   make our preferred criterion win proves nothing. Proposal: fit its noise and staircase parameters to
   arm A's *observed* series, then verify the criterion on curve families the generator was NOT fitted to.
2. **We will need to collect true-plateau curves.** The unbounded TinyStories control (10 epochs, no early
   stop) is the cheapest way to obtain at least one genuine converged curve, and it doubles as a test of
   whether the `= 2.0` hint's ledger is artifact. **Do this before finalising the criterion**, and treat
   any criterion validated only on simulation as provisional.

**[Q5]** Is one converged curve enough to validate against? Almost certainly not — but it is infinitely
more than zero, and each additional one is a full training run. What is the minimum viable positive set?

## 4a. The reframe that may replace all of §5: novelty, not progress-through-training

Contributed in review, and it is better motivated than the global-slope family below.

**The current framing asks "how far through training am I?".** Every mechanism in §5 watches one scalar —
corpus-average val_nelbo — and tries to infer saturation from its shape. But that scalar is an **average
over a heterogeneous corpus**, and averages hide exactly what we need: different parts of the corpus are
learned at different times, so a flat average can mean "everything is done" or "some parts are still
improving while others plateaued long ago". Those need different decisions and the average cannot
distinguish them.

**The proposed framing asks "is what I am seeing novel?".** A concept that does not fit existing knowledge
embeds slowly at first (foundations must form), then learning accelerates, then it saturates — an S-curve
*per concept*. Weight belongs on **each window/sample**, giving feedback at every step, rather than on a
global stage counter. Under this view:

> **A plateau is not a flat loss curve. It is the point at which the corpus has consistent coverage** —
> every part has been learned as well as it is going to be.

### Why this is likely right: it explains the staircase

Arm A's series is not smooth decay. It is flat stretches (jitter ~0.001) punctuated by step-downs ~0.005,
five times over. The global-slope family has no explanation for that shape and has to be defended against
it (§P3). The novelty framing predicts it directly: **a step-down is a cluster of related documents
crossing the knee of its S-curve together.** Different clusters knee at different times; between knees the
average is flat.

**That is a testable prediction on data we already hold**, and it should be tested before committing to
any design: instrument per-document or per-cluster loss, replay arm A, and check whether the step-downs
coincide with groups of items converging. If they do, the novelty framing is the correct model of this
process and §5's mechanisms are curve-fitting a symptom.

### Prior art to ground it

* **RHO-LOSS** (reducible holdout loss) — select on loss *reducibility* rather than raw loss, separating
  "not yet learned" from "unlearnable".
* **Automated curriculum learning** (Graves et al.) — learning-progress signals as the driver.
* **Example forgetting** (Toneva et al.) — per-example learning/forgetting dynamics over training.

A useful simplification for *our* purpose: RHO-LOSS separates unlearnable from unlearned because it is
doing **data selection**. For **stopping**, both mean the same thing — no more to gain here — so we can
skip the irreducible-loss model that makes RHO-LOSS expensive.

### The cheap implementable version

Full per-document tracking is millions of items of state and needs held-out evaluation to mean anything
(training loss on seen data measures memorization). Proposal instead:

* a **fixed stratified probe set** — a few thousand held-out documents sampled across the corpus's
  clusters, scored periodically at the existing eval cadence;
* per-probe loss curves, each reduced to a **converged / still-improving** verdict by the same
  noise-adaptive test §5 needs anyway;
* **coverage = fraction of probes converged**, and *that* is the stop signal.

This reuses the eval machinery, costs a bounded amount per eval, and yields a directly interpretable
number. It also degrades gracefully: with one probe cluster it reduces to today's global criterion.

### It also answers the overfit requirement

Raised in the same review: the corpus is iterated many times rather than streamed, so **the stop must
double as an overfit guard**. A probe set is held out by construction, so a probe whose loss starts
*rising* is a direct per-cluster overfit signal — detectable long before the corpus-average train/val gap
widens enough to notice. On arm A that gap is currently *closing*, so there is no overfit pressure yet;
this gives us the instrument to see it when there is.

**[Q7]** Probe granularity: how are clusters defined — by source (fineweb/cosmopedia/minipile), by
document length, by embedding cluster, or simply at random? Random is the honest default and needs no
extra machinery; source-stratified is nearly free and probably more informative.

**[Q8]** Per-probe noise is far higher than the corpus average, so per-probe convergence verdicts will be
weak individually. Does the **fraction converged** aggregate cleanly enough to be a stable stop signal, or
does it need its own smoothing? Simulation should answer this before implementation.

**[Q9]** What happens when coverage stalls below 100% — some probes never converge because they are
unlearnable at this capacity? The stop condition cannot be "all probes converged". Likely it is
"coverage fraction itself has plateaued", which reintroduces a curve-shape test on a *better-conditioned*
series (bounded [0,1], monotone-ish) rather than eliminating it.

### Extensibility seams to preserve

Per the same review, build these as extension points rather than a closed design:

* the **per-item verdict** (converged?) should be a pluggable predicate, so §5's mechanisms can be swapped
  underneath without touching the aggregation;
* the **aggregation** (coverage) should be separable from the **decision** (stop / trigger cooldown);
* the **probe set definition** should be data, not code, so stratification can change without a rebuild;
* the per-item ledger is independently useful for **data selection and curriculum** later, not only for
  stopping — which is a reason to build it even if the stopping criterion ends up simple.

## 4b. FIRST EXPERIMENT: TinyStories smoke collection

Agreed in review as the entry point, and it is well chosen — it does **three jobs in one run**.

TinyStories suits this better than the production corpus for reasons that matter here:

* **stories are coherent self-contained blocks**, so a document *is* the semantic unit. That answers
  **[Q7]** for the first experiment with no clustering machinery — probe = story;
* it **trains fast at small scale**, so a genuinely converged curve is hours not days;
* it is the corpus family the `DEFAULT_EXPECTED_PLATEAU_EPOCH = 2.0` ledger was largely built from, so
  whatever it shows bears directly on whether that ledger is artifact.

### What one run yields

1. **A true-plateau positive control.** Run unbounded (no early stop, generous cap). If the global curve
   keeps improving past ~1.9 epochs, the ledger entries are stopping-rule artifacts. If it genuinely
   flattens, we finally own **one converged curve** — the thing §4's validation gap says we cannot
   manufacture.
2. **A direct test of the novelty framing.** With per-probe losses logged, check whether global step-downs
   coincide with groups of probes crossing their knee. This is the prediction that distinguishes §4a from
   §5, and it is much easier to read on short self-contained stories than on web text.
3. **A first look at coverage as a signal** — does *fraction of probes converged* move smoothly enough to
   drive a decision (**[Q8]**), and where does it stall (**[Q9]**)?

### Instrumentation needed

Deliberately minimal, and additive — nothing on the training path changes:

* a **fixed held-out probe set** sampled once and stored as data (not code), so stratification can change
  without a rebuild;
* **per-probe loss logged at eval cadence** to a series file alongside `train.log`. Scoring goes through
  batched `forward()`, matching the standing policy for scoring, and reuses the existing eval seam.

Cost is the thing to watch: a few thousand probes at every eval is not free. Start with a **small probe
set and a coarser probe cadence than the main eval** (e.g. once per epoch), since the S-curve knees we are
looking for are epoch-scale, not eval-scale. Raise it only if the signal is too sparse to read.

**[Q10]** Should probes be excluded from training, or is it enough that they are held-out documents the
sampler never draws? They must be genuinely unseen for the overfit-detection half of §4a to work — worth
confirming the window sampler can be told to skip them, rather than assuming the held-out split already
does.

## 5. Candidate mechanisms, to be narrowed

Not a menu to implement — a list to argue down to one.

| # | Mechanism | Adapts to data? | LR-shape invariant? | Weakness |
|---|---|---|---|---|
| M1 | Slope-vs-`L(t)` + t-test on the regression slope | yes (uses residual noise) | yes | with enough samples any slope is significant → never stops; needs a practical bound |
| M2 | Model comparison: "descending" vs "flat" on the recent window, stop when flat is favoured | yes | yes if x-axis is `L(t)` | evidence-ratio cutoff is still a parameter, though a principled one |
| M3 | Projected remaining gain over remaining budget < worthwhile | yes | yes | requires a budget, i.e. collapses toward design (A) |
| M4 | Keep patience, replace only the trend test | partly | no | inherits the confound; minimal change, lowest risk |

**Recommendation for review: M1 or M2 on an `L(t)` x-axis, with M3 as the outer economic bound.** M4 is the
fallback if validation fails and we need something better than today without a redesign.

**[Q6]** M1's "never stops" failure and M3's "needs a budget" both push toward the same place — an explicit
compute budget. If we need a budget anyway, does that collapse the whole design into (A), and is the
adaptive machinery earning its keep? **I think this is the question that decides the project.**

## 6. Staging

1. **Extract** every existing curve (arms A/B/C/D, historical ledger) into a reusable series file. No
   redesign on argument — get the trends captured first.
2. **Build the simulator** and the replay harness, with the existing curves as false-positive controls.
3. **Run the unbounded TinyStories control** for at least one true-plateau positive.
4. **Settle [Q1], [Q2] and [Q6]** — these change what gets built.
5. Implement behind the same runtime toggle as the LR schedule work, default = current behaviour, so arm A
   remains a valid baseline and the existing arms stay resumable.
6. A/B against the arm A baseline, three pillars.

## 7. References

* [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/pdf/2606.19348)
  — fixed 33T/32T token budgets, cosine cooldown to 10% of peak; Muon for most parameters with AdamW for
  embeddings/head/RMSNorm, which is essentially this project's existing split.
* [Understanding Warmup-Stable-Decay: A River Valley Loss Landscape Perspective](https://arxiv.org/pdf/2410.05192)
* [Training Dynamics of the Cooldown Stage in WSD](https://arxiv.org/pdf/2508.01483)
* [Power Scheduler: A Batch Size and Token Number Agnostic LR Scheduler](https://arxiv.org/pdf/2408.13359)
* [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining (ICLR 2026)](https://arxiv.org/abs/2511.18903)
* See also `docs/LR_SCHEDULE.md` — the schedule change is a **prerequisite** for this work, not a sibling:
  a plateau is only well-defined at constant LR.
