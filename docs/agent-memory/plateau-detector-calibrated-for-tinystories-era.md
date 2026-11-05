---
name: plateau-detector-calibrated-for-tinystories-era
description: "BACKLOG — the plateau detector's window/patience are EVAL-COUNT denominated, so they shrink as a fraction of the run as runs lengthen; calibrated on ~2-epoch TinyStories runs, biased toward premature stopping on the 5+ epoch fineweb regime"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T20:49:23.327Z
---

Identified 2026-07-30 while arm A ran past 5 epochs still setting new bests.

## The evidence

`DEFAULT_EXPECTED_PLATEAU_EPOCH = 2.0` (train_stage.cpp) documents its own ledger:

```
d128 2.00ep, d192 1.70, d256 2.00, d320 1.80, d448 1.90, d512 2.50,
d768 2.30, ce256_prod_fixed (blend+content-embed) 2.80
-- every run confirmed in [1.7, 2.8], centered on ~2
```

Arm A reached **5.3 epochs still improving** — roughly double the top of that range, so the hint is outside
its own stated evidence for this corpus. The ledger even contains the predictive trend: the most complex
corpus in it (`ce256_prod_fixed`, blend + content-embed) plateaued **latest at 2.80**. TinyStories is
synthetic, low-entropy and templated so it saturates fast; a fineweb_edu/cosmopedia/minipile blend at
~0.78 bits/byte has far more to extract at fixed capacity. **>5 epochs is expected here, not anomalous.**

## The miscalibration that actually bites — NOT the hint, the WINDOW

Evals run every ~10% of an epoch, and `PLATEAU_WINDOW = 6` / `PLATEAU_PATIENCE = 3` are counted in
**evals**, so as a fraction of the run they shrink as runs lengthen:

| run length | trend window | patience |
|---|---|---|
| 2 epochs (TinyStories era) | 30% of training | 15% |
| 6 epochs (arm A) | **10%** | **5%** |

The detector judges "has it plateaued" over a progressively smaller slice of training precisely on the
longer runs this corpus needs. Systematic bias toward premature stopping.

It compounds: being far past the hint selects the **loosest** trend threshold
(`PLATEAU_MIN_REL_AT_HINT = 0.02` rather than `PLATEAU_MIN_REL_FAR = 0.002`), so the trend test has been
trivially satisfied for most of the run. **Only the patience guard has prevented four premature stops.**
Arm A hit 2-of-3 patience four separate times and broke out each time with a step-down of -0.0063,
-0.0048, -0.0048 and -0.0022. That guard — added after two arms stopped while hitting a new best at every
eval — is currently carrying the entire detector on a corpus its window was never sized for. See
[[loopsplit-3arm-clean-result-looping-works]] for the earlier false-plateau history.

This also vindicates the guard being **threshold-free**: a minimum-margin on "new best" would have stopped
arm A at least three times, each time prematurely.

## Also noted, no evidence of harm

`LR_BASE = 0.001` at batch 8 scaled by `sqrt(batch/8)` is a **56x extrapolation** of the heuristic at
batch 448. Loss descends smoothly with no instability, so this is a flag rather than a finding — but it is
an extrapolation, not a measurement, and it dates from the same era.

## DO NOT make the window fraction-of-run — that recommendation was WRONG

An earlier version of this note proposed fraction-of-run denomination. **The user correctly rejected it**:
a run-denominated window GROWS without bound, so at 20 epochs it would span multiple epochs and the
detector would become progressively LESS able to fire — a worse failure than being twitchy. And the
mechanism argues for epochs: a plateau is the exhaustion of new signal from RE-SEEN data, an epoch-scale
phenomenon, so an epoch-denominated window measures on the axis the phenomenon actually lives on.
Fraction-of-run measures on an axis with no mechanistic meaning.

The real requirement is neither: it is **signal-to-noise**. The window must be long enough that
`drift x window > noise`. Here noise is ~0.001-0.002 and drift ~0.002/eval, so 6 evals is marginal —
which is why the trend test fires while the run is still descending and the patience guard has to carry it.

## The deeper problem that subsumes the window question

**A plateau cannot be cleanly detected under a decaying LR** — flattening is confounded with the decay.
Measured on arm A: improvement fell to 0.35x while LR fell to 0.66x, so part of the flattening was step
size, not saturation.

Worse, under inverse-sqrt the cumulative learning distance `integral(lr dt) ~ sqrt(t)` **diverges**, so the
loss creeps down forever, ever more slowly. There is no true flat line to detect, only asymptotic
approach — which is why every magnitude threshold feels arbitrary: it necessarily IS.

**Therefore the LR change is a PREREQUISITE for a well-posed detector, not a separate improvement.** Under
a constant-LR stable phase, "stopped improving" becomes a real property of model+data rather than a
schedule artifact. Detect the plateau at constant LR, THEN trigger the cooldown — which is the actual
reason WSD and plateau-stopping pair naturally, a better argument than the cooldown-loss one. The 10% LR
floor helps for the same reason: it stops the schedule strangling learning before the real flat line.
See [[lr-schedule-wsd-cooldown-backlog]].

## MEASURED 2026-07-30 (`beb1ba0`, docs/PLATEAU_DESIGN.md §4c) — and it INVERTS the plan

Rolling OLS fit of val_nelbo vs eval index over all four arms' `train.log` series.

1. **Noise is ~0.002–0.004 nelbo**, stable across window widths (arm A: 0.00188 at W=4 → 0.00215 at
   W=12), so it is scatter, not curvature. `PLATEAU_MIN_REL_FAR = 0.002` sits right on top of it.
2. **The WINDOW of 6 is the miscalibration, quantified.** A slope-significance test at W=6 fires **five
   times** on arm A (epochs 4.2, 5.0, 5.1, 5.3, 5.6) while it was still descending. **W=8 never fires.
   W=12 never fires.** Six evals cannot resolve ~0.0007/eval drift against ~0.002 noise.
3. **Arms B/C/D were unambiguously still descending when stopped.** A significance test never fires on
   any of them at ANY window width. Three of four arms truncated with no trend-based basis.
4. **A significance test alone would NEVER stop anything here.** At W≥8 the slope stays significantly
   negative for every arm including arm A at 5.6 ep. It is a good FALSE-POSITIVE SUPPRESSOR and a bad
   stopping rule — so **the practical-gain bound must be PRIMARY**, the reverse of what the note below
   proposed. Calibration from arm A: stopping at 2/3/4/5 ep forgoes 0.0537 / 0.0351 / 0.0159 / 0.0039,
   marginal gain per epoch roughly halving late (~0.019 → ~0.012 → ~0.0065).

### READ FIRST — these curves came from the DELETED inverse-sqrt schedule

**Every series was produced under a monotonically FALLING LR**, so the curves flatten for two reasons at
once and nothing here separates them (measured: improvement fell to 0.35x while LR fell to 0.66x). This
is the same confound that motivated WSD — and I calibrated the replacement criterion on it anyway until
the user caught it.

What survives: **Finding 1 (noise ~0.002–0.004)** — scatter is a property of the eval set and batch, not
the schedule. **Findings 2–3** survive as evidence the CURRENT detector is broken (they are about runs it
truncated), and Finding 2's direction holds, though its exact misfire epochs are shape-dependent.

What does NOT survive: **Finding 4's diminishing-returns table is a LOWER bound.** Under WSD's constant
stable phase the late-run curve keeps a fixed step size instead of shrinking into the floor, so a
constant-LR run would still have had more to give than 0.0039-at-5-epochs suggests.

**The criterion redesign is PARKED until a WSD-era trajectory exists.** Sequence (user's, 2026-07-30):
1. TinyStories unbounded control — the fast smoke check, still answers whether
   `DEFAULT_EXPECTED_PLATEAU_EPOCH = 2.0` is observation or detector artifact.
2. **Arm A rerun under the full break set (schemeV5 + WSD), MANUALLY MONITORED, not auto-stopped** — to
   capture a real constant-LR trajectory to and past its true plateau. That is the calibration dataset.
3. **Synthetic trajectories with a KNOWN ground-truth plateau** for unit tests — a real trace shows what
   the curve looks like, but only a synthetic one lets a test assert "fired within N evals of the true
   knee", because only there is the knee known. Real data to CHOOSE the criterion, synthetic to PIN it.

Also: one corpus, one scale, and three series are short *because* the detector truncated them.

## Candidate criterion (SUPERSEDED by §4c above — significance is the suppressor, not the rule)

Threshold **significance, not magnitude**: stop when the fitted slope is no longer statistically
distinguishable from zero given observed noise (a t-test on the regression slope). Adapts automatically to
noise level and window length. **Caveat that must be handled or it never stops**: with enough samples any
nonzero slope becomes significant, so pair it with a PRACTICAL bound — stop when projected remaining gain
over the available budget is not worth the compute.

## Backlog actions (NOT mid-sweep — all arms share these constants)

1. **FIRST: run an unbounded TinyStories control** (e.g. 10 epochs, no early stop). If it keeps improving
   past the ledger's 1.9, then every "plateau epoch" in that ledger is **detector-induced rather than
   observed** — a lower bound, not a measurement — and `DEFAULT_EXPECTED_PLATEAU_EPOCH = 2.0` has no
   evidence base. Cheap, and it tests the foundation before anything is redesigned.
2. Capture the trends already available (every arm's train.log plus the historical ledger) before choosing
   a criterion. Do not redesign on argument.
3. Then re-derive the hint from a fineweb-era ledger, or make it corpus-aware.
4. Expect the **6-epoch step cap to bind for every arm** in the current sweep, so report them as "trained
   to a fixed generous cap", NOT "converged".

Related: [[lr-schedule-wsd-cooldown-backlog]] (the models also never run a cooldown),
[[tinystories-sweep-saturation]], [[production-line-pivot-medium-model]], [[autosize-research-grounded-formula]]
