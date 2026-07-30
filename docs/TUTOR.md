# Tutor — per-entry adaptive LR from a corpus mastery surface

**Status: BACKLOG, spike-shaped. Nothing here is agreed.** Proposed 2026-07-30, refined the same day.
The interesting question is not "can this be built" — it can — but **"does the feedback pay for its own
cost, and does it beat a global schedule at matched tokens"**. Both open.

**Working name: "Tutor"**, for the scheme (the schedule stops being a clock and becomes a teacher in a
feedback loop — checking progress, finding gaps, rehearsing them). The per-entry state it maintains is
the **mastery surface**. Provisional; alternatives considered were *isoprogress* (after the goal —
equal learning velocity everywhere) and plain *corpus ledger* (accurate, but says nothing about the
sentiment). Spike name would follow the repo's existing convention: `tutorspike`.

## The idea

Today the learning rate is a **global function of step**: warmup → stable → cosine cooldown
(`sub0/lr_schedule.hpp`). Every token in every document trains at whatever the clock says, regardless of
whether the model knows that material cold or is meeting it for the first time.

Tutor inverts that. The corpus carries a **mastery surface** — per entry (document, or window), a record
of how that entry is currently progressing. The training signal follows **where learning is still
happening**, so each entry effectively gets its own warmup and anneal, there is **no global progress
signal**, and well-learned entries are *rehearsed* rather than re-taught.

## The metric: learning VELOCITY, not loss level

This is the correction that makes the scheme viable, and it dissolves the failure mode that kills naive
versions.

The obvious design — weight by NELBO — chases **unlearnable** data: corrupted text, boilerplate, other
languages, OCR noise all score badly forever. RHO-LOSS exists to separate "not yet learnt" from "not
learnable", and pays for a held-out reference model to do it.

**The goal is not a flat NELBO across the corpus. It is a flat/consistent LEARNING GRADIENT across the
corpus.** An unlearnable entry has a high NELBO whose value *does not shift*. So the driving figure is
the **rate of change**, not the level:

| entry state | NELBO level | ΔNELBO | weight |
|---|---|---|---|
| not yet learnt | high | falling | **high** — the information is here |
| unlearnable (noise, foreign, corrupt) | high | ~zero | **low** — no return on compute |
| mastered | low | ~zero | **low** — rehearse, don't re-teach |
| regressing (forgotten) | rising | rising | **high** — rehearsal is overdue |

Level alone cannot distinguish rows 1 and 2; velocity separates them with no reference model. It also
gets **forgetting** for free, which a level-based rule handles only by accident.

## The normalization trap (must not be skipped)

Velocity must be measured **per unit of APPLIED LEARNING**, not per visit or per step.

Otherwise the scheme has a self-reinforcing false-mastery loop: down-weight an entry → it receives less
effective learning → its NELBO stops moving → **it looks mastered** → down-weight it further. An entry
can be driven to zero weight purely by having been given a low weight, and nothing in the signal reveals
that it happened. This is the mechanism by which a feedback controller confounds its own actuator with
its measurement.

So the denominator is the **accumulated applied learning** for that entry — Σ(effective lr × tokens
trained), not visit count and not wall-clock:

```
velocity_i  =  -Δ NELBO_i  /  Δ (Σ effective_lr × tokens)_i
```

That requires the ledger to track accumulated applied learning per entry, which is also exactly what the
diagnostic below wants to display. The same field serves both.

Second-order concern, noted not solved: this normalises against *our own* LR, but the model's global
state is also moving underneath every entry between visits. Velocity is therefore never a clean partial
derivative. Whether that matters is an empirical question for the spike.

## The implementation insight that makes a PoC cheap

**A per-entry learning rate is a per-entry LOSS WEIGHT.** Gradient contribution scales linearly, so
scaling entry *i*'s loss by `w_i` is equivalent to training it at `w_i · lr` — no optimizer change, and
no need for different LRs on different rows of one batch (which does not compose).

This project **already has the seam**: the per-window loss mask (`tests/loss_mask_tests.cpp`). Weighting
generalises it from `{0,1}` to a scalar. The gradient path needs no new concept.

## Why this is worth taking seriously

**It structurally removes a named, published failure this repo already cites.** `docs/LR_SCHEDULE.md`
records ICLR 2026 arXiv 2511.18903 (*How LR Decay Wastes Your Best Data in Curriculum-Based LLM
Pretraining*): curriculum gains collapse because data introduced late is seen at too low an LR. That is
**caused by the LR being global**. Per-entry schedules do not have the failure mode — late data arrives
with its own fresh schedule. Stronger than any efficiency argument.

It is also `docs/PLATEAU_DESIGN.md` §4a's coverage reframe applied one level down: if coverage is the
right *stopping* signal, the same surface is the right *training* signal. One mechanism, two uses.

## The corpus for the experiment: SPLICED, not synthetic-only

Rather than a purely synthetic toy, splice a corpus with a **real difficulty gradient**:

* **TinyStories sections** — simple, templated, low-entropy. Should saturate fast and fall in weight.
* **cosmopedia sections** — substantially more complex. Should retain weight far longer.
* **a deliberate unlearnable slice** — shuffled tokens / random bytes. High NELBO forever, **velocity
  ~zero**. This is the population that must NOT dominate.

Two of the three populations are real text, so the result speaks to real training rather than to an
artefact of generated data — while the ordering prediction (cosmopedia outlasts TinyStories) is known in
advance and therefore falsifiable. The unlearnable slice is the direct test of the velocity metric: a
level-based rule up-weights it, a velocity-based rule must not.

## The mastery-surface heat map — worth building FIRST, independent of the rest

Per document/window, the ledger can carry and display:

* visits / tokens trained
* **accumulated applied learning** (Σ effective lr × tokens) — the velocity denominator
* current NELBO, and its trend
* current weight / effective LR
* learning velocity

Rendered as a **corpus surface**, this is a diagnostic in its own right: it shows *where* the model has
and has not learned, which is precisely the coverage question `PLATEAU_DESIGN.md` §4a raises and which
nothing in this project currently instruments.

**This is the lowest-risk first deliverable and should probably lead.** It is read-only — it changes no
training behaviour — so it can be built and trusted before any feedback loop exists, and it is useful
even if Tutor is never adopted: it would let a plateau be *seen* (is the corpus uniformly covered, or is
loss dominated by a stubborn slice?) rather than inferred from a single scalar. It also de-risks the rest,
because a feedback controller whose state you cannot visualise is very hard to debug.

## PoC staging

1. **Ledger + heat map, read-only.** No weighting. Confirm the surface says something sane on the spliced
   corpus: TinyStories mastering faster than cosmopedia, the garbage slice pinned at high NELBO / zero
   velocity.
2. **Velocity metric**, normalised by applied learning. Verify the three populations separate.
3. **Weighting v0**: bounded, clamped `w ∝ f(velocity)`. Three falsifiable checks — duplicates/simple
   text collapse in weight; the unlearnable slice does **not** dominate; matched-token NELBO beats global
   WSD.
4. Only then: revisit scheduling ("rehearsal"), span-level granularity, feeding coverage to the stopping
   criterion.

## Remaining risks, named

* **Cost of measurement** — use the training loss already computed per window (free, optimistically
  biased) rather than extra forward passes. If the free signal is too biased, the scheme likely dies on
  cost.
* **Staleness** at 44 GB, where an entry may be visited only a few times in a whole run.
* **Stability** — loss weighting changes effective batch composition step to step, interacting with the
  `sqrt(batch)` LR heuristic and grad-norm clipping. Effective LR could drift with nothing logging it.
* **Reproducibility** — the surface is feedback state, so it must be checkpointed exactly or matched-arm
  A/Bs quietly stop being matched.

## Explicitly out of scope for the spike

Per-token weighting, learned/meta-learned policies, a second reference model in the training loop, and
any change to the mainline training path. A negative result is a successful spike; this doc is the record
of why.
