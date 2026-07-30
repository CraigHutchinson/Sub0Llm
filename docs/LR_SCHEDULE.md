# LR schedule: what this project does, and what 2025-26 practice actually is

Researched 2026-07-30 to check whether `lr_schedule()`'s inverse-sqrt is still defensible. Short answer:
**it is the legacy choice, and the specific concerns raised against it are both named problems in the
current literature.** Two of them have a single fix that also happens to suit this project's stopping
criterion better than what it does now.

## What Sub0Llm does today

`train_stage.cpp:488` — linear warmup over `0.25 * epoch_steps`, then horizon-free inverse-sqrt
(`peak * sqrt(warmup/step)`). This is the 2017 Noam/Transformer schedule.

Verified separately (see the batch-units analysis): because warmup is epoch-denominated and the decay is
referenced *to warmup*, the batch terms cancel and the schedule is effectively token-denominated. The only
surviving batch dependence is the deliberate `sqrt(batch/8)` peak scaling. So the "step is an arbitrary
unit" smell does **not** bite here — but note that the general concern is real enough to have its own
paper, the **Power Scheduler**, whose title is literally "A Batch Size and Token Number Agnostic Learning
Rate Scheduler". The smell is well-founded even though this instance is safe.

## Reference point: DeepSeek-V4 (the current de facto)

`DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence` (arXiv 2606.19348), V4-Pro and
V4-Flash pretrained on 33T / 32T tokens. Three details matter here, and two of them temper the
recommendation below rather than reinforcing it.

**Schedule.** Peak LR 2.7e-4 (Flash) / 2.0e-4 (Pro), held, then **decayed by a COSINE schedule in the
final stretch** to 2.7e-5 / 2.0e-5. Two corrections to the naive WSD reading:

* the cooldown is **cosine, not linear**;
* it decays to **10% of peak, not to ~0** — the same 10% floor Chinchilla used. A decay-to-zero proposal
  is *not* what the de facto reference does.

The shape is still stable-then-cooldown, i.e. WSD-family, so Finding 1 stands; but if this project adopts
it, copy V4's form (cosine, floor at 10%) rather than the textbook linear-to-zero.

**Optimizer — and this project is already aligned.** V4 uses **Muon for the majority of parameters and
AdamW for the embedding module, the prediction head, and all RMSNorm weights**. That is essentially
Sub0Llm's existing hybrid Muon(matrices) + AdamW(everything else) split. Worth knowing the arrangement
arrived at here independently matches the current reference implementation.

**The caveat that most affects the plan below.** V4 reports Muon gives *"reduced sensitivity to learning
rate schedule hyperparameters"*. Since Sub0Llm already runs Muon on the matrices, the inverse-sqrt-vs-WSD
gap should matter **less here than for an AdamW-only run** — which lowers the expected payoff of the
change and strengthens the case for A/B-ing it rather than adopting it on authority.

**Not applicable at this scale**, but noted: V4 extends sequence length progressively (4K → 16K → 64K →
1M) and warms up with dense attention for the first 1T tokens before switching on sparse attention.

## Finding 1: inverse-sqrt is not current practice — WSD is

**Warmup-Stable-Decay** (warmup, then a long *constant* phase, then a short sharp decay) is the 2025-26
standard, adopted by DeepSeek-V3 and ERNIE 4.5. Its headline property is exactly what this project needs:
**the total step count does not have to be known in advance.** Cosine requires committing to a horizon;
inverse-sqrt is horizon-free but never actually finishes.

That matters here more than it would elsewhere, because Sub0Llm stops on a **plateau detector**, not a
fixed budget. Today the detector halts the run at an arbitrary point on a slowly-decaying inverse-sqrt
tail, so the model never gets a cooldown. The literature on the WSD **cooldown stage** finds a
disproportionate share of the final loss drop happens during that decay. So this project's models may be
systematically *under-finished* — stopped mid-tail rather than annealed.

WSD composes with plateau-stopping almost perfectly: hold LR constant until the detector fires, then run a
short linear decay to ~0 over a final fraction of steps, and report the post-cooldown number.

## Finding 2: the front-loading concern is real, and named

The observation that a monotonically decaying LR weights the *early* passes over later ones is the subject
of **"How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining"** (ICLR 2026).
Its finding: curriculum ordering beats random shuffling under a *constant* LR, but the advantage
**collapses under standard LR decay**, because data arriving late is seen at an LR too low to learn from.

Sub0Llm's regime is a multi-epoch reshuffle of a fixed 40% subset rather than a quality curriculum, so the
effect is not "best data wasted" but "**later epochs systematically down-weighted**" — the same mechanism,
without the quality ordering to make it costly. Each epoch re-presents the same distribution at a lower LR
than the last, so pass 5 contributes structurally less than pass 1 regardless of content.

The paper's remedies are worth noting because they are *not* what intuition suggests:

* **moderate LR decay** — i.e. decay less, which is what WSD's constant phase gives you;
* **model averaging** across checkpoints, unified with the above as Curriculum Model Averaging.

Notably **not** per-epoch restarts or a sawtooth. Cyclical schedules are not where this line of work went.

## Finding 3: decay-free variants are an active area

* **WSO** (Warmup-Stable-Only, no decay at all) reportedly outperforms decay-based schedules *after
  supervised fine-tuning*, even when decay looks better at the end of pretraining. Not directly relevant —
  this project does not SFT — but it reinforces that aggressive decay is not free.
* **WSM** (Warmup-Stable-Merge) replaces the decay phase with checkpoint merging, and can emulate cosine,
  linear or inverse-sqrt decay after the fact.

## What NOT to change

**Warmup length.** Current practice is *short* warmup — a small percentage of total steps. Sub0Llm's
0.25 epoch is already ~6% of a 4-epoch run, on the long side. Extending it to a full epoch would move
against practice, not toward it.

## The change, as LANDED 2026-07-30

Replace inverse-sqrt with a **DeepSeek-V4-shaped schedule**: warmup (unchanged) → constant at peak →
**cosine cooldown to 10% of peak**, triggered by the plateau detector instead of by a step count. This
addresses Findings 1 and 2 together: the constant phase is the "moderate decay" remedy for epoch
down-weighting, and the cooldown supplies the anneal the models currently never get.

Follow V4's form specifically — cosine, floored at 10% of peak — not the textbook linear-to-zero. The
floor is what the de facto reference actually ships.

Expected payoff is **lower here than the literature headline**, because V4 reports Muon reduces LR-schedule
sensitivity and this project already runs Muon on the matrices. That is a reason to A/B it, not a reason
to skip it: the cooldown claim is about a phase this project never runs at all, which is a different thing
from being less sensitive to the shape of a decay you are already doing.

### Status: LANDED, superseding the "do not land mid-sweep" hold below

This section previously read **"Do not land this mid-sweep"** -- correct while the LoopSplit sweep was
the thing being measured. That hold is DISCHARGED, not ignored: the user parked arms A-E for a full
retrain after the backlog window and explicitly waived backward compatibility, so there is no live
comparison left to invalidate. Implementation is `include/sub0/lr_schedule.hpp` (pure, tested in
`tests/lr_schedule_tests.cpp`), wired in `src/train_stage.cpp`.

Two things the design above did NOT say, both found in review before landing:

* **A fixed-budget run has a known horizon, so it must schedule its own anneal.** Triggering the
  cooldown solely from the plateau detector left every `--steps`/`--epochs` run at constant peak LR with
  NO anneal at all -- strictly worse than the inverse-sqrt it replaced, and it silently broke the very
  A/B this doc prescribes ("at matched tokens", i.e. `--epochs`). Fixed-budget runs now anneal over the
  last `COOLDOWN_FRACTION` of their budget, landing on the floor at the final step.
* **Entering the cooldown must not be a one-way door.** It is triggered by a detector this project's own
  notes call miscalibrated, so the state at the moment of triggering is saved to
  `<model>.preanneal.ckpt` -- deliberately not `.step`-named, so it is never pruned and never silently
  auto-resumed. A false plateau now costs one anneal instead of the run, and the cooldown length/floor
  become an A/B-able axis from an identical stable-phase state.

It remains a good candidate for its own A/B -- inverse-sqrt vs WSD at matched tokens on one arm -- since
the cooldown claim predicts a measurable final-loss gain, and this project's standing policy is to
measure rather than adopt on authority. That A/B has NOT been run.

## Sources

* [DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence](https://arxiv.org/pdf/2606.19348)
* [Power Scheduler: A Batch Size and Token Number Agnostic LR Scheduler](https://arxiv.org/pdf/2408.13359)
* [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining (ICLR 2026)](https://arxiv.org/abs/2511.18903)
* [Understanding Warmup-Stable-Decay: A River Valley Loss Landscape Perspective](https://arxiv.org/pdf/2410.05192)
* [Training Dynamics of the Cooldown Stage in WSD](https://arxiv.org/pdf/2508.01483)
* [WSM: Decay-Free LR Schedule via Checkpoint Merging](https://arxiv.org/pdf/2507.17634)
* [Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](https://openreview.net/forum?id=JnebU2QLdH)
