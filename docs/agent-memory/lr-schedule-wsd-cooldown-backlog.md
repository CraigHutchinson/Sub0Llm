---
name: lr-schedule-wsd-cooldown-backlog
description: "BACKLOG — swap inverse-sqrt for a DeepSeek-V4-shaped schedule (constant, then COSINE cooldown to 10% of peak, triggered by the plateau detector). Sub0Llm's models currently never get a cooldown at all"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-30T20:48:27.882Z
---

## SHIPPED 2026-07-30 — `f9c5981` + `90a74b2`

Landed during the user's breaking-change window (arms A–E parked for retrain, backward compat waived, so
the "must not land mid-sweep" hold below is DISCHARGED). Implementation: `include/sub0/lr_schedule.hpp`
(pure, engine-free), tested in `tests/lr_schedule_tests.cpp`, wired in `src/train_stage.cpp`.
`CKPT_VERSION` 5→6.

**A fable review caught four real defects before landing** — worth repeating the pattern:
1. Cooldown entry was gated `!fixed_budget`, so every `--steps`/`--epochs` run trained at constant peak
   LR and stopped with **NO anneal at all** — worse than the inverse-sqrt it replaced, and it silently
   broke the matched-token A/B the doc itself prescribes. Fixed-budget runs now anneal over the last
   `COOLDOWN_FRACTION` of their budget, landing on the floor at `max_steps`.
2. Resuming past a completed cooldown stopped instantly at the LR floor, closing the resume-and-continue
   path arm A used for 5.6 further epochs. Now clears the cooldown and returns to the stable phase.
3. **My CKPT_VERSION justification was WRONG**: I claimed schemeV5 had already invalidated old
   checkpoints via the `nfloat` guard so the bump was free. schemeV5 changes WHICH strings occupy the
   vocab, not necessarily how many, so `PARAM_FLOATS` need not move. The bump is the sole cause of
   stranding the arms — taken deliberately, not free. Asserting an untraced safety claim is exactly what
   AGENTS.md §3 forbids.
4. Stale log line still printed "inverse-sqrt decay" every run.

**User's addition, and it removes the sharpest risk:** the state at the instant of triggering is saved to
`<model>.preanneal.step{:09d}.ckpt` — step-indexed so repeated anneals don't overwrite each other and so
the rollback point is interpretable after a resume renumbers steps; deliberately not `.step`-named so
`list_step_ckpts` can't see it (never pruned, never auto-resumed). A false plateau now costs one anneal
instead of the run, and cooldown length/floor become an A/B-able axis. The naming invariant **verifies
itself** after writing rather than relying on a comment.

**NOT DONE: the A/B.** inverse-sqrt vs WSD at matched tokens has not been run. The under-finishing
argument is sound and V4-sourced, but standing policy is to measure, not adopt on authority — especially
since V4 itself notes Muon reduces schedule sensitivity and we already run Muon.

---

Researched 2026-07-30, written up in `docs/LR_SCHEDULE.md`. (Historical: **must NOT land mid-sweep** — it
changes every arm's optimization trajectory and would invalidate the LoopSplit comparison. Discharged
above.)

## The finding that motivates it

`train_stage.cpp:488` uses linear warmup then horizon-free inverse-sqrt (`peak * sqrt(warmup/step)`) —
the 2017 Noam schedule. Because this project stops on a **plateau detector**, the run halts at an
arbitrary point on a slowly-decaying tail and **never runs a cooldown phase at all**. Work on the WSD
cooldown stage finds a disproportionate share of the final loss drop happens during that decay, so these
models may be systematically **under-finished** — including every LoopSplit arm.

That is the actionable part. It is not "our decay has the wrong shape"; it is "we omit the anneal".

## What DeepSeek-V4 actually does (arXiv 2606.19348, the de facto reference)

* Peak 2.7e-4 (Flash) / 2.0e-4 (Pro), held, then **COSINE cooldown to 10% of peak** (2.7e-5 / 2.0e-5).
* **Not linear, and NOT to zero.** The 10% floor is what Chinchilla used too. An earlier draft of the doc
  proposed decay-to-zero; that was wrong and is corrected.
* Optimizer: **Muon for most parameters, AdamW for embeddings, prediction head, all RMSNorm weights** —
  essentially the hybrid split this project already arrived at independently ([[muon-optimizer]]).

## The caveat that lowers the expected payoff

V4 reports Muon gives *"reduced sensitivity to learning rate schedule hyperparameters"*, and Sub0Llm
already runs Muon on the matrices. So the schedule-shape gap should matter **less here** than the
literature headline suggests.

But that is a reason to A/B, not to skip: **reduced sensitivity to the SHAPE of a decay says nothing about
OMITTING the decay entirely**, which is the actual situation. Keep those two claims separate.

## Two concerns raised by the user, both confirmed as named problems

1. **"`step` is an arbitrary unit"** — real enough to have its own paper (*Power Scheduler: A Batch Size
   and Token Number Agnostic LR Scheduler*, arXiv 2408.13359). **But it does NOT bite in this codebase**:
   warmup is epoch-denominated (`lr_warmup_steps = LR_WARMUP_EPOCHS * epoch_steps`, train_stage.cpp:1914)
   and the decay is referenced to warmup, so batch terms cancel and the schedule is effectively
   token-denominated. Only the deliberate `sqrt(batch/8)` peak scaling survives. Residual leak: Adam bias
   correction uses the raw step counter and does not cancel — a bounded early transient, not worth fixing.
2. **"Decay front-loads the early epochs"** — this is *How Learning Rate Decay Wastes Your Best Data in
   Curriculum-Based LLM Pretraining* (ICLR 2026, arXiv 2511.18903). Curriculum gains collapse under
   standard decay because late data is seen at too low an LR. Here the regime is multi-epoch reshuffling
   rather than a quality curriculum, so the effect is "later epochs systematically down-weighted".
   Its remedies are **moderate decay + model averaging** — explicitly NOT per-epoch restarts or a
   sawtooth. Do not build a cyclical schedule.

## Do NOT change

**Warmup length.** Practice is short warmup; 0.25 epoch is already ~6% of a 4-epoch run. Extending toward
a full epoch moves against practice.

## PROMOTED 2026-07-30: this is the NEXT work, ahead of finishing arms B/C/D/E

User decision: rather than spend days completing the remaining arms under a recipe now known to be
suboptimal, use **arm A's completed run as the baseline** and optimize training effectiveness first.
Arms B/C/D/E stay resumable and can be finished later if wanted.

### Requirement: ship it as a TOGGLE, default = current behaviour

Recipe axis, not an architecture axis — it does not change `PARAM_FLOATS` or model shape — so it belongs
with `--lr`/`--batch`/`--seed` as a **runtime `RunConfig` field recorded in config.json**, NOT as a
compile-time `POS_ENCODING`-style constexpr. Each model then records which schedule trained it. Default
must keep existing behaviour **bit-identical** so arm A stays a valid baseline and the other arms stay
comparable.

### THE CONSTRAINT THAT MUST NOT BE MISSED: do NOT bump CKPT_VERSION

A WSD cooldown triggered by the plateau detector makes LR depend on **when the cooldown started**, which
is run state — not a pure function of step as `lr_schedule()` is today. That state must persist or a
resumed mid-cooldown run silently reverts to constant LR.

**Put cooldown-start in `state.json`, NOT in the checkpoint.** `CKPT_MAGIC`/version is EXACT-match only
with no back-compat readers, so adding a checkpoint field would **invalidate every existing .ckpt —
including all four LoopSplit arms**, destroying exactly the resumability this plan depends on.

There is a direct precedent to follow: `tokens_seen` was deliberately carried via `state.json` for this
same reason (train_stage.cpp comment, 2026-07-29) — "A bump would have invalidated every existing .ckpt --
including the LoopSplit arms' checkpoints, which exist precisely so those runs can be resumed."

## Ordering

After the LoopSplit sweep completes. Then A/B inverse-sqrt vs V4-shaped at matched tokens on ONE arm,
per [[three-pillar-shootout-policy]] and the standing preference to measure rather than adopt on
authority. Related: [[roadmap-2026-07-21]], [[optimizer-alternatives-backlog]].
