# Per-entry adaptive LR from a corpus knowledge ledger — SPIKE / SANDBOX, not mainline

**Status: BACKLOG, proof-of-concept shaped.** Proposed 2026-07-30. Nothing here is agreed, and it should
land in a sandbox spike before any of it is considered for the training path. The interesting question
is not "can this be built" — it can — but **"does the feedback pay for its own cost, and does it beat a
global schedule at matched tokens"**. Both are open.

## The idea

Today the learning rate is a **global function of step**: warmup → stable → cosine cooldown
(`sub0/lr_schedule.hpp`). Every token in every document is trained at whatever the clock says, regardless
of whether the model already knows that material cold or is seeing it for the first time.

The proposal inverts that. The corpus carries a **ledger** — per entry (document, or span), a record of
how well-learned it currently is, measured by its own ELBO/NELBO. The training signal for a piece of text
is then driven by **how novel it is**, not by how far through training we are:

* well-learned entries contribute little — they are *rehearsed*, not re-taught;
* poorly-learned entries contribute more — they are where the remaining information is;
* each entry effectively gets **its own warmup and anneal**, on its own schedule;
* "training" becomes **flattening the ELBO across the corpus** rather than descending a global curve;
* there is **no global progress signal at all** — the ledger (a coverage map / heatmap) is the state.

The framing the user gave is the useful one: the schedule stops being an arbitrary clock and becomes a
**teacher in a feedback loop** — checking progress, finding gaps, and rehearsing them deliberately.

## Why this is worth taking seriously rather than filing as blue-sky

**It is the structural fix for a named, published problem this project has already cited.**
`docs/LR_SCHEDULE.md` records *How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM
Pretraining* (ICLR 2026, arXiv 2511.18903): curriculum gains collapse under standard decay because data
introduced late is seen at too low an LR. That failure is **caused by the LR being global**. A per-entry
LR does not have the failure mode at all — data introduced late arrives with its own fresh schedule.
That is a stronger argument for this design than any efficiency claim.

It is also the same reframe as `docs/PLATEAU_DESIGN.md` §4a, applied one level down. §4a argues a plateau
is **corpus coverage**, not a flat curve. If coverage is the right stopping signal, the same ledger is
the right *training* signal — and having one mechanism serve both is a real simplification, not two
features.

## The implementation insight that makes it cheap

**A per-entry learning rate is a per-entry LOSS WEIGHT.** A training example's contribution to the
gradient scales linearly in its loss coefficient, so scaling entry *i*'s loss by `w_i` is equivalent to
training that entry at `w_i * lr` — without touching the optimizer, the schedule, or the parameter
update, and without needing different LRs for different rows of one batch (which does not compose).

That matters because this project **already has the seam**: the per-window loss mask
(`tests/loss_mask_tests.cpp`, the padded-short-document path). Weighting is a generalisation of masking
from `{0,1}` to a scalar. The gradient path needs no new concept.

So the PoC does not require optimizer surgery. It requires a ledger and a policy.

## The hard problems, named honestly

1. **Measuring novelty is a forward pass.** If the ledger is refreshed by evaluating entries, the scheme
   can trivially cost more than it saves. The whole spike hinges on this. Cheapest option: use the
   **training loss already computed** for each window as a free (if biased) novelty estimate — no extra
   compute at all. It is measured on data being simultaneously fitted, so it reads *optimistically*, but
   it is free, and "free and biased" may beat "accurate and doubling the cost".
2. **Staleness.** An entry's score is only current at the moment it was last visited; the model moves
   underneath the whole ledger continuously. How stale is tolerable is an empirical question, and it
   interacts with corpus size — at 44 GB, an entry may be revisited only a few times in the whole run.
3. **High loss ≠ novel.** This is the failure mode that kills naive versions: up-weighting whatever
   scores worst chases **unlearnable** data — corrupted text, boilerplate, other languages, OCR noise —
   not informative data. RHO-LOSS (Mindermann et al., *Prioritized Training on Points that are Learnable,
   Worth Learning, and Not Yet Learnt*) exists precisely to separate "not yet learnt" from "not
   learnable", using a held-out reference model. Any version of this that skips that distinction should
   be expected to fail, and the spike should test it deliberately rather than discover it.
4. **Stability.** Loss-weighted training changes the effective batch composition step to step, which
   interacts with the `sqrt(batch)` LR heuristic and with gradient-norm clipping. Effective LR could
   drift without anything logging that it had.
5. **Reproducibility.** The ledger is run state that feeds back into training, so two runs with identical
   seeds diverge unless the ledger is checkpointed exactly. This project's A/B methodology depends on
   matched arms, so the ledger has to be part of the run's persisted state from day one.
6. **Does it actually gain anything?** Unknown. It must be A/B'd at matched tokens against the global WSD
   schedule, three-pillar (quality + throughput + memory), per standing policy.

## Suggested PoC shape (sandbox spike, small and falsifiable)

The point of a spike is a result, not a feature. Build the smallest thing that can be **wrong**:

1. **Toy corpus with KNOWN structure.** Synthesise documents where novelty is ground truth: a set of
   unique documents, plus deliberate exact duplicates and near-duplicates, plus a slice of deliberate
   garbage (random bytes / shuffled tokens) that is *unlearnable*. Ground truth is then known for all
   three populations, which is what makes the result interpretable — the same argument as the synthetic
   plateau trajectories in `PLATEAU_DESIGN.md` §4c.
2. **Ledger v0**: one scalar per document, EMA of its observed training loss. Free — no extra forward.
3. **Policy v0**: weight ∝ a bounded function of (entry loss − corpus mean loss), clamped hard.
4. **The three checks that make it falsifiable:**
   * duplicates should have their weight collapse quickly (the scheme notices "known");
   * the garbage slice must **NOT** dominate the weighting (problem 3 — if it does, the naive version is
     dead and the spike has earned its keep by showing that cheaply);
   * at matched tokens, does it reach a given NELBO sooner than the global schedule?
5. **Only if all three pass**, consider the harder questions: span-level rather than document-level,
   revisit scheduling ("rehearsal"), and how coverage feeds the stopping criterion.

## Relationship to existing work here

* `docs/PLATEAU_DESIGN.md` §4a — the novelty/coverage reframe for *stopping*. Same ledger, other end.
* `docs/LR_SCHEDULE.md` — the global WSD schedule this would replace or modulate, and the ICLR 2026
  decay-wastes-curriculum-data finding that motivates it.
* `sub0/blend_schedule.hpp` — already does deficit/weighted-fair scheduling **between sources** with
  equal-epoch-coverage targets. This is the same idea at per-entry granularity, and that machinery is the
  natural host for a revisit policy.
* The loss mask — the existing `{0,1}` seam that a scalar weight generalises.

## Explicitly out of scope for the spike

Per-token weighting, learned/meta-learned policies, anything requiring a second reference model in the
training loop, and any change to the mainline training path. If the toy result is negative, this doc is
the record of why — which is a successful outcome for a spike.
