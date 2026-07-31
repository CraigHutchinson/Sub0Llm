# tutorspike — build log and findings

Working record for the spike proposed in [TUTOR.md](TUTOR.md). That document is the design and the
argument; this one is what was actually built, what it measured, and what changed as a result.

**Status: Stage 0 and Stage 1 landed. Stage 2 (the ledger) is next.** Nothing here changes training
behaviour yet, by construction — the read-only surface leads, per TUTOR.md's own staging.

Branch `claude/tutorspike`, worktree `.claude/worktrees/tutorspike`.

## Staging

| stage | deliverable | state |
|---|---|---|
| 0 | per-window loss readout on the CPU and CUDA loss paths | **done** (`b3a8a41`) |
| 1 | spliced 3-population corpus + population manifest | **done** |
| 2 | the mastery-surface ledger, read-only | next |
| 3 | live heat map + the falsifiable three-population read | after 2 |
| 4 | weighting v0 — only if stage 3 reads sane | not started |

## Stage 0 — the measurement did not exist

TUTOR.md assumes the per-window training loss is free. It is computed, but it was not *reachable*:
both paths reduced to one scalar before the host could see anything. `ce_backward_kernel` atomicAdds
into a single `double` already weighted by `1/(batch*denom)`, and `train_batch` returns a lone `float`.

So stage 0 is a readout — an optional trailing `out_win_loss` array threaded through `train_batch`,
`sub0_dev_train_step`/`backward`/`forward_loss`, the CUDA kernel and the mock backend. `nullptr`, which
is what every pre-existing caller passes, leaves both paths bit-identical. The gate is an identity
rather than an approximation:

```
mean over b of win_loss[b]  ==  the batch mean already reported
```

pinned on the CPU across dense, ragged and masked batches ([win_loss_tests.cpp](../tests/win_loss_tests.cpp))
and on real hardware across the eval branch, the *chunked* gradient branch, and a partially-masked batch
([cuda_tests.cpp](../tests/cuda_tests.cpp)).

**Agreement is 1e-6 relative, not exact, and the reason bounds what the surface can resolve.** The
scalar accumulates `w * nll` in float32 (left alone deliberately, so the reported loss does not move);
the readout divides in double. Measured divergence at a masked batch is ~6e-9 relative. That is the
noise floor from arithmetic alone — small, but worth knowing before attributing a small velocity to
learning. It also answers a question raised during the build: **there is no point storing the ledger's
loss fields in double.** The measurement is float-limited at ~1e-7 relative, so `float` per entry is
free accuracy-wise and halves the largest structure the scheme needs.

## Stage 1 — the spliced corpus, and a boundary rule that had to be checked

`sub0llm-tutorspike` writes `data/tutorspike_corpus.txt` plus `data/tutorspike_manifest.json`. Three
populations, interleaved in runs of 64 documents so ordinal never coincides with population:

| population | documents | role | registered prediction |
|---|---|---|---|
| tinystories | 24000 | simple, templated | masters FAST — level falls, then velocity → 0 |
| cosmopedia | 8000 | complex real prose | retains velocity far longer |
| shuffled | 4000 | the unlearnable slice | level stays HIGH, velocity → 0 early |

Documents are drawn by seeking to random offsets across the whole source file, never from a prefix —
the same reasoning [window.hpp](../include/sub0/window.hpp)'s `doc_in_subset` gives, since neither
source is shuffled.

**The garbage slice is word-shuffled real text, not random bytes.** Random bytes are unlearnable in an
uninteresting way: they miss the vocabulary, so a high score might only mean "out of distribution".
Shuffling words holds the vocabulary and the unigram distribution fixed and destroys only sequential
structure — so the model *can* still learn the unigram statistics. Its loss therefore falls a little,
early, and then stops. That is exactly the state a level-based rule mistakes for "still has information
to give", which is what makes it the sharp test rather than a strawman.

### The finding: the document boundary is `<|endoftext|>`, not a blank line

The first version split on blank lines. That was wrong in three compounding ways, and it is worth
recording because none of them would have failed loudly:

* `src/tokenizer.cpp`'s `scan_doc_boundaries` recognises exactly one boundary, the `TOK_EOS` token.
  Blank lines are ordinary paragraph breaks and all three sources contain them freely.
* Splitting the *sources* on blank lines cut documents mid-story and let a sampled "document" span a
  real boundary — precisely the cross-document contamination `window.hpp` exists to prevent.
* Emitting blank-line-separated documents would have left the configurator seeing **one** document, so
  every ordinal in the manifest would have been meaningless.

And the case that made it visible: shuffling words scattered `<|endoftext|>` into the middle of the
garbage documents, which would have injected spurious boundaries into the one population whose whole
job is to have no structure. The tool now splits on the marker, strips any embedded occurrence, and
re-emits the marker as the separator.

### The manifest join, and how it is verified rather than trusted

The manifest maps document **ordinal** to population as run-lengths, because ordinal is the only join
key available: the configurator records document starts in the order it reads them, so document *i* in
the corpus file is document *i* in `corpus.tok`. `total_docs` exists so a consumer can check that
instead of assuming it.

Verified: 36000 documents written, and the configurator reports **36001** `doc_starts`. The extra entry
is the trailing phantom boundary `scan_doc_boundaries` documents (a final EOS pushes a start equal to
the token count). So the check the ledger must make is `doc_starts.size() - 1 == total_docs`, not
equality — if the configurator ever drops or merges a document, the counts disagree loudly instead of
silently mislabelling the whole surface.

Configured at `--dmodel 256 --vocab 16384` → vocab 16517, d256 L8 H2, 13.3M tokens. **Both overrides
are deliberate.** Auto-sizing on a 53 MB corpus picked d128/vocab 3718, and both would have wrecked the
experiment: a model too small to learn cosmopedia makes cosmopedia look *unlearnable*, which is the one
distinction the spike exists to test, and AGENTS.md §7 is explicit that a small config must stay
realistic in the axes you are not varying — vocabulary especially.

## The measurement problem Stage 2 has to face

Raised in review during the build, and it is the sharpest open issue in the design. TUTOR.md names it
briefly ("velocity is never a clean partial derivative"); it deserves more than that.

**An entry's NELBO is recorded as of the last time it was trained. Between visits the model moves
underneath it**, pulled by every other document's gradients. So the next measurement can come back
*higher* through no property of that entry at all. The measured ΔNELBO confounds two things:

1. what this entry's own gradient did — the signal, and
2. drift from everything else trained in between — interference.

At a low weight or a long gap, (2) dominates. And it is not simply noise to be filtered: that
cross-correlation is part of how representations settle, and under a global schedule it is applied
evenly in every direction. A per-entry controller stops being even-handed about it.

Three consequences for the ledger, all of which it should be built to answer rather than assume:

* **Velocity is only meaningful above the drift floor.** Which means the floor has to be measured, not
  guessed. The instrument is nearly free now that stage 0 exists: hold out a small set of documents that
  are **never trained**, and score them at the same cadence through `forward_loss`. Their ΔNELBO is
  interference *alone*, with no own-gradient term. Anything smaller than that is not a reading. This is
  also the probe set [PLATEAU_DESIGN.md](PLATEAU_DESIGN.md) §4a asks for and question **[Q10]** answers —
  one mechanism, both uses.
* **Sign may carry more than magnitude.** A candidate rule worth recording before the data arrives: it
  is not the size of the delta that says "train this more", it is whether the delta is positive at all
  (rising = actively being forgotten = rehearsal overdue). That is a different, cheaper, and much more
  interference-robust statistic than a normalised rate, and the ledger should record enough to evaluate
  both from one run.
* **Start the reading with the garbage slice.** Given enough capacity, *every* learnable population
  eventually goes flat — mastered and unlearnable look alike in velocity, and are told apart only by
  level. The unlearnable slice is the population whose predicted behaviour is unambiguous from early on
  (high level, velocity → 0 while the others are still moving), so it is the first thing that can
  confirm or falsify the surface, and it does not need the run to approach convergence to do it.

## Repeated visits: one instrument, two curriculum arms

Proposed in review: instead of one training visit per document, visit each document N times — either a
warmup sweep of several cycles over the whole corpus, or a per-entry repeat count that varies through
training. The stated motivation is the sharp part: **with two consecutive visits, the second reading is
taken with no other document's training in between**, so it measures that entry's own learning rather
than the drift that contaminates a reading taken many steps after the last visit.

That is three separable ideas with very different risk, so they are recorded separately.

### A. The back-to-back reading, as an instrument — adopt

This is the right attack on the interference problem above, with one correction that makes it stronger
rather than weaker.

**It is not fully clean.** The update between the two readings is driven by the whole batch — ~448
windows — so the model still moves under the entry because of its batch-mates. What changes is the
*size* of the contamination: from "an unknown number of steps since this entry was last drawn" (at 36000
documents, hundreds of steps) to **exactly one step**. And one step of drift is precisely what the
never-trained probe set measures. So the decomposition becomes well-posed:

```
Δ_measured (own gradient + 1 step of drift)  −  Δ_probe (1 step of drift alone)  =  own learning
```

Two unknowns became one measurement and one control, which is what the raw revisit delta never was.

**And it does not require repeating the training.** A forward-only re-score of the same batch
immediately after the step yields the identical measurement without changing what training does — which
keeps the whole of stage 2 read-only, the property TUTOR.md leads with. Cost: a forward is ~1/3 of a
forward+backward, so ~+33% on the steps where it runs, and at a cadence of every 20th step ~1.7%
overall. That is affordable in a way TUTOR.md's own cost risk ("if it needs extra forward passes it
probably dies") suggested it would not be — because the pass is amortised over a cadence, not paid every
step.

Caveat to carry: a reading taken immediately after training on those exact tokens is **maximally**
optimistically biased. TUTOR.md already notes the training loss is optimistic; back-to-back is the worst
case of it. The probe set is held out and therefore unbiased, which is a second reason to have one.

### B. A systematic coverage sweep instead of random draws — stage 4 arm

Random window sampling gives Poisson coverage: after one epoch's worth of steps, roughly 1/e ≈ **37% of
documents have never been drawn at all**. For a surface whose first job is to say where the model has
and has not learned, "never visited" is a confound that is indistinguishable from "no velocity" unless
the visit count is read alongside. A deterministic sweep removes it outright.

Cheap and appealing, but it changes training order, so it is not stage 2. Stage 2 keeps random sampling
and records `visits` per entry, which makes the confound *visible* — and quantifying how much of the
early surface is zero-visit is itself the evidence for or against doing the sweep.

### C. Multi-cycle per-document blocking, as a curriculum — defer, with a specific warning

"Teach one topic, then the next" is the risky one, and **this repository has already been burned by the
same shape**. `blend_schedule.hpp`'s header records it: a small source blended against a huge one got
statistically re-covered every few steps, "hit a memorization phase-transition almost immediately, and
its cratering loss contribution produced a visible cliff-then-plateau in the blended training loss while
the base corpus was still a tiny fraction into its own first epoch — a real incident". Repeating a
*single document* N times consecutively is a sharper version of that, not a milder one, and the
tell-tale is identical: a collapsing training loss on the repeated entry that means memorization, not
learning. Whatever else happens, this arm must be judged on held-out probes, never on its own training
loss.

**A finding worth recording separately, because it breaks an equivalence TUTOR.md relies on:** repeat
count is *not* interchangeable with loss weight. TUTOR.md's implementation insight — that a per-entry
learning rate is a per-entry loss weight, since gradient contribution scales linearly — holds for
weighting, but N sequential steps at `lr` is not one step at `N·lr` under Adam: the optimizer's moment
normalisation makes the two differ, and they diverge further the larger N gets. So repeats are a
genuinely separate actuator with its own dynamics, not a reparameterisation of the one already planned.
That is a reason to keep them out of the first weighting experiment rather than to skip them forever.

## The reframe: transfer is a product, not an error term

Raised in review after the drift-floor design above, and it changes what the ledger is *for*.

The between-visit change was being treated as contamination to subtract. But an entry is not trained
between its own visits, by definition — so whatever moved it came from **everything else**. Read on its
own, that is a direct measurement of how the rest of the corpus acts on this document:

* **moved down more than the corpus-wide floor** → other documents taught this one. Reinforcement,
  redundancy, shared structure. It is being learned for free and may deserve *less* training.
* **moved up** → conflict. Something else is competing for the same capacity and displacing it. That is
  where more training is actually indicated, and no level-based rule and no velocity reading can see it —
  velocity only observes what happens across an entry's own visits.

Same measurement, two readings: the corpus-wide **mean magnitude** is the noise floor that bounds
velocity, and each entry's **deviation from it** is that entry's relationship to the training set. The
second may be the more valuable output of the whole exercise, because nothing in this project can
currently see corpus structure at all — and unlike the weighting scheme, it is a pure diagnostic that
needs no controller to be useful.

Splitting the two needs exactly the reading already planned in **A** above, because the loss recorded at
a visit is the training forward's, taken *before* that step's update:

```
own learning  =  nelbo_post(visit k)  −  nelbo(visit k)        its own gradient
transfer      =  nelbo(visit k+1)     −  nelbo_post(visit k)   everything else's
```

Without the post reading the two terms stay summed and neither is recoverable — so `Surface::record`
deliberately records **no** transfer at all when a post reading is absent, rather than attributing an
interval that still contains the entry's own learning. That is pinned by a test, because silently
attributing it is the failure that would look most plausible.

`Entry` therefore carries `nelbo_post`, a running-mean `transfer`, and `global_mark`; transfer is
normalised by **global** applied learning, not the entry's own, because the thing being attributed is
everything else's training. 44 bytes/entry.

## Stage 2 — the ledger

`include/sub0/tutorspike.hpp` (surface, drift probe, manifest) + `src/tutorspike.cpp` (simdjson manifest
parse, versioned sidecar persistence, snapshot writer). Engine-free, so all of its arithmetic is
testable with no model — which matters because every way it can be wrong (a velocity normalised by the
wrong denominator, a transfer term that quietly includes own learning, a dropped sign turning forgetting
into mastery) still produces a plausible-looking heat map.

Nine tests, all passing; frontend suite 181 → 190 cases, 114495 → 114523 assertions, no failures.

Decisions worth recording:

* **Persistence is a sidecar, not a `.ckpt` field.** TUTOR.md requires the surface to be restored
  exactly or matched-arm A/Bs stop being matched — but adding a variable-length section to the
  fixed-size checkpoint struct is the highest-blast-radius change in this codebase (AGENTS.md §3), and a
  spike has no business stranding in-progress production runs to store state the mainline never reads.
  Versioned, and a mismatch refuses to load rather than resuming against mismatched feedback state.
* **The velocity mark threshold is scale-derived**, a multiple of what one typical visit applies, not a
  constant. Recomputing velocity every visit divides a float-limited numerator by a near-zero
  denominator, which yields a large number made entirely of rounding — it would read as a spectacular
  learning rate rather than as an absent measurement.
* **The drift floor is a mean ABSOLUTE delta.** A signed mean cancels: half the probes drifting up and
  half down would report "no drift" while the model was moving under every one of them.
* **Probes are excluded by construction**, not merely unlikely to be sampled — PLATEAU_DESIGN.md §4a's
  **[Q10]**, answered the strict way. A probe trained even occasionally contributes an own-gradient term
  and stops being a drift reading.

## Stage 2b/3 — wired in, and the first reading

`--tutor-manifest` turns the surface on. Per window it records into the ledger; every 20th step it
re-scores that step's windows through the eval seam for the post-update reading; at the eval cadence it
scores the never-trained probes, updates the drift floor, writes the sidecar and rewrites
`<model_dir>/tutor_surface.json`. `tools/tutor_heatmap.html` polls that snapshot — self-contained, no
CDN, no build step; serve the model dir over http and open it.

**One bug worth recording**, because it produced a confident, entirely wrong log line: the probe window
plan was built *before* `DriftProbe::reset()` populated the probe set, so it planned over an empty list.
The startup banner still reported "282 never-trained probes" (it reads the probe set, which was
populated a moment later) while the eval line reported "0 probes" and a drift floor of exactly zero. A
floor of zero is not obviously wrong — it reads as "the model is not drifting" — which is precisely the
kind of quiet failure the read-only stage exists to catch before a controller is consuming the number.

### First reading — 400 steps, 0.26 epochs, still in warmup

Smoke run (`--batch 32 --allow-concurrent`, alongside another trainer, so throughput here means nothing).

| population | docs | seen | nelbo | velocity | transfer | n |
|---|---|---|---|---|---|---|
| tinystories | 24000 | 8591 | 3.956 | 0.0000 | **−3.76e-03** | 145 |
| cosmopedia | 8000 | 6677 | 5.122 | 0.0000 | **−2.78e-03** | 470 |
| shuffled | 4000 | 1638 | 5.458 | 0.0000 | **+1.93e-03** | 32 |

Coverage 47.0%, drift floor 1.65e-04 per unit applied.

**The level ordering is as registered** — tinystories < cosmopedia < shuffled — but that is the weak
prediction, and at 400 steps it mostly reflects entropy rather than learning.

**The transfer term splits by sign on the first run, and that is the interesting result.** Both real
populations are *negative* — they improve while not being trained, i.e. the rest of the corpus teaches
them — and the shuffled slice is *positive*: it is actively displaced by everything else. That is
exactly the reinforcement/conflict distinction the review proposed, appearing before the model has
learned much of anything, and it is a distinction no level-based or velocity-based reading can make.

Read it cautiously for now:

* **velocity is still identically zero everywhere**, because the mark threshold (4 typical visits) has
  not been reached at ~1.5 visits/document. Velocity needs a much longer run; nothing here speaks to it.
* **n is small and very unevenly distributed** (145 / 470 / 32), so the shuffled figure in particular
  rests on 32 intervals.
* **A confound, now visible in the numbers**: coverage is 36% for tinystories but 83% for cosmopedia.
  Windows are drawn by token position, so a document's draw probability is proportional to its LENGTH,
  and cosmopedia's documents are much longer. Per-document visit rates therefore differ by population
  for a reason with nothing to do with learning. This affects visits, coverage and how many transfer
  intervals each population accumulates — it does not explain a sign flip, but any per-population
  average must be read with it in mind. It is also a direct argument for the coverage sweep (arm **B**).
* The run is inside warmup, so the learning rate — and hence applied learning per visit — is still
  rising.

### The back-to-back reading is bounded by MEMORY, not by time

Found on the first full-scale run, and it corrects the cost argument in **A** above.

The re-score was costed in *time* — a forward is ~1/3 of a forward+backward, so ~1.7% at a cadence of 20
— and that part holds. The part that was missed is the **footprint**. `sub0_dev_forward_loss` goes
through `fwd_alloc(full=true)`, which materialises an **unchunked** `[batch*T, VOCAB]` logits buffer.
Training never pays that because it chunks its head (`head_ce_chunked`); the inference forward does not.
At this run's effective batch that buffer is tens of GB, on top of a training allocation already sized
to fill VRAM:

```
cuda error: cudaMalloc(&g_fwd.logits, MV * sizeof(float)) ... -> out of memory
```

This is precisely the hazard `eval.hpp`'s `Session` comment already warns about — the note explaining
why a trainer's own evals stay on the CPU. The warning was read and its *time* implication acted on; its
memory implication was not.

**Fix:** re-score only the first `n` windows, where `n` comes from `eval::device_batch` — the existing
derivation that bounds this same buffer to `DEVICE_LOGITS_BUDGET_BYTES` — capped at 64. Sampling a
subset is not a compromise: the transfer term needs *some* visits to carry a post reading, not all of
them, which is what the cadence constant already said.

**And it failed silently**, which matters more than the bug. `rescore` returned false, training carried
on, the surface kept updating, and the transfer column simply stopped filling — presenting as "no
conflict or reinforcement found" rather than "the measurement never ran". Same shape as the probe-plan
ordering bug earlier: a broken instrument that reads as a benign result. Both now log once, loudly. Two
such failures in one stage is the strongest evidence yet for TUTOR.md's instinct to make the surface
read-only and visualised *before* anything consumes it.

## THE RESULT — 10 epochs, 1110 steps, exclusive GPU

`--epochs 10 --batch 448`, d256 L8, vocab 16517, 13.3M-token spliced corpus. 1110 steps, zero device
failures, zero re-score failures. `val_nelbo` best **1.8858**, still improving at the end — no plateau
and no overfit within 10 epochs, so nothing here is read off a saturated model.

Coverage 91.1%: **8.9% of documents were never sampled at all in ten epochs**, which is the
length-proportional draw combined with Poisson coverage, and is itself an argument for arm 4 below.

| population | docs | visits/doc | nelbo | velocity (raw) | transfer |
|---|---|---|---|---|---|
| tinystories | 24000 | 19.1 | 1.550 | 0.2254 | +5.30e-06 |
| cosmopedia | 8000 | 82.6 | 2.535 | 0.0132 | +3.63e-05 |
| shuffled | 4000 | 21.4 | 4.392 | 0.1218 | −8.91e-07 |

Drift floor 8.50e-06 per unit applied.

### Velocity at matched applied learning — the claim, and it holds

| applied | tinystories | cosmopedia | shuffled |
|---|---|---|---|
| 5–10 | 0.3590 (n=2133) | — | **0.2001** (n=422) |
| 10–20 | 0.2690 (n=12569) | 0.2144 (n=73) | **0.1413** (n=2403) |
| 20–50 | 0.0181 (n=3778) | 0.0150 (n=2759) | **0.0104** (n=677) |
| 50–100 | 0.0078 (n=230) | 0.0094 (n=4403) | **−0.0048** (n=34) |
| 100+ | — | 0.0074 (n=660) | — |

**The unlearnable slice carries the HIGHEST level (4.392) and the LOWEST velocity in every bucket where
it has data.** A level-based rule would weight it hardest of all three; the velocity reading ranks it
last everywhere. That is the spike's central claim and it survived the test.

It goes further at 50–100, where shuffled velocity is **negative** — the slice is actively regressing
under continued training while both real populations still creep forward. That is the "forgetting comes
free with velocity" property, observed rather than argued. Caveat: n=34 in that bucket.

### What did NOT replicate

**TUTOR.md's prediction that cosmopedia retains velocity longer than tinystories is unsupported.** At
matched applied learning tinystories is *ahead* at 10–20 and 20–50; cosmopedia edges it only at 50–100.
And in raw terms cosmopedia has the LOWEST velocity of all three (0.0132) — the opposite of the
prediction — because its documents are far further along their own curves (82.6 visits vs 19.1). What
actually distinguishes cosmopedia is that it keeps *receiving* learning, not that it retains velocity
per unit of it. That is a sampling property, not a learnability one, and it would have been reported as
a confirmed prediction by anyone reading the raw column.

**The transfer sign is not stable across training**, and this is the weakest part of the result:

| read | tinystories | cosmopedia | shuffled |
|---|---|---|---|
| 0.26 ep | −3.76e-03 | −2.78e-03 | +1.93e-03 |
| 3.8 ep | −1.98e-06 | +1.82e-05 | +5.77e-06 |
| 10 ep | +5.30e-06 | +3.63e-05 | −8.91e-07 |

Only "cosmopedia is the most conflicted" holds across the last two. Every other ordering flips. Two
things are likely responsible, and one is a design defect:

* **`Entry::transfer` is a running mean over every interval since step 0**, so a late reading is a
  lifetime average dominated by accumulated history rather than a statement about the model's current
  state. It cannot show what it is being read as showing. It should be an EMA with a horizon, or —
  better, and already proposed below — an event stream normalised post-hoc.
* n is small and uneven (869 / 1204 / 162), single seed, no error bar. Which is exactly what the
  duplicate-content control exists to supply.

### Three pillars

* **Correctness** — engine suite unchanged at the same 2 pre-existing failures; frontend 190 cases /
  114523 assertions green; the surface's own 9 arithmetic tests green. Zero device or re-score failures
  across 1110 steps.
* **Performance** — 113,747 tok/s mean over 101 evals on an exclusive GPU, with the surface on. The
  re-score at cadence 20 and the 282-probe CPU scoring at eval cadence are inside that figure. No
  clean surface-off baseline was taken at this config, so the *overhead* is not yet quantified — that
  is an honest gap, not a claim of "free".
* **Memory** — the ledger is 1.58 MB for 36000 documents (44 B/entry), and the snapshot JSON 1.11 MB.
  Extrapolated to fineweb's ~40M documents the flat array is ~1.76 GB, which is where aggregation or
  sampling stops being optional.

## What to control against next — the axes, then the arms

Raised in review while the 10-epoch run was in flight. The interim reading already showed why it
matters: the raw per-population velocity ordering was *wrong* (shuffled appeared to out-learn
cosmopedia) and only came right after binning on applied learning. A correction that large, found by
accident, means the normalisation is doing more work than the measurement.

### The axis problem: velocity has two, and we normalise by one

`velocity = -dNELBO / d(own applied learning)` treats an entry's own progress as the only thing that
moves its loss. But an entry's reading also depends on **how mature the model was when the reading was
taken**. Two documents both at `applied = 10` are not comparable if one arrived there in epoch 1 (long
document, sampled constantly, learning from a young model) and the other in epoch 7. Binning on applied
learning collapses exactly that second axis, so the corrected table above is still confounded — less
than the raw one, but not cleanly.

So the control axis worth adding is **global progress** (global applied learning, or step), giving a 2D
surface `velocity(own_applied, global_applied)` rather than a single number per entry.

**The correction that follows, and it reuses a trick already in the design.** Fit the corpus-wide
expectation `E[velocity | own_applied, global_applied]` and report each entry's **residual** against it.
That makes velocity comparable across populations by construction, with no binning and no hand-chosen
buckets — and it is precisely what the transfer term already does by reading each entry's deviation from
the corpus-wide drift floor. One idea, applied to both halves of the surface.

**What that needs from the ledger:** `Entry` keeps only the LATEST velocity, which cannot support the
2D fit. The cheap fix is not a bigger `Entry` but an append-only **velocity event stream** — one line per
velocity update (`doc, population, global_applied, own_applied, nelbo, velocity`). A few million lines
over a run, written at the same cadence as everything else, and it makes every normalisation choice a
post-hoc analysis decision rather than something baked into the ledger and impossible to revisit.

### Control arms, ranked by what they would actually settle

1. **A second seed. Do this one first.** Everything above is a single run, and this project has been
   here before: the GQA A/B was left unresolvable because seed noise ran 1.73x the signal, and the
   LoopSplit 3-arm result is flagged SINGLE SEED in the notes for the same reason. Nothing in the table
   deserves interpretation until a replicate says the ordering is stable. Cheapest arm, largest effect
   on what may be claimed.

2. **A duplicate-content control — the missing error bar.** Splice each of ~500 documents in TWICE, at
   different ordinals. The two copies are identical text, so any divergence in their measured velocity
   or transfer is pure noise: sampling variance, window placement, ordering, arithmetic. That yields the
   per-document error bar the surface currently lacks entirely, and it answers a question the drift
   probe cannot — the probe measures interference on *untrained* documents, whereas this measures
   reproducibility on *trained* ones, which is what every per-entry claim actually rests on. Nearly
   free: a splice-tool flag.

3. **Complete the 2x2: shuffled COSMOPEDIA.** Today the garbage slice is shuffled tinystories, so
   "shuffled" differs from "cosmopedia" in two ways at once — structure *and* source. Adding a
   shuffled-cosmopedia population makes it factorial ({tinystories, cosmopedia} x {intact, shuffled}),
   and the unlearnability claim becomes the **interaction term** rather than a difference of differences
   between non-comparable groups. This repo already learned this lesson structurally — the depth-axis
   plan records arm E as REQUIRED because it was the 2x2's missing cell, with an explicit instruction to
   read the interaction, not the raw arm difference.

4. **A length-matched corpus — fix the sampling confound at source.** Windows are drawn by token
   position, so draw probability scales with document length; that single fact produced the 7.3 vs 31.5
   visits-per-document gap and hence the wrong raw ordering. Truncating every spliced document to a
   common token length equalises visit rates by construction, and removes the need for the post-hoc
   binning that is currently load-bearing. Cheaper and more honest than correcting for it forever.

5. **A frozen-model null.** Score the same probe set twice with no training in between. The delta should
   be exactly zero; whatever it is instead is the measurement's own nondeterminism (bf16 accumulation,
   atomics ordering), and it is a floor underneath *every* reading including the drift floor itself.
   Minutes to run, and it is the only one of these that can invalidate the instrument rather than
   refine it.

Arms 1 and 5 are cheap enough to run regardless. Arms 2-4 are all splice-tool changes plus a rerun, and
2 is the one that turns every number in this document from a point estimate into a claim.

## The sampler: an epoch permutation, not independent sampling

Raised in review after the 10-epoch run, and it is the right primitive. Independent sampling was
producing three separate defects at once, all of them properties of drawing *with replacement*:

* **Coverage.** Poisson coverage left 8.9% of documents never sampled after ten epochs, reading as zero
  velocity — indistinguishable, in the surface, from "nothing left to learn".
* **Variance.** A document's visit count was Poisson-distributed around its expectation, so two
  *identical* documents accumulate materially different applied learning by luck alone, and that
  variance lands in every per-document reading.
* **The probe rejection loop** (11386 redraws) disappears entirely: probe documents are omitted when the
  plan is built, so exclusion is exact rather than statistical.

`EpochPlan` shuffles every window tiling the corpus and draws without replacement. The unit is the
**window**, not the document — a long document contributes proportionally more windows, which is correct
rather than a confound, and velocity's applied-learning denominator already handles length by design.

### Three defects found while building it

**The first epoch was not shuffled at all.** Reshuffling only on wrap left epoch 0 running in corpus
order — and since the splice interleaves populations in runs of 64, that would have trained them in
near-blocks for a whole epoch. Caught by a test asserting the permutation differs between epochs.

**`epoch()` read one too low at the boundary.** Counting wraps means that after consuming exactly
`size()` slots — one complete epoch — no wrap has yet occurred. Now derived from the draw count.

**A window longer than the step's own T corrupted the next window.** `seq_t` is randomised per step
(`vary_seq`), while `EpochPlan` tiles at a fixed `SEQ_LEN`, so a step running at T=110 could be handed a
256-token window. Every window-fill loop indexes row `b` as `b*T + s` while iterating `s < len`, so
`len > T` writes straight into row `b+1`. Training ran to completion with no error and the only symptom
was `val_nelbo` 2.1284 against 1.8858 — which was very nearly attributed to the permutation being worse
for the model. This is AGENTS.md §10 exactly: `win_len` is a shared surface, and its consumers'
constraint on it was not enumerated before changing what fills it. Fixed at two levels — `vary_seq` is
pinned off whenever the surface is active (independently right: a random per-step width is an
uncontrolled axis in the measurement, since a window's loss depends on how much context it had), and the
invariant is stated at the assignment rather than left implicit in a distant flag.

### The coverage metric had the wrong denominator

Under the permutation, coverage still read 91.1% — *identical* to the random sampler, which looked like
the permutation had failed. It had not: of 3189 unvisited documents, 281 are probes and 2908 form a
**contiguous tail from index 33069**, which is the validation split. Every reachable document was
visited.

`coverage()` was dividing by the total document count, including documents training can never reach, so
100% was unattainable and the permanent shortfall read as a coverage failure. Worse, it coincidentally
matched the old sampler's *genuine* Poisson shortfall almost exactly — a metric with the wrong
denominator concealing the regression it exists to detect. It now divides by the documents the plan
actually reaches (32810 of 36000) and reports 100.0%, with the unreachable count stated at startup.

### What the permutation costs — do not read val_nelbo across samplers

Two things make the sampler arms **not** comparable at equal step counts, and both are properties of
tiling rather than of the seeds:

* **Token accounting over-counts.** The trainer adds `batch_t * seq_t` per step, but tiled windows
  average ~208 tokens rather than 256 (short documents make partial windows), so ten *counted* epochs
  are ~8.16 real passes — roughly 23% fewer tokens actually trained. `EpochPlan::epoch()` is the true
  count and is logged alongside.
* **Fixed cut points remove an augmentation.** Independent sampling drew windows at random offsets
  inside long documents, so the model saw many distinct views of the same text; tiling always cuts at
  the same boundaries, so it sees 60948 fixed windows repeatedly. That is a real loss of diversity, and
  a candidate refinement is a per-epoch offset jitter — though jitter and exact-once coverage cannot
  both hold at the document edges, so it is a trade rather than a free win.

The seed pair below is internally valid because both arms share all of this; comparisons against the
earlier random-sampling run's `val_nelbo` are not, and are not made.

## Review of the counters and the recording capture

A deliberate audit of every counter and statistic the spike maintains, rather than continuing to find
defects reactively. Three of the four bugs found so far — the probe-plan ordering, the silent re-score
OOM, the `win_len > T` corruption — shared one property: **the instrument failed in a way that read as a
benign result.** That is the failure mode this section is organised around.

### Defects found and fixed

| # | Defect | Why it was dangerous |
|---|---|---|
| 1 | `nelbo_post == 0.f` used as "no reading" | 0.0 is a *legitimate* loss — `cross_entropy` returns exactly 0 for a fully loss-masked window and the readout reports it faithfully. A real reading would be classified absent, and transfer would silently never record for those entries. |
| 2 | Event buffer dropped silently when full | A diagnostic stream with an unrecorded hole is worse than no stream: downstream it is invisible and reads as an absence of *events* rather than an absence of *recording*. Now counted, reported every eval, and carried in the snapshot. |
| 3 | Sidecar wrote raw `Entry` with no layout stamp | Adding a field without bumping the version would reinterpret every entry at the wrong stride and load **silently**. The version alone cannot protect against this, because forgetting to bump it *is* the mistake. `sizeof(Entry)` now rides the header. |
| 4 | Magic checked *after* `entry_size` | On a foreign or truncated file every later field is garbage; reporting "wrong entry size" for what is actually "not a surface file" sends the reader after the wrong problem. |
| 5 | CSV header written per *process*, not per *file* | A resumed run injected a second header row partway down the stream, which every CSV reader parses as a data row of garbage. |
| 6 | `DriftProbe::observe` folded in non-finite scores | `nelbo_cpu_each` writes NaN for an ungradeable window; one NaN makes the whole floor NaN, which then compares false against every velocity — silently disabling the guard that says which readings are real. |
| 7 | `d_applied = applied - applied_mark` | Catastrophic cancellation at scale: `applied` grows without bound (~1e6 here) while the threshold stays a few units, so the velocity denominator degrades in proportion to run length — worst exactly where readings matter most. Now accumulated forward and reset at each mark. |
| 8 | Coverage divided by total documents | Covered above: made 100% unattainable and hid a regression behind a coincidence. |

### Known limitations, recorded rather than fixed

* **`Entry::transfer` is a lifetime running mean** since step 0, so a late read is an average over the
  whole run, not current state. Deliberately left as-is now that the event stream carries per-interval
  values — the events are the source of truth and any horizon can be applied post-hoc. The ledger field
  is a summary, and the docs say so.
* **`Entry::tokens` is `uint32`.** ~2.6e7 at fineweb-scale visit counts, so ~160x of headroom. Recorded,
  not defended.
* **`visits` in a transfer event is the count *before* that visit**, since transfer is computed before
  the visit is folded in; a velocity event's is *after*. Documented rather than forced into agreement,
  because both are the natural value at their respective moment.
* **A re-score covers the first `n` windows of a batch**, which is an unbiased sample only because the
  batch order *is* the shuffled permutation. That is now load-bearing on the sampler and is stated here
  so a future change to draw order does not quietly bias the transfer readings.

### A methodology error in my own reporting

The "113,747 tok/s mean over 101 evals" figure was extracted with `grep -oE "\(1[0-9]{5} tok/s\)"` —
which silently **excludes any eval below 100,000 or above 999,999 tok/s**. It is a mean over the
qualifying subset, biased upward by construction, and it was presented as a run-wide mean. Recomputed
without the filter below. Numbers taken by ad-hoc log-scraping deserve the same scepticism as numbers
taken by the instrument.

### The recording design

Four artefacts per run, each self-describing, in the model directory:

| file | shape | purpose |
|---|---|---|
| `tutor_run.json` | one object, `schema` versioned | **Identity.** Label, seed, batch, peak lr, seq len, manifest + its seed, windows/epoch, reachable documents, and every cadence constant. Written once at startup so it survives a run that dies before its first eval. |
| `tutor_surface.json` | column-major arrays | **Current state**, for the live viewer. Now also carries `reachable`, so the viewer distinguishes "training never reaches this" from "not visited yet" — both are `visits == 0`. |
| `tutor_events.csv` | flat table | **History.** One row per velocity/transfer update with both normalising axes (`own_applied`, `global_applied`) plus `win_len`, `visits`, `step`, `pop`. This is the retrospective record; the ledger is only its summary. |
| `tutor_surface.bin` | versioned binary | **Exact resume state**, layout-stamped. |

The principle behind the split: identity is written once and never changes, state is overwritten, and
history only ever appends. A reader six months from now needs all three, and the most common way to lose
a diagnostic is to have kept only the middle one.

## THE SEED-CONTROLLED RESULT

Two runs, identical but for the seed (1 and 2), 10 counted epochs each (8 real plan epochs), epoch
permutation, re-score every step, 100% coverage, zero failures, zero dropped events. This is the control
ranked first, and it settles both of the spike's claims — in opposite directions.

### The central claim SURVIVES

> At matched applied learning, the unlearnable slice has the lowest learning velocity — while carrying
> the highest loss.

| applied | seed 1 | seed 2 | verdict |
|---|---|---|---|
| 5–10 | 0.1532 (n=678) | 0.1516 (n=668) | **holds in both** |
| 10–20 | 0.0971 (n=2624) | 0.1000 (n=2633) | **holds in both** |
| 20–50 | 0.0074 (n=285) | 0.0071 (n=281) | **holds in both** |
| 50–100 | 0.0117 (n=14) | −0.0120 (n=12) | split — n≈13 |

Shuffled is the lowest of the three populations in every bucket with a usable sample, in both seeds,
while its NELBO (4.16) is the highest by a wide margin (tinystories 1.29, cosmopedia 2.45). **A
level-based rule weights this population hardest of all three; the velocity reading ranks it last
everywhere.** That is TUTOR.md's core proposition, and it is now replicated rather than observed once.

Reproducibility is tight where it matters: NELBO agrees to 0.0–0.5% across seeds, velocity to 0.1–5.2%.

### The transfer term FAILS its control

| population | seed 1 | seed 2 | |
|---|---|---|---|
| tinystories | −2.18e-06 | −9.46e-07 | 57% apart |
| cosmopedia | +3.51e-06 | +1.04e-05 | 196% apart |
| shuffled | −3.26e-06 | **+4.06e-06** | **sign flip** |

With ~1,350–7,300 readings per population — twenty times the previous run — the transfer term still does
not replicate. The shuffled population changes sign between seeds. **Every per-population transfer
number reported earlier in this document is withdrawn**: the reinforcement/conflict split at 0.26
epochs, the "cosmopedia is most conflicted" reading, all of it. They were seed noise.

That is a negative result about the *measurement*, not about the idea. Transfer as a concept is
untouched; what is refuted is the claim that this estimator resolves it at per-population granularity.

### Why it fails, from an independent direction

`docs/TUTOR_SWEEP.md` reached the same conclusion structurally, before these numbers existed. The
transfer interval is one **full epoch** long: at 60,948 windows per epoch over 32,810 documents the
median document is visited once per epoch, so its measurement row spans essentially the entire corpus.
Row density ≈ 100%, therefore `rank(A) ≈ 1`, and a matrix of that rank can estimate exactly one thing —
a global drift scalar. Per-population differences at 1e-6 are below what it can resolve, so what we were
reading as structure was the residual of a single number.

A structural prediction and an empirical replication failure agreeing is much stronger evidence than
either alone, and it points at the fix: **shorten the interval, do not restructure the sweep.** Nothing
about ordering changes row density, because one update happens per step; only re-scoring a held-out
panel *between consecutive steps* does (0.735% density at batch 448).

### Registered predictions, scored

| prediction | outcome |
|---|---|
| shuffled: high level, lowest velocity | **confirmed**, both seeds |
| tinystories masters fast | confirmed (lowest NELBO, 1.285) |
| cosmopedia retains velocity longer than tinystories | **not supported** — and the seed *disagreements* fall exactly here, tinystories vs cosmopedia swapping at 5–10 |
| transfer separates reinforcement from conflict per population | **falsified at this resolution** |

## Open items

* Ledger memory at scale: ~40 B/entry is nothing here (36k documents) but is ~1.6 GB at fineweb's ~40M
  documents. Aggregation or sampling, not a flat array — TUTOR.md already flags this as the staleness
  risk. Narrowing the loss fields to `float` (see stage 0) is the first easy cut.
* Any throughput number taken while another trainer shares the GPU is meaningless. Correctness work is
  unaffected; the three-pillar performance figures wait for an idle device. A concurrent training run
  needs `--allow-concurrent` and a small batch.
