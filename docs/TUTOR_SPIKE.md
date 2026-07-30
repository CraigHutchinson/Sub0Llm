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

## Open items

* Ledger memory at scale: ~40 B/entry is nothing here (36k documents) but is ~1.6 GB at fineweb's ~40M
  documents. Aggregation or sampling, not a flat array — TUTOR.md already flags this as the staleness
  risk. Narrowing the loss fields to `float` (see stage 0) is the first easy cut.
* Any throughput number taken while another trainer shares the GPU is meaningless. Correctness work is
  unaffected; the three-pillar performance figures wait for an idle device. A concurrent training run
  needs `--allow-concurrent` and a small batch.
