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

## Open items

* Ledger memory at scale: ~40 B/entry is nothing here (36k documents) but is ~1.6 GB at fineweb's ~40M
  documents. Aggregation or sampling, not a flat array — TUTOR.md already flags this as the staleness
  risk. Narrowing the loss fields to `float` (see stage 0) is the first easy cut.
* Any throughput number taken while another trainer shares the GPU is meaningless. Correctness work is
  unaffected; the three-pillar performance figures wait for an idle device. A concurrent training run
  needs `--allow-concurrent` and a small batch.
