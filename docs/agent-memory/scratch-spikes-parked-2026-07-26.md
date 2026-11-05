---
name: scratch-spikes-parked-2026-07-26
description: "PARK NOTE for the whole scratch-token/packing spike line (factspike Phases A-I, KV-trace memoization, SPELL-marker finding, periodic re-injection) -- exact state, what's built vs. uncommitted, and the prioritized continuation path. Read this FIRST before resuming any scratch-token work."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-26T16:17:09.730Z
---

**2026-07-27 -- factspike is now DISABLED IN THE BUILD, not merely parked (`55412b8`).** Its tests sit
behind `SUB0_BUILD_SPIKE_TESTS` (OFF by default); revive with `cmake -DSUB0_BUILD_SPIKE_TESTS=ON`, which
restores the exact pre-change assertion counts. The objective rule applied: a spike qualifies for the
gate only once NOTHING in `src/` or `tools/` includes its header. factspike passed that test (its finding
had already been merged into `corpus_collapse.hpp`); **scratchspike and corpus_collapse did NOT** --
`train_stage.cpp` consumes both as blend sources, so their tests still cover a production path and stay
in the default build. `core.hpp`'s KV-splice API is kept (it is this work's substrate) but now documents
that its only consumer is spike-gated. Default baselines moved: d196check 44009991 / 125.


**Status: PARKED 2026-07-26 at the user's request, mid-thread, with a live positive result in hand.**
Not abandoned, not blocked -- deliberately paused to work on Nanbeige-4.5-style architecture features
instead. Everything below is the state to resume from.

## Where the investigation actually got to

Axis 9 (attention-capacity preservation -- does a packed slot preserve the multi-hop computation a real
multi-piece word gets?) is **root-caused, and one real fix now has evidence**:

1. **The bottleneck is fixed-size COMPRESSION, not missing information.** Three independent experiments
   converge: candidate 1 (feed the compressor richer input -- real per-layer K/V instead of raw
   embeddings) didn't help; Phase H (give mechanism A + candidate 1 the SPELL markers too, i.e. MORE real
   signal) made both measurably WORSE; candidate 2 (remove compression entirely) is exact.
2. **Candidate 2 (full n-position KV splice) is provably EXACT** -- cos_sim 1.000, zero exceptions, 9/9
   subjects -- but only once SPELL markers are included, and it costs O(n), the same as not packing. It
   solves fidelity by giving up the entire point (axis 10, cost efficiency).
3. **The SPELL_START/SPELL_END finding** (user's own hypothesis, confirmed by controlled swap-and-
   remeasure): `detail::word_span` strips the wrapper markers a real forward pass actually processes.
   EVERY packing mechanism in the project composes from that marker-stripped basis. Including them closed
   candidate 2's entire remaining gap to exact 1.000. Production `corpus_collapse.hpp` + the live
   `decode.hpp` word-collapse path are both exposed to the same gap; `wordspike.hpp`'s own spike data is
   NOT (it bypasses tokenization, using raw bytes).
4. **Context-sensitivity is NOT the wall people might assume**: an isolated-captured trace, markers
   included, spliced into a genuinely DIFFERENT real context still reaches ~0.98 cos_sim (F=0.919
   pieces-only vs G=0.983 marker-inclusive). Markers matter more than context does. This is the
   load-bearing evidence for any "cache a word's trace once, reuse it across many mentions"
   prefill-compute-amortization idea.
5. **Phase I -- the first real O(1)-cost positive.** Periodic re-injection (Nanbeige-inspired, see
   [[nanbeige-architecture-reference]]): re-add the packed vector into the slot's hidden state every
   `stride` layers instead of relying on one layer-0 injection. Scale-adaptive (fraction of `h`'s own
   current norm -- a first fixed-embedding-scale version was a near no-op because the residual stream's
   norm grows with depth and dwarfed it). Result: mean cos_sim 0.831->0.860 monotonic across dose, and
   **accuracy 3/9 -> 5/9 correct** at the 40% dose, EVAL-ONLY on a model never trained for the signal.

## What is actually BUILT (and its exact scope)

Engine primitives, all in `include/sub0/core.hpp` + `src/backend_cpu.cpp`, all CPU-only and all
`forward_one`-only (the incremental decode path) -- **none of this touches the training graph**
(`Model::forward()`/`train_batch`, a structurally separate hand-written autodiff implementation):
- `kv_krow_ptr(layer,pos)` / `kv_vrow_ptr(layer,pos)` -- read raw KV-cache rows.
- `kv_rope_rotate(row,pos)` -- thin export of the existing `rope_row`; `pos` may be negative (exact
  inverse, verified algebraically AND by an always-on round-trip test).
- `kv_splice_row(layer,pos,k_canonical,v)` -- write a (pooled or full) trace row directly into the cache.
- `set_scratch_reinject(stride,scale)` -- Phase I's periodic re-injection; `scale` is a FRACTION OF `h`'S
  OWN CURRENT NORM, not an absolute constant. `(0, 1.0f)` restores default behavior exactly.

Test harness in `tests/factspike_engine_tests.cpp` (hidden/opt-in TEST_CASEs, plus two always-on ones):
- `[.factspikekvtrace]` -- Phase G, mechanisms A/B/C/D/E/F/G side by side on ONE trained model.
- `[.factspikemarkers]` -- Phase H, separate marker-inclusive training run.
- `[.factspikereinject]` -- Phase I, dose-response sweep.
- `[factspike][kvtrace]` (ALWAYS-ON, 2 cases) -- `kv_rope_rotate` round-trip identity, and the
  tokenization-boundary assumption `hidden_state_cosine_kvtrace_full` depends on.
- `corpus_collapse::build_dataset_markers` + a 3rd capstone arm in `tests/corpus_collapse_engine_tests.cpp`.

Docs: `docs/FACTSPIKE.md` (Phases A-I, full honest write-ups including the negatives and the two
diagnosed false starts), `docs/SCRATCH_TOKEN_FRAMING.md` (criteria/axis table + candidates 1/2/3),
`docs/CORPUS_COLLAPSE.md` (production cross-check entry).

## Committed at park time (clean tree)

All of the above is COMMITTED on branch `claude/llm-transformers-1bit-qf3t3i` as two commits:
- `0d0d36e` -- engine primitives + factspike Phases G/H/I + the SPELL-marker finding + both docs.
- `c273144` -- the `corpus_collapse` marker-inclusive cross-check (production-scale, honest negative).

Working tree was clean at park time. Full regression was green at park time (124 test cases on
factspike96, 125 + ~44M assertions on d196check) EXCEPT one confirmed pre-existing, unrelated failure
(`tokenizer_tests.cpp` "case markers stay atomic" -- fails identically on a clean checkout of the
factspike96 toy config; verified by stash/rebuild, not caused by any of this work).

## Continuation path, in priority order

1. **Wire periodic re-injection into the TRAINING graph with a LEARNED gate.** The single highest-value
   next step. Phase I's positive came from an ungated flat heuristic on a model that never saw the signal
   in training -- a learned gate (Nanbeige's `NanbeigeNgramLayerFusion` shape: normalize both sides, gate
   on a learned dot-product, project the injected value through a learned matrix) trained end-to-end
   should do strictly better. NOTE the real cost: `Model::forward()` is a separate hand-written autodiff
   path from `forward_one`, so this needs a forward AND a hand-derived backward, plus CUDA parity, plus
   new learned params (which reopens the `TODO(charencoder-production)` checkpoint-persistence gap --
   see [[slot-encoder-checkpoint-persistence-design]]).
2. **Multi-seed replication of Phase I** before over-trusting 3/9->5/9 at n=9, one seed. This project has
   been burned by single-seed results before (the HRR-vs-MeanPool history).
3. **Sweep the re-injection dose further** -- only 5%/15%/40% were tried, all monotonically improving. No
   ceiling or breaking point found yet; a crude ungated heuristic must eventually inject enough noise to
   hurt, and knowing where that is informs the learned-gate design.
4. **Polysemy risk is UNTESTED and architecturally the biggest open threat.** Every result so far used
   invented proper nouns with zero polysemy by construction. Memoizing a genuinely ambiguous word's trace
   (bank/spring/lead) and splicing it into a context implying the other sense could misfire badly -- this
   is arguably a hard limit on what packing can ever do for common vocabulary, and would scope the whole
   mechanism to low-polysemy vocabulary (rare terms, names, jargon -- which is what scratch tokens have
   always targeted, likely not coincidentally).
5. **HRR-crosstalk analytical cross-check** -- still not done despite being flagged repeatedly. Cheap, no
   training: fit the existing 9-point `(n_pieces, cos_sim)` data against Plate's `1/sqrt(n)` curve.
6. **Marker-inclusive fix for the PRODUCTION path** is deliberately NOT recommended yet -- the real-scale
   `corpus_collapse` NELBO cross-check showed no measurable effect (identical to 4 decimals), so there's
   no evidence it currently harms production. Revisit only if an axis-9 fix validates at toy scale first.

Related: [[factspike-piece-transfer-validated]] (the full phase-by-phase detail),
[[scratch-tokens-context-translation-layer]], [[nanbeige-architecture-reference]],
[[roadmap-2026-07-21]] (all other Sub0Llm areas).
