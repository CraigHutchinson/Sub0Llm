---
name: persistent-slot-range-engine-substrate
description: "DONE (engine substrate + spike test, committed) - unbounded id>=VOCAB scratch-slot range for a future persistent compound-word cache. A first attend-only resolve curriculum trained on it (K=6/30) shows real but unsaturated held-out generalization. train_batch/win_persist MT path + parallel eval DONE 2026-07-18 (988s for 3-seed K=30, was previously not finishing 1 seed in 10min). Still NOT built: the compound-word cache/DB, encode-time substitution, live decode-time collapse."
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

The "persistent compound-word cache" idea (a recurring multi-token OOV, e.g. "Rumpelstiltskin", gets ONE stable id forever instead of costing N tokens every mention) was designed in depth across several conversation turns but never durably saved — the design lived only in a plan file (`smooth-noodling-kurzweil.md`) that later got overwritten for an unrelated task, and in transcript. This memory is the durable record, now that a real chunk of it is actually implemented.

**The core design question, answered**: does a scratch-slot-like id need to live inside the fixed tokenizer vocab/marker range (bounded, needs a format-bump to grow), or can it be an "extended" id >= VOCAB (open-ended, no format bump)? Answer: **extended ids work, and are structurally cheaper than growing the reserved marker block**, because:
- `VOCAB` is a **compile-time constant** sizing `tok_emb`/`lm_head`/`logits`/`sample_token` — an id >= VOCAB can **never be sampled** by the model (`sample_token` is hard-bounded to `[0,VOCAB)`), only ever **injected** by a harness. This is the same "predictor swap" pattern the whole [[deterministic-mechanisms-thesis]] already runs on.
- `encode_slot`/`encode_slot_bwd` (the mean-pool/CharEncoder/Scalar compose functions) are **already id-agnostic** — they take a fragment span + a `tok_emb` pointer, never the slot's own id — so they needed ZERO changes to serve this second range.
- Growing the EPHEMERAL reserved-marker pool (`SCRATCH_SLOT_BASE..+COUNT`) is a tokenizer-FORMAT bump (`n_base`/VOCAB shift, invalidates every trained checkpoint). An id >= VOCAB is NOT — it lives in a range the base encoding never touched, so it's compatible with already-trained checkpoints with no retrain.
- Extending an EXISTING type (`ScratchBindings`, with a runtime-configurable base) was considered and rejected: the two ranges have different LIFETIMES (ephemeral = thread_local, rebound per window; persistent = global, read-once, immutable for the process) and different scale characteristics (a handful of slots vs. a potentially large, monotonically-growing table) — forcing them into one type would blur both. Built as a SEPARATE type instead (`PersistentBindings`).

**What's DONE (2026-07-16), engine-level substrate + spike test, all committed-pending**:
- `include/sub0/scratch_slots.hpp`: `PersistentBindings` (fragment table + a RUNTIME `base` field, unlike `ScratchBindings`' compile-time `SCRATCH_SLOT_BASE` — this header deliberately has no `core.hpp`/VOCAB dependency, so callers that DO know VOCAB pass it explicitly), `is_persistent_slot(token, base)`, `persistent_fragments(pb, token)` (null/unbound-safe by construction — always returns empty rather than requiring every call site to branch on `pb` being null).
- `include/sub0/core.hpp`: `set_persistent_bindings(const PersistentBindings*)`, forward-declared like `set_scratch_bindings`.
- `src/backend_cpu.cpp`: a plain (NOT thread_local) global `g_persistent_binds`, matching the "global, read-once, immutable" lifetime. Wired into all 3 embedding-lookup sites (`op_embed` forward, `Op::Embed` backward, `forward_one` decode) with `is_persistent_slot` checked **unconditionally** for any id >= VOCAB — this closes a real, previously-latent OOB risk: before this change, NOTHING stopped an id >= VOCAB from falling through to the raw `tab[id,...]` lookup and reading past the `[VOCAB,C]` table's bounds if one ever arrived. Now such an id always composes (real fragments if bound, an all-zero row via `encode_slot`'s existing empty-fragments contract if not) and never touches the raw table.
- `tests/persistent_slots_engine_tests.cpp` (5 cases, 27,096 assertions, all green): pure-function correctness of `is_persistent_slot`/`persistent_fragments`; an engine differential proving a bound persistent slot's forward embedding matches `encode_slot`'s own mean-pool output (via a substituted-ordinary-token reference arm, since a persistent id has no real row to overwrite for the usual scratch-slot differential technique); a real forward+`cross_entropy`+`backward`+`reduce_gradients` cycle proving gradient scatters into the fragment rows (caught a real test-design bug of my own along the way: a persistent id can never legally be a `cross_entropy` TARGET, since it has no logit column — had to reposition it to input-only in the sequence); an unbound/no-table-installed safety test (finite logits, no crash); and a no-persistent-binding invariance test (bit-identical when the feature is untouched).

**What's still NOT built** — this is a proven, tested SUBSTRATE, not the compound-word-cache FEATURE itself:
- The actual DB (word text -> canonical fragment ids -> assigned persistent id). `TOK_SPELL_START`/`TOK_SPELL_END` (casing.hpp) already flag candidate multi-token OOV/proper-noun spans for free — a frequency threshold over those is the cheap first-cut mining strategy, not yet implemented.
- Encode-time dictionary substitution (recognizing a known compound in a corpus/prompt at tokenize time and substituting its persistent id directly) — not implemented.
- Live decode-time retroactive collapse (recognizing a JUST-GENERATED repeat and binding it mid-generation) — [[repeatspike-harness-collapse-result]] proved the MECHANISM works (harness-driven, mid-stream binding), but that spike used the EXISTING bounded 6-slot ephemeral pool, not this unbounded range — the two haven't been connected yet.
- `corpus.tok`'s on-disk width has no reserved slack (`tokmap.hpp`'s `tok_bytes_for_vocab` sized only to the tokenizer's own vocab) — whatever eventually assigns real persistent ids must feed the true max-id-plus-one into `TokWriter`, or risk silent truncation. Not yet a real concern since nothing constructs such ids outside tests yet.
- CUDA: zero scratch-mechanism awareness on the device path (confirmed, unchanged) — this stays CPU-only, same as the rest of the scratch-embedding work.

**Multi-scale verified (AGENTS.md §7 -- engine bugs can be dims-dependent)**: the 5-case test suite passes
at both d128 (`gsm8k` build dir, 27,096 assertions) and a freshly-configured d196/L4/H7 CPU-only build
(198,432 assertions) -- same test, different scale, both clean. Full suite also green at d196 (26,929,410
assertions, 88 cases -- fewer cases than the CUDA-enabled `gsm8k` build since this one was configured
`-DSUB0_COMPUTE=CPU`, excluding cuda_tests.cpp; expected, not a gap).

**First curriculum/harness built on top of the substrate (2026-07-16, same session, later)**: narrowed to
attend-only resolve tasks only (see [[persistent-slot-selection-problem-backlog]] for why the "choose which
entity" half of scratchspike's task family can't transfer — a persistent id has no logit column, so it can
never be a model's predicted answer). `include/sub0/persistent_scratchspike.hpp` (new curriculum, `base`
threaded as an explicit param so the header stays engine-free) + `tests/persistent_scratchspike_engine_tests.cpp`
(`[.persistent_scratchspike]`, 2 cases: K=6 baseline, K=30 past the bounded pool's ceiling). `train_batch`
doesn't accept `PersistentBindings` (confirmed, matches this memory's own "What's still NOT built" list
below being about encode-time/decode-time work, not batched-training support, which is now ALSO
confirmed missing) — training drives the low-level forward/backward API in a manual per-window loop
instead (same shape `train_ce_steps` already uses for CharEncoder's own un-batchable gradient).

**Result (SUPERSEDED same session by the 4-encoder multi-seed shootout below -- the first MeanPool-only
read was misleading)**: the initial MeanPool-only numbers (K=6 0.125, K=30 0.225 held-out) suggested "real
but weak/unsaturated." The full shootout showed MeanPool was simply the WRONG ENCODER for this task.

**4-ENCODER SHOOTOUT RESULT (2026-07-16/17, K=30 at 3 seeds each -- user-directed multi-seeding,
decisively vindicated)**: at K=6, **ConvPool effectively SOLVES resolve** (held-out 0.917, drilled 0.943
tracking, clean phase transition ~step 1800), HRR strong-and-climbing (0.542 at cutoff), MeanPool/Hash
stuck at 0.125. **The encoder ranking is TASK-DEPENDENT**: this INVERTS content-select (where HRR 0.744
beat ConvPool 0.458) -- per-task shootouts are the method, never a one-time crowning. At K=30/3000 steps,
seed variance (Hash 0.117-0.583 across seeds!) exceeds every between-encoder gap -- unsaturated,
inconclusive, the longer-budget saturation run is the real K=30 decision point. **Selected default:
ConvPool for the persistent range** (spike/Phase-2 use -- enc_w trains in-process, no persistence needed);
**HRR = parameter-free fallback** wherever enc_w checkpoint persistence (still-open
TODO(charencoder-production)-class work) isn't built.

**EPOCH-WALK FOLLOW-UP (2026-07-17, user-hypothesized)**: re-ran the two wildest-variance K=30 arms with
an exactly-once-per-epoch shuffled document walk (SampleMode::EpochWalk) instead of i.i.d. random windows.
Verdict is a split: the variance hypothesis was NOT confirmed (seed spread persists undiminished --
0.350/0.525/0.875 MeanPool, 0.108/0.167/0.467 Hash -- so the seed sensitivity is OPTIMIZATION-BASIN entry,
not coverage luck; the chain A/B's zero variance at 1.000x3 corroborates: variance lives only in
capability-edge tasks). BUT the walk consistently HELPED MeanPool (worst walk seed 0.350 > random's median
0.225; mean 0.294->0.583) and its seed 3 hit **0.875 held-out at K=30 with plain MeanPool** -- a clean
phase transition, the best K=30 number from ANY configuration. Two consequences: (1) EpochWalk is now the
recommended spike-training default (removes a confound, never worse, plausibly raises basin-entry odds);
(2) **K=30 resolve is SOLVABLE and the wall was never encoder choice or capability -- it's basin entry
within budget**, which reframes the shootout's whole (already-flagged-inconclusive) K=30 tier. The K=6
encoder ranking (ConvPool 0.917 > HRR > MeanPool/Hash) stands. Getting ConvPool runnable required real plumbing
(PersistentBindings had no enc_w/enc_w_grad; all 3 backend call sites hardcoded nullptr -- selecting
ConvPool was UB before the 2026-07-16 fix). MeanPool seed-1 reproduced EXACTLY across the plumbing change
(differential proof the threading is inert for param-free encoders). Full table:
docs/DETERMINISTIC_MECHANISMS.md; raw logs: out/build/gsm8k/persistent_shootout_multiseed.log.

**Deprecation plan for the 6-in-vocab ephemeral pool (2026-07-16, user-directed, CONDITIONAL -- revised
same day)**: the user wants the 6-slot in-vocab approach deprecated "if things work fine" on the
persistent range. First analysis claimed op-collapse chaining was a HARD blocker (the model emits `S0` as
a step-2 operand -- needs a logit column -- so the in-vocab pool survives as the model's "registers").
**The user challenged this and is RIGHT**: verified at op_curriculum.hpp:127, the graded `g(S0)` step-2
reference is a synthetic-template artifact -- the trace generator bakes "re-emit the raw slot token after
the separator" because the in-vocab pool made that trivially available, NOT because raw-id emission is
the right orchestration interface. The op executes in the scalar node OUTSIDE the model, so the reference
convention is a harness-side design choice: plain-vocab HANDLES (ordinal bytes '0'/'1' -- already real
byte tokens -- or an implicit "previous result" default for linear chains) dereferenced by the node work
just as well and need NO slot logit columns. Consequence: **full deprecation of the in-vocab pool is
genuinely reachable**, path = (1) compaction role migrates to the persistent range (pending shootout +
saturation results); (2) the current S0-emission op-collapse curriculum is SIDELINED pending
reimplementation with handle-based references + an A/B against the S0-emission form (not yet built; keep
running the existing form in production until the replacement proves out -- don't rip out a working
mechanism before its successor exists); (3) selection-style capabilities re-express over the same handles
(see [[persistent-slot-selection-problem-backlog]], where the handle idea is now the designated direction,
not speculation). Same day: ConvPool enc_w/enc_w_grad plumbing was added to PersistentBindings + the 3
backend_cpu.cpp dispatch sites (closing the "setting encoding=ConvPool on a persistent id is UB" hazard)
and ConvPool joined the shootout as a 4th arm with a single-threaded encoder-AdamW training loop.

**How to apply**: the substrate + a first (narrow, attend-only, unsaturated) curriculum result now both
exist. Remaining real next steps, in rough dependency order: (1) more training budget on the same
attend-only curriculum to see if it saturates closer to the bounded pool's own numbers; (2) ~~wiring
PersistentBindings into train_batch~~ DONE, see below; (3) the actual compound-word DB / encode-time
substitution / live decode-time collapse work this file's "What's still NOT built" list above already
describes -- still entirely unstarted; (4) the selection-task research question, tracked separately in
[[persistent-slot-selection-problem-backlog]].

## UPDATE (2026-07-18): train_batch DOES accept PersistentBindings now, spike converted to the MT path

Item (2) above landed on 2026-07-17 (commit a80156e, alongside the SentinelBindings work) --
`train_batch` gained a `win_persist` per-window array, the same shape `win_binds`/`win_sentinel` already
had, but this memory (and the test file's own header comment, and scratch_slots.hpp's own doc comment on
`PersistentBindings`) never got updated to say so -- three separate stale "doesn't accept PersistentBindings"
claims found and fixed together, see [[sub0-dev-seam-consumer-migration]]'s sibling commit
5c0a146 for the actual conversion. `persistent_scratchspike_engine_tests.cpp`'s `train_steps` now drives
the parameter-free arms (MeanPool/Hash/HRR) through `win_persist` (all cores); ConvPool (learned enc_w)
keeps the manual single-threaded loop, matching the documented per-thread-reduction limit. `eval_multi`
also gained `#pragma omp parallel for` scoring (mirroring pairspike_engine_tests.cpp's own eval_select) --
this turned out to be the LARGER of the two wins: the live greedy-decode eval tasks (280+120 per round x
10 rounds x up to 3 seeds) dominate wall-clock more than the kBatch=16 training windows do. Measured: full
3-seed K=30 MeanPool arm 988s (previously didn't finish even one seed within a 10-minute budget under the
old train-serial/eval-serial path). Correctness unaffected -- task-generation rng order is untouched, only
scoring parallelizes, and K=6 MeanPool/Hash/ConvPool sanity arms all produce the same shape of sensible,
non-degenerate learning curves as before.
