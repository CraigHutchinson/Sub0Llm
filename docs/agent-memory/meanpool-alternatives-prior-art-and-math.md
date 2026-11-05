---
name: meanpool-alternatives-prior-art-and-math
description: "Alternatives to MeanPool/CharEncoder for scratch-slot content embeddings: Hash/RoPE (modest), ConvPool char-CNN (stronger), and HRR circular-convolution binding (DECISIVE, mean 0.744 vs chance 0.333) all spike-verified; HRR is now WIRED INTO PRODUCTION as the default (RunConfig::content_embed_kind, resume-safe, verified via real train+gen smoke run)"
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

BACKLOG (added 2026-07-16 at explicit user request: "add a task to look at meanpool alternatives -- check
prior art and also from the mathematical standpoint for algorithms that could fit the required goals").
Context: [[scratch-content-embedding-meanpool-vs-charencoder]] (the corrected finding this extends),
[[deterministic-mechanisms-thesis]], `include/sub0/scratch_slots.hpp` (`encode_slot`/`encode_slot_bwd`).

**Motivation reframed (user correction, 2026-07-16): axis (b) below is NOT "otherwise unanswerable."**
`combine`/`uncombine` already give the model a RESOLVE-then-attend route to ANY sub-character/positional
query (starts-with, ends-with, Nth-char): emit `UNCOMBINE` on a slot, the interceptor injects its real
fragment bytes into the stream in order, then ordinary attention over those now-literal tokens can answer
positional questions trivially -- this is EXACTLY the mechanism the production curriculum's #4b
"contains X via LOCAL-GROUNDED CoT" already validates (resolve each slot, then reason). So a one-hop
content-derived embedding (MeanPool/CharEncoder/Hash) is NOT the only path to positional reasoning, and
framing it that way overstates the gap. What a working positional encoding would actually buy, if it beats
chance on `content_select_task`, is EFFICIENCY -- answering in one attention hop over the slot token
instead of spending a multi-step resolve trace (tokens + KV-cache) per query. That's a real but narrower
motivation than "the only way to solve this" -- keep this framing when writing up any finding below.

## The problem, precisely

`encode_slot(tok_emb, C, frags, enc, out)` must map a variable-length sequence of fragment token embeddings
`e_1..e_n ∈ R^C` (rows already trained via the LM objective) to ONE fixed vector `s ∈ R^C`, such that:

  a. **Presence/set discrimination** — "does this OOV contain character X anywhere" — CharEncoder already
     gives a modest, real lift here (0.325 -> 0.392 vs ~0.333 chance, d128) over plain MeanPool.
  b. **Positional/order discrimination** — "is X the FIRST/Nth character" — UNADDRESSED by anything built
     today. MeanPool and CharEncoder are BOTH provably permutation-invariant (Deep Sets canonical form, see
     below), so this is a hard representational ceiling, not a training-scale problem.
  c. Bounded output magnitude regardless of fragment count (fragments run ~1-6 today: OOV bytes or digit
     strings) — must stay compatible with the model's token-embed scale (see [[conditioning-embedding-amplitude]]).
  d. Cheap: CPU, no heap alloc in the hot path, ideally O(C) or O(C²) at these tiny C/n, not O(C³) or worse.
  e. **Compositional generalization to UNSEEN fragment combinations** — this is the hard constraint that
     rules out anything that could memorize whole sequences (a big RNN/lookup-table) instead of composing
     from per-fragment pieces; it's the exact property scratchspike's held-out-OOV probes exist to test.
  f. Differentiable (a working analytic backward, gradient-checked against finite differences, matching
     this codebase's standing verification bar — see [[verify-correctness-against-reference-before-perf]]).

These are TWO separate axes (a) and (b), not one problem — see "Recommendation" below.

## Prior art (verified via web search 2026-07-16, not just recalled)

- **Deep Sets** (Zaheer et al. 2017, NeurIPS, arXiv:1703.06114) — proves `f(S) = ρ(Σ_{s∈S} φ(s))` is the
  necessary+sufficient canonical form for any permutation-INVARIANT set function. CharEncoder (`φ = relu(W·)`,
  `ρ = identity`, sum-pool) and MeanPool (`φ = identity`, mean-pool) are BOTH literally this form — this is
  the rigorous grounding for why neither can be positional, not just an empirical observation.
- **Holographic Reduced Representations** (Plate 1995, IEEE Trans. Neural Networks; see also the Redwood
  Center's hosted PDF) — circular convolution as a `bind(role, filler)` operator, sum as `bundle` (superposition);
  compresses an arbitrary-length structure into ONE fixed-width vector with an approximate inverse
  (circular correlation) to later ask "what filled role p". Parameter-free, well-studied, from the Vector
  Symbolic Architecture (VSA) / hyperdimensional computing literature.
- **Fourier HRR / FHRR** (Plate; formalized further by Frady et al.) — the complex-domain form of HRR:
  each channel is `e^{jθ}` (unit-magnitude, phase carries the information); binding = elementwise complex
  multiply = PHASE ADDITION; bundling = complex superposition. This is mathematically the SAME operation as
  a rotary positional encoding: bind(position, fragment) via a position-dependent phase rotation, then sum
  across positions. **This codebase already has RoPE (`op_rope`) implemented, tested, and proven for
  token-position encoding in attention** — the same rotation machinery could plausibly be reused to bind a
  FRAGMENT's position within its slot's own short sequence, before pooling, which is the cheapest,
  most in-house-consistent way to break the permutation symmetry (see Candidate 1 below).
- **Character-Aware Neural Language Models** (Kim, Jernite, Sontag, Rush 2016, AAAI, arXiv:1508.06615) — a
  CNN (+ highway network) over character embeddings composes a word representation; on-par or better than
  word/morpheme-level LSTM baselines with 60% fewer params; shown to encode BOTH semantic and orthographic
  (i.e. positional/spelling) information from characters alone. This is the closest literal prior art to
  "compose a scratch slot's byte fragments into one vector" — convolution's local receptive field is
  intrinsically position-sensitive (unlike sum/mean pooling).
- **Set Transformer** (Lee et al. 2019, ICML) — small self-attention block with a learned pooling query;
  CAN be made order-sensitive by adding positional encodings to the keys/values before attention. Most
  expressive option surveyed, also the most expensive (attention params + compute) — likely overkill given
  fragment sequences here are tiny (3-6 elements).

## Mathematical candidates, ranked by fit for THIS codebase

1. **Positional-tagged pooling (RoPE-style rotation before sum).** Apply the SAME rotation `op_rope` already
   applies to token positions in attention, but to each fragment's embedding by its POSITION WITHIN THE
   SLOT (0, 1, 2, ...) before summing (equivalently: FHRR phase-binding + bundling, see above). Zero new
   learned params if the rotation is fixed (matching RoPE's own non-learned convention here), reuses
   already-tested/verified in-house math, directly targets axis (b) while (a) stays intact (magnitude
   after rotation-then-sum still reflects which characters are present). **Try this first** — cheapest,
   most consistent with [[compile-time-decisions-preferred]] and [[keep-simd-kernel-primitives]].
2. **Holographic/VSA circular-convolution binding.** Mathematically the most elegant, parameter-free,
   well-studied approximate-invert properties (could later support "what's at position p" queries beyond
   just presence). Higher first-implementation risk/novelty for this codebase (an FFT or O(C²) direct
   circular-conv step has no existing analog here, unlike RoPE).
3. **1D conv + maxpool over fragment embeddings (Kim et al. char-CNN style).** Well-proven specifically for
   THIS exact problem shape (characters -> word vector) in real language-modeling literature. Modest new
   params (a few small conv kernels), naturally order-sensitive (local receptive fields), bounded via
   maxpool. A good SECOND candidate if rotation-based binding underperforms.
4. **DeepSets-generalized pooling (add a post-pool MLP: `ρ` is currently identity for both MeanPool and
   CharEncoder — make it a learned MLP).** NOT a fix for axis (b) — still permutation-invariant by the same
   theorem — but the cheapest INCREMENTAL improvement to axis (a) specifically (better presence/contains
   discrimination), worth a fast A/B before investing in a structurally different mechanism.
5. **Small attention-pooling block (Set-Transformer-style, positions encoded on K/V).** Most expressive
   ceiling, most expensive; likely disproportionate to fragment sequences this short (3-6 elements) and
   against this project's stated preference for cheap/deterministic mechanisms over general learned
   capacity (see [[deterministic-mechanisms-thesis]]) — keep as a last resort, not a first try.

Note: `scratch_slots.hpp`'s `SlotEncoding` enum already reserves a `Hash` arm ("order-aware pooling without
new params... falls through to MeanPool until implemented") — Candidate 1 (rotation-based binding) is a
strong, well-grounded way to REALIZE that already-reserved slot rather than inventing a new enum value.

## Recommendation

Treat (a) presence/contains and (b) positional/order as separate spikes, not one task:
- (a): try Candidate 4 (post-pool MLP on top of the existing CharEncoder) first — smallest possible change.
- (b): try Candidate 1 (RoPE/FHRR-style positional binding before pooling) first — realizes the reserved
  `Hash` arm, zero new params, reuses proven in-house rotation math. Candidates 2 and 3 are the next two
  things to try if Candidate 1 underperforms on a held-out positional probe (`content_select_task` / #4 in
  `scratchspike.hpp`, currently excluded from the production curriculum precisely because nothing solves it).
- Verify the SAME way this project always has: an A/B against `content_select_task`'s held-out probe (the
  #4 "starts with X" task, currently EXCLUDED from `build_dataset_scratch` because nothing beats chance at
  it) at spike scale first (matching `b7896ab`'s own methodology), gradient-check against finite differences
  before any perf work, and multi-scale test per [[engine-tests-gradient-check-dims-dependent]]'s standing
  caution that engine bugs can be dims-dependent.

## RESULT (2026-07-16): Candidate 1 (Hash/RoPE positional binding) implemented + spike-verified, positive
## but NOISY -- multi-seed check tempers the first single-run number, direction still holds

`SlotEncoding::Hash` is now real: `encode_slot`/`encode_slot_bwd` rotate each fragment's row by a
position-dependent angle (interleaved-pair convention, same as `op_rope`) before summing -- position 0 is
the identity rotation, so it passes through unrotated. No learned params. Gradient-checked against finite
differences AND confirmed genuinely order-sensitive by direct test (reordering the SAME fragments changes
the output; MeanPool provably does not) in `tests/scratch_embed_tests.cpp`.

**A/B spike** (`tests/scratchspike_engine_tests.cpp`, "CONTENT (starts-with) reasoning -- Hash/RoPE
positional-binding A/B", d196, K=3, 3000 steps) on `content_select_task` (#4, chance ~0.333):

  - First single run: MeanPool held-out 0.375, Hash held-out 0.450 -- a promising-looking 0.075 gap.
  - **Multi-seed follow-up (3 training-order seeds -- `build_model()`'s weight init is deterministic/
    unparameterized here, so only training ORDER varies across "seeds")**: seed 1 MeanPool 0.375 / Hash
    0.450 (+0.075, matches the first run); seed 2 MeanPool 0.325 / Hash 0.325 (+0.000 -- BOTH arms
    collapsed to the same degenerate fixed-slot output, the exact failure mode `b7896ab`'s own CONTAINS
    A/B first noted); seed 3 MeanPool 0.367 / Hash 0.433 (+0.066). **Mean held-out: MeanPool 0.356, Hash
    0.403 (gap +0.047)**. A paired t-test on these 3 differences gives t≈2.0 on 2 degrees of freedom --
    directionally consistent (Hash never underperforms MeanPool, wins clearly in 2/3 seeds) but NOT
    statistically decisive at n=3. The original single-run gap was on the favorable end of the spread, not
    fully representative -- this is exactly the kind of thing multi-seed checking exists to catch.

**Honest read**: the theoretical claim (MeanPool structurally cannot carry position, Deep Sets canonical
form; Hash can, rotation is order-sensitive) is unchanged and still well-grounded -- but the EMPIRICAL
margin at this tiny spike scale (d196, K=3, 3000 steps) is real-but-modest and noisy enough that one
seed's degenerate-collapse failure mode can erase it entirely. This is "promising direction confirmed
across seeds, not yet a strong/reliable effect" -- weaker confidence than the first run alone suggested,
matching (if anything slightly below) `b7896ab`'s own CharEncoder CONTAINS finding in modesty. Do not cite
the single-run 0.450 number without this context.

**Remember the reframed motivation** (see above): this is not "the only way" to answer positional queries
-- combine/uncombine's resolve-then-attend route already handles that, at the cost of a multi-step trace.
Hash's payoff, if it holds up at scale, is answering in ONE hop instead of a multi-step resolve.

**Not yet done** (as of the Hash result alone; see Candidate 3's RESULT below for what changed next):
(1) more seeds and/or more steps to sharpen the statistics (n=3 is still thin; the degenerate-collapse
failure mode itself may be worth understanding/mitigating before adding more seeds blindly); (2) wire
`SlotEncoding::Hash` as a selectable option in `train_stage.cpp`'s content-embed path (currently hardcodes
MeanPool, same as CharEncoder's own unfinished production wiring -- see
[[scratch-content-embedding-meanpool-vs-charencoder]]); (3) re-add `content_select_task` (#4) to
`build_dataset_scratch`'s production mix is premature until an effect is more reliable.

## RESULT (2026-07-16, same day): Candidate 3 (ConvPool, char-CNN style) implemented + spike-verified --
## STRONGER and more consistent than Hash, now the leading candidate

`SlotEncoding::ConvPool` is real: a learned width-2 convolution over consecutive fragment pairs (TWO
`[C,C]` projections, packed into `enc_w` as `[2,C,C]` -- reuses CharEncoder's own `enc_w`/`enc_w_grad`
pointers, no new `ScratchBindings` fields), relu, then MAX-POOLED across window positions (a lone fragment
with no width-2 window falls through as its raw row). Gradient-checked against finite differences AND
confirmed order-sensitive by direct test in `tests/scratch_embed_tests.cpp`.

**A/B spike** (`tests/scratchspike_engine_tests.cpp`, "CONTENT (starts-with) reasoning -- ConvPool (char-CNN)
A/B", d196, K=3, 3000 steps, multi-seeded from the start this time, baseline = fully PLAIN/no-content-embed
rather than MeanPool, matching CharEncoder's own A/B methodology): seed 1 plain 0.358 / ConvPool 0.433
(+0.075); seed 2 plain 0.325 / ConvPool 0.517 (+0.192); seed 3 plain 0.350 / ConvPool 0.425 (+0.075).
**Mean held-out: plain 0.344 (right at chance 0.333, as expected -- zero content signal), ConvPool 0.458
(gap +0.114)**. ConvPool wins on EVERY seed -- no ties, no degenerate collapse (unlike Hash's seed 2).
Paired t-test t≈2.9 on 2 df -- still n=3, still not conventionally "decisive," but a meaningfully stronger
and more consistent result than Hash's own (+0.047 mean gap, one seed tied at chance).

**Why ConvPool likely beats Hash** (plausible, not proven): ConvPool has LEARNED parameters (twice
CharEncoder's count) that can shape a task-adaptive "does this look like it starts with X" decision
boundary; Hash's rotation is entirely FIXED/deterministic, pushing all the interpretive work onto the
shared downstream model layers. This mirrors the ORIGINAL CharEncoder-vs-MeanPool finding on the presence
axis (learned beats parameter-free, given enough capacity/data) now showing up on the order axis too.
Maxpool's hard per-channel window selection may also help by avoiding the linear superposition/interference
between rotated fragments that Hash's plain sum is exposed to.

**Superseded same day by Candidate 2 (HRR) below -- ConvPool is no longer the leading candidate.** Kept
for the record; see the HRR RESULT section for the current ranking and the "why ConvPool beats Hash"
reasoning above turned out to UNDERSTATE the picture (learned params are not the deciding factor -- HRR
has none and wins by a much larger margin, see below).

## RESULT (2026-07-16, same day, later still): Candidate 2 (HRR, circular-convolution binding) implemented
## + spike-verified -- DECISIVELY stronger than both Hash and ConvPool, now the clear leading candidate

`SlotEncoding::HRR` (Holographic Reduced Representations, Plate 1995) is real: `bind(role_p, filler_p)` =
circular convolution of a FIXED pseudo-random unit-RMS "role" vector (keyed by fragment position, from a
lazily-built, program-lifetime-cached table, `hrr_role_table`) with the fragment's `tok_emb` row, then SUM
(bundle) across fragments. No learned params (like Hash). Gradient-checked against finite differences AND
confirmed order-sensitive by direct test, including an independent re-derivation of a single fragment's
encoding (not just self-consistency with `encode_slot`) in `tests/scratch_embed_tests.cpp`.

**A/B spike** (`tests/scratchspike_engine_tests.cpp`, "CONTENT (starts-with) reasoning -- HRR
(circular-convolution binding) A/B", d196, K=3, 3000 steps, multi-seeded from the start, baseline =
MeanPool matching Hash's own methodology): seed 1 MeanPool 0.375 / HRR 0.600 (+0.225); seed 2 MeanPool
0.325 / HRR 0.892 (+0.567); seed 3 MeanPool 0.367 / HRR 0.742 (+0.375). **Mean held-out: MeanPool 0.356,
HRR 0.744 (gap +0.388)** -- more than DOUBLE chance (0.333), consistent and strong across all 3 seeds (no
ties, no degenerate collapse), with held-out tracking drilled closely in every seed (0.600/0.582,
0.892/0.839, 0.742/0.764) -- genuine generalization to unseen OOVs, not memorization. This gap is roughly
3.4x ConvPool's (+0.114) and 8x Hash's (+0.047) -- not a marginal improvement, a qualitatively different
result.

**Why HRR likely wins by so much** (plausible, not proven): the classic VSA "quasi-orthogonality"
property -- independent random high-dimensional vectors have near-zero expected dot product (std ~
1/sqrt(C)) -- means HRR's per-position role vectors interfere with each other much LESS than Hash's
structured rotation (a low-rank, only-2-channels-at-a-time transform, and adjacent positions' rotation
angles are close together in the slower/higher-index channel pairs -- less separable) or ConvPool's
local width-2 window (which only ever looks at ADJACENT fragment pairs, missing longer-range positional
structure a query like "starts with X" needs to distinguish position 0 from position 4 or 5). HRR's
circular convolution is also a FULL C x C entanglement per binding (every output channel is a function of
every input channel via the role vector) -- a much richer, higher-rank transform than either Hash's
per-pair rotation or ConvPool's per-window-position local view. Learned params turned out NOT to be the
deciding factor after all (HRR has none, ConvPool has many, HRR wins by a wide margin) -- the STRUCTURAL
choice of binding operator matters far more than whether it's learned.

**DONE (2026-07-16, same day, later still): HRR is wired into production.** `train_stage.cpp`'s
content-embed path now defaults to `SlotEncoding::HRR` for a fresh run (no checkpoint-persistence
dependency, since it has no params); a new `RunConfig::content_embed_kind` field persists the actual
choice, with resume-reconciliation and a MeanPool struct-default so old content_embed=on models (trained
before this field existed) keep decoding with what they actually trained on, never silently drifting to
HRR. `gen_stage.cpp` reads the same persisted choice. Verified with a real train+gen smoke run (d196,
`--scratch-mix 0.3 --content-embed`, 40 steps): training log shows `content-derived scratch embeddings ON
(hrr, CPU-forced)`, `config.json` shows `content_embed_kind: "hrr"`, gen's log shows `content-embed hrr` --
train/gen loop closed correctly. See [[scratch-content-embedding-meanpool-vs-charencoder]] for the
detailed wiring description.

**Remaining next steps**: (1) more seeds/steps to sharpen statistics further, though a gap this large is
unlikely to be noise; (2) re-admit `content_select_task` (#4) into `build_dataset_scratch`'s production
curriculum mix now that HRR gives it a real answer -- NOT yet done, still requires deciding the mix
weight against the other two production capabilities (mirroring how `contains_k` was tuned in); (3) a
natural follow-up question: does HRR ALSO help the order-agnostic CONTAINS task (#4b) as well as or
better than CharEncoder, now that a much stronger order-sensitive mechanism exists? (4) Candidate 4
(post-pool MLP on CharEncoder) is now lower priority -- HRR's result suggests the STRUCTURAL binding
operator, not added learned capacity, is what actually matters for order-sensitivity.
