---
name: factspike-piece-transfer-validated
description: factspike (d96 toy model) validates that piece-embedding transfer into a scratch slot preserves a learned factual association above chance
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-21T21:30:48.422Z
---

**Result: DONE, Phase 0/A/B/C all run on real hardware (2026-07-19).** Piece-embedding transfer — binding
a subject's OWN already-trained vocab-piece id(s) directly to a scratch slot (no byte decomposition) —
lets the model retrieve a taught fact through the slot alone, with zero textual restatement anywhere in
the prompt. Peak accuracy 0.56 (5/9) vs. 0.125 chance (4.4x), and this landed repeatedly across many
separate training rounds (not one lucky round), which is what makes it a real signal. Held-out negative
control's own peak (0.33, n=3) is statistically consistent with pure small-n sampling noise (P(>=1/3
correct by chance) ~= 33%), not leakage — Phase 0's leakage check only covered drilled-subject-vs-color
piece correlations, so this is a soft conclusion, not a proven one. Full trajectories and design record in
docs/FACTSPIKE.md.

**Mechanism required NO engine changes**: `ScratchTable::bind()` (scratch.hpp) stores ids verbatim;
`encode_slot()` (scratch_slots.hpp) never cared whether bound ids were byte codes or real vocab-piece ids
— every prior curriculum (scratchspike/repeatspike/wordspike/op_curriculum) just always happened to pass
byte-decomposed fragments via `ScratchTable::expand()`.

**Why**: initial plan draft mistakenly proposed testing byte/character-composed embeddings ("can a
representation rebuilt from individual character embeddings coincidentally hold the same meaning"). User
corrected this directly: the mechanism should transfer the word's OWN already-trained piece embedding(s)
("the weights that would be in the context"), not rebuild meaning from characters. Relevant to
[[deterministic-mechanisms-thesis]] and [[persistent-slot-selection-problem-backlog]].

**How to apply**: this is real, positive evidence that a scratch slot composed from a word's own trained
embedding(s) preserves enough of a learned association for retrieval — relevant to
[[scratch-tokens-context-translation-layer]] design work and any future persistent-slot/compound-word
work. Two reusable techniques from this experiment:

1. **Peak, not final-round, accuracy for judging volatile toy-model evals.** Training at this small scale
   (d96, tiny corpus) is highly volatile round-to-round — a first factspike run's baseline-arm trajectory
   went 0.00,0.00,0.00,0.11,0.11,0.11,0.00,0.00,0.44,0.78,0.00,... — checking only the final round would
   have reported a false negative on a real, demonstrated capability. Track peak across all rounds, still
   report the full trajectory for human judgment.
2. **Decouple vocab-learning repetition from training-data repetition** when an experiment needs a vocab
   deliberately forced multi-piece. `min_merge=2` (sub0-configure default) lets ANY word occurring >=2
   times earn its own whole-word merge — so running vocab-learning against the same high-repetition corpus
   needed for fact-teaching training defeats "forced multi-piece" (every subject got its own merge).
   Fixed by running sub0-configure's vocab-learning step against a SEPARATE, low-repetition corpus (each
   subject attested exactly once), then building the real high-repetition training data separately
   in-memory (bypassing corpus.tok entirely — matching every other spike curriculum's own pattern).

Also surfaced (and fixed) a real, separate bug during this work: `train_batch`'s `lengths` parameter is
NOT optional when documents are shorter than the sampled window — passing `nullptr` caused an
out-of-bounds read when a short document was also the last one in the flat token array. Fixed in
factspike_engine_tests.cpp's own `train_steps`; the SAME bug pattern still exists, unfixed, in
wordspike_engine_tests.cpp's own `train_steps` (never triggered there by luck) — flagged, not yet fixed.

**Follow-on (2026-07-19, same day): "Pack-Aware Training" (PAT) idea, QAT-grounded.** User reframed
"scratch" slots as semantically "packing" (many-to-one lossy composition of trained embeddings, like
MeanPool/HRR) and asked whether QAT (quantization-aware training) ideas could close the scratch-vs-baseline
gap. Researched mid-2026 QAT literature (StableQAT's Fourier surrogate gradient, Bit-by-Bit progressive
QAT, a sub-100M-scale schedule study finding bit-width doesn't need a special LR schedule). Key finding:
our packing operator (MeanPool/HRR) is ALREADY exactly differentiable — `encode_slot_bwd` is a real
analytic adjoint, not an STE approximation — so QAT's hard part (routing gradient through a non-
differentiable `round()`) doesn't apply here; we get it "for free." The actual gap: QAT computes the REAL
task loss through the quantized forward pass; our existing slot-exposure training (Phase C) only ever
grades GENERIC FILLER text after the packed slot ("plays outside every day", unrelated to the fact) for a
DISJOINT exposure-subject pool — weak, task-decoupled gradient, never reaching the drilled subjects the
eval actually queries.

Added `hrr_unbind()` (scratch_slots.hpp) — literally `encode_slot_bwd`'s HRR circular-correlation math,
applied to the packed VALUE instead of a gradient (this is HRR's canonical unbind op) — as a diagnostic:
does per-round reconstruction fidelity (cosine sim to the true piece row) correlate with scratch-arm
accuracy? Real result: NO positive correlation (Pearson r=-0.23 for both drilled and held-out). More
informative than the r itself: drilled-subject fidelity is nearly STATIC across 20 rounds (0.378-0.387)
while accuracy swings 0.00-0.56 in the same window — consistent with "nothing currently trains packed-
vector fidelity for drilled subjects at all," matching the task-decoupled-gradient diagnosis above.

**PAT (Phases D/E/F) run and PARKED (2026-07-20/21)**: built the task-contingent exposure-document fix
(graded text after the slot = the fact itself, not filler), then chased a dilution confound across three
matched-budget A/Bs (Phase D: flat mix, collapsed everything, including baseline — a real tell that
something other than "the harder task hurt scratch" was going on; Phase E: warm-start+ramp partially
recovered it, isolating pool DILUTION — widening the plain-text pool for exposure subjects under a fixed
budget — as the dominant cause, cold-mixing as secondary; Phase F: fixed dilution properly via a 3-way
weighted sampler, scratch kept improving monotonically (0.00→0.11→0.22) but baseline hit its WORST point
across all four regimes, backwards from what dilution predicts). Conclusion: likely hitting single-seed
volatility limits at this model scale — this project has hit this exact wall before (HRR-vs-MeanPool: "a
single run looked strong... a 3-seed follow-up tempered that", see [[meanpool-alternatives-prior-art-and-math]]).
User explicitly parked this axis: it doesn't move the mechanism itself, only how hard training pushes the
model to use it, and any fix would need to generalize to arbitrary words sharing these piece tokens
elsewhere in the vocab, not just this experiment's own subjects.

**Attention-capacity axis (2026-07-21): the strongest signal in the whole investigation.** User's own
hypothesis: a genuine n-piece word gets ~N_LAYERS*(n-1) sibling-attention hops (later positions re-attend
every earlier piece's already-once-refined state each layer, compounding); a packed slot has NONE of this
structurally (one position, no same-word siblings, composition frozen into a fixed formula before layer 1
runs). Prediction: packed-vs-normal representational gap should SCALE WITH piece count n. Tested by
extending the hidden-state diagnostic (`sub0::last_hidden_ptr()`, compares the model's actual pre-readout
representation, not raw embeddings) to record exact n per subject: **Pearson r(n_pieces, cos_sim) = -0.614**
— moderate-to-strong, exact predicted direction, far cleaner than similarity-vs-correctness (r=0.24 from
the prior diagnostic pass). Borderline significance at n=9 (p≈0.08) but the first result here with both a
precise mechanistic story AND a clean quantitative match. Motivated a full requirements/criteria writeup,
[[scratch-tokens-context-translation-layer]] (docs/SCRATCH_TOKEN_FRAMING.md).

**Phase G (2026-07-21): KV-trace memoization spike — candidate 1 falsified as stated.** User green-lit
building the framing doc's "most promising mechanism" (mathematically exact alternative to embed-level
composition: capture a word's REAL per-layer (K,V) trace from an isolated forward pass instead of
approximating from raw embeddings) and asked to "identify DRY collapse/abstraction/extension points"
before implementing. Built 3 minimal new engine primitives (`kv_krow_ptr`/`kv_vrow_ptr`, `kv_rope_rotate`
— a thin export of the existing `rope_row`, confirmed algebraically as its own exact inverse under
negation — and `kv_splice_row`, `core.hpp`/`backend_cpu.cpp`), deliberately NOT wired into any bindings
struct or `forward_one` dispatch branch (spike-scoped, orchestrated directly by the test harness) per this
project's "don't build production abstractions before validated" discipline. Pooling reused
`encode_slot`'s own HRR math unmodified (captured per-layer K/V rows treated as a synthetic `[n,D_MODEL]`
table) — zero new pooling math needed, one clean DRY win.

Real result (same run/weights as mechanism A, matched comparison): mean cos_sim A=0.83 vs B=0.77 (WORSE,
not better), r(piece_count,cos_sim) A=-0.614 vs B=-0.624 (slightly stronger, not weaker as candidate 1
predicted), accuracy tied 3/9 on different subjects. A single n=8 outlier (Ceaiaenze) explains most of the
mean-gap (nearly vanishes excluding it: 0.859 vs 0.850) but NOT the correlation gap, which survives
outlier exclusion. **Candidate 1 as specified — swap the pooling INPUT (raw embeddings -> real per-layer
K/V), keep HRR bundling — does not close the gap.** Mechanistic reframe: this points at HRR's own
fixed-capacity n->1 bundle as the bottleneck, independent of what's bundled — a direct, cheap, not-yet-run
analytical link to the framing doc's open "HRR crosstalk theory" question (Plate's 1/sqrt(n) result)
against this same 9-point dataset, no new training needed. Full numbers: docs/FACTSPIKE.md Phase G entry,
tests/factspike_engine_tests.cpp `[.factspikekvtrace]`. Next steps NOT yet decided/authorized: the
HRR-crosstalk curve-fit, a different pooling operator, or candidate 2 (no pooling, O(n) cost accepted).

**Candidate 2 spiked same day, user: "spike candidate 2" — and a decisive SPELL-marker finding fell out
of it.** No pooling at all: splice the FULL n-position per-layer trace (n real KV-cache rows per layer,
O(n) cost accepted) instead of compressing to 1. Reused ALL THREE candidate-1 primitives with ZERO new
engine code — splice writer called n times per layer instead of once, exactly the extension point flagged
when those primitives were first built. Trivial case (isolated capture, position-0 splice — should be
mathematically exact) first landed at mean cos_sim 0.947, not the predicted ~1.0. Root-caused, not left
mysterious: (1) a suffix-tokenization boundary confound (`" loves the color "` tokenizes differently alone
vs. as the tail of the full sentence — fixed by slicing the real suffix out of the combined tokenization
instead of re-tokenizing it; pinned by a new always-on regression test, `[factspike][kvtrace]`); (2) after
that fix, the three single-piece subjects hit EXACT 1.000 (proves the splice/rotation math itself is
correct) while every multi-piece subject stayed at 0.87-0.96.

**User flagged the SPELL_START/SPELL_END hypothesis directly** ("the spell start/end could be carrying
some weight — this could be a finding to apply retrospectively to all prior tests if it's founded").
Tested it as a controlled swap, not a correlation: `subject_piece_ids` (the basis EVERY mechanism in this
whole investigation — A, candidate 1, candidate 2 as first specified — captures/composes from) strips the
SPELL_START/SPELL_END wrapper tokens a real forward pass over a multi-piece subject actually processes.
Replaying the marker-INCLUSIVE span instead closed the ENTIRE remaining gap to exact 1.000 for every
single drilled subject, multi-piece included, zero exceptions. **Founded, not speculative**: every packing
mechanism built in this investigation has been operating on an incomplete basis by construction.
Context-sensitivity probe (D) still confounded by the same marker omission (0.948, indistinguishable from
C's own 0.947) — needs its own marker-inclusive re-run before it answers the "how context-sensitive is a
word's own trace" question it was built for.

**Phase H, same day: the re-test, run — result is WORSE, not better, and it completes the picture rather
than contradicting it.** User: "Continue, also consider if there are earlier scratch token investigations
that could be valid to recompute, iterate." Built a genuinely SEPARATE matched-budget training run
(`build_slot_exposure_dataset_markers`, factspike.hpp) — the model must be TAUGHT to read a
marker-inclusive-bound slot, not just evaluated with one. Result: mean cos_sim A_markers=0.695 (vs. 0.83
piece-only), candidate-1 B_markers=0.630 (vs. 0.77), 0/9 accuracy this snapshot — worse across the board.
**Why this doesn't contradict candidate 2**: candidate 2 (full splice, no compression) has no capacity
limit, so more real content only helps (exact 1.000). Mechanism A and candidate 1 still have to COMPRESS
everything into one fixed-D_MODEL vector via HRR bind — giving that already-bottlenecked compressor
(candidate 1's OWN finding above: richer input didn't help there either) two MORE genuine items to
squeeze in makes its job harder, not easier. **Three independently-designed experiments now converge**:
candidate 1 (richer compression input — no improvement), Phase H (more real signal into the same
compression — measurably worse), candidate 2 (no compression at all — exact reconstruction). The
bottleneck was never which tokens get composed — it's the fixed-size compression STEP itself. Single-seed
caveat as always; the DIRECTIONAL agreement across three separate designs is the real signal, not any one
number.

**Codebase-wide survey (background agent, same day) + production-scale cross-check.** Surveyed every
`word_span`/`ScratchTable::expand` call site for exposure to the same gap. **`wordspike.hpp`'s own spike
training data is NOT exposed** — builds bindings from raw ASCII bytes of the subject string, bypassing
tokenization (and SPELL markers) entirely; the spike tested a narrower question than the mechanism it
validated for. **`corpus_collapse.hpp` (production, "COMMITTED") and its live-generation counterpart in
`decode.hpp` ARE exposed** — direct callers of the same `word_span` extraction. Added
`corpus_collapse::build_dataset_markers` and re-ran the existing `[.corpus_collapse]` capstone as a 3-arm
A/B/C at REAL TinyStories scale (out/build/d196check, d196/11 layers): held-out NELBO identical to 4
decimal places between marker-stripped and marker-inclusive arms (-0.0075 both vs. base-only). A real,
honest negative — not a contradiction of the factspike finding, but a coarser instrument (aggregate NELBO
over a whole val set, only 401/1693 docs even had a collapse-eligible word) failing to detect an effect
that factspike's surgical single-position probe was built specifically to catch. Full numbers + per-
subject breakdown: docs/FACTSPIKE.md (Phase G + Phase H + production cross-check entries),
docs/SCRATCH_TOKEN_FRAMING.md (axis 9 fully updated), docs/CORPUS_COLLAPSE.md (new dated entry).

**Context-sensitivity RESOLVED, same day, user: "Continue" (picking up the prefill-reuse question from
the ELI5 exchange).** D's own context-sensitivity number (0.948) carried an UNFIXED second confound (same
class as C's own suffix-tokenization bug). Building the marker-inclusive re-test surfaced a THIRD,
distinct instance: the fixed prefix "Everyone knows that " has its own trailing-space token absorbed/
merged away when immediately followed by a real subject, so `prefix_ctx.size()` (tokenized alone) is NOT
a reliable offset into the combined tokenization -- caught because a first fix attempt bailed on 0/9
subjects (correctly read as "too clean to be real per-subject noise," i.e. a bug signal, not a genuine
tokenizer property). Fixed by trusting only the ONE property verified twice (a subject's own
tokenize-alone form appears as an exact contiguous match in any longer string it's part of) and deriving
prefix/suffix by SEARCHING for that match rather than assuming any piece's separate tokenization length
lines up.

Real result (`context_sensitivity_cosine2`, spliced into the real different prefix): **F (pieces only)
mean 0.919, G (marker-inclusive) mean 0.983**. Single-piece subjects (no markers to omit) sit at
F≈G≈0.995-0.997, matching their own same-context C≈E≈1.000. Every multi-piece subject shows F clearly
below its own same-context C (real, non-negligible context-sensitivity cost on top of marker-omission),
but G recovers nearly all of it (0.95-0.997 per subject). **Markers matter MORE than context-sensitivity
does**: the E→G same-context-vs-real-context gap (~0.017) is far smaller than the F→G gap from including
markers (~0.064) -- most of what looked like context-sensitivity cost in the original confounded D was
still the marker-omission problem bleeding through. **A trace captured once in isolation, WITH markers,
reproduces ~98% of the true recomputed representation when reused in a different real sentence** -- a
genuinely positive, concrete answer to whether one precomputed trace generalizes well enough to reuse
across many real mentions (the load-bearing assumption for prefill-compute-amortization), not just a
plausible-sounding idea anymore. Same n=9/one-prefix/one-toy-model caveats as everywhere else here.

**Phase I (2026-07-21), user: "Continue with the next spike with new findings and direction" —
Nanbeige-inspired periodic re-injection, the FIRST real O(1)-cost positive in the whole investigation.**
Motivated directly by [[nanbeige-architecture-reference]]'s `NanbeigeNgramLayerFusion`: instead of trying
to make ONE upfront injection survive N_LAYERS of computation (what mechanism A/candidate 1 both do,
and what Phase G/H diagnosed as the real weakness), RE-INJECT the same packed vector as a residual
addition at multiple depths. Built `set_scratch_reinject(stride, scale)` — zero new learned parameters,
EVAL-ONLY (wired into `forward_one`, NOT the training graph — `Model::forward()`/`train_batch` is a
structurally separate autodiff implementation) — on the ALREADY-trained Phase-C model, no retraining.

First version (fixed embedding-scale addition): near no-op (0.831→0.831 mean cos_sim even at full
strength every layer). Correctly diagnosed as an artifact, not accepted as a real negative: the residual
stream's own norm grows across depth (why `ln_f` exists before the head at all), dwarfing a small fixed
addition by mid-stack. Corrected to SCALE-ADAPTIVE (fraction of `h`'s own CURRENT norm at each layer,
still zero learned params) — real, clean, monotonic dose-response: mean cos_sim 0.831→0.833→0.840→**0.860**
(5%/15%/40% of h's norm), and **task accuracy jumped from 3/9 to 5/9 correct** at the highest dose, on a
model NEVER trained to expect this signal. Ceaiaenze (n=8, the worst-performing subject throughout this
whole investigation) flipped from wrong to correct. That a genuine improvement shows up even without
training exposure is the encouraging part — it suggests the model's existing weights already have latent
capacity to use a stronger signal when presented, even via a crude, ungated heuristic. **This is the
first mechanism in the entire axis-9 thread to show a real positive while keeping O(1) visible cost** —
unlike candidate 2 (exact but O(n)) and unlike every tried compression-operator variant (candidate 1,
marker-inclusive mechanism A, both WORSE). Well-motivated, NOT YET STARTED next step: wire a proper
LEARNED gate (matching Nanbeige's own design) into the training graph so the model can learn when/how much
to lean on the reinforced signal, rather than a flat heuristic. Caveats: single seed, n=9, only 3 doses
swept (no sense of a ceiling/breaking point yet). Full numbers: docs/FACTSPIKE.md Phase I entry,
docs/SCRATCH_TOKEN_FRAMING.md's new "candidate 3."
