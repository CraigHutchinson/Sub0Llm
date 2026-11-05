---
name: scratch-tokens-context-translation-layer
description: "backlog idea — a reversible context-local \"translation state\" between raw input and the model lexicon; an OOV word CONSTRUCTED from char/fragment tokens (e.g. SPELL_START..SPELL_END) collapses to one context-specific reserved token for the rest of the context -- context-window + KV-cache efficiency, not just a cleaner model-facing stream. TANDEM with [[spelling-via-injected-char-word-training]] (the inverse, word->chars, direction)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-21T13:26:18.173Z
---

**Reframed 2026-07-10, confirmed to work IN TANDEM with [[spelling-via-injected-char-word-training]]**
(user's own words: "they both work in tandem") — the two directions of one char<->word capability:
- **This memory (scratch tokens)**: chars/fragments -> word. Once an OOV word has been
  CONSTRUCTED/expressed from its character or sub-token fragments (this project's existing
  SPELL_START..SPELL_END mechanism is exactly that construction), collapse it to a single
  context-local reserved token so it doesn't keep paying its full multi-token cost for the rest of
  the context.
- **The other memory (spelling spike)**: word-token -> chars, via a reasoning/thinking harness.
  This direction is plausibly a PREREQUISITE for the one below: minting a scratch token for a
  freshly-spelled word, and later correctly recognizing/expanding it, both need the model to
  already have a working char<->word mapping — not just a related idea, likely a dependency.

**Concrete efficiency payoff (user's own example, 2026-07-10, sharpens the original "cleaner
model-facing stream" framing below into a measurable resource cost):** a 6-character OOV word like
"aabbcc", spelled out via SPELL_START..SPELL_END, costs a **6-token window of context AND
KV-cache** every time it's referenced. Once the model has constructed/established that word once
within a context, collapsing subsequent references to ONE context-specific reserved token means:
(a) the context window holds 1 slot instead of 6 for every later mention, directly extending
effective usable context length, and (b) attention only needs to attend to ONE cached KV entry
instead of 6 for every later mention, an actual memory + compute saving in the KV-cache, not just a
context-length nicety. This is the load-bearing NEW justification for scratch tokens beyond "the
model reasons over a cleaner stream" (the original framing) — it is also a real efficiency
mechanism, closer in spirit to this project's own [[hardware-efficiency-iteration-standing-task]]
than the original write-up positioned it.

**Original framing, still the right mental model for the mechanism itself:** the encoder should NOT
transcode raw input directly into the model's internal lexicon. Instead a context-specific
**translation state** sits between them. Novel/rare/OOV words (e.g. a cat named
"Jingelbruitiupss") collapse into a reusable, context-local **scratch-token** (slot), so the model
reasons over a clean in-vocabulary stream ("the cat ENTITY_1 chased ENTITY_2") instead of a 12-char
SPELL shatter. Information that is technically noise/malformation is stripped into the side table
too. The table is stored but **not necessarily fed to the model**; at decode the translation state
restores the input **verbatim and losslessly**.

**Honest framing / design tension:** a pure opaque slot blinds the model — it then can't do
morphology ("Jingelbruitiupss's"), rhyme, or spell-back. So the scratch-token's EMBEDDING should be
CONTENT-DERIVED (a learned hash/char-encoder fingerprint of the surface), giving the model a
compressed semantic signal, while the EXACT bytes live in the side table for lossless decode.
Identity-consistency matters (same surface → same slot within a context). This is essentially a
copy/pointer mechanism + entity-slot memory + learned OOV hash-embedding (prior art:
pointer-generator copy, char-CNN/hash OOV embeddings, entity abstraction). The reserved marker
headroom Stage 2 just built (`TOK_RESERVED_*`, see [[tokenizer-redesign-implementation-complete]])
is NOT the right mechanism for this — those are fixed, global, compile-time marker ids; a scratch
token needs to be minted DYNAMICALLY per-context (bound to a specific surface form only for the
duration of one context), closer to a KV-cache slot allocation than a vocabulary entry.

**Relations:** reversible/context-local upgrade of the currently-LOSSY `normalize_text` (would make
curly-quote/whitespace folding restorable). See [[ch32-char-composition-oov-falsified]] and the
acronym note for the char-composition caveat that also applies here. Separates meaning from surface
like [[gist-is-compute-context-primitive]] and the [[verbosity-time-slider-from-gist]] realization
axis. A generalization of factored/separable tokens (tokenizer-normalization-and-factored-tokens,
NOTE: that memory turned out to be mis-filed under the diffusion project's session lineage on
review 2026-07-10 — re-verify it actually applies to Sub0Llm before leaning on it further).

**SPIKE STARTED 2026-07-11 (tandem dependency now CLEARED -- the spelling/combine-uncombine spike is
RESOLVED SUCCESS + PRODUCTIONIZED, see [[spellspike-poc-results]] + [[corpus-blending-and-spell-integration]]).**
User: "pick up and plan out the scratch token spike, follow the same work method as the combine/uncombine
spike before we merge to production" -- i.e. isolated CPU spike, engine-free curriculum + hidden engine
harness, held-out generalization = success measure, prove BEFORE touching production.

KEY DESIGN INSIGHT: scratch = combine/uncombine + a per-context BINDING TABLE, and it needs NO engine
or decode.hpp change to spike -- kv_decode_generate's expand/combine are std::function callbacks, so a
STATEFUL lambda (owning the table) supplies it. Interceptor logic: combine(OOV frags that are NOT a
vocab piece) -> assign next free reserved SCRATCH SLOT + record slot->frags; expand(scratch slot) ->
its bound frags. So an OOV costs 1 token per mention after being defined once. Spike question: can a
small model treat a FRESH, embedding-less scratch slot as a valid in-context symbol -- resolve a
reference to it via its earlier binding -- and GENERALIZE to HELD-OUT OOVs never bound in training
(the clean no-memorization probe)? This isolates whether dynamic in-context binding works BEFORE the
content-derived-embedding engine change (resolution here is purely by attention to the binding).

SCOPE NOTE: this first cut isolates RESOLUTION (the novel part). The DEFINE region is CONTEXT (masked);
the model is graded only on INVOKING uncombine to resolve a slot reference + the answer -- mechanically
close to the combine/uncombine spike, the new bit being the slot's meaning is purely contextual (no
trained embedding) + generalization to held-out bindings. Multi-binding disambiguation (bind 2-3 OOVs
to 2-3 slots in one context, query mixed -> a real in-context symbol TABLE) is the stretch after single
works. Scratch pool for the spike = TOK_RESERVED_4..7 (4 slots); production wants a bigger reserved
range (tokenizer-budget follow-up) -- NOT renamed in casing.hpp yet (spike-local, prove first).

RESULT 2026-07-11: SPIKE SUCCESS -- scratch resolution GENERALIZES. Built `scratchspike.hpp` (engine-free
curriculum) + `scratchspike_tests.cpp` (5 structure tests) + `scratchspike_engine_tests.cpp` (hidden
[.scratchspike] harness: stateful binding-table `ScratchOps` -- expand(scratch slot)->bound frags,
combine(OOV)->assign/return slot -- plugged into kv_decode_generate's std::function callbacks, so NO
engine/decode.hpp change). Harness pre-binds slot0 to each task's OOV (models a prior in-context define);
curriculum = `# n <slot>` / `~ <slot>` with the OOV bytes ONLY reachable by uncombining the slot (the
isolation the first-cut in-stream-define lacked). d128/vocab774 CPU, 3000 steps, masked.
- FIRST TRY (60-OOV pool, 42 drilled): MEMORIZED -- DRILLED nth 0.72 but HELD-OUT nth stuck at ~0.10-0.16
  floor, held-out round-trip ~0. Two variants (in-stream-define AND clean pre-bound) both failed the same
  way -- the classic unmasked-spellspike memorization signature.
- FIX = DRILL DIVERSITY (same lever spellspike needed): 400-OOV pool / 280 drilled -> HELD-OUT nth
  0.12->0.455 and STILL RISING at 3000 steps, held-out round-trip 0->0.358 rising. DRILLED nth 0.67.
  42 distinct entities was too few (memorizable); 280 forces the general "index the resolved bytes" op.
- CONCLUSION: a small model resolves a DYNAMICALLY-bound, EMBEDDING-LESS scratch slot for OOVs it never
  saw bound -- the context-translation RESOLUTION mechanism works via binding-table + combine/uncombine,
  no content-derived-embedding change needed. Mechanically it's spellspike with WORD->a constant reserved
  slot0 + interceptor-side binding table; the win is proving an embedding-less per-context slot is
  resolvable + the N-tokens-per-mention -> 1-token compression is real.
COMMITTED 5482158 (single-binding scaffold + result).

MULTI-BINDING RESULT 2026-07-11 (committed next): the "harder" disambiguation test turned out EASIER --
K=3 distinct OOVs pre-bound to slots 0..2, a symbol-table preamble [slot_0 slot_1 slot_2] then a query on
a RANDOM one -> HELD-OUT nth hits **1.000 by step 1800** (drilled 1.000 too), vs single-binding's 0.455.
So the model resolves the RIGHT slot among competitors PERFECTLY, generalizing to held-out OOVs. Caveat on
the single-vs-multi gap (partly confounded, don't over-read the delta): the multi dataset is PURE nth (2x
the nth examples) while single mixes nth+rt, AND the varying query-slot + preamble force a cleaner
"answer = injected_bytes[n]" indexing op than single-binding's CONSTANT slot0 (which can hide a degenerate
shortcut). The unambiguous takeaway: multi-binding resolution works + generalizes perfectly -> scratch
tokens are robust to multiple competing bindings, not degraded by them. (Why perfect: the interceptor
supplies the queried slot's bytes on uncombine; the model only has to emit uncombine on the queried slot
+ index -- a clean generalizing op. This is genuine resolution: no uncombine = no injected bytes = can't
answer.) `scratchspike.hpp` gained multi_nth_char_task/pick_multi_task/build_dataset_multi + Task.binds
(harness pre-binds ALL slots); second [.scratchspike] harness case.

CONFOUND RESOLVED 2026-07-11: ran the multi harness at K=1 (pure-nth, single slot) -> HELD-OUT nth also
**1.000** (by step ~2400). So the single-binding test's 0.455 was ENTIRELY the nth+rt task MIXTURE
(build_dataset splits capacity between nth and the harder round-trip), NOT a scratch limitation. Clean
picture: pure nth-resolution is PERFECT held-out at K=1 AND K=3; the round-trip leg (uncombine then
combine-back a NOVEL byte string -> slot) is the genuinely harder sub-task that only partially generalizes
(~0.36) -- combine-back of unseen bytes is harder than index-the-injected-bytes. Bottom line: scratch-slot
RESOLUTION is solidly proven (a small d128 model perfectly resolves dynamically-bound, embedding-less
slots for held-out OOVs, 1..K competing bindings); the open work is binding POLICY + reason-over-slots.

#1 + #2 RESULTS 2026-07-11 (both NEGATIVE at d128 -- copy is the bottleneck):
- #1 MODEL-DRIVEN BINDING (the model EMITS combine to define an OOV; interceptor's combine callback now
  assigns a free slot on an unseen OOV): DRILLED 0.98 but HELD-OUT ~0.15 (floor). The model memorizes
  drilled OOV copies but can't COPY a NOVEL OOV into the combine region to bind it. define_task/
  build_dataset_define + eval_define check the resulting binding == the OOV.
- #2 REASON-OVER-SLOTS (select): K=3 tagged slots [slot_i tag_i]*, query a tag -> output the matching
  slot. STUCK AT CHANCE (~0.33 = 1/K) drilled AND held-out -- never learns the tag->slot associative
  lookup. select_task/build_dataset_select + eval_select; answer IS the slot (no resolution needed).
- ROOT CAUSE (unifying): both require COPYING A TOKEN CONDITIONED ON CONTEXT (reproduce the OOV bytes /
  copy the tag-matched slot) -- an induction/copy op -- whereas the SUCCESSFUL tasks only needed
  index-the-injected-byte. So at d128/L4/H4 the model does RESOLUTION+INDEXING perfectly but COPY-based
  ops don't generalize. Copy/induction heads plausibly form better at larger scale/more training, so #1
  and #2 are INCONCLUSIVE-at-spike-scale, NOT proven-impossible -- retest at production dims (d448+)
  before relying on them. (Contrast: resolution generalized at this same d128, so it's a real
  difficulty gap, not just "d128 too small for everything".)

*** CORRECTION 2026-07-11 -- a WARM-START CONFOUND invalidated several numbers above; here are the honest
COLD results. *** ROOT CAUSE: build_model re-inits WEIGHTS deterministically (fixed-seed 1234
init_weights) BUT the AdamW moment arenas (g_param_m/g_param_v) are GLOBAL and persist across TEST_CASEs
in one process -- so a test warm-started from the PRIOR test's optimizer momentum. It cut BOTH ways:
- multi-binding "1.000" was INFLATED (warm-started by the single-binding test that ran first). COLD = 0.643
  drilled / 0.308 held-out (generalizes, still rising -- NOT perfect).
- #2 reason-over-slots "chance ~0.33" was DEFLATED (contaminated by #1/define's momentum). COLD = **1.000
  drilled AND held-out** -- reason-over-slots (associative recall: K tagged slots, query a tag -> output the
  matching slot) actually works PERFECTLY. This REVERSES the earlier "#2 fails" conclusion.
FIX: reset_opt_state() zeros the Adam arenas after build_model in every scratch TEST_CASE -> results are
now order-INDEPENDENT + reproducible run-alone or in-suite (verified). See [[cross-test-optimizer-state-contamination]].

HONEST COLD RESULTS (d128/vocab774, 3000 steps, each test cold):
- single-binding resolution: held-out nth 0.455, rt 0.358 (generalizes, rising).
- multi-binding (K=3) resolution: held-out 0.308 (generalizes, rising; the "1.000" was warm).
- #1 model-driven binding (copy a NOVEL OOV into combine): held-out 0.158 -- FAILS (copy bottleneck, real).
- #2 reason-over-slots (select by tag): held-out **1.000** -- WORKS, and associative recall needs NO
  content-derived embedding (it's oov-content-independent). Content-derived embeddings would only be needed
  for CONTENT-based reasoning (e.g. "which slot's name has a vowel"), not associative recall.
Corrected takeaway: the model RESOLVES slots (partial, rising -> stronger at scale) and REASONS over them by
association (perfect), but cannot CREATE a binding by copying a novel OOV (#1). Binding is best encoder-side
(deterministic), which sidesteps #1; resolution improves with scale; associative reasoning already works.

FOUNDATION ENGINE SCAFFOLD STARTED 2026-07-11 (committed 0bc3a01) -- content-derived slot embeddings, the
one delineated engine change (fixes content-reasoning #4). User picked "shared scaffold first" + "start
small but keep foundational scope to test all encoder options later". Shipped: include/sub0/scratch_slots.hpp
(engine-safe: casing.hpp+std only) = the slot RANGE as a range-agnostic first-class constant + a pluggable
`SlotEncoding{MeanPool,CharEncoder,Hash}` (only MeanPool implemented; others reserved dispatch arms) + a
tokenizer-free `ScratchBindings` view + pure `encode_slot`/`encode_slot_bwd` (mean of the fragments' tok_emb
rows + adjoint). backend_cpu: a thread_local binding context (`set_scratch_bindings`, exported via core.hpp
with a forward-declared ScratchBindings) + a bound-scratch-slot branch in op_embed / Op::Embed backward /
forward_one -- a bound slot embeds from its fragments (grad flows back to them); null bindings = byte-for-byte
today's path (pos_emb naturally inert). Tests scratch_embed_tests.cpp: exact mean-pool fwd+adjoint, a forward
DIFFERENTIAL proving the engine feeds a bound slot's fragment-mean in (excludes the slot's own logit col --
the TIED head reads the slot ROW, a separate concern), + no-binding bit-identity. NEXT on this foundation
(all designed to drop in): CUDA embed branch (parity-gated); training integration (train_batch builds
per-window ScratchBindings from the curriculum + set_scratch_bindings, so --scratch-mix learns WITH content
embeddings -> re-run #4, expect it off chance); gen wiring (ScratchTable::to_bindings()); other encoders
(CharEncoder/Hash arms); reserved-range growth (a casing marker-count bump = new tokenizer format). NOTE:
the tied HEAD still reads a slot's fixed row -- a content-derived HEAD is a further extension if slots need
to be PREDICTED, not just consumed.

TRAINING INTEGRATION DONE + KEY FINDING (committed e11e0da): train_batch gained an optional per-window
`win_binds` (each OMP worker installs its window's ScratchBindings around that window's forward+backward);
scratchspike Dataset carries per-doc bindings (doc_bindings via append_doc); harness train_steps(content_embed)
+ run_task feed them at train AND eval. A DIRECT test ("scratch embed: train_batch applies per-window
content bindings") proves the bindings change the forward loss -- wiring is LIVE. FINDING at d128: mean-pool
content embeddings do NOT deliver content reasoning -- BOTH first-letter (#4; needs ORDER mean-pool discards)
AND order-agnostic "contains X" (new content_contains_task; A/B without==with==chance, degenerate fixed-slot
output) stay at chance. NOT a no-op; a single char is too diluted in a mean of ~5 byte rows for a d128 model
to extract. => the reserved learned CharEncoder arm (per-char/order-aware features over the fragment rows +
params + a tape node + backward) is the NEXT LEVER and the real test of whether content-derived embeddings
deliver -- exactly the pluggable extension the scaffold was built for; mean-pool is just the too-weak encoder.
(Supersedes the "re-run #4 expect off chance" hope above: done, mean-pool did NOT move it.)

CHARENCODER DONE + POSITIVE (committed eb9feee math+FD test, b7896ab training+finding): encoder = sum over
fragments of relu(enc_w[C,C] . row). Spiked with NO model-format change -- enc_w lives in ScratchBindings
(op_embed/backward/forward_one thread it to encode_slot/_bwd); harness trains it SINGLE-THREADED
(train_ce_steps) with its OWN AdamW (enc_w_grad has no per-thread reduction -> a multi-threaded train_batch
would RACE; promote to a model param before production). FINDING (d128, order-agnostic "contains X", A/B):
mean-pool 0.325 (chance ~0.333); SGD-CharEncoder 0.308 (UNFAIR, under-trained -- model uses AdamW so the
encoder never catches up); **AdamW-CharEncoder 0.392** -- ABOVE chance AND plain. Modest (d128 content-match
capacity is limited) but the DIRECTION is real: a learned per-fragment encoder delivers content signal a mean
cannot. LESSONS: encoder is the lever mean-pool wasn't; the encoder's OPTIMIZER matters (SGD failed, AdamW
worked -- fair-test discipline); scale is the amplifier. NEXT: promote enc_w to a MODEL PARAM (layout +
per-thread grad reduction -> ends the single-thread limit + race -> multi-threaded train + --scratch-mix
integration); d448+ scale run; CUDA embed branch; gen wiring (ScratchTable::to_bindings); reserved-range
growth; order-aware encoder (Hash/positional -- for the FIRST-LETTER task both mean-pool AND this CharEncoder
miss, being order-agnostic).

ROUTE A vs ROUTE B -- BOTH SHARE ONE CAPACITY WALL (committed cb50795 test, d4b4125 docs; user asked
"test the thinking-uncombine spell first -- learning spelling via embeddings is inefficient when a perfect
Route A exists"). TWO routes to slot-content reasoning: A = uncombine+loss-masking (model resolves the slot,
interceptor injects the spelling, then reads it -- exact, O(K*len) tokens); B = content-derived embeddings
(bake content into the slot vector -- O(1) token, lossy). Built content_contains_reason_task (resolve EVERY
slot then select the one whose spelling contains the queried char) + build_dataset_contains_reason +
eval_contains_reason; factored pick_contains (ContainsPick) so single-hop + reasoning builders share
selection. FINDING (d128/4-layer, 3000 steps): the resolve PROTOCOL is learned FLAWLESSLY at every K -- the
model reliably emits <slot> UNCOMBINE and every spelling enters the stream, HELD-OUT too (dumped the
generated traces to PROVE it, not infer it). But the downstream char-match-and-select is CAPACITY-bound:
K=2 (chance .50) held-out ~0.72 WORKS + generalizes; K=3 (chance .33) COLLAPSES to a degenerate constant
slot (always S0, ~chance) even with all 3 spellings in-stream. So Route A is NOT mechanism-limited
(resolution perfect at both K) -- its ceiling is model capacity for the K-way match, the SAME wall Route B
hit (CharEncoder 0.392). KEY REFRAME: the routes differ in inference COST + signal fidelity, NOT reasoning
ceiling -- they share it; content being present (A) is no easier than content being embedded (B); the
bottleneck is the composed match-and-select, not delivery. => SCALE is the deciding experiment for BOTH, and
ROUTE A CAN BE SCALE-TESTED IMMEDIATELY with NO engine changes (plain masked training on the reason
curriculum at a bigger model) -- promoted to next-step #1 above enc_w promotion. METHOD LESSON: when an A/B
sits at chance, DUMP what the model generates before concluding "too hard" -- a flat-drilled chance result
looked like failure but the traces showed a perfect protocol + a degenerate always-S0 answer (a capacity
shortcut), which is a totally different diagnosis than a broken mechanism.

SCALE VERDICT -- d256 clears K=2, NOT K=3 (committed bdde67b; user: "tackle scaling with a medium d256
model first, quick train"). Configured a fresh d256 CPU build (out/build/spike_d256; sub0llm-configure
--dmodel 256 --seq 64 --tie-embeddings 1 on english.txt) and ran the Route A two-K test. THREE runs cleanly
separated recipe from capacity: (1) d256 auto-config = 8L/2H @ lr0.003 (d128 recipe) -> REGRESSED, collapsed
to always-S0 even at K=2 (0.72->0.43); (2) d256 pinned 4L/8H (scaled version of the WORKING d128 4L/4H
shape) @ lr0.003 -> ALSO collapsed K=2 (0.48) -- RULES OUT head-count/shape; common factor = 2x width at a
stale LR; (3) d256 4L/8H @ lr0.0015 (==0.003*128/256, muP ~1/width) -> K=2 RECOVERED to peak 0.83 (BEATS
d128's 0.72), K=3 STILL collapses (~chance, stuck from step 0). => doubling width DONE RIGHT recovers/
improves K=2 but does NOT buy the 3-way match; K=3 is genuine capacity beyond 2x width. BIG LESSON:
hyperparameters DON'T TRANSFER across scale -- a d128-tuned LR silently lands a wider model in the
degenerate basin and it NEVER escapes (naive scale-up ACTIVELY BREAKS a capability that worked small). Baked
2 fixes into the test so future scale runs aren't sabotaged: kLr auto-scales 0.003*128/D_MODEL (no-op at
d128, validated d256); report tracks BEST held-out not last (K=2 peaks ~step1500 then OVERFITS/decays to
0.575 -- last-round understated it). NEXT for K>=3: bigger still / more steps / a K=2->K=3 CURRICULUM
(bootstrap reasoning at low fan-out then widen). Related: this is the same scale-doesn't-transfer lesson as
[[central-run-config-json]]'s per-model recipe; d256 build is out/build/spike_d256 (CPU, reusable).

ALL/NONE DECOMPOSITION -- the wall is LOCALIZATION, not checking (committed 78e7188 code, ccc3185 docs;
user: "did we train k=2 with neither-word cases? add all & none to the regime"). The user caught that
pick_contains GUARANTEED exactly one match, so K=2 was solvable by ELIMINATION ("not word 0 -> word 1", a
single check that never looks at word 1 and does NOT compose to K=3 -- the old K=2 0.72 was THIS shortcut,
not real reasoning). Added NONE (char in no oov -> ANS_NONE='_') + ALL (char in every oov -> ANS_ALL='@')
answer tokens (single opaque byte tokens that acquire meaning from consistent use, like role/control
tokens) over the same resolve-then-answer trace; content_contains_reason_task now takes an explicit answer
token; pick_content_contains_reason_task draws a ONE-heavy 60/20/20 mix; harness reports per-kind
(one/none/all) breakdown. FINDING (d128): the task DECOMPOSES -- (1) PRESENCE DETECTION (NONE/ALL, a global
"present anywhere/everywhere?" OR) is learned NEARLY PERFECTLY (~1.0) at K=2 AND K=3 -> the model now
GENUINELY checks every word (the user's goal, achieved); (2) LOCALIZATION (ONE -- WHICH slot) is the REAL
wall -- per-kind one-rate stays ~random slot-guess (~0.45 K=2, ~0.25 K=3) even ONE-heavy; with an EQUAL 1/3
mix it doesn't even TRY, taking a present->ALL/absent->NONE heuristic (~0.67 overall, one~0.05). So the
bottleneck is a POINTER/argmax-over-words op (find the matching row, report its label), NOT checking and NOT
K-specific. Weighting toward the HARD case (ONE-heavy, mirrors build_dataset_scratch's 3:1) forces the model
to point but it points ~randomly -- localization genuinely unlearned at d128. OPEN next probes (no engine
change): ONE-heavy @ d256 LR-scaled (does one-rate rise?); MORE HEADS (parallel pointer lanes); pointer-
focused/localize-only curriculum. METHOD WIN: per-kind breakdown + generated-trace dump turned a flat
"~chance" into a precise decomposition (presence vs localization) -- reinforces [[check-captured-logs-before-process-archaeology]]-style "look at the actual output".

LOCALIZATION WALL = SEGMENT-LOCAL MATCHING (committed f1264dc code, 5255dee docs). Two probes, both
NEGATIVE: (1) SCALE -- d256 (4L/8H, auto LR 0.0015) ONE-heavy leaves one-rate ~random (0.48 K=2 / 0.30 K=3),
UNCHANGED from d128 -> 2x width + 8 heads do NOT buy localization. (2) CoT VERDICT SCAFFOLD -- added a
per-slot verdict (VERD_YES='+' / VERD_NO='-', content_contains_cot_task + build_dataset_contains_cot;
factored pick_contains_case so reason & CoT share selection; eval takes a ReasonPicker fn-ptr; run loop
promoted to shared route_a_run()): right after resolving each slot the model emits +/- (does THIS slot
contain c), so the final answer is "copy the slot that got +". The final COPY works when verdicts are right,
BUT the verdicts come out WRONG (traces: "% t -> uwqcos:+ tzi:-", inverted) -> one-rate STILL ~random (0.44
K=2 / 0.25 K=3). WHY: emitting slot i's verdict needs to isolate THAT slot's just-injected bytes -- the SAME
segment isolation the model can't do. REFINED DIAGNOSIS: the wall is binding a content-match to a SPECIFIC
slot's byte-region -- the model does a GLOBAL scan (is char anywhere -> NONE/ALL perfect) but not a
SEGMENT-LOCAL one (is char in THIS word). ROBUST: survives 2x scale, more heads, ONE-heavy, AND the CoT
scaffold -- NOT a quick recipe/scaffold fix. REMAINING IDEAS (all plain masked-training, no engine work):
local query RESTATEMENT (re-emit the char NEXT TO each segment so the check is over adjacent tokens -- but
overflows the 40-tok window at K=3, needs wider window / trimmed prompt); structural SEGMENT MARKERS /
positional cues; or a MUCH larger model (>=4-8x, since 2x didn't move it). NOTE: kWindowT=40 vs SEQ_LEN=64
is the binding constraint on trace length for these variants.

LOCALIZATION SOLVED -- LOCAL GROUNDING is the fix (committed aa1c8ba local variant, 22ff4ca control, ae3aaf3
docs). The "segment-local matching wall" was really NON-LOCAL BINDING: the model couldn't hold the queried
char in mind while scanning segments listed FAR earlier. FIX: RESTATE the queried char immediately before
each per-slot verdict (trace: <bytes_i> <c> <verdict_i>, check over ADJACENT tokens) + drop the redundant
front slot-list to fit the 40-tok window (content_contains_cot_task now has restate_query + drop_front bools;
pick_content_contains_cot_local_task = both true). RESULT @ d128 (the TINY 4-layer model): plain CoT one-rate
~0.44 K=2 / ~0.25 K=3 -> LOCAL one-rate 0.91 K=2 / **1.00 K=3, held-out 1.000** -- and it GENERALIZES to
never-bound OOVs (real reasoning, trace "want S2 | % z ... cstk z- ... tzi z+ = S2" flawless). CONFOUND
CONTROL (pick_content_contains_cot_ctrl_task = drop_front only, NO restate): one-rate stays ~0.35 (random) --
proves the RESTATEMENT (locality), not the shorter context, is the driver. => localization is NOT a capacity
wall; the d128 model does it perfectly when the reasoning is GROUNDED LOCALLY (query next to the data). Scale
(d256), more heads, ONE-heavy, and plain CoT all failed; local grounding worked at the smallest size. GENERAL
LESSON (likely transfers beyond scratch tokens): for multi-item content reasoning, put the query ADJACENT to
each item rather than stating it once far away -- distance, not capacity, was the bottleneck. Refactor:
build_dataset_contains_via(picker) is the one generic contains-curriculum builder (reason/CoT/local/control
share it). This mechanism-over-scale win is the seed of the [[deterministic-mechanisms-thesis]].

MAXIMAL-K SWEEP + PRODUCTION MERGE (committed 6376ed0 infra, 7ef712b merge). SCRATCH_POOL bumped 4->6 (full
reserved range TOK_RESERVED_4..9 = mechanism's max K without a tokenizer-format change). Parameterized the
training window (train_steps/route_a_run gain `window`) + a seq128 build (d128/L4/H4) so long local-CoT
traces (~5+11K) fit. SWEEP (seq128, K=2..6, single run each so NOISY): one-rate 0.667/0.905/0.439/0.750/
0.268 at K=2/3/4/5/6 -- NO hard cliff (all above random 1/K) but margin DEGRADES + noisy past K~4 (K=3
robustly ~0.9-1.0); consistent with the final argmax-over-K-verdicts getting harder -> cap production
contains_k at 4. MERGE: build_dataset_scratch gains a contains_k component -> --scratch-mix teaches THREE
capabilities. MIX+BUDGET were the crux: content AND resolution BOTH hard + COMPETE (esp. shared UNCOMBINE
pattern) -- 20% content->content ~random; 30% resolution->resolution ~random. FIX: 45% content/45%
resolution/10% associative + enough budget. VALIDATED @7200 steps (COMBINED test): resolution 1.000,
associative 1.000, content one 0.95-1.00 -- all three together, no interference. The 3000-step starvation
was a BUDGET artifact (production 10x+ longer). train_stage: contains_k=min(pool,4,(SEQ_LEN-8)/11), omit <2.

FULL ASPECT SWEEP COMPLETE 2026-07-11 (committed 821d9e5). #4 CONTENT reason-over-slots (K slots, NO
tags, query a first-letter -> the slot whose OOV STARTS with it): CHANCE (~0.33) drilled AND held-out --
FAILS. So the boundary is crisp: ASSOCIATIVE recall (tag->slot, in-context info) works perfectly;
CONTENT-based reasoning (needs the slot's meaning AT the slot token, without resolving) fails because a
generic reserved-id embedding carries no content -- THIS is precisely + only what content-derived slot
embeddings would fix. CAPABILITY MAP (d128): resolution WORKS (high-variance, mix-sensitive: 0.13-1.000);
disambiguation WORKS; associative reason WORKS (1.000); content reason FAILS (needs engine change);
model-driven binding-create FAILS (copy bottleneck -> encoder-side); round-trip partial.
MERGE: build_dataset_scratch (a 3:1 RESOLUTION-heavy mix of resolution + associative) is what --scratch-mix
now blends. CURRICULUM-BALANCE finding: a naive 50/50 mix let the EASY associative task's gradient STARVE
the harder resolution one (resolution 0.31 solo -> 0.13 mixed); 3:1 resolution-heavy restored it (->1.000
this run, associative ~0.48). d128 resolution is high-variance; don't over-read single-run numbers --
production scale should stabilize both. All working features are now in the production curriculum.

STATE: RESOLUTION + disambiguation = SOLID (mergeable mechanism, no engine change). Binding-CREATION (#1)
+ reasoning-over-slots (#2) = copy-bottlenecked at d128, need scale and/or the content-derived-embedding
engine change. Production-merge plan reflects this split (resolution now; #1/#2 gated on a scale retest).
All committed on branch claude/llm-transformers-1bit-qf3t3i (spike scaffold 5482158, multi 67786ab, #1+#2
next). NEXT if pursued: retest #1/#2 at d448+ (production build); OR the content-derived-embedding engine
change (slot embedding = a char-encoder fingerprint of its bytes) which directly targets the copy/reason
gap; THEN production merge of the resolution half (binding table into decode.hpp + gen_stage).

(Superseded status line: was "BACKLOG, gated on the spelling spike" -- gate now cleared, spike active.)

**FRAMING DOCUMENT ADDED 2026-07-21 (docs/SCRATCH_TOKEN_FRAMING.md), criteria-first not chronological.**
Written after [[factspike-piece-transfer-validated]]'s attention-capacity finding (Pearson
r(piece_count, cos_sim) = -0.614 -- packed-vs-normal representational fidelity degrades with piece
count, the cleanest signal that whole investigation produced). Captures 15 axes a scratch mechanism
needs to satisfy (this memory's own history covers most of them empirically) as a table with status +
evidence pointers, so a future session doesn't have to re-derive the requirement set from this log.

Core reframe, the user's own "Rumplestiltskin" synopsis: if a word were a single vocab token its
knowledge would just be one embedding row; split into pieces, the knowledge is NOT in any one row, it's
the PROCESS of N_LAYERS of sibling-attention hops between pieces. Current mechanism (MeanPool/HRR over
raw pre-transformer embeddings, `encode_slot()`) is a crude one-shot APPROXIMATION of that process,
computed before any of the real multi-hop computation happens -- explains why fidelity degrades with n
(more real computation being approximated away). The mathematically EXACT alternative, not yet built:
KV-TRACE MEMOIZATION -- run the real forward pass over a word's pieces once, cache the resulting
per-layer (K,V) trace (this engine's own `KVCache` is `[N_LAYERS][SEQ_LEN][D_MODEL]`, already used for a
same-context retroactive splice in wordspike's live collapse), and splice that trace into a DIFFERENT
context later -- reusing real computation instead of guessing at it. Honest tension: full n-position KV
splice is exact but costs the same O(n) budget as not packing at all (defeats the point); current
1-position embed composition is compact but lossy; a `k`-position compromise (considered, rejected by
user -- marginal gain, real residual loss, not principled) is not the answer either.

Two candidate mathematically-motivated middle grounds, neither built yet: (1) post-hoc per-layer KV
pooling -- let the real n-position computation happen once, THEN compress per-layer K/V (not raw
embeddings) for storage; (2) landmark-style transparent expansion -- keep the full trace cached
out-of-band, the visible scratch token is a pointer that expands attention into it on demand (cites
Memorizing Transformers / landmark attention as prior art). Open math questions recorded in the doc: how
context-sensitive is a word's own KV trace (testable via `hidden_state_cosine()`/`last_hidden_ptr()`
extended to intermediate layers); where does trace information concentrate by layer; does the measured
r=-0.614 match known HRR-crosstalk theory (~1/sqrt(n) falloff, though HRR literature assumes D far above
this project's toy D_MODEL=96) -- a cheap analytical curve-fit against the existing 9-point dataset, not
yet done. See also [[persistent-slot-selection-problem-backlog]] for the sentinel-pair addressing design
this framing complements.

**KV-TRACE MEMOIZATION SPIKED 2026-07-21 -- candidate 1 (post-hoc per-layer KV pooling) FALSIFIED AS
STATED.** User green-lit building the framing doc's "most promising mechanism" ("green lit even if it
needs lots of work"), explicitly asking first to "identify DRY collapse/abstraction/extension points"
before implementing. Built minimally: 3 new engine primitives (`kv_krow_ptr`/`kv_vrow_ptr`,
`kv_rope_rotate` -- a thin export of the existing `rope_row`, confirmed algebraically as its own exact
inverse under negated position -- and `kv_splice_row`, all in `core.hpp`/`backend_cpu.cpp`), deliberately
NOT wired into any `ScratchBindings`-style struct or `forward_one` dispatch branch -- orchestrated
directly by the test harness instead, since a 4th production bindings type before the mechanism is even
validated would be premature (this project's own YAGNI discipline, [[only-add-arguments-we-need]]).
Pooling needed ZERO new math: captured per-layer K/V rows are fed straight into `encode_slot`'s existing
HRR math (treated as a synthetic `[n,D_MODEL]` table) -- the same operator mechanism A already uses, one
layer deeper.

Real result (both mechanisms measured in the SAME run/weights -- a fairer matched comparison than
re-quoting a prior session's number, given this project's repeated single-seed-volatility lesson): mean
cos_sim A(embed-compose)=0.83 vs B(kv-trace)=0.77 -- WORSE, not better; r(piece_count,cos_sim)
A=-0.614 vs B=-0.624 -- slightly MORE negative than the predicted-weaker direction; accuracy tied 3/9 on
DIFFERENT subjects. A single n=8 outlier explains most of the mean-gap (nearly vanishes excluding it:
0.859 vs 0.850) but not the correlation gap. **Candidate 1's specific hypothesis -- that richer pooling
INPUT (real per-layer K/V vs raw embeddings) narrows the degradation -- is falsified.** Mechanistic
reframe this motivates: the bottleneck looks like HRR's own fixed-`D_MODEL`-capacity n->1 BUNDLE itself,
not what's being bundled -- a direct, not-yet-run, purely analytical link to the still-open "does r=-0.614
match textbook HRR crosstalk (~1/sqrt(n))" question against the same dataset, no new training needed.
Full numbers + outlier analysis: docs/FACTSPIKE.md Phase G entry, `[.factspikekvtrace]` in
tests/factspike_engine_tests.cpp. Next steps NOT yet decided: the HRR-crosstalk curve-fit, a different
pooling operator, or candidate 2 (no pooling at all, accept O(n) cost).

**Candidate 2 spiked same day (2026-07-21) -- decisive SPELL_START/SPELL_END finding.** No pooling, full
n-position splice, zero new engine code (reused all 3 candidate-1 primitives, splice writer called n
times instead of once). User flagged directly: "the spell start/end could be carrying some weight."
Founded via a controlled swap, not inference: `subject_piece_ids` (what EVERY mechanism here captures
from) strips SPELL_START/SPELL_END wrapper markers a real forward pass actually processes; replaying the
marker-inclusive span closed a real, root-caused gap (0.87-0.96 -> exact 1.000) for every multi-piece
drilled subject, zero exceptions. Every packing mechanism in this whole line of work has been operating
on an incomplete basis by construction. Full detail: [[factspike-piece-transfer-validated]]'s own
"Candidate 2 spiked" section, docs/FACTSPIKE.md Phase G, docs/SCRATCH_TOKEN_FRAMING.md (axis 9 + open-
questions updated).

**Phase H (2026-07-21), same day: the retest, run — RESULT IS WORSE, and it resolves the mechanism, not
just the axis.** Marker-inclusive composition made mechanism A/candidate-1 WORSE (0.83→0.695, 0.77→0.630),
not better, because both still have to COMPRESS everything into one fixed-D_MODEL HRR bundle — giving an
already-bottlenecked compressor more real content to squeeze in doesn't help. Candidate 2 (no compression)
benefits freely from the same markers because it has no capacity limit to hit. Three separate experiments
(candidate 1's richer-input swap, Phase H's marker-add, candidate 2's no-compression splice) now converge:
**the bottleneck was always the fixed-size compression STEP, never which tokens feed it.** Codebase survey
found `wordspike.hpp`'s own spike training data is NOT exposed to the marker-omission gap (bypasses
tokenization, uses raw bytes) but `corpus_collapse.hpp` (production, COMMITTED) and its live-generation
counterpart in `decode.hpp` ARE. Production-scale cross-check (real TinyStories, d196/11L): held-out NELBO
identical to 4 decimals between marker-stripped/inclusive arms — a real negative at that coarse, aggregate
metric, not a contradiction of factspike's surgical single-position finding. Full detail:
[[factspike-piece-transfer-validated]], docs/FACTSPIKE.md, docs/CORPUS_COLLAPSE.md new entry.
