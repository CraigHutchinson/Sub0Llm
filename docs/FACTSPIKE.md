# Factspike — does a factual association survive piece-embedding transfer into a scratch slot?

Scope note: this is the design record for `factspike`, an experiment testing whether the model's own
already-trained embedding for a word's real vocab piece(s) — literally the weights that would sit in
context if the word were spelled out normally — still carries a learned factual association when
repositioned into a scratch-slot token instead of the byte-decomposed spelling every other scratch
mechanism uses. It complements [SCRATCH_TOKENS.md](SCRATCH_TOKENS.md) (the scratch-slot mechanism this
reuses unmodified) and [CORPUS_COLLAPSE.md](CORPUS_COLLAPSE.md) (the "ask to see real examples, don't
trust an aggregate metric alone" discipline this follows). Written first, before any code.

## Status: DONE — Phase 0/A/B/C all run on real hardware; positive result (2026-07-19)

Piece-embedding transfer works, at least intermittently: bound directly (no textual restatement anywhere
in the prompt), a subject's own piece embeddings retrieve its taught fact well above chance (peak 0.56 vs.
0.125 chance, 4.4x), while a held-out negative control's single above-chance reading is consistent with
pure n=3 sampling noise. See the Status log for the full numbers and the reasoning behind that call.

## Why this exists, and a real mechanism correction made during planning

Every scratch-slot mechanism proven so far (`scratchspike`/`repeatspike`/`wordspike`/`op_curriculum`) tests
either exact byte-recovery (UNCOMBINE gets back the same bytes that were bound) or structural composition
(route to a node that dereferences a bound value). Neither tests whether the model's *learned world
knowledge* — an association it picked up the ordinary way, from many plain-text training examples — still
fires when the subject is replaced by a content-derived scratch slot.

The first draft of this plan mistargeted the mechanism: it proposed testing whether `content_embed`'s
*existing* behavior (`ScratchTable::expand()` always decomposes a bound fragment down to individual base
bytes before `encode_slot()` composes MeanPool/HRR over them) happens to preserve enough meaning — i.e.
"can a representation rebuilt from individual character embeddings coincidentally hold the same meaning as
the word." That is not what was asked. The actual question: does the word's **own already-trained piece
embedding(s)** — not a representation rebuilt from characters — still carry the association when moved to
a slot token.

**A genuinely useful finding from investigating this: no engine code needs to change.**
`ScratchTable::bind(slot, frags)` (`include/sub0/scratch.hpp`) stores whatever ids it's given verbatim
(`bindings[i] = frags`) — it has no opinion about what those ids mean. `encode_slot()`
(`include/sub0/scratch_slots.hpp:389-485`) composes a slot's embedding by indexing `tok_emb[id]` for each
bound fragment id and MeanPool-averaging or HRR-circular-convolving the rows — it doesn't know or care
whether those ids are byte codes or real vocab-piece ids. Every existing curriculum happens to pass
byte-decomposed fragments (via `expand()`), but nothing forces that. Binding a subject's **own tokenized
piece ids directly** (skip `expand()`'s decomposition entirely) makes the slot's embedding a literal
composition of the word's own already-trained piece rows — exactly "transfer/merge the weights that would
be in the context into the token after the slot." Extracting a clean piece-id list (markers stripped) from
an encoded word is also already built: `sub0::detail::word_span(tok::encode(tk, subject), 0).second` —
reused as-is, matching this project's own "never re-derive something the tokenizer already exposes"
precedent.

## Why a toy, forced-multi-piece model, not TinyStories

A small dedicated model (`--dmodel 96` or similar) trained on a small dedicated fact-only corpus with a
vocab target deliberately too small for whole-word merges makes "compose meaning across a few piece
embeddings" the **normal** way the model represents any word — not a rare special case reserved for OOV
names. That makes a positive or negative result cleanly attributable to the mechanism itself, not an
artifact of using pieces the model rarely sees in ordinary text. Verified via the configurator's own
"word sub-token N (by occ)" reporting line (`tools/configurator.cpp:1276`, corrected earlier this session
to report N2+N>=3 as the real SPELL rate) — Phase A confirms most subject/fact words are genuinely
multi-piece under the chosen vocab target before proceeding.

## Why the model still needs SOME slot-exposure training

`op_embed`/`set_scratch_bindings` (`src/backend_cpu.cpp`) *replaces* a scratch-slot token's embedding
entirely with `encode_slot()`'s computed vector — the slot id's own dedicated learned row is never used
when bindings are active. So surrounding attention/FFN layers have no generic "this is always token 282"
prior to fall back on; they need SOME learned machinery for "a content-derived vector can appear at this
position, here's how to use it," or the fact-recall test would fail for a reason unrelated to the deeper
question (a pure "never saw a slot at all" confound). To keep the test clean, slot-reading exposure in
this plan uses the SAME piece-id-binding convention the eval uses — not reused `wordspike` documents,
which train exclusively on byte-decomposed bindings and might not transfer to piece-composed vectors.

## Staged plan

Each phase gates the next — this is explicitly an open experiment, not a guaranteed-positive validation.

### Phase 0 (free, no model): piece-balance leakage pre-check

Generate the `FactPair` set (invented subject, fact drawn from a small fixed color vocabulary, assigned
uniformly at random, independent of subject spelling) and verify no single piece id (or small piece
combination) disproportionately predicts a color beyond chance across the drilled set, and that held-out
subjects don't cluster near one color's piece pattern. Cheap; iterate on the generation/assignment scheme
until clean before touching a model.

### Phase A: build the toy corpus + a forced-multi-piece model config

Generate factspike's raw fact-teaching TEXT (actual corpus text, not just an in-memory `Dataset`), run
`sub0-configure --corpus <factspike.txt> --dmodel 96 --vocab <small target>`, confirm via the
configurator's own reporting that most/all subject and fact words are genuinely multi-piece.

### Phase B (cheap, real training): does the curriculum teach a persistent fact at all?

Baseline-only pilot — train on JUST the fact-teaching documents (ordinary text, mask=1 throughout, no
scratch slots anywhere yet), small pool (~8 drilled / 4 held-out subjects, 6-8 colors),
`corpus_collapse`-scale budget (~600 steps) to start. Eval: ONLY the baseline arm (subject spelled out in
full) — confirm clearly-above-chance accuracy for drilled subjects before spending more budget. This is
the load-bearing premise everything else depends on.

### Phase C (only if B succeeds): slot-reading exposure + the real 3-arm A/B

Extend the curriculum with documents that ALSO teach slot-reading using the SAME piece-id-binding
convention as the eval — structurally mirroring `corpus_collapse`'s "second mention collapses to a slot,
mask=0, surrounding text graded" document shape, but binding via
`detail::word_span(tok::encode(tk, subject), 0).second` (piece ids) instead of `expand()` (bytes).
Critically, this exposure must use subjects/content UNRELATED to the taught facts, so the fact is never
restated in the same document as a slot-collapse — avoiding an in-document shortcut. Three arms per
drilled subject:

- **Baseline**: subject spelled out in full → generate → check the fact color.
- **Scratch (piece-transfer)**: same prompt shape, subject's own piece ids bound directly to a slot → generate
  → check the fact color. No textual restatement anywhere in this prompt — this is the actual claim.
- **Held-out** (negative control): same slot mechanism, subject never fact-taught → expect near-chance
  accuracy; a result meaningfully above chance would indicate leakage Phase 0 didn't catch.

Report real numbers for all three arms side by side, whatever they are — a weak/negative scratch-arm
result is still a genuinely informative finding about representational alignment (or the lack of it), not
a failed experiment.

## Verification plan

- Phase 0 green before any training.
- Phase A: configurator's own report confirms forced multi-piece vocabulary.
- Phase B pilot: drilled-subject baseline accuracy judged clearly above chance before Phase C.
- Phase C: baseline/scratch/held-out accuracy + wall-clock reported here, honestly, whichever way it lands.
- Full `ctest`/`sub0_tests` regression green throughout.

## Status log

- **2026-07-19**: design planned and corrected after user clarification (mechanism is piece-embedding
  transfer, not byte/character composition — no engine changes needed).

- **2026-07-19**: Phase 0 (piece-balance leakage pre-check) green on first try — no vocab piece is shared
  by all drilled subjects of one color and absent from every other color; held-out subjects don't fully
  overlap one color's piece set. 5 assertions, seed 20260719, 24 subjects, 75% drilled.

- **2026-07-19**: Phase A — built a dedicated `d96` toy config (`D_MODEL=96, N_LAYERS=8, N_HEADS=2,
  VOCAB=493`) from a low-repetition vocab corpus (each subject attested once, decoupling vocab-learning
  repetition from the high-repetition training corpus needed for Phase B/C — see the technical-concepts
  note on this in the session; a first attempt at `vocab_target=320` against the real 30-rep training
  corpus gave every subject its own whole-word merge via `min_merge=2`, defeating "forced multi-piece").
  Confirmed via a standalone checker: 9/12 subjects genuinely multi-piece (piece counts 1–8; Hpfds/Xtora/
  Elgux stayed single-piece).

- **2026-07-19**: Phase B (baseline-only pilot, fact-teaching text only, no slots) — first 600-step run
  plateaued at/under chance (0.00–0.11 vs. 0.125 chance), looking like a clean negative. Extended to the
  full 2000 steps per the plan's own "expect to measure and possibly raise this" guidance: the real
  trajectory is highly volatile (`0.00,0.00,0.00,0.11,0.11,0.11,0.00,0.00,0.44,0.78,0.00,0.11,0.67,0.44,
  0.11,0.11,0.00,0.00,0.00,0.00`), peaking at **0.78** (7/9) at step 1000. A final-round-only check would
  have reported a false negative on a real, demonstrated capability — switched the pass criterion to peak
  (not last-round) accuracy across all 20 rounds, with the full trajectory still reported for human
  judgment. Also found and fixed a real, separate crash bug during this phase: `train_steps` dropped
  `sample_window`'s `.len` and passed `nullptr` lengths to `train_batch`, which then over-read past the
  end of the flat token buffer when the sampled short document was also the last one in the array — fixed
  by threading `lengths` through (this same bug pattern exists unfixed in `wordspike_engine_tests.cpp`'s
  own `train_steps`, never triggered there by luck; flagged, not yet fixed, out of scope for this task).

- **2026-07-19**: Phase C (the real 3-arm capstone: baseline / scratch-slot piece-transfer / held-out) —
  9 drilled + 3 held-out eval subjects, 6 separate slot-exposure subjects (disjoint from both, per the
  "avoid an in-document shortcut" requirement), 2000 combined-source training steps (50/50 fact-teaching
  vs. slot-exposure documents per batch element). Full per-round trajectory (chance ≈ 0.125):

  ```
  s100: baseline=0.00 scratch=0.00 held_out=0.00        s1100: baseline=0.89 scratch=0.11 held_out=0.00
  s200: baseline=0.22 scratch=0.22 held_out=0.00        s1200: baseline=0.89 scratch=0.44 held_out=0.00
  s300: baseline=0.11 scratch=0.11 held_out=0.00        s1300: baseline=0.11 scratch=0.00 held_out=0.00
  s400: baseline=0.11 scratch=0.11 held_out=0.00        s1400: baseline=0.89 scratch=0.44 held_out=0.33
  s500: baseline=0.89 scratch=0.44 held_out=0.33        s1500: baseline=1.00 scratch=0.22 held_out=0.00
  s600: baseline=0.00 scratch=0.11 held_out=0.00        s1600: baseline=0.67 scratch=0.11 held_out=0.00
  s700: baseline=0.67 scratch=0.22 held_out=0.33        s1700: baseline=0.44 scratch=0.00 held_out=0.00
  s800: baseline=0.00 scratch=0.00 held_out=0.00        s1800: baseline=0.89 scratch=0.33 held_out=0.00
  s900: baseline=0.00 scratch=0.00 held_out=0.00        s1900: baseline=0.56 scratch=0.56 held_out=0.00
  s1000: baseline=0.11 scratch=0.00 held_out=0.00       s2000: baseline=0.56 scratch=0.33 held_out=0.00
  ```

  peak: baseline **1.00**, scratch **0.56** (5/9), held-out **0.33** (1/3).
  last: baseline 0.56, scratch 0.33, held-out 0.00.

  **Reading**: baseline confirms the fact is learnable via ordinary text at all (peak 1.00, repeatedly
  hitting 0.89 across many rounds — not a one-off). The scratch arm — a subject introduced to the model
  *only* via a slot bound to its own already-trained piece embedding(s), zero textual restatement anywhere
  in the prompt — repeatedly lands at 0.33–0.56 across many separate rounds (not just once), 2.6–4.4x
  chance. That repetition is what makes this a real signal rather than a lucky single round: the core
  claim (piece-embedding transfer preserves enough of a learned factual association for retrieval) holds,
  at least intermittently, on real hardware. The held-out arm's own peak (0.33) is also above chance, but
  with only 3 held-out subjects, "≥1 correct by pure chance" happens ~33% of the time on its own
  (`1 - 0.875^3 ≈ 0.33`) — and unlike scratch, held-out hits a non-zero value in only 3 of 20 rounds and
  ends at 0.00, consistent with noise rather than a sustained signal. Phase 0's leakage check only covered
  drilled-subject-vs-color piece correlations, not exposure-subject or held-out-subject correlations —
  worth tightening if this experiment is extended, but the round-by-round pattern here doesn't support
  leakage as the more likely explanation over small-n noise.

  Training itself is volatile at this model/dataset scale (matching Phase B's own finding) — a real,
  separate finding from the mechanism question itself, and a natural next step if this line of work
  continues (longer runs, LR schedule, or a less aggressively small model).

  All engine-side regression tests remained green throughout (the `backend_cpu.cpp` OpenMP loop-variable
  fix, made to unblock this build, is exercised extensively by these runs with no regressions observed).

- **2026-07-19**: reconstruction-fidelity diagnostic — does per-round HRR-unbind fidelity correlate with
  scratch-arm accuracy? Motivated by a "Pack-Aware Training" (PAT) discussion drawing on the QAT
  literature (StableQAT, Bit-by-Bit progressive QAT, and a sub-100M-scale schedule study — see PR/commit
  discussion for citations): QAT's dequantize step is inline forward-pass reconstruction feeding the same
  task loss, and HRR's `encode_slot_bwd` already performs the structural equivalent (its backward pass is
  literally circular correlation by the role vector — the canonical HRR unbind operation, applied to the
  gradient instead of the packed value). Added `hrr_unbind()` (`scratch_slots.hpp`, applies that same
  correlation to the packed *value* instead of a gradient) and measured, each of the 20 training rounds,
  the mean cosine similarity between `hrr_unbind(packed_vector, position_p)` and the TRUE `tok_emb` row for
  piece p, across every (subject, piece) pair — for both drilled and held-out subjects — then correlated
  that trajectory against the matching accuracy trajectory (Pearson r).

  ```
  reconstruction-fidelity vs accuracy Pearson r: drilled(scratch)=-0.233  held_out=-0.229
  ```

  **Reading**: no positive correlation — if anything, weakly negative for both arms. This diagnostic does
  NOT support "raw packed-vector fidelity to its constituent piece rows is the direct lever" as a strong,
  standalone hypothesis. More strikingly: `fid_drilled` barely moves across all 20 rounds (0.378–0.387,
  a ~2% band) despite scratch accuracy swinging from 0.00 to 0.56 in the SAME window — the fidelity metric
  is essentially static while accuracy is highly volatile, so no correlation is really available to find
  either way. That flatness is itself informative: nothing in the current Phase C curriculum meaningfully
  shapes drilled-subjects' own packed-vector fidelity, because (as identified in the same PAT discussion)
  the only downstream-graded text after a slot-exposure document's packed slot is generic filler ("plays
  outside every day") — unrelated to the fact, weak task-contingent gradient, and moreover only ever
  applied to a DISJOINT exposure-subject pool, never to the drilled subjects the eval actually queries.
  A near-static fidelity trajectory is exactly what that predicts. This diagnostic doesn't contradict the
  proposed fix (restructure exposure documents so the graded continuation after the slot is the fact
  color itself, giving real task-contingent gradient into the packed vector) — it's consistent with "there
  currently is no real training pressure on packed-vector fidelity to have shaped it in either direction."
  Caveat: raw cosine similarity to a piece's own full row is a coarse proxy — the model may only need a
  task-relevant SUBSPACE of that row preserved, not the whole vector, so a low/flat score here doesn't
  rule out task-relevant information being preserved; it just isn't informative either way. The actual
  next test is whether task accuracy (and this fidelity metric) starts moving once the exposure-document
  fix is in place — see Phase D below.

- **2026-07-20**: Phase D ("Pack-Aware Training") — the exposure-document fix, run for real. Same
  budget/seeds as Phase C. `build_slot_retrieval_dataset()` replaces `build_slot_exposure_dataset()`:
  slot-retrieval documents are now `[SLOT] loves the color {fact}.` with NO full-text restatement in that
  document (mask=1 graded on the fact color itself, task-contingent gradient into the packed vector).
  Exposure subjects' full-text fact teaching moved to a separate document via a WIDENED `build_dataset()`
  pool (drilled + exposure subjects together, held-out still never included).

  ```
  peak: baseline=0.222222 scratch=0.000000 held_out=0.000000
  last: baseline=0.000000 scratch=0.000000 held_out=0.000000
  reconstruction-fidelity vs accuracy Pearson r: drilled(scratch)=0.000000 held_out=0.000000
  (Phase C comparison: peak baseline=1.00 scratch=0.56 held_out=0.33, r_scratch=-0.23)
  ```

  **Reading: this made everything WORSE, not better** — including baseline (subject spelled out in full,
  no slot involved at all), which collapsed from peak 1.00 down to peak 0.22. Scratch collapsed to 0.00
  (r=0.0 is degenerate here, not "no correlation" — scratch accuracy was constant zero, so there was no
  variance to correlate against). `fid_drilled` stayed in roughly the same flat band as Phase C
  (0.385–0.393), so packed-vector fidelity still barely moved — the harder objective didn't even
  meaningfully shape fidelity, let alone accuracy.

  That baseline collapsed too, not just scratch, is the important tell: this isn't cleanly "the harder
  slot-retrieval task hurt scratch specifically," it's "the combined training regime got substantially
  worse at teaching the fact AT ALL," including through plain text. Two candidate explanations, not yet
  disambiguated: (1) widening the plain-text pool from 9 to 15 subjects under the same fixed 50%-of-budget
  share dilutes each drilled subject's own repetition by ~40%, and this model is already known to be
  highly repetition/budget-sensitive at this scale; (2) mixing a much harder, noisier, information-dense
  objective (predict the fact color from a packed vector with almost no other context) 50/50 from a
  RANDOMLY-INITIALIZED model, flat from step 1, destabilizes shared weights before either objective can
  settle — exactly the failure mode the QAT literature's progressive-schedule work (Bit-by-Bit) warns
  about: don't apply the harder transform-aware objective cold and flat; ramp it in, ideally from a
  checkpoint already trained on the easier objective. This second explanation was flagged as a design risk
  in the original PAT proposal, before this result existed, and this result is consistent with it having
  mattered — though not proof of it over the dilution explanation.

  Next candidate experiment (not yet run): warm-start from a Phase-B-trained checkpoint (or an initial
  baseline-only phase within the same run) before introducing slot-retrieval training at all, ramping
  slot_frac up from ~0 rather than starting flat at 0.5. Full engine regression suite confirmed green
  throughout (`hrr_unbind` and the new dataset builder don't affect anything outside factspike's own
  files).
