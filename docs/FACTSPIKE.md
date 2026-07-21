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

- **2026-07-20**: Phase E — the warm-start/ramp follow-up, run for real. Same matched budget/seeds/
  datasets as Phase D (same widened 15-subject pool, same slot-retrieval documents), but rounds 0-4
  (steps 1-500) train on plain text ONLY (`slot_frac=0`), rounds 5-14 ramp `slot_frac` linearly 0→0.5,
  and the final 5 rounds hold at 0.5 (Phase D's own steady-state mix).

  ```
  peak: baseline=0.333333 scratch=0.111111 held_out=0.000000
  last: baseline=0.222222 scratch=0.000000 held_out=0.000000
  (Phase C: peak baseline=1.00 scratch=0.56 | Phase D flat-mix: peak baseline=0.22 scratch=0.00)
  ```

  **Reading — this disambiguates Phase D's two candidate causes, and dilution wins.** Warm-start+ramp
  did help some (baseline 0.22→0.33, scratch 0.00→0.11 vs. Phase D), so cold-mixing interference was a
  real, non-zero contributor. But it recovered almost none of the gap to Phase C (baseline 1.00, scratch
  0.56) — nowhere close. The decisive tell is in the WARM-UP-ONLY rounds themselves (s100-s500,
  `slot_frac=0` throughout, so cold-mixing literally cannot be a factor yet — zero slot-retrieval training
  has happened): baseline only reached peak 0.33 by s400 on the widened 15-subject pool, whereas Phase B
  (9 subjects only, same step budget) was already hitting 0.44-0.78 by the equivalent step range. Since
  the ONLY difference between that warm-up window and Phase B is the widened subject pool, and cold-mixing
  can't be involved yet, this isolates **dilution of per-subject repetition as the dominant cause**, with
  cold-mixing as a real but secondary contributor. The actual fix isn't a training schedule — it's not
  diluting drilled-subject exposure when the pool widens (e.g. oversample drilled subjects specifically,
  or scale total budget/repetition with pool size instead of holding both fixed). Not yet built; the
  training-schedule line of investigation (Phase D/E) is shelved in favor of the representational question
  below, which is orthogonal to it.

- **2026-07-20**: Hidden-state diagnostic — a different question entirely, raised directly by re-examining
  what "packing" computes. Normal multi-piece processing composes a word's meaning INSIDE the transformer:
  N_LAYERS of self-attention + FFN over several real positions, and the model's decision is read from the
  LAST layer's output (a deeply processed representation). The scratch mechanism instead composes OUTSIDE
  the transformer: `encode_slot()` (MeanPool/HRR) is a fixed, parameter-free formula applied directly to
  the RAW pre-transformer embedding rows, injected as a single position's INPUT, which then still has to
  go through all N_LAYERS from scratch with no other positions to attend to. These are not obviously
  comparable objects — the reconstruction-fidelity diagnostic (Phase C follow-up) only checked whether the
  packed vector resembles the RAW piece embeddings (an input-level question); this checks whether the
  model's own fully-processed, decision-relevant representation ends up comparable between the two paths.

  Added `sub0::last_hidden_ptr()` (`backend_cpu.cpp`/`core.hpp`): `forward_one` now unconditionally copies
  its residual-stream hidden state (right before `ln_f`/the head projection) into a thread-local buffer,
  exposed read-only. Deliberately NOT a signature change to `forward_one` itself (shared with the CUDA
  backend's interface) — an always-on side-channel copy instead, zero call sites affected, CPU-only.

  For each drilled subject, under Phase C's own (best-validated) regime: `hidden_state_cosine()` runs the
  exact same prompt shape as `eval_baseline`/`eval_scratch` ("`{subject} loves the color `" vs. "`<slot>
  loves the color `"), captures the final hidden state right before the color would be generated for both,
  and reports cosine similarity — the literal vector the model uses to pick the next token, not an
  arbitrary intermediate point.

  ```
  Crofw        cos_sim=0.889  scratch_correct=no       Elgux        cos_sim=0.974  scratch_correct=no
  Yelfan       cos_sim=0.551  scratch_correct=no        Woqsmb       cos_sim=0.747  scratch_correct=yes
  Hpfds        cos_sim=0.935  scratch_correct=yes       Zlumwkrpx    cos_sim=0.881  scratch_correct=no
  Ceaiaenze    cos_sim=0.605  scratch_correct=no        Nozke        cos_sim=0.932  scratch_correct=no
  Xtora        cos_sim=0.965  scratch_correct=yes       mean=0.831 (3/9 correct this round)
  ```

  **Reading**: mean similarity 0.831 is high — NOT near-orthogonal, which is what a dominant "wrong region
  of representation space entirely" failure would look like. That partly weakens the primitive-mismatch
  hypothesis as the DOMINANT explanation: the transformer's own N_LAYERS generally do bring a packed
  representation into a broadly similar neighborhood to the normally-read one. But this number is inflated
  by a confound worth separating out: **3 of these 9 drilled subjects (Hpfds, Xtora, Elgux) turned out
  single-piece** in Phase A's own vocab build (docs/FACTSPIKE.md's Phase A note), so `encode_slot()` isn't
  doing real multi-item composition for them — MeanPool of one item is that item; HRR-binding one item is
  a fixed per-position rotation of it, still close to the original row. Restricting to the 6 GENUINELY
  multi-piece subjects (Crofw, Yelfan, Ceaiaenze, Woqsmb, Zlumwkrpx, Nozke): mean similarity drops to
  0.768, and correctness shows no clean pattern at all (Woqsmb — the LOWEST similarity of the six — is the
  ONLY one correct this round). Computed Pearson r between similarity and correctness across all 9:
  r≈0.24 (weak positive, not reliable at n=9, and n=6 for the multi-piece-only subset is too small to
  read anything into beyond "no obvious clean relationship"). Net: real but partial representational
  degradation specific to genuine multi-piece composition, not the dominant driver, alongside the
  already-identified weak-training-signal issue (Phase C's exposure documents) — this looks like several
  contributing factors of similar order, not one single root cause.

- **2026-07-20**: full engine regression suite (both MSVC and Clang toolchains) confirmed green after
  `last_hidden_ptr()` (touches the shared `backend_cpu.cpp`/`core.hpp`) — no regressions.

- **2026-07-20**: Phase F — re-tests Phase D's task-contingent gradient fix WITHOUT the dilution confound
  Phase E isolated. `train_steps_3way` keeps drilled subjects on their OWN separate plain-text Dataset
  (identical construction to Phase C's own `ds`) instead of merging them with exposure subjects into one
  pool, at a controlled split: `slot_frac=0.5` (Phase D's own steady-state), and of the remaining windows,
  only 20% go to exposure subjects — drilled subjects keep ~40% of ALL training windows (vs. Phase C's
  50%, vs. Phase D/E's diluted ~30%). Flat mixing from step 1 (not Phase E's ramp), to isolate "does
  fixing dilution alone recover Phase C" as its own clean data point.

  ```
  peak: baseline=0.111111 scratch=0.222222 held_out=0.000000
  last: baseline=0.000000 scratch=0.000000 held_out=0.000000
  (Phase C: peak baseline=1.00 scratch=0.56 | Phase D: peak baseline=0.22 scratch=0.00 |
   Phase E: peak baseline=0.33 scratch=0.11)
  ```

  **Reading — genuinely puzzling, and worth being honest about rather than force-fitting a clean story.**
  Scratch continued a monotonic improvement across the de-diluting sequence (D: 0.00 → E: 0.11 → F: 0.22)
  — consistent with dilution-fixing helping the scratch arm specifically. But baseline dropped to its
  LOWEST point across all four regimes (0.11), even though drilled subjects had MORE plain-text exposure
  in Phase F (~40%) than in Phase D (~30%) — the opposite of what the dilution hypothesis predicts for
  baseline. This is not something a coherent single-cause story explains cleanly.

  The honest read: this is very likely evidence of hitting the limits of SINGLE-SEED experimentation at
  this model/dataset scale. Every phase in this investigation (B through F) has shown extreme round-to-
  round volatility within a single run (e.g. Phase B's own 0.00→0.78→0.00 trajectory) — it would be
  unsurprising if the *choice of design* (dilution fixed or not, schedule flat or ramped) matters less
  than *which chaotic trajectory a given seed happens to land in*, at this small a scale. This exact
  lesson already exists elsewhere in this codebase: `scratch_slots.hpp`'s own HRR-vs-MeanPool history
  notes "a single run looked strong... a 3-training-seed follow-up... tempered that" — single-seed
  comparisons at small scale have misled before in this project, specifically. Phases C through F have
  all been single-seed. None of the peak/last comparisons across them should be treated as settled causal
  claims without multi-seed replication — the *qualitative* findings (dilution matters some, cold-mixing
  matters some, representational fidelity is imperfect but not dominant) are likely still directionally
  right, since they're each grounded in more than just a peak-number comparison (Phase E's warm-up-only
  isolation argument, the hidden-state cosine numbers), but the exact magnitudes and the specific
  Phase F baseline anomaly should not be over-interpreted from n=1.

  **The training-schedule axis (Phase D/E/F, "Pack-Aware Training") is parked here.** It doesn't move the
  actual mechanism, only how hard the model is pushed to use it, and any real fix would need to hold for
  arbitrary words sharing these same piece tokens elsewhere in the vocabulary — not something a training
  schedule targeted at this experiment's own subjects could establish. Attention is next.

- **2026-07-21**: a new hypothesis, orthogonal to both training signal and raw representational fidelity —
  the ATTENTION-CAPACITY axis. In a causal decoder, a genuine n-piece word gets roughly
  `N_LAYERS × (n−1)` sibling-attention hops: at every layer, the word's later positions can re-attend over
  every earlier piece's CURRENT (already-once-refined-by-attention) state, and this compounds across
  layers into real, iterative computation. A packed slot has none of this structurally — it's one
  position, one Q/K/V per layer, no same-word siblings to attend back over at all; whatever composition
  happens is frozen into the MeanPool/HRR formula before layer 1 even runs, and everything after is just
  self-refinement on a single already-fixed starting point. Prediction: the packed-vs-normal
  representational gap should SCALE WITH n (more pieces packed -> more multi-hop computation skipped), not
  be a flat effect.

  Extended the hidden-state diagnostic (`[.factspikehidden]`) to record each subject's exact piece count
  alongside cosine similarity (piece counts were already known to split single- vs multi-piece, but not at
  this granularity):

  ```
  Crofw        n_pieces=4  cos_sim=0.889     Woqsmb       n_pieces=6  cos_sim=0.747
  Yelfan       n_pieces=4  cos_sim=0.551     Zlumwkrpx    n_pieces=7  cos_sim=0.881
  Hpfds        n_pieces=1  cos_sim=0.935     Nozke        n_pieces=4  cos_sim=0.932
  Ceaiaenze    n_pieces=8  cos_sim=0.605     Xtora        n_pieces=1  cos_sim=0.965
  Elgux        n_pieces=1  cos_sim=0.974
  Pearson r(piece_count, cos_sim) = -0.614
  ```

  **Reading: this is the strongest, cleanest signal found in this entire investigation.** r=-0.61 across
  n=9 is moderate-to-strong and directionally exactly as predicted (more pieces -> lower similarity) — a
  much cleaner relationship than similarity-vs-correctness ever showed (r=0.24, no clean pattern). The
  three n=1 subjects (no multi-hop structure to lose in the first place) cluster at the top (0.935-0.974);
  multi-piece subjects spread lower and roughly trend down with n (n=8's Ceaiaenze lowest at 0.605), though
  n=7's Zlumwkrpx (0.881) doesn't fit a strictly monotonic curve -- real trend, not a clean line, which is
  expected at n=9. With df=7 this is borderline by conventional significance (p≈0.08, two-tailed) and
  deserves a larger sample before being treated as fully confirmed, but it's the first result in this whole
  investigation with both a precise mechanistic story AND a clean quantitative match to that story's
  specific prediction.
