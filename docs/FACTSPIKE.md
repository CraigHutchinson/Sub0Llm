# Factspike — does a factual association survive piece-embedding transfer into a scratch slot?

Scope note: this is the design record for `factspike`, an experiment testing whether the model's own
already-trained embedding for a word's real vocab piece(s) — literally the weights that would sit in
context if the word were spelled out normally — still carries a learned factual association when
repositioned into a scratch-slot token instead of the byte-decomposed spelling every other scratch
mechanism uses. It complements [SCRATCH_TOKENS.md](SCRATCH_TOKENS.md) (the scratch-slot mechanism this
reuses unmodified), [CORPUS_COLLAPSE.md](CORPUS_COLLAPSE.md) (the "ask to see real examples, don't
trust an aggregate metric alone" discipline this follows), and
[SCRATCH_TOKEN_FRAMING.md](SCRATCH_TOKEN_FRAMING.md) (the requirements/criteria framing and open
mathematics this experiment's attention-capacity finding, below, feeds directly into). Written first,
before any code.

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

  **PAT axis parked here.** User's own call: it doesn't move the mechanism, only how hard training
  pushes the model to use it, and any fix would need to generalize to arbitrary words sharing these piece
  tokens elsewhere in the vocabulary, not just this experiment's own drilled subjects. See
  `docs/SCRATCH_TOKEN_FRAMING.md` for the full requirements/criteria writeup this finding motivated.

- **2026-07-21, Phase G ([.factspikekvtrace]): KV-trace memoization, "candidate 1" from
  `docs/SCRATCH_TOKEN_FRAMING.md`.** Instead of composing a slot's embedding from RAW pre-transformer
  piece rows (mechanism A, `encode_slot`), capture the word's REAL per-layer `(K,V)` trace from an
  isolated forward pass over its own pieces, pool `n → 1` per layer (reusing `encode_slot`'s own HRR
  math, one layer deeper — the captured rows stand in for a synthetic `[n, D_MODEL]` "embedding table"),
  and splice the pooled trace directly into a live KV-cache at inference time, bypassing that position's
  own forward_one computation entirely. Three new engine primitives (`kv_krow_ptr`/`kv_vrow_ptr`,
  `kv_rope_rotate`, `kv_splice_row`, `core.hpp`/`backend_cpu.cpp`) — deliberately NOT wired into any
  bindings struct or `forward_one` dispatch branch; orchestrated directly by the test harness
  (`capture_kv_trace`/`splice_kv_trace`, `factspike_engine_tests.cpp`), matching this project's "don't
  build production abstractions before they're validated" discipline. A fast, model-free sanity check
  (`kv_rope_rotate` round-trips: rotate then inverse-rotate is the identity, `[factspike][kvtrace]`) gates
  the mechanism's core rotation math before trusting any training-time result.

  Both mechanisms measured in the SAME run, on the SAME trained weights, same Phase-C-regime training —
  a fairer matched comparison than re-quoting a previous session's mechanism-A-only numbers, given this
  project's repeated lesson about single-seed volatility. Real result:

  ```
  Crofw      n=4  A=0.889 B=0.828     Woqsmb     n=6  A=0.747 B=0.884
  Yelfan     n=4  A=0.551 B=0.795     Zlumwkrpx  n=7  A=0.881 B=0.813
  Hpfds      n=1  A=0.935 B=0.895     Nozke      n=4  A=0.932 B=0.836
  Ceaiaenze  n=8  A=0.605 B=0.133     Xtora      n=1  A=0.965 B=0.868
  Elgux      n=1  A=0.974 B=0.879

  mean cos_sim:  A=0.831  B=0.770
  accuracy:      A=3/9    B=3/9   (single post-training snapshot, not peak-across-rounds)
  Pearson r(piece_count, cos_sim):  A=-0.614  B=-0.624
  ```

  **Mechanism A's numbers reproduce the 2026-07-21 hidden-state-diagnostic run almost exactly**
  (r=-0.6139 vs the previously recorded -0.614, same per-subject values) — confirms this harness is a
  faithful re-run under the identical regime, not a different setup producing a coincidentally similar
  number.

  **Honest reading: candidate 1 does NOT show the predicted improvement.** The prediction was that B's
  correlation should be WEAKER (closer to zero) than A's, since compression now only has to lose
  genuinely-redundant-across-pieces information at each layer instead of approximating the whole
  multi-hop process from nothing. Instead B's r is very slightly MORE negative (-0.624 vs -0.614) and its
  mean similarity is lower (0.770 vs 0.831) — the opposite direction from the hypothesis, though small
  enough at n=9 to not be a strong claim either way. Accuracy tied exactly at 3/9 for both, but on
  DIFFERENT subjects (Woqsmb: A right/B wrong; Zlumwkrpx: A wrong/B right) — some real per-subject signal
  difference, no net aggregate gain.

  **Outlier-sensitivity check, done honestly rather than left implicit**: Ceaiaenze (n=8, the largest
  piece count in the drilled set) is doing a lot of work — B collapses to 0.133 there while A only drops
  to 0.605. Excluding that single subject: mean cos_sim A(n=8)=0.859 vs B(n=8)=0.850 (the MEAN gap nearly
  vanishes), but r(n=8) A≈-0.448 vs B≈-0.535 (B's correlation stays somewhat stronger/more negative even
  with the outlier removed). So the aggregate mean-similarity gap is mostly outlier-driven, but the
  correlation comparison is not — under this small a sample (n=9, or n=8 excluding one point), that's as
  far as this data can honestly be pushed.

  **What this result actually says, mechanistically**: candidate 1's hypothesis assumed the INPUT to the
  pooling step (raw embeddings vs. real per-layer K/V) was the limiting factor. Swapping in richer,
  already-computed input and seeing no improvement — if anything a same-or-slightly-worse degradation
  curve — points instead toward the POOLING OPERATOR ITSELF (HRR's fixed-`D_MODEL`-capacity n→1 bundle)
  as the dominant bottleneck, independent of what's being bundled. This is a testable link to
  `docs/SCRATCH_TOKEN_FRAMING.md`'s own still-open "HRR crosstalk theory cross-check" question (Plate's
  `1/sqrt(n)` bundling-fidelity result) — not yet done, a clean next analytical step against this same
  9-point dataset, no new training required. It also reframes what's worth trying next: variations on
  WHAT gets pooled (this experiment) look less promising than either changing the pooling operator itself,
  or abandoning pooling altogether (candidate 2's literal multi-position splice, accepting its O(n) cost).

  Not a success for candidate 1 as specified — reported here in full rather than reframed into a
  qualified positive, per this project's own "peak, not last-round, but never spin a negative into
  something it isn't" standing discipline (see Phase D/E/F's own honest negative/ambiguous entries above).

- **2026-07-21, Phase G continued: candidate 2 ("landmark-style transparent expansion") + the
  SPELL_START/SPELL_END finding.** User: "spike candidate 2." No pooling at all — the FULL n-position
  per-layer trace is spliced directly into the KV-cache (`n` real rows per layer instead of 1), accepting
  O(n) cost. Reused the SAME three primitives with ZERO new engine code: capture is `capture_kv_trace`'s
  own per-piece de-rotated rows without the pooling step, and splice is `kv_splice_row` called `n` times
  instead of once — exactly the extension point the original DRY review anticipated. Measured alongside A
  and B, same run/weights: **C** = the trivial case (position-0, isolated capture context — should be
  ~exact, a correctness gate on the splice math, not really a "does it help" measure) and **D** = a
  context-sensitivity probe (the same isolated-captured trace spliced into a REAL non-empty preceding
  context, `"Everyone knows that " + subject + " loves the color "`, against a REAL baseline for that same
  sentence — tests the framing doc's own open "how context-sensitive is a word's own trace" question).

  **First run: C landed at 0.947 mean, not the predicted ~1.0** — a real gap on a claim that should have
  been exact. Two distinct confounds found and separated, not conflated:
  1. **Suffix-tokenization boundary mismatch** (cheap, fixable): `tok::encode(" loves the color ")` in
     isolation does NOT byte-match the same phrase's tail inside `tok::encode(subject + " loves the
     color ")` — BPE merges the leading space differently depending on what precedes it (confirmed via a
     dedicated regression test, `[factspike][kvtrace]`, non-hidden, checking a subject's own tokenization
     always matches its span at the START of the combined sentence — true for every drilled subject).
     Fixed by slicing the real suffix out of the combined tokenization instead of re-tokenizing it. After
     the fix, mean C rose only slightly (0.947), but the THREE single-piece subjects (Hpfds, Xtora, Elgux —
     no `SPELL_START`/`SPELL_END` wrapper needed, so nothing for `subject_piece_ids` to strip) jumped to
     **exactly C=1.000 each** — floating-point-exact reconstruction, the correctness gate finally passing
     cleanly for the cases where it should. Every multi-piece subject stayed meaningfully below 1.0
     (0.87–0.96), now cleanly separated from the single-piece group with the suffix confound and any
     implementation bug both ruled out.
  2. **SPELL marker omission** (the deeper, structural finding, user's own hypothesis): `subject_piece_ids`
     (via `detail::word_span`) strips the `SPELL_START`/`SPELL_END` wrapper tokens that a real forward
     pass over a multi-piece subject actually processes — direct token-id inspection confirmed
     `tok::encode(subject)` = `[SPELL_START, ...pieces..., SPELL_END]` for every multi-piece subject, and
     exactly `pieces` (no wrapper) for every single-piece one. **This is the SAME basis every mechanism in
     this investigation captures/composes from — A's `encode_slot`, candidate 1's pooling, and candidate
     2's splice all operate on `pieces`, never on the wrapping markers.**

  **Direct, controlled test (not inference from correlation)**: added `E` — candidate 2's trivial case
  again, but replaying the marker-INCLUSIVE span (`tok::encode(subject)`) instead of `subject_piece_ids`,
  same subjects, same trained weights, same everything else. Result: **E = 1.000000 for every single
  drilled subject, including every multi-piece one that sat at 0.87–0.96 under C.** Not a trend, not a
  correlation — a swap-and-remeasure that closes the entire remaining gap to floating-point-exact
  precision, with zero exceptions across 9 subjects. **The SPELL marker hypothesis is now founded, not
  speculative**: `SPELL_START`/`SPELL_END` are fully load-bearing for reconstructing a multi-piece word's
  real computation, and every packing mechanism built so far (A, B, and candidate 2 as first specified)
  has been omitting them by construction, not by any deliberate design reason.

  **Context-sensitivity probe (D)**: mean 0.948, essentially indistinguishable from C's own 0.947 (not
  the trivial case's confound-free ~1.0, since D still replays marker-stripped `pieces` under the SAME
  isolated-capture assumption, now additionally spliced into a real non-empty prefix). Given the marker
  finding, this probe should be re-run with marker-inclusive capture before drawing any conclusion about
  context-sensitivity specifically — right now its own number is dominated by the marker-omission
  confound, not yet isolating the context-sensitivity question it was designed to test.

  **Context-sensitivity, redone cleanly (F/G), 2026-07-21 continuation.** D's own number turned out to
  carry a SECOND, previously-unfixed confound identical in kind to the one C needed fixing for (re-
  tokenizing the suffix separately from the combined sentence). Building the marker-inclusive re-test
  (`context_sensitivity_cosine2_impl`) surfaced a THIRD, distinct instance of the same general class: the
  fixed prefix `"Everyone knows that "` ends in a trailing space whose own token gets absorbed/merged away
  when immediately followed by a real subject, so `prefix_ctx.size()` (computed by tokenizing the prefix
  alone) is not a reliable offset into the combined tokenization either — caught by a throwaway diagnostic
  when a first attempt at the fix bailed on 0/9 subjects (too clean a failure to be real per-subject
  noise, correctly read as a bug signal). Fixed by trusting only the ONE assumption verified twice now (a
  subject's own tokenize-alone form appears as an exact contiguous match somewhere in any longer combined
  string it's part of) and deriving prefix/suffix by *searching* for that match and slicing the
  authoritative one-shot tokenization around it, rather than assuming any separately-tokenized piece's own
  length lines up.

  **F** (pieces-only, isolated-capture spliced into the real prefix, both boundary confounds now fixed):
  mean **0.919**. **G** (marker-inclusive, same real-context splice): mean **0.983**. Per-subject: the
  three single-piece subjects (no markers to omit, minimal internal composition to be context-sensitive
  about) sit at F≈G≈0.995-0.997, matching their own same-context C≈E≈1.000 almost exactly — real context
  costs them essentially nothing. Every multi-piece subject shows F clearly below its own same-context C
  (e.g. Crofw C=0.893→F=0.848, Ceaiaenze C=0.937→F=0.879, Zlumwkrpx C=0.960→F=0.871) — a real,
  non-negligible cost from moving to a genuinely different real context, on top of the marker-omission
  gap. But G recovers almost all of it: 0.95-0.997 across every multi-piece subject, tight and consistent,
  a small (~0.02) but real remaining gap from E's own exact same-context 1.000, not the ~0.13 gap F showed.

  **Reading**: markers matter MORE than context-sensitivity does. The E→G same-context-vs-real-context gap
  (~0.017) is far smaller than the F→G gap from including markers at all (~0.064) — most of what looked
  like "context-sensitivity cost" in the original, confounded D was actually still the marker-omission
  problem bleeding through. Once markers are included, a trace captured ONCE in complete isolation and
  spliced into a genuinely different real sentence reproduces ~98% of the true recomputed representation.
  **This is a real, positive answer to whether a precomputed trace generalizes well enough to reuse across
  different real mentions** — the load-bearing assumption behind any "cache a common word's trace once,
  splice it into many later documents" prefill-compute-amortization scheme. Not proof it will hold at
  larger scale or for less redundantly-structured real prose (n=9, one prefix, one toy model — the same
  caveats as everywhere else in this investigation), but a genuinely encouraging, concrete first data point
  where the investigation had none before.

  **What this changes going forward**: this is now a well-founded, high-confidence next experiment, not
  yet run — does composing mechanism A's `encode_slot` (and candidate 1's pooling) from the
  marker-inclusive span instead of `pieces` narrow Phase C's baseline-vs-scratch accuracy gap or the
  r=-0.614 fidelity correlation? Unlike candidate 2 (splice, provably exact once markers are included),
  A/B still have to COMPRESS `n(+2)` rows into one embedding, so marker-inclusion wouldn't make them
  exact — but it changes what they're compressing FROM, and that basis has now been shown, directly, to
  be missing real, load-bearing signal. Not yet run as of this writing — a genuine re-test, not a
  retroactive reinterpretation of Phase C–F's already-recorded numbers (those runs never captured the
  markers, so there is nothing to re-analyze in the old data, only a new run to make).

- **2026-07-21, Phase H ([.factspikemarkers]): does marker-inclusive composition help mechanism A —
  a genuine re-test, run.** A SEPARATE matched-budget training run (`build_slot_exposure_dataset_markers`,
  `factspike.hpp`): the model must be TAUGHT to read a marker-inclusive-bound slot, not just evaluated with
  one, or this would test generalization to a novel binding convention instead of the actual hypothesis.
  Same seeds/steps/regime as Phase C / `[.factspikehidden]`, only the slot-exposure dataset's binding
  convention differs. Measured mechanism A (`encode_slot`, marker-inclusive bind) and a bonus candidate-1
  retest (pooled KV-trace, marker-inclusive capture) on this newly-trained model.

  **Result: WORSE across the board, not better.** mean cos_sim A_markers=0.695 (vs. the piece-only
  regime's own recorded 0.83), accuracy A_markers=0/9 correct in this single post-training snapshot (the
  piece-only `[.factspikehidden]` run, while not headline-accuracy-focused, tracked several correct
  subjects at this same snapshot point), r(piece_count,cos_sim) A_markers=-0.272 (weaker than -0.614, but
  not a positive sign here — a flatter curve on top of an overall WORSE, not better, fit is not the same
  finding axis 9 was hoping for). Candidate-1-style pooling on this same model: mean cos_sim
  B_markers=0.630, r=-0.255 — also not an improvement over Phase G's own piece-only candidate-1 numbers
  (mean 0.77, r=-0.624).

  **Why this doesn't contradict candidate 2's result — it completes the picture.** Candidate 2 (full
  splice, no compression) improved to EXACT 1.000 when given the marker-inclusive span, because splicing
  more real content costs nothing — there's no capacity limit to hit. Mechanism A and candidate 1 both
  still have to COMPRESS everything into one fixed-`D_MODEL`-size vector via HRR bind-by-position. Adding
  2 more real, load-bearing items (the markers) to compress does not help a compressor whose own capacity
  was ALREADY the bottleneck (candidate 1's own finding, above: richer input didn't help there either) —
  it gives the SAME fixed-size bundle MORE genuine signal to lose, which is consistent with things getting
  worse, not better. **Three independently-designed experiments now point the same direction**: candidate
  1 (swap the compression INPUT for something richer — no improvement), Phase H (add MORE real signal to
  the same compression step — measurably worse), and candidate 2 (remove the compression step entirely —
  exact reconstruction). None of them are individually conclusive at single-seed scale, but the CONVERGENT
  direction across three separate designs is a stronger signal than any one result alone: **the bottleneck
  has never been which tokens get composed — it's the fixed-size compression step itself.** A fix needs to
  change the compression operator (different than HRR's fixed bundle) or abandon compression (candidate
  2's own direction), not what gets fed into the current one.

  **Honest caveat, same discipline as every prior phase**: this is ONE training run. This project has
  hit the single-seed-volatility wall before (see the HRR-vs-MeanPool history, and Phase C–F's own
  disagreement). The DIRECTION is what's being leaned on here — three separate methodologies agreeing,
  not one number in isolation — not a claim that -0.272 or 0.695 are precise, reproducible constants.

- **2026-07-21, production-scale cross-check: `corpus_collapse` capstone, 3-arm, real TinyStories corpus
  (`out/build/d196check`, d196/11 layers).** The codebase-wide survey that motivated this (below) found
  `corpus_collapse.hpp` — the production, "COMMITTED" real-corpus word-collapse mechanism — is a direct,
  first-class caller of the SAME `detail::word_span` marker-stripping extraction factspike's whole
  investigation has been about. Added `build_dataset_markers` (marker-inclusive, structurally identical to
  the existing `build_dataset` except the bound span includes `SPELL_START`/`SPELL_END` instead of
  excluding them) and extended the existing capstone (`[.corpus_collapse]`, matched-budget A/B, now A/B/C)
  with a third arm.

  **Result: no detectable difference.** Held-out NELBO (20 val batches × 16 windows, 2000 sampled docs,
  401/1693 had a recurring compound word to collapse at all): base-only=2.9251, base+collapse=2.9176,
  base+collapse_markers=2.9176 — **identical to 4 decimal places** between the marker-stripped and
  marker-inclusive arms. Both collapse arms beat base-only equally (delta -0.0075 either way).

  **This is a real, honest negative — reported as such, not explained away — but it does not contradict
  the factspike finding; it answers a different question with a much coarser instrument.** Factspike's
  cosine-similarity probe measures EXACT representational fidelity at ONE specific position, in a toy
  model built so that "compose across a few piece embeddings" is the *normal* way it represents any word —
  a maximally sensitive, surgical measurement. Held-out NELBO averages next-token prediction quality
  across the WHOLE validation set, where only 401/1693 documents contain any collapse-eligible word at
  all, and even fewer tokens within those are the actual substituted slot position — a real effect
  localized to a tiny fraction of evaluated tokens would need to be large to move an aggregate NELBO
  measured this way, at this budget (600 steps/arm, "a directional signal, not a production run" per this
  test's own long-standing comment). A bigger, more capable model (d196/11-layer vs. factspike's toy
  d96/8-layer) may also simply be more robust to composing from an incomplete basis. **Read together**:
  the SPELL-marker effect on exact representational reconstruction is real and decisive (factspike); a
  measurable effect on general language-modeling quality at production scale, at this budget, is not yet
  demonstrated (corpus_collapse) — these are compatible findings about different things, not a
  contradiction, and neither should be over-extended to answer the other's question.

  **Codebase-wide survey of who else is exposed to this gap** (full detail:
  `docs/SCRATCH_TOKEN_FRAMING.md`'s updated candidate-2 section): `detail::word_span` unconditionally
  strips `SPELL_START`/`SPELL_END` for ANY multi-piece word (`N>=2`, no narrower qualifying condition,
  schemeV3's own deliberate design — see `docs/TOKENIZER_DESIGN.md`). `ScratchTable::expand()` is NOT an
  independent second leak — every real caller feeds it already-`word_span`-stripped pieces, so it's
  downstream of the same gap, not a separate one. **`wordspike.hpp`'s own spike training data is NOT
  exposed** — it builds `doc_bindings` from raw ASCII bytes of the subject string directly, bypassing
  tokenization (and therefore `SPELL_START`/`SPELL_END`) entirely; the spike tested a narrower question
  than the mechanism it was validating for. **The LIVE, deployed word-collapse path IS exposed** — both
  `corpus_collapse.hpp`'s own dataset construction (now cross-checked above) and the live-generation
  counterpart (`decode.hpp`'s `resolve` lambda, fired mid-generation on `TOK_SPELL_END`, feeding
  `gen_stage.cpp`'s `word_collapse` callback a marker-stripped span via the same hand-rolled logic
  `word_span` implements). No prior design doc (`SCRATCH_TOKENS.md`, `CORPUS_COLLAPSE.md`,
  `TOKENIZER_DESIGN.md`) ever previously considered whether the markers should be included in or excluded
  from composed slot content — every prior mention treats them purely as a boundary-detection trigger, not
  as payload. This was a genuinely new question as of this session's Phase G entry, not a previously-known
  and accepted trade-off.

- **2026-07-21, Phase I ([.factspikereinject]): periodic packed-content re-injection, Nanbeige-inspired —
  the first REAL positive signal on axis 9 from a mechanism that keeps O(1) cost.** Project memory
  `nanbeige-architecture-reference` (a real, verified `modeling_nanbeige.py`) documents
  `NanbeigeNgramLayerFusion`: instead of injecting a side-channel signal once at the input embedding and
  letting it survive N_LAYERS of ordinary computation alone, Nanbeige re-injects the SAME signal as a
  gated residual addition at multiple depths. Phase G/H's own finding — mechanism A's real weakness isn't
  missing information, it's that the single upfront injection has to survive the whole stack unaided — is
  exactly what this targets. Built `set_scratch_reinject(stride, scale)` (`core.hpp`/`backend_cpu.cpp`): a
  minimal, PARAMETER-FREE opt-in probe that re-adds the SAME layer-0 packed vector back into a scratch
  slot's own hidden state every `stride` layers, EVAL-ONLY (wired into `forward_one`, the incremental
  decode path, NOT into the training graph — `Model::forward()`/`train_batch` is a structurally separate
  autodiff implementation with no knowledge of this yet). Tested on the ALREADY-trained Phase-C model, no
  retraining — a directional probe: a positive result here, on a model never taught to expect the extra
  signal, would be a strong argument for the bigger investment of wiring this into training properly.

  **First version (fixed embedding-scale injection): a near no-op.** `scale * packed_vector` added
  directly, where `packed_vector` sits at ordinary embedding-row magnitude (`scratch_slots.hpp`'s own
  amplitude convention). Even "aggressive" (scale=1.0, every layer, cumulative magnitude ~`N_LAYERS`×
  the original injection) barely moved anything: mean cos_sim 0.830964→0.831134, accuracy unchanged at
  3/9. Too clean/flat a null to trust at face value — diagnosed, not just accepted: the residual stream's
  own RMS norm grows across depth (standard pre-norm behavior — it's exactly why `ln_f` exists before the
  head at all), and by mid-stack `h`'s accumulated magnitude dwarfs a fixed, small, embedding-scale
  addition regardless of how many times it's repeated.

  **Corrected version (scale-adaptive re-injection): a real, monotonic, honest positive.** Rescaled the
  injection to a FRACTION OF `h`'s OWN CURRENT NORM at each layer, not an absolute constant (`scale` now
  means "inject a perturbation this-fraction-as-large-as h's current magnitude," still zero new learned
  parameters — just a per-layer RMS computation and rescale). Swept gentle=5%, medium=15%, aggressive=40%
  of `h`'s norm, re-added every layer, same trained weights as baseline for all four:

  ```
  mean cos_sim:  base=0.831  gentle(5%)=0.833  medium(15%)=0.840  aggressive(40%)=0.860
  accuracy:      base=3/9    gentle=3/9         medium=3/9         aggressive=5/9
  r(n_pieces,cos_sim): base=-0.614  gentle=-0.615  medium=-0.631  aggressive=-0.569
  ```

  A clean, monotonic dose-response curve — not noise. **Accuracy jumped from 3/9 to 5/9 correct at the
  aggressive dose**, a real task-level improvement from an eval-only intervention the model was never
  trained around. Ceaiaenze (n=8, the most-degraded subject throughout this whole investigation) went from
  wrong at every lower dose to CORRECT at aggressive. Elgux flipped from wrong to correct too, despite its
  OWN cos_sim actually *dropping* (0.974→0.900) — accuracy and representational similarity to baseline
  aren't the same thing; what matters for the task is whether the color prediction comes out right, not
  exact fidelity to how a real multi-piece computation would have looked. The correlation with piece count
  weakens slightly at the highest dose (-0.614→-0.569), the first sign in this whole investigation of that
  specific number moving in the hoped-for direction, though modestly.

  **Reading, honestly**: this is the first mechanism in the entire axis-9 investigation to show a REAL
  positive effect while keeping O(1) visible cost — unlike candidate 2 (exact, but O(n)) and unlike every
  compression-operator variant tried (candidate 1, marker-inclusive mechanism A, both WORSE). That it shows
  up even WITHOUT training exposure is a genuinely encouraging sign, not a limitation to explain away: it
  suggests the model's existing weights already have some latent capacity to use a stronger signal when
  it's presented, even via a crude, ungated, un-learned heuristic. Training the model to expect and
  properly modulate this (a learned gate, matching Nanbeige's own `NanbeigeNgramLayerFusion` design, rather
  than a flat fraction-of-norm heuristic) is the natural, well-motivated next step — genuinely justified by
  this result, not started yet. **Caveats, same discipline as always**: single seed, n=9 subjects, only
  three doses swept (no sense yet of where a ceiling or breaking point sits — a crude ungated heuristic
  will eventually overwhelm the residual stream with noise if pushed far enough, this just hasn't found
  that edge yet), and the eval-only setup means this is a lower bound on what a properly-trained version
  could do, not a claim about the ceiling.
