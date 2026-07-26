# Corpus-scale word collapse — scaling wordspike's proven mechanism to real corpus text

Scope note: this is the design record for `corpus_collapse`, a train-time curriculum source that applies
`include/sub0/wordspike.hpp`'s proven harness-driven compound-word collapse mechanism to a sampled subset of
the REAL base corpus, instead of only a synthetic GSM8K-style template. It complements
[SCRATCH_TOKENS.md](SCRATCH_TOKENS.md) (the scratch-slot mechanism this builds on) and project memory
`wordspike-natural-prose-collapse`. Written first, before any code, per this project's own "prove the
mechanism before the engineering" sequencing — see the plan-mode review that produced this doc.

## Status: PLANNED, not yet implemented (2026-07-18)

## Why this exists

`SCRATCH_TOKENS.md`'s "Natural-prose word collapse" section (written when wordspike shipped, committed
09fa775) flagged a deliberately-deferred idea: apply the same collapse mechanism to the real base corpus,
"to teach the pattern at full corpus scale instead of a curriculum's few thousand examples." wordspike's own
`[.wordspike]` capstone since proved the mechanism composes cleanly with op-delegation (op accuracy 1.00 in
both FUZZY and COLLAPSE arms) — this is the gated follow-on that result unblocked.

## A scope correction made during planning

The original deferred note sketched this as a **configure-time** pass baking collapsed slots directly into
`corpus.tok` itself (`tools/configurator.cpp`'s tokenize pass). Investigating the actual pipeline found a
real constraint that changes the design:

- `corpus.tok`'s on-disk format (`include/sub0/tokmap.hpp:8-21`, v2 "S0T2") carries **no mask array at
  all** — a 32-byte header, byte-aligned token ids, and `doc_starts`. Every token loaded from it is always
  fully graded (`BlendSource::masked()`, `include/sub0/blend.hpp:53`, is `!mask.empty()`; the base-corpus
  source's mask span is always empty — `src/train_stage.cpp:1726`).
- Baking collapse directly into `corpus.tok` therefore forces a choice between (a) a breaking on-disk format
  change (new mask array, touching `TokWriter`/`TokView`/`TokMap` and every consumer), or (b) silently
  making the model actually GRADED on predicting scratch-slot emission from real corpus exposure — a
  fundamentally different, unproven capability claim. wordspike's own capstone deliberately never tested
  this: its slots are always `mask=0`, harness-injected, never a prediction target (see
  `wordspike.hpp`'s `EvalProblem` comment — an earlier version of that very capstone's eval assumed
  otherwise and was found to be testing an unsupported claim; masked positions carry zero gradient signal
  about what to predict there, in either arm).

**The lower-risk, still-real version**: a new train-time curriculum source that reads the corpus a run is
*already* training on (already-tokenized, already mmap'd) and applies wordspike's exact proven mechanism to
a sampled subset of its real documents — same architecture as `wordspike`/`scratchspike`/`op_curriculum`
(each builds an in-memory `Dataset` once per run and registers as a `BlendSource`), just sourcing real prose
instead of a synthetic template.

- No `corpus.tok` format change.
- No new capability claim beyond what wordspike already proved — this scales the PROVEN mechanism to more
  diverse, realistic text (real rare/compound words, not one template's invented names), which is real,
  testable value on its own.
- The bigger "teach spontaneous slot-prediction from broad real-corpus exposure" idea, and the
  configure-time/format-changing version, stay **explicitly out of scope** — recorded below as named,
  gated follow-ons, not attempted here.

## Masking discipline

Reuses the precedent from `op_curriculum.hpp`'s own result-collapse (`[op math] <result>`), **not**
`wordspike.hpp`'s/`scratchspike.hpp`'s more aggressive whole-mention masking:

- op_curriculum masks ONLY the injected result span itself; surrounding real/templated prose stays graded.
- wordspike/scratchspike mask essentially all prose immediately around a mention because their content is
  synthetic, arbitrary, and genuinely unlearnable either way (a random invented name, a control-symbol
  passage).
- Real corpus prose is normal, predictable, learnable language — masking it out would throw away real
  training signal for no reason.

**Rule: `mask=1` for everything (including a word's own first, spelled-out mention, kept as its real
tokens) except the exact position(s) where a later recurrence gets replaced by its slot token (`mask=0`
there).**

## The mechanism

New header `include/sub0/corpus_collapse.hpp` (engine-free, mirrors `wordspike.hpp`'s shape):

```cpp
namespace sub0::corpus_collapse {
struct Dataset {
    std::vector<int> tokens;
    std::vector<std::uint8_t> mask;
    std::vector<std::uint64_t> doc_starts;
    std::vector<std::vector<std::vector<int>>> doc_bindings;
};
struct Options {
    std::uint64_t seed = 0;
    int n_docs = 3000;
    int max_doc_tokens = 0;   // caller derives from SEQ_LEN; 0 = no cap (test-only)
};
Dataset build_dataset(const tok::Tokenizer& tk, sub0::TokView base_tokens,
                       std::span<const std::uint64_t> doc_starts, std::size_t train_tok,
                       const Options& opt);
}
```

`build_dataset`:

1. Samples up to `opt.n_docs` document indices uniformly at random (via `opt.seed`) from `doc_starts` — a
   production corpus can have millions of documents; building from all of them every run would be far too
   expensive/memory-heavy, matching every other generator's own bounded `n_examples`-style cap.
2. For each sampled document, clamps its end to `train_tok` — mirrors `window.hpp:56`'s own
   `if (de > train_tok) de = train_tok;` exactly. `doc_starts` is the FULL corpus's boundary list while
   `train_span`/`train_tok` (`src/train_stage.cpp:1280-1328`) is the truncated training region (the tail is
   held out for validation) — this clamp is load-bearing, not optional, or a document straddling the split
   would leak val-region tokens into the training curriculum.
3. **Prefix-truncates** any document longer than `opt.max_doc_tokens` rather than skipping it — preserves
   yield from long documents. `sample_window` (`window.hpp:59-63`) snaps an arbitrary in-document crop for
   any document longer than the window width; without this cap, a sampled training window could show a
   collapsed slot with no preceding spelled mention anywhere in view. Every other generator's documents are
   always short enough that this case never arises — real corpus documents are the first source where it
   can (see Gaps below — this is a real, accepted v1 limitation, not silently glossed over).
4. Reads each document's tokens out of the `TokView` (byte-packed 2/3/4-byte ids, `tokmap.hpp:62-97` — not
   a raw `int` span) into a small `std::vector<int>` before walking it.
5. Per document: a **fresh** `ScratchTable table; table.tk = &tk;` (reset per document — recurrence must
   never leak across unrelated documents; `table.tk` must be set or `combine_recurrence` would wrongly treat
   every ordinary single-piece word as a bind candidate instead of recognizing it via `tk->piece_index`).
   Walks the span via the EXISTING `sub0::detail::word_span` + `table.combine_recurrence`
   (`include/sub0/scratch.hpp` — both already built for `prefill_collapse`, zero new low-level logic
   needed; this is the same walk, just per-document instead of over one flat prompt). Emits `mask=1` by
   default, `mask=0` only at a substitution. After each document,
   `ds.doc_bindings.push_back(std::move(table.bindings))` — confirmed safe independently of how a
   document's own table was built (`blend.hpp`'s `doc_of`/binding lookup,
   `train_stage.cpp:1975-1980,2060-2063`, only needs one bindings-vector per document, in document order).

## Wiring

Mirrors wordspike's own wiring, plus two extra gates a design review surfaced (both would silently no-op
this feature if skipped):

- **`src/train_stage.cpp`**, the existing source-building loop (~1808, alongside `scratchspike`/
  `op_curriculum`/`wordspike`): new `else if (spec.generator == "corpus_collapse")` branch. Reuses the
  ALREADY-COMPUTED `train_span`/`doc_index` locals from earlier in `sub0_train_stage` (`:1280-1328`) — no
  separate corpus load, this runs against the SAME corpus.tok the run's mandatory "base" source already
  uses. **Explicit `doc_index.empty()` check → hard error (`return 1`), not a warn-and-skip**: the
  `on_demand`/no-corpus.tok fallback path never populates document boundaries, and a source with a
  zero-length view can make the WFQ scheduler's `pick_source_staged` pick it forever (a `size == 0` source
  normalizes to `0.0` and ties always resolve to it), which would then hand `sample_window` a `train_tok`
  of 0 — `train_tok - full` underflows to a huge `size_t` and the returned window indexes a zero-length
  `TokView` out of bounds. A loud, explicit refusal is the only safe response here; there is no meaningful
  degraded behavior to fall back to.
- **`src/train_stage.cpp:1572`**, `have_binding_source` lambda: add `|| s.generator == "corpus_collapse"`.
  Without this, `content_embed_active` never turns on for a schedule using only `corpus_collapse`, and
  every slot gets a generic, content-blind learned embedding — silently defeating the whole point, since a
  content-derived embedding is what lets the SAME slot id mean something different per document.
- **`src/gen_stage.cpp:161`**, `word_on`: add `|| generator_ever_active(schedule, "corpus_collapse")`.
  Without this, a model trained with `corpus_collapse` but no `wordspike` source never gets
  `prefill_collapse`/`word_collapse` wired at real generation time — silently leaving the live mechanism
  (already proven, already shipped) untested for exactly the models this effort trains.
- **`include/sub0/blend_schedule.hpp`**: extend the `generator` field's comment list; reuse the EXISTING
  `n_examples` knob for `n_docs` (no new `SourceSpec` field, no `blend_schedule.cpp` parser changes needed —
  there is no generator-name allowlist there; validation is purely the `else if` chain in
  `train_stage.cpp`).
- **`docs/SCRATCH_TOKENS.md`**: correct the framing of the existing deferred-idea paragraph — this is a
  train-time sampled curriculum sourced from real corpus text, not a configure-time, format-changing,
  literal-full-corpus pass; the configure-time version stays noted as a further, separate promotion gated
  on this proving out first.

## Gaps, risks, and extensibility

Reviewed before implementation, not discovered after:

- **WFQ re-covers a small sampled subset repeatedly, not the whole real corpus.** `corpus_collapse`'s
  `n_docs`-bounded sample (e.g. 3000 documents) is tiny next to the base corpus it's drawn from. Once the
  WFQ scheduler (`blend_schedule.hpp`'s own header comment, written for exactly this shape of problem — "a
  small curriculum blended against a huge base corpus... hit a memorization phase-transition almost
  immediately") finishes one epoch over this source, further draws re-cover the SAME sampled documents — a
  real memorization risk on those specific documents' specific collapse patterns, not just a pacing problem
  WFQ already solves. v1 accepts this (matches every other synthetic generator's own "build once per run"
  precedent) but the capstone should explicitly watch for it (val_nelbo trend over the LATER portion of a
  longer training run, not just an early snapshot). **Extension**: periodic re-sampling of a fresh document
  subset (every N steps or every schedule stage) for broader real coverage over a long run — not needed for
  v1, a natural follow-up knob.
- **Prefix truncation vs. where recurrence actually lives in long documents.** Truncating a document to fit
  one window (`max_doc_tokens`) means only the document's OPENING portion is ever collapsed — but in real
  prose (news articles, longer stories), a name is often introduced once near the start and referenced many
  times much later, well past the truncation point. v1's simple prefix truncation is safe and correct but
  systematically UNDER-exercises exactly the long-range recurrence pattern this whole effort cares about. A
  more sophisticated alternative (collapse the FULL document first, THEN window-crop the RESULT)
  reintroduces the same "does this crop's start show the establishing mention" problem in a different
  place, and isn't a clean win — a known, documented v1 limitation and a named extension point, not solved
  now.
- **Collapse quality on messy real text — validate qualitatively, not just PASS/FAIL.** Real corpus text
  will surface `TOK_SPELL_START/END`/`TOK_JOIN`-marked spans wordspike's clean synthetic names never did:
  URLs, malformed tokens, abbreviations, foreign words, code-like fragments embedded in prose. Exact
  byte-match recurrence (`combine_recurrence`) can't conflate different words, so this is safe, but could
  waste slot budget on low-value recurring fragments instead of meaningful entities. **Verification
  addition**: the capstone should log/report a SAMPLE of what actually got collapsed, for a human sanity
  check alongside the quantitative val_nelbo numbers — not accuracy alone, per the three-pillar shootout
  policy. Also expect (and treat as fine, not a bug) some real documents to exhaust the 6-slot pool —
  `combine_recurrence`'s existing "pool full → identity, not a repeat" fallback already degrades
  gracefully.
- **Explicitly NOT the same substrate as the other slot-related backlog items.** This design deliberately
  reuses wordspike's BOUNDED, per-document-reset 6-slot pool — it is not a step toward the separate,
  already-partially-built "persistent slot-range" substrate (unbounded ids ≥ VOCAB, cross-document, its own
  "compound-word cache/DB still not built" backlog item — project memory
  `persistent-slot-range-engine-substrate`). Good sign for future extensibility:
  `detail::word_span`'s detection logic and `combine_recurrence`'s piece_index-based ordinary-word exclusion
  are already cleanly separated from the bounded-pool STORAGE (`ScratchTable`) they currently write into —
  if/when the persistent substrate's own cache gets built, the SAME detection walk should be pluggable into
  that unbounded storage with no rewrite, only a different `combine_recurrence`-shaped adapter.
- **The natural "if this works" next step**: promoting this from a train-time sampled curriculum to a real
  configure-time, format-changing pass (baking collapse into `corpus.tok` itself, at full corpus scale, not
  a 3000-document sample) — now informed by actual empirical results from this cheaper version first,
  exactly the "prove the mechanism before the engineering" sequencing this whole document family already
  follows. Gated on this plan's own TinyStories capstone result.
- **Resume/reproducibility**: confirm during implementation that `corpus_collapse`'s document sampling
  reseeds identically on a resumed run (matching every other generator's `seed ^ magic-constant` precedent)
  rather than silently drawing a different subset — a resume-time divergence here would be a real, subtle
  correctness gap, not a style nit.

## Verification plan

- **`tests/corpus_collapse_engine_tests.cpp`**, fast + always-run (engine-free, no model): hand-built
  `TokView`/`doc_starts` fixtures proving (a) two documents each containing the same repeated OOV do NOT
  collapse against each other (per-document table reset); (b) the `train_tok` val-split clamp is respected;
  (c) a document longer than `max_doc_tokens` is truncated, not dropped, and the truncated span's
  mask/tokens stay consistent; (d) mask is 1 everywhere except exactly the substituted slot position(s).
- **`[.corpus_collapse]` hidden capstone on TinyStories** (this project's own default analysis corpus) —
  matched-budget A/B, base-only vs base+corpus_collapse, reporting per the standing three-pillar shootout
  policy: correctness (val_nelbo delta — does adding this source measurably help or hurt ordinary graded
  prediction on the SAME real documents; not accuracy alone), performance (wall-clock for `build_dataset` at
  a couple of `n_docs` sizes), memory (peak `Dataset` footprint vs corpus.tok's own mmap footprint). Do not
  touch the production `fineweb_edu_workflow` corpus/schedule until this is clean.
- Full `ctest`/`sub0_tests` regression suite green on both CPU-only (`tok_cpu`) and CUDA (`gsm8k`) build
  configs.

### Implementation order

1. This doc, first.
2. `include/sub0/corpus_collapse.hpp` + `tests/corpus_collapse_engine_tests.cpp` (fast unit tests green
   before touching train_stage/gen_stage).
3. Wire `src/train_stage.cpp` (source-loop branch + `have_binding_source`) and `src/gen_stage.cpp`
   (`word_on`).
4. `include/sub0/blend_schedule.hpp` comment, `docs/SCRATCH_TOKENS.md` framing correction.
5. `[.corpus_collapse]` capstone on TinyStories; update this doc's status log with the real result before
   considering any production schedule change.

## Status log

- **2026-07-18**: design planned and reviewed (this doc). Not yet implemented.
- **2026-07-18 (later same day)**: IMPLEMENTED and validated. `include/sub0/corpus_collapse.hpp` +
  `tests/corpus_collapse_engine_tests.cpp` (4 fast unit tests, always green) written and wired into
  `src/train_stage.cpp` (source-loop branch, `have_binding_source`) and `src/gen_stage.cpp` (`word_on`).
  One refinement made during implementation: `doc_index.empty()` is a **hard error** (`return 1`), not the
  originally-planned warn-and-skip -- a zero-size `BlendSource` can make the WFQ scheduler's
  `pick_source_staged` pick it forever (ties resolve to a `size==0` source's always-`0.0` normalized
  progress), which then hands `sample_window` a `train_tok` of 0 and underflows to an out-of-bounds
  window. A loud refusal is the only safe response; there is no meaningful degraded fallback here.

  **`[.corpus_collapse]` capstone result** (TinyStories, `out/build/d196check`, D_MODEL=196, matched
  total budget of 600 steps/arm, equal-weight base:collapse blend -- a deliberately AGGRESSIVE exposure,
  see the Gaps section above):
  - **Correctness**: held-out NELBO base-only = 2.9315, base+collapse = 2.9454 (delta +0.0139 nats). A
    small, plausibly noise-level increase from a single-seed, 600-step run under an aggressive 50/50
    exposure to a small (98K-token) sample -- not a red flag; well inside the `< base + 0.5` gate.
  - **Performance**: `build_dataset` over 2000 sampled real documents (98,456 tokens after collapse) took
    235.2 ms -- a negligible one-time cost at the start of a real training run.
  - **Memory**: the sampled Dataset is ~0.47 MB, against the base corpus's ~2.49 GB of tokens (mmap'd,
    not resident) -- utterly negligible footprint.

  Full regression suite green on both CPU-only (`tok_cpu`, 123 cases) and CUDA (`gsm8k`, 162 cases) build
  configs after wiring. **Not yet run against the production `fineweb_edu_workflow` schedule** -- per this
  doc's own gate, that step is still pending a deliberate decision, not something to do by default.

- **2026-07-19: root-caused and fixed a real mistargeting bug the qualitative check above should have
  caught immediately.** Asked to show real examples, a standalone diagnostic against real TinyStories text
  revealed the mechanism was NOT targeting rare/OOV entities: 1548/1693 sampled documents (91%) hit a
  collapse candidate, and the actual examples were ordinary common words with trailing punctuation
  (`<~room.~>`, `<~stream.~>`, `<~said,~>`), not names. Root cause, confirmed by measuring a 5M-token real
  sample: `encode_join`'s old 2-piece-word encoding (a bare `TOK_JOIN` between the two piece ids, no
  wrapping markers) was byte-identical to an ordinary single-piece word glued to trailing punctuation via
  the SAME general-glue `TOK_JOIN` used everywhere else in the tokenizer. Measured split of that shape:
  853 unambiguous `TOK_SPELL_START..END` words (N≥3) vs 576,013 occurrences of the ambiguous 2-piece shape,
  of which only 6.1% (35,339) were genuine word splits and 93.9% (540,674) were the false positive.

  **Fix** (`casing.hpp`'s `kSchemeVersion` 2→3, a foundational tokenizer change, not scoped to this file):
  every multi-piece word (N≥2, not just N≥3) now gets `TOK_SPELL_START..END` wrapping; `TOK_JOIN` narrows
  to general glue only, never a word-boundary signal. `sub0::detail::word_span` (`scratch.hpp`) simplified
  to match (the ambiguous 2-piece branch is gone). Re-tokenizing TinyStories confirmed no vocab/relearn
  impact (same 16535-token vocab, 21s to re-tokenize 671M tokens) and the fix measured exactly as
  predicted: `TOK_SPELL_START..END` occurrences jumped 853 → 35,342 (absorbing the 35,339 genuine 2-piece
  words), and the residual `TOK_JOIN` shape dropped to 99.9% pure glue.

  **Re-run `[.corpus_collapse]` capstone result** (same TinyStories build, re-tokenized, same seed/budget):
  documents hitting a collapse candidate dropped from 1548/1693 (91%) to 401/1693 (24%); real examples now
  show genuinely recurring nouns (`<~dive~>`, `<~clay~>`, `<~oak~>`, `<~reef~>`) and, notably, actual proper
  names collapsing correctly on their second mention (`<~Beth~>`, `<~Jase~>`, `<~Mae~>` twice). Three-pillar
  re-measurement: correctness — held-out NELBO delta flipped from +0.0139 (collapse slightly hurt) to
  **-0.0075** (collapse slightly helped, both plausibly noise-level for a single-seed run, but no longer
  clearly negative); performance — `build_dataset` dropped from 235.2ms to **25.4ms** for the same 2000
  sampled documents (far less spurious binding work with the false positives gone); memory — unchanged,
  ~0.47MB. A design-review subagent separately caught that the TOKSTAMP reuse-cache assumption "the
  tokenizer scheme is always JOIN, so it is not a variable" (`configurator.cpp`) was now false and would
  have silently kept serving stale wrong-scheme `corpus.tok` files with no warning -- fixed alongside (see
  `tools/configurator.cpp`'s `TOKSTAMP_VERSION` 1→2).

  Full regression suite re-confirmed green on `tok_cpu`, `gsm8k`, and `d196check`/TinyStories after the
  tokenizer fix. **Lesson worth keeping**: asking to see real examples, not just trusting an aggregate
  metric, is what surfaced this -- the original +0.014 NELBO delta alone looked like a mundane "no harm"
  result and gave no hint the mechanism was mistargeting by this much.

- **2026-07-21, SPELL-marker cross-check (docs/FACTSPIKE.md's own investigation)**: a toy-model spike
  (factspike, d96) found that `word_span` -- the SAME extraction this file's own `build_dataset` calls
  (line ~119) -- unconditionally strips `TOK_SPELL_START`/`TOK_SPELL_END` from what gets bound to a
  collapsed slot, and that these markers are fully load-bearing for exact representational reconstruction
  (a controlled swap-and-remeasure closed a 0.87-0.96 cosine-similarity gap to exact 1.000, zero
  exceptions, once markers were included). Since this file is a direct, first-class caller of that same
  extraction, added `build_dataset_markers` (marker-inclusive, structurally identical to `build_dataset`
  except the bound span includes the wrapper markers) and re-ran the `[.corpus_collapse]` capstone as a
  3-arm A/B/C (base-only / base+collapse / base+collapse_markers), same TinyStories build/budget as above.

  **Result: no detectable difference.** Held-out NELBO identical to 4 decimal places between
  marker-stripped and marker-inclusive arms (both -0.0075 vs. base-only). This is a real, honest negative
  for THIS metric at THIS budget -- not evidence the SPELL-marker finding is wrong, but evidence that its
  effect (measured, in the toy model, via a surgical single-position cosine-similarity probe) doesn't
  show up in an aggregate NELBO averaged across a whole validation set, where only 401/1693 documents even
  contain a collapse-eligible word. Read alongside factspike's own follow-up (Phase H): compression-based
  mechanisms (this file's own `combine_recurrence`+byte-fragment binding, structurally the same
  fixed-capacity-bundle shape as factspike's mechanism A) getting WORSE, not better, when given the same
  markers to compress suggests the relevant bottleneck for THIS kind of mechanism was never "which tokens
  get bound" -- it's the fixed-size compression step itself. Not a reason to revert this file's own
  marker-stripped convention (no evidence it's currently harming production NELBO), but a flag that a
  future compression-mechanism change should account for this rather than re-litigate it from scratch.
  Full detail: `docs/FACTSPIKE.md`'s own entry, `docs/SCRATCH_TOKEN_FRAMING.md` axis 9.
