---
name: autosize-research-grounded-formula
description: sub0::config::autosize() rewritten from a 5-bucket ladder to a formula grounded in real reference models (TinyStories paper, SmolLM2, FineWeb-Edu ablation), with a hardware-aware VRAM clamp, a --size-scale lever, and a per-corpus calibration ledger (not a running sum)
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Pass 4 (2026-07-07, `a3d2b1e`): calibration became a per-corpus LEDGER, not a running sum -- real
bug found live.** Pass 3's `TokenCalibration{total_bytes, total_tokens}` accumulated on every fresh
tokenization. Caught in the act: reconfiguring `ts_smoke` to pin dims matching an old checkpoint forced
a fresh re-tokenize of `tinystories.txt` (already counted once from the original seed), and the flat
accumulator silently ADDED its contribution again -- confirmed by an exact delta match
(+2,226,847,660 bytes / +667,617,584 tokens, exactly tinystories.txt's known real numbers). User
independently spotted the same class of risk before this was reported back: "the tokenizer_calibration
will keep getting this issue... we need it to store the info as a corpus ledger with the total so it
can be recalculated correctly as we train tinystories a lot at the moment." Redesigned:
`TokenCalibration` is now `std::vector<TokCorpusEntry{corpus, bytes, tokens}>`, keyed by
`std::filesystem::weakly_canonical(corpus)` (absolute, cwd-independent). `upsert_token_calibration()`
REPLACES an existing corpus's entry rather than adding to a total -- idempotent, so re-tokenizing the
same corpus from any number of build dirs counts it exactly once.
`bytes_per_token_calibrated()` still sums bytes/tokens ACROSS entries (sum-then-divide, unchanged
semantics, still correctly weights a huge corpus over a tiny one). File format: tab-separated
`path\tbytes\ttokens`, one line per corpus, `#`-comment header. Reseeded from the exact same real
tinystories.txt (2,226,847,660 B / 667,617,584 tok) + fineweb_edu.txt (46,119,439,400 B /
13,402,891,893 tok, back-derived from the pass-3 committed total minus tinystories.txt's known
contribution) -- combined ratio unchanged at 3.436. Verified end-to-end via the real
`sub0llm-configure` binary: fresh-tokenized a tiny throwaway corpus (`data/english.txt`) twice with
different pinned dims (forcing two independent fresh tokenizations, not cache hits) and confirmed its
ledger entry was upserted (replaced), not duplicated, both times -- 3 entries throughout, never 4.

**Two-pass change (2026-07-07), both committed.** Pass 1 (`e6feac5`'s predecessor, same-day) replaced
the old 5-bucket hand-tuned ladder with a smooth formula: Chinchilla-informed (20 tokens/param)
target-parameter budget -> width/depth decomposition (fixed aspect ratio 40, head_dim=64) -> Heaps'-law
vocab growth -> mild seq_len-vs-width scaling. Pass 2 (`e6feac5`) revised it after the user asked "how
do we know 128/196 isn't more correct for tinystories" and flagged a real hardware concern.

**Research findings (WebSearch/WebFetch, cited in `config_util.hpp`'s own doc comment):**
- [TinyStories paper](https://arxiv.org/abs/2305.07759) (Eldan & Li 2023): real reference configs keep
  `n_layers=8` FIXED from 1M to 28M params, only widening d_model (64/128/256) -- i.e. small models
  should be proportionally much DEEPER (narrower aspect ratio) than a fixed-ratio formula assumes.
  Models under 10M params already produce coherent short stories on this exact dataset. Vocab=10K
  (GPT-Neo tokenizer), context length 512.
- SmolLM2 family: 135M (d=576, 9 heads -> head_dim=64, "prioritize depth over width" strategy),
  1.7B (d=2048, L=24, 32 heads -> head_dim=64, aspect ratio ~85). Both trained WAY past
  Chinchilla-optimal: 135M on 2T tokens (~14,800:1), 1.7B on 11T tokens (~6,470:1) -- real small-model
  practice massively over-trains relative to compute-optimal because data is cheap relative to keeping
  the model small/deployable.
- [FineWeb-Edu paper](https://arxiv.org/html/2406.17557v1)'s own ablation model: 1.82B params, 350B
  tokens (~192:1) -- the closest-in-spirit reference point to this project (a research/ablation-scale
  run, not a dedicated mega-lab campaign), still ~10x above pure Chinchilla.

**Formula changes from pass 1 -> pass 2:**
- `kTokensPerParam`: 20 (pure Chinchilla) -> 100 (deliberately conservative middle ground vs. the
  192-14,800 range real practice actually uses, since this project's corpus is fixed/finite, not an
  unlimited web stream).
- Aspect ratio (d_model/n_layers): was a single fixed constant (40) -> now log-linearly interpolated
  in `target_params` between `kAspectMin=16` (TinyStories' own ~1M-param scale) and `kAspectMax=85`
  (SmolLM2-1.7B's own real ratio). Vocab and seq_len formulas unchanged from pass 1 (Heaps'-law /
  mild-width-scaling already reasonably grounded).
- Result at 22MB (TinyStories-paper-scale): now d128/L8 -- matches their real ~2.5M-param reference
  config almost exactly. At the real tinystories.txt corpus (~2.2GB): d256/L8 (~19M params) -- much
  closer to what real small-model research on this KIND of data uses than the pass-1 Chinchilla-strict
  formula's d448/L11 (~41M params) suggested.

**Real hardware-unreachability bug found and fixed:** the user flagged "the FineWeb model default is
unreachable on this hardware" -- confirmed CONCRETELY by pinning the pass-1 formula's real prediction
for this project's actual 46.1GB `fineweb_edu.txt` (d1216/L30/H19, ~652M params) against a fast test
corpus and running `sub0llm-configure`: it hard-failed with "the model needs ~11158 MiB for even batch
1 but only 8151 MiB VRAM is available." The revised (100:1) ratio alone already fixes THIS specific
case (real corpus now auto-sizes to d832/L14/H13, ~198M params, ~3.6 GiB) -- but a genuine
**hardware-aware VRAM clamp** was added as a real safety net for anything bigger or a smaller card:
`autosize(corpus_bytes, vram_mb, size_scale)` now takes an optional VRAM budget (0 = no-op, e.g.
CPU-only builds where GPU_VRAM_MB is correctly 0 per [[cpu-only-build-cuda-facts-leak-fixed]]) and
shrinks the shape ONE HEAD AT A TIME (recomputing everything derived from it; vocab held FIXED since
it's a separate axis and doesn't compete for the capacity budget) until the realized param count's
approximate footprint (calibrated against a real measurement: 651.9M params needed ~11,158 MiB, i.e.
~17.1 bytes/param; 18 used for margin) fits. Wired into `tools/configurator.cpp`: only applies when
`compute` RESOLVES to GPU (an explicit `--compute 0` on a CUDA-capable build correctly skips the VRAM
clamp, since that run trains on CPU/RAM instead). An explicit `--dmodel` override that intentionally
exceeds VRAM still hits the configurator's existing hard-error path downstream, correctly -- only the
AUTO suggestion is now hardware-aware. Verified end-to-end: `--size-scale 200` on the real 2.2GB
tinystories.txt corpus (deliberately forcing a huge target) clamped to d1344/L17/H21 and configured
successfully instead of hard-failing.

**`--size-scale` lever (user's own suggestion, "minimal vs ideal"):** a multiplier (default 1.0) on the
target-parameter budget -- same formula throughout, not a second scheme. Verified: vocab stays
untouched by the lever (correctly, since it's corpus-token-derived not capacity-derived); d_model moves
monotonically with the scale.

**n_heads plateau question (user's own follow-up):** does NOT need an explicit plateau mechanism --
`head_dim=64` stays FIXED across all scales (matching SmolLM2-135M through SmolLM2-1.7B's own real
architectures, both head_dim=64), so `n_heads = d_model/64` grows only as fast as `d_model` does, and
the existing `kMaxDModel=2048` safety clamp already implicitly caps n_heads at 32 -- which happens to
exactly match SmolLM2-1.7B's own real head count at a comparable scale. A separately-designed plateau
would just be reinventing what the existing bound already does correctly.

**Testing:** `tests/config_tests.cpp` -- exact-value regression pins at the real tinystories.txt/
fineweb_edu.txt corpus scales (not just synthetic round numbers), a dedicated VRAM-clamp test (confirms
monotonic-in-budget, vocab untouched, floor respected, a genuinely huge synthetic corpus clamps to fit
this machine's real 8151 MiB), and a size_scale test. All values were computed independently in Python
first (matching the exact C++ arithmetic) before being hardcoded as test assertions, then verified the
C++ implementation reproduces them exactly -- caught one real bug this way (the VRAM clamp's first cut
estimated the budget from target_params alone, missing vocab's own separate VRAM contribution, so the
realized footprint could still overshoot at extreme scale -- fixed by switching to a post-hoc iterative
shrink over the REALIZED shape instead of a one-shot pre-estimate).

**Pass 3 (2026-07-07, `ba6dea1`+`43260b9`): boy-scout reconciliation + persistent bytes/token
calibration.** User: "always resolve inconsistencies when found -- keep a clean ship, boy scout rules."
Found 3 more, all fixed, all by SHARING one definition rather than patching a second copy of the same
number (matching the pattern the `preview()` temp-mismatch fix already established this session):
- `sub0llm report`'s corpus-fit diagnostic (`train_stage.cpp`) still quoted raw "Chinchilla ~20
  tokens/param" verbatim and used an independently-hardcoded `C/40.0` aspect ratio in its "suggested
  next size" search -- both silently stale vs the revised formula. Fixed by hoisting `kTokensPerParam`
  and the aspect curve (as `aspect_for_params()`, extracted from a local lambda) to NAMESPACE SCOPE in
  config_util.hpp so both `autosize()` and `report` share the same definition. The data-rich/
  data-limited classification thresholds (were fixed 40/5) are now 2x/0.25x multipliers of the shared
  ratio, preserving the band's shape as its center moved.
- `report`'s "suggested next size" told users to run `cmake --preset native -DSUB0_D_MODEL=...` --
  flags that do nothing since SUB0_AUTO_CONFIGURE was removed (confirmed zero references anywhere in
  CMakeLists.txt). Replaced with the real current workflow: `sub0llm-configure --dmodel ... --layers
  ... --heads ...` then `cmake --build --preset native`.
- Verified live against a real trained model via `sub0llm report` -- all corrected messages/math check
  out (target-ratio text, 0.25x threshold classification, the real configure command, aspect=58 from
  the shared curve vs the old fixed 40).

Then, separately: `kBytesPerToken=4.0` (autosize()'s a-priori token-count estimate) was a generic
guess. User: "calculate a more accurate token-per-byte... use fineweb as the biggest reference... we
can iterate on improving this as we get more data" -- explicitly wanted PERSISTENT, ACCUMULATING
calibration, not a one-off constant bump. Built `data/tokenizer_calibration.txt` (committed, NOT
gitignored -- a project-wide shared prior, unlike the per-corpus `.model`/`.words` sidecars): a running
(total_bytes, total_tokens) sum, `TokenCalibration` struct + parse/format in config_util.hpp (pure,
matching the `.model` sidecar's own convention), `sub0::config::bytes_per_token_calibrated()`.
`tools/configurator.cpp` reads it before auto-sizing (falls back to 4.0 if absent) and APPENDS this
run's own real (norm_bytes, token_count) measurement to it right after a fresh full tokenization (never
on a cache-hit reuse -- re-configuring an unchanged corpus must not double-count it). Seeded with real
measurements from this project's own two corpora: tinystories.txt (3.336 B/tok, 2,226,847,660 B /
667,617,584 tok) + fineweb_edu.txt (3.441 B/tok, computed from corpus.tok's real size: 26,805,783,786 B
/ 2 [16-bit tokens, vocab=16489] = 13,402,891,893 tok) -> combined 3.436 B/tok, dominated by
fineweb_edu since it's ~20x the token volume (sum-then-divide, not an average of averages, so this
weighting happens automatically). Verified end-to-end: a fresh test corpus correctly read the seed for
sizing, then correctly appended its own measurement afterward (both totals grew by exactly its
contribution) -- reverted that test corpus's contribution afterward since it wasn't a real addition
worth keeping, restoring the clean two-corpus seed.
