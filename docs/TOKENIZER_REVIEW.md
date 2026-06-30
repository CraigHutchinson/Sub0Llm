# Tokenizer Code-Quality Review

Scope: the `sub0::tok` library — `include/sub0/casing.hpp` (reversible text transform +
truecasing), `include/sub0/tokenizer.hpp` (the `Tokenizer`, `Scan`, `LearnOptions` types) and
`src/tokenizer.cpp` (`learn`, `encode`/`detokenize`, the JOIN FSMs, (de)serialize). Goal: an
implementation that operates optimally, with SIMD/CUDA opportunities flagged for later. The
contract throughout is the round-trip `detokenize(encode(x)) == normalize_text(x)`, now guarded
by `tests/tok_lib_tests.cpp` (pinned) + `tests/tok_roundtrip_fuzz.cpp` (dogfood + 8000 fuzz).

This is a review + plan; the zero-risk annotations (SIMD/CUDA `TODO`s) are already in the code.
Ordered by value/effort.

## 1. Remove the legacy (non-JOIN) tokenizer path  *(major refactor; highest cleanup value)*

**Phase A — library + tests + configurator: ✅ DONE.** `learn` always mints the full 256-byte base +
markers; `encode`/`detokenize` collapsed to the single JOIN path; `deserialize` no longer scheme-detects;
`Tokenizer::join_scheme`, `LearnOptions::join_scheme` and the now-unused `sym_to_base` are gone. Tests
dropped the `.join_scheme = true` opt (JOIN is the only scheme) and the field asserts became `join_id >= 0`.
Verified: 224 engine-free assertions incl the 8000-case round-trip fuzz, and the configurator round-trips.

**Phase B — build/registry plumbing: ⏳ remaining.** Retire the now-vestigial flags: `SUB0_JOIN_TOKENIZER`
+ `SUB0_JOIN_FLAG` + the `--join` CLI (now metadata-only, the library ignores it), the `JOIN_TOKENIZER`
constexpr, `registry.hpp`'s `join_tokenizer` field / `j` tag / `compatible()` arg, and `train_stage`'s
pass-through. Needs a full build (engine + registry + the model-naming change) so it is its own commit;
update `frontend_tests` (the `rj` dir-name + the `compatible()` join arg) with it.

The original rationale (kept for context): the JOIN scheme is the default and **all old models were
discarded** (no backwards compat); the legacy "space-as-a-byte-token" scheme forked every core routine
behind a `join_scheme` flag, doubling the surface that must stay correct.

**Extent (what the legacy path touches):**
- `tokenizer.cpp`: the `else` branch in `learn` (corpus-derived partial alphabet, no markers),
  the `else` loop in `encode`, the `else` in `detokenize` (`casing::detokenize`), and the
  scheme-detection in `deserialize`.
- `tokenizer.hpp`: `LearnOptions::join_scheme`, `Tokenizer::join_scheme`, and `sym_to_base`
  (only the legacy `encode` uses it).
- `casing.hpp`: `detokenize` (the legacy inverse) becomes unused by the library — keep only if a
  test/tool still needs the marker-stream inverse, else drop.
- Ripple (build/tooling): `tools/configurator.cpp` `--join` flag + its legacy emission/round-trip
  branch; `CMakeLists.txt` `SUB0_JOIN_TOKENIZER` option + `--join ${flag}`; `registry.hpp`
  `join_tokenizer` field/tag/`compatible()` arg; `train_stage.cpp` passing `JOIN_TOKENIZER`.

**Plan:** make JOIN the *only* scheme in the library first (delete the branches; `learn` always
mints the full 256-byte base + 13 markers), update `tok_lib_tests` (its "current scheme" cases
currently exercise the legacy `learn(corpus)` with no opts), and confirm green with the
engine-free `sub0_frontend_tests` (fast, no GPU). Then retire the build/registry plumbing
(`SUB0_JOIN_TOKENIZER`, the `j` tag, `--join`) in a second commit, since that needs a full
configure+build. Net: ~one parallel code path and several conditionals deleted, no behaviour
change for current (JOIN) models.

## 2. DRY the encode/decode FSM  *(correctness robustness)*

`encode_join` and `detokenize_join` are hand-mirrored: the encoder tracks the decoder's
pending-space (`dps`), and any divergence is a silent round-trip bug (exactly the class the
fuzz harness exists to catch — e.g. the NASA's UP/apostrophe fix). The fuzz net makes this safe
today, but the two could share a single declarative **token spec** — for each special token: its
marker id, the literal bytes it emits, and its effect on `(pending_space, in_spell, recase)`.
The decoder becomes a table walk; the encoder picks the token whose spec realizes the next gap.
Lower priority than #1 (it's a structural nicety, well-fenced by tests), but it would make new
specials (the SPACE_N family, future bracket regions) additions to a table rather than edits to
two mirrored switch statements.

## 3. Performance — the corpus-tokenization hot path  *(SIMD/CUDA, flagged in-code)*

The configurator runs `normalize_text → truecase_tokenize → encode (→ bpe_encode_word)` over the
**entire corpus** (GBs) at configure time — this is the throughput-critical path, not per-prompt
encode. Annotated `TODO`s are in place:

- **`bpe_encode_word` memoization (biggest win, algorithmic).** It re-encodes every word
  occurrence from scratch at O(N²·merges); corpora are Zipfian, so a `word-bytes → ids` cache
  amortizes the per-unique-word cost to ~zero. The unique-word table already exists in `Scan`
  during `learn` — the emission path just doesn't reuse it. `TODO(perf, hot)` at the function.
- **`normalize_text` SIMD.** A per-byte scan for two rare lead bytes (0xE2, 0x60); a vectorized
  "find next 0xE2/0x60, bulk-copy the run" makes it near-memcpy on the >99% pass-through bytes.
  `TODO(simd, hot)`.
- **`truecase_tokenize` SIMD + CUDA.** Alpha-run/case classification is vectorizable (16/32 bytes
  at once via is_alpha/is_upper masks). The whole pipeline is embarrassingly parallel across
  newline-aligned chunks (already threaded), so a **CUDA tokenizer (one block per chunk)** is a
  natural later lever — the merge table + attested set are read-only and the work is regular.
  `TODO(simd/cuda, hot)`.

Runtime (per-prompt) `encode` is not hot; leave it simple. Keep `learn`'s incremental
heap-based BPE as-is (already optimized — see the configurator-ingest memory).

## 4. Smaller cleanups (do alongside #1)

- `Tokenizer` carries many parallel `*_id` marker fields; once legacy is gone, consider a small
  `markers` struct or an `enum`-indexed array so adding a special is one line, not four
  (field + learn mint + deserialize detect + decode case).
- `seq_key` / the per-word `std::string` key churn in `Scan` — fine for `learn`, but if `Scan`
  ever feeds a memoization cache (#3), share one keying helper.
- Marker codes are already named constants (`TOK_*`) — good; keep that discipline as the set
  grows (SPACE8/TAB8, bracket regions).

## Priority

1. **#1 legacy removal** — deletes a whole parallel path; engine-free tests cover it; do first.
2. **#3 `bpe_encode_word` memoization** — the real configure-time speedup; measurable via the
   configurator timing already printed.
3. **#3 SIMD** of `normalize_text` / `truecase_tokenize` — once memoization lands and BPE is no
   longer dominant, these byte scans become the next ceiling.
4. **#2 FSM table** — structural; schedule when adding the next special-token family.
5. **CUDA tokenizer** — large, later; aligns with the CUDA-first-for-iteration-time goal.
