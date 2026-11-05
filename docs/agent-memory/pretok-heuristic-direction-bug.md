---
name: pretok-heuristic-direction-bug
description: should_pretokenize (config_util.hpp) assumed corpus.tok is ~2x the source size; real measurement on two different corpora shows it's ~0.6x. Fixed to 0.75x conservative estimate. Caught via a real production configure log.
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User caught this from a real `sub0llm-configure` log line on the production `fineweb_edu_workflow`
run: `corpus.tok: --corpus-pretok=AUTO -> skip (on-demand) (corpus 46119 MB est-tok 92239 MB vs 34056
MB half-RAM)` -- the estimated corpus.tok size (92GB) was exactly 2x the corpus (46GB), and that 2x
assumption is what pushed AUTO to skip pre-tokenizing a corpus that in fact fits fine.

**Root cause**: `sub0::config::should_pretokenize` (`include/sub0/config_util.hpp`) computed
`est_tok_bytes = corpus_bytes * 2` with the comment "corpus.tok ~ 2x the source text". Backwards --
corpus.tok is SMALLER than the source, not larger. The v2 on-disk format (`tokmap.hpp`) packs 2
bytes/token for any vocab up to 65536 (every entry in `autosize()`'s dims ladder stays under that),
and real corpora in this project compress to ~3.3-3.8 source bytes/token (JOIN scheme + Unigram
pieces) -- so corpus.tok lands around 0.55-0.6x the source size. Confirmed with two independent real
measurements: the 46.1GB fineweb corpus produced a 26.7GB corpus.tok (13,364,203,461 tokens × 2
bytes), and a 2.23GB corpus (this repo's current `data/tinystories.txt`, actually FineWeb-scale
despite the name) produced a 1.34GB one (667,617,584 tokens × 2 bytes) -- both landing at ~0.58-0.6x.

**Likely origin of the wrong "2x"**: probably inherited from an earlier era where corpus.tok used
4-byte int32 tokens (the legacy "S0TK" format `tokmap.hpp` still reads for backward compat) at a
lower-compression tokenizer, where 4 bytes/token ÷ ~2 source-bytes/token really would give ~2x. The
v2 byte-aligned format + the Unigram/JOIN scheme's much better compression invalidated that ratio, but
the comment/constant were never revisited.

**Fix**: `est_tok_bytes = corpus_bytes * 3 / 4` (0.75x) -- a deliberately conservative upper bound
above the ~0.6x actually measured (covers a corpus that compresses worse than this project's prose
corpora), but far closer to reality than 2x. The same wrong "~2x-on-disk token copy" phrasing had
propagated into three other comments/CLI help strings (`configurator.cpp`'s `--corpus-pretok` help
text and Pass-3 comment, `train_stage.cpp`'s on-demand-tokenization comment) -- all fixed to reference
the corrected ~0.6x figure and point at `should_pretokenize` as the source of truth.

**Not yet done**: no unit test added for `should_pretokenize` (it's flagged in its own comment as
"pure... unit-testable" but has zero test coverage in `tests/config_tests.cpp` currently) -- worth
adding given this was a real, silently-wrong heuristic for a while. Related:
[[model-sizing-levers-from-corpus-stats]], [[corpus-tok-reuse-stamp]].
