---
name: corpus-tok-reuse-stamp
description: sub0llm-configure now skips the scan/learn/tokenize pass and reuses existing corpus.tok+tokenizer.tok when a stamp confirms the corpus + vocab_target + min_merge + emit_tok are unchanged. ~20s -> ~0.1s on a re-run that only touches an unrelated knob.
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User-driven feature request: "the corpus tokenization is not depending on all the settings of the
model but presently we re-process even when there is already a valid tokenizer and corpus - could we
consider reusing existing when we detect that they are from the same mould?" Real pain point: a
`sub0llm-configure` re-run made ONLY to change an unrelated knob (a `--dmodel` pin, `--prec-gemm`,
tune-cache) unconditionally re-tokenized the whole corpus, which for a FineWeb-scale (46GB) corpus
means ~19s of Pass-3 tokenize dominating the run every single time.

**Design**: tokenizer.tok/corpus.tok are pure functions of exactly three inputs beyond the corpus
bytes: `vocab_target`, `min_merge`, `emit_tok` (the JOIN tokenizer scheme is the only one, not a
variable). Added a stamp file (`generated/tokenizer.stamp`, `tools/configurator.cpp`, `TOKSTAMP_MAGIC`
"S0TS") recording corpus size+mtime + those three inputs, following the SAME validated-by-size+mtime
pattern the pre-existing `<corpus>.words` scan cache already used (`save_scan_state`/
`load_scan_state`, not touched). On a fresh run, `tok_stamp_matches()` checks the stamp; if it matches
AND `tokenizer.tok` deserializes AND (when `emit_tok`) `corpus.tok` opens via `TokMap` with a matching
vocab, the ENTIRE scan+learn+Pass-3-tokenize block is skipped -- just loads the existing tokenizer and
reports a short summary instead of the full diagnostic dump (which needs scan-derived stats that
don't exist on the reuse path).

**Verified end-to-end** against the real `data/tinystories.txt` (currently a 2.2GB corpus, not the
small canonical TinyStories set -- see [[tinystories-default-analysis-corpus]] for context if that
matters later): fresh run ~20s (BPE-learn ~1s + Pass-3 tokenize ~19s), immediate re-run with identical
args ~0.1s (stamp hit), re-run with an unrelated `--prec-gemm 1` flag added ~0.1s (still hits --
confirms the actual fix works), re-run with `--vocab 8192` correctly falls through to a full ~19s
regen (stamp correctly invalidates on the inputs that actually matter).

**Implementation gotcha worth remembering**: the original code declared `const tok::Tokenizer tkz =
tok::learn(...)` INSIDE what became the `else` (fresh-path) branch, shadowing an outer `tok::Tokenizer
tkz;` declared before the reuse-check -- this compiles fine (shadowing is legal C++) but silently
means the outer `tkz` used by every downstream consumer (header emission, Pass 3, the final summary
line) is NEVER populated on the fresh-learn path. Caught before ever running it, by re-reading the
diff line-by-line rather than trusting the restructure was mechanical. The general lesson: when
hoisting a variable out of a branch to share across an if/else, grep for every `const <Type> name =`
of that SAME name still inside the branches -- a re-declaration silently defeats the hoist instead of
erroring.

Not yet done: no dedicated unit test for `tok_stamp_matches`/the reuse fast-path (verified only via
real CLI invocations, not `tests/config_tests.cpp` or `tests/frontend_tests.cpp`) -- worth adding if
this area gets touched again. Related: [[configurator-ingest-optimization]] (the scan-cache /
parallel-streaming work this builds on) and [[sub0-auto-configure-removal]].
