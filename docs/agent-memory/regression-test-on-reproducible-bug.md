---
name: regression-test-on-reproducible-bug
description: "When a bug is reproducible via input/mocks, always add a regression test alongside the fix, not just the fix itself"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Whenever a bug can be reproduced through input or mocks (not just observed once in a live run), add
a test that pins the exact failing case before/alongside fixing it — don't just patch the code and
move on.

**Why:** the user's explicit standing instruction (2026-07-02). Fixes without a regression test are
silent regressions waiting to happen; the project already has strong test infrastructure (Catch2,
[[tokenizer-dogfood-fuzz-and-source-pathologies]]'s round-trip dogfood/fuzz) that makes this cheap.

**How to apply:** any time a Read/investigation/build run surfaces a concrete failing input (a crash,
a wrong output, a round-trip mismatch, an assertion failure), capture that exact input as a new test
case in the relevant suite before considering the bug closed. Example: the EOS-token decode bug found
2026-07-02 ([[eos-token-and-fixed-marker-scheme]]) — `detokenize_join` dropped a pending space before
a literal `<|endoftext|>` because the eos_id decode case didn't check `dps` like the general word path
and `odquote_id` do. The dogfood test caught it (source comments literally contain `<|endoftext|>`),
but the specific case (`"word <|endoftext|>"` with a single leading space) should also get a pinned
unit case in tok_lib_tests.cpp / tok_roundtrip_fuzz.cpp, not just rely on dogfood incidentally covering
it.
