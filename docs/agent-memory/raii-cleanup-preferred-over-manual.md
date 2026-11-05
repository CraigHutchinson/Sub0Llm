---
name: raii-cleanup-preferred-over-manual
description: "User preference -- use RAII guards for cleanup instead of manual calls at a function/test's tail end, since a thrown exception (e.g. Catch2's REQUIRE) skips manual cleanup silently"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Prefer RAII (a guard object whose destructor does the cleanup) over a manual cleanup call placed at
the end of a function/test body, whenever the code in between can throw or early-return.

**Why:** found concretely in tests/cuda_tests.cpp (2026-07-02): ~19 CUDA tests each called
`sub0_cuda_shutdown()` as their literal last line, after a body full of `REQUIRE(...)` assertions.
Catch2's `REQUIRE` throws on failure, so any test that failed BEFORE reaching its own tail-end
shutdown() left the cuBLAS handle (and its math mode / a captured CUDA graph) alive for whichever test
ran next in the same process -- a real, reproducible test-isolation bug masquerading as a numerical
regression (see [[cuda-padded-window-gradient-mismatch]], which looked like a broken kernel until this
was found). The user's explicit instruction on seeing this: "remember we prefer raii concepts to avoid
these issues if it's practicable here."

**How to apply:** when writing or reviewing code (not just tests) with a "setup, do risky things,
cleanup" shape, default to a guard type (destructor-does-cleanup) over a manual call at the end,
unless there's a concrete reason RAII doesn't fit (e.g., the cleanup itself needs to run at a specific
mid-function point under a condition RAII can't express cleanly, or the resource genuinely never
throws in between). This applies to test code as much as production code -- test cleanup skipped by a
failing assertion is exactly this pattern.
