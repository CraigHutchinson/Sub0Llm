---
name: resolve-preexisting-issues-blocking-our-work
description: "When pre-existing issues block or touch work we created, fix them rather than working around them"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2c2568c0-ff36-4396-a4a1-461041aa521f
---

When pre-existing problems (broken builds, failing/unbuildable tests, etc.) sit in the path of work we're doing, resolve them — don't just route around them and move on.

**Why:** The user considers a task incomplete if our work can't be properly validated because of adjacent breakage. Working around (e.g. building only one target to skip broken siblings) hides regressions and leaves the project in a worse state.

**How to apply:** If verifying our change requires a green build/test run and something pre-existing breaks it, fix that too as part of the task. Surface it, then resolve it — e.g. the ch21/ch22/ch25 chapter `main.cpp` compile errors that blocked `ctest --preset debug`.

**BROADENED boy-scout rule:** this  applies even when the issue does NOT block current work — compiler warnings (clang and nvcc), pre-existing test flakiness, latent perf/occupancy limits noticed in passing. If we're already in the file/build/test area and spot something wrong, fix or at minimum triage+record it in the same session rather than walking past it. Don't go out of scope hunting for unrelated issues across the whole codebase — this is "clean up what you touch or trip over," not a mandate to audit everything.
