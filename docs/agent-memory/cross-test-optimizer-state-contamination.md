---
name: cross-test-optimizer-state-contamination
description: "Multi-TEST_CASE training harnesses in ONE process silently warm-start each other: build_model re-inits WEIGHTS deterministically (fixed-seed init_weights) but the AdamW moment arenas g_param_m/g_param_v are GLOBAL and persist, so a test inherits the PRIOR test's optimizer momentum. Zero the Adam arenas (reset_opt_state) after build_model per test, else results are order-dependent + non-reproducible (both inflated AND deflated cases seen)."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Discovered 2026-07-11 while running the scratch-token spike ([[scratch-tokens-context-translation-layer]]).
A multi-TEST_CASE engine harness (train a small model per case, report metrics) gave ORDER-DEPENDENT,
NON-REPRODUCIBLE numbers: the same test read 1.000 in-suite but 0.643 run-alone.

**Why:** `sub0::build_model()` calls `g_model.init_weights()` which is DETERMINISTIC (a fixed-seed
`std::mt19937 rng(1234)`), so the WEIGHTS are identical every call -- the once-per-process randomize latch
was itself removed precisely so each TEST_CASE gets a fresh weight baseline (see backend_cpu.cpp's own
comment). BUT the AdamW optimizer's first/second-moment arenas (`g_param_m`, `g_param_v`, reached via
`adam_m_ptr()/adam_v_ptr()`) are GLOBAL and are NOT touched by build_model or by constructing a fresh
`sub0::AdamW opt(lr)`. So a test's optimizer starts with the PREVIOUS test's accumulated momentum -> it
warm-starts. Constructing a new AdamW object does not help: the momentum lives in the shared arenas, not
the object.

**It cut BOTH ways (this is the trap -- not a uniform bias you can mentally subtract):**
- INFLATED: a resolution test warm-started by a similar prior test converged to a spurious 1.000 (really
  ~0.31 cold).
- DEFLATED: a different-task test contaminated by a prior task's momentum got stuck near chance (really
  1.000 cold). This one REVERSED a "the model can't do it" conclusion into "it does it perfectly".

**Fix:** zero the Adam arenas after build_model in every TEST_CASE:
```cpp
void reset_opt_state() {
    const std::size_t n = sub0::trainable_floats();
    std::fill(sub0::adam_m_ptr(), sub0::adam_m_ptr() + n, 0.f);
    std::fill(sub0::adam_v_ptr(), sub0::adam_v_ptr() + n, 0.f);
}
```
After this, in-suite == run-alone == reproducible (verified). PRODUCTION stages are immune (each runs
build_model exactly once per process then load_checkpoint), so this is a TEST-harness hazard only.

**Standing lesson:** any in-process harness that trains multiple models must reset ALL shared training
state between runs -- weights AND optimizer moments (and any other global arena) -- and a result should be
confirmed RUN-ALONE before it's trusted, exactly like interleaving A/B trials for
[[thermal-confounds-ab-wallclock-testing]]. Order-dependence is the tell.
