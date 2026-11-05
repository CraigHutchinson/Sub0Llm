---
name: cuda-padded-window-gradient-mismatch
description: "Full-suite-only CUDA gradient/logit-diff test flakiness (marginal cosine/tolerance checks pass isolated, fail mixed-suite) -- 2026-07-02 instance RESOLVED (CudaGuard); 2026-07-06/2026-07-09 recurrences ROOT-CAUSED AND FIXED (c713f0b): build_model()'s once-only weight randomization, not a hardware/thermal effect"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Found (2026-07-02)** while running the full `sub0_tests` suite on ts_gpu after unrelated tokenizer
changes: `tests/cuda_tests.cpp` "CUDA backward matches the CPU gradient with short padded windows"
fails --  `padded grad rel-L2 = 0.96354  cos = 0.387049  loss = 20.3627`, well under the BF16 branch's
`cos > 0.7` gate. The test batches 4 windows of mixed length (8/5/3/7 at T=8) and checks the GPU's
masked/padded-window gradient agrees with the CPU reference (padding should contribute nothing to the
gradient).

**Confirmed pre-existing, not a regression from this session's work:** `git diff`/`git log` show
neither `tests/cuda_tests.cpp` nor `src/backend_cuda.cu` were touched in this session (which only
changed tokenizer.hpp/cpp, casing.hpp, configurator.cpp, gen_stage.cpp, engine_core.cpp/core.hpp, and
test files for the EOS feature -- see [[eos-token-and-fixed-marker-scheme]]). Last commit touching
those CUDA files is 528c32a ("eliminate the dq kernel's register spill via a warp-cooperative channel
split"), from earlier in the same day's session, before the EOS thread started.

**Not investigated further** -- flagged to the user, not chased, since it's outside the session's
active focus (EOS feature, then a training-loop GPU-occupancy investigation, see
[[training-step-host-device-overlap]] if that memory exists yet). The unusually high loss (20.36, vs
the ~7-10 range typical of untrained-model tests elsewhere in the suite) may be a clue worth checking
first when this is picked up -- could indicate the padding mask is leaking un-masked logits/targets
into the loss for this specific short-window batch, not just a numerically-noisy gradient.

**RESOLVED (2026-07-02, later same day):** NOT a kernel bug. Root cause was cuda_tests.cpp test
isolation: `REQUIRE` throws on failure (Catch2), so any test whose own `sub0_cuda_shutdown()` sat at
the tail end left the cuBLAS handle (and whatever math mode / captured graph it held) alive for
whatever test ran next in the same process, if an earlier `REQUIRE` in THAT test threw first. Fixed
with an RAII `CudaGuard` (destructor calls `sub0_cuda_shutdown()`) declared at the top of every CUDA
test — see [[raii-cleanup-preferred-over-manual]]. This test now passes reliably (cos=0.94-0.9999
across runs, was 0.39 intermittently). Discovered as a side effect of the lm_head tensor-core
investigation ([[training-step-nsys-occupancy-findings]]), not independently chased.

**RECURRENCE, same symptom class (2026-07-06, found during [[qk-norm]] CUDA verification):** full
mixed-file suite on both the qk-norm-off default build AND the pre-session baseline binary shows 3
marginal gradient-cosine `REQUIRE`s failing (`cuda_tests.cpp` "attention-only gradient", "tied-embedding
gradient" region, "AdamW step" — cos landing at 0.53-0.68 against a `>0.7` gate) plus the logit-diff
tests (0.019-0.024 against a `<0.01` gate) -- but every one of these passes cleanly when the `[cuda]`
tag is run alone (30/30, both scales) or when the single failing test case is rerun in isolation with a
fresh RNG seed. **Confirmed this time it is NOT a missing-CudaGuard regression** -- grepped
`cuda_tests.cpp`, every single `TEST_CASE` in the file already declares its own `CudaGuard`, including
all the tests added this session. Working hypothesis (not yet confirmed): these specific gradient/logit
checks use untrained random-init weights and sit close to their own tolerance boundary by nature: cuBLAS
FP32 algorithm selection is warm-state-dependent (confirmed separately by [[qk-norm]]'s own
investigation for the logit-diff pair), so running after a full suite's worth of prior GPU workload can
shift the selected algorithm/clock state enough to move a marginal cosine from ~0.85 down to ~0.55 for a
sensitive (near-degenerate, untrained) gradient. This is a DIFFERENT mechanism than the original
cuBLAS-handle-leak bug above (that one was cross-test STATE persistence; this one looks like cross-test
WARM-STATE/thermal sensitivity hitting an inherently marginal tolerance) -- not chased further, flagged
as open follow-up. If picked up again: try (a) tightening these tests' random init to a less-degenerate
starting point so the cosine has more margin, or (b) forcing a fixed cuBLAS math mode in the test binary
so algorithm selection stops depending on prior warm-up.

**ROOT-CAUSED AND FIXED (2026-07-09, commit `c713f0b`) -- the "warm-state/thermal" hypothesis above was
WRONG.** Same symptom hit again (3 tests: 2 forward-parity logit-diff checks + the batch-grow/graph-
recapture test, all in the `sub0_tests` full-suite run, all passing clean in isolation) while working
through the [[gpu-failure-detection-hardening]] nodiscard pass, this time actually bisected instead of
flagged-and-moved-on: confirmed deterministic reproduction (`[cuda]`-tag-only run = always green; full
suite = always the same 3 failures) by running progressively larger prefixes of the full test list ahead
of the failing tests via Catch2's comma-OR filter syntax, then reading `sub0::build_model()`
(`backend_cpu.cpp`). Found the real bug: `build_model()` gated `g_model.init_weights()` behind
`std::atomic<bool> g_params_init` -- a "randomize once per PROCESS" latch. Every production caller
(train/gen/report/...) genuinely calls `build_model()` exactly once per process, so the latch was
invisible there. But EVERY TEST_CASE in `cuda_tests.cpp`/`engine_tests.cpp` calls
`sub0::build_model()` expecting "random-init params" (the literal comment at the call site) -- after
the first TEST_CASE in the whole `sub0_tests` binary, every later `build_model()` call was a silent
no-op for weights, so "fresh model" tests were actually running against whatever gradient/AdamW/Muon-
step or saturating-activation mutations EARLIER tests had left in the shared param arena. Params
drifting far enough from init-scale (0.02 std) is exactly what pushes a bf16-tensor-core-vs-FP32
absolute-logit-diff margin (tuned assuming init-scale weights) past its bound -- no thermal/algorithm-
selection mechanism needed. Fix: removed the once-latch, `build_model()` now unconditionally
re-runs the (already fixed-seed 1234, fully deterministic) `init_weights()` every call -- a verified
no-op for every real production path (each calls it once, and where a model already exists on disk
the immediately-following `load_model()`/`load_checkpoint()` overwrites the reinit anyway), restoring
true test isolation. Verified: full suite green across 2 reruns with different `--rng-seed`, and
`sub0_frontend_tests` (81/81) unaffected. The two OLDER instances above (2026-07-02/07-06) most likely
had the same underlying cause, misdiagnosed each time because the investigation stopped at "passes
isolated, fails mixed" without checking whether "fresh model" was actually fresh.
