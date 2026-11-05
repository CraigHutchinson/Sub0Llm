---
name: muon-optimizer
description: Muon optimizer (orthogonalized-momentum via Newton-Schulz) added as a hybrid Muon(hidden 2D matrices)+AdamW(embed/head/norm/bias) option, --optimizer muon, CPU-only; math+convergence verified, newton_schulz5 optimized to zero-per-call-alloc+SIMD (~11x -> ~3.2x slower than AdamW at d128, bit-identical loss trajectory pre/post optimization)
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User asked (2026-07-04) to survey architecture ideas from current LLM research; explicitly called
Muon "notable" and asked to prioritize it among a 6-item list (SwiGLU-as-default, weight tying,
QK-norm, attention sinks, corpus-driven sizing round out the rest, see
[[architecture-ideas-backlog]]).

**Verified against the actual reference source first, not recalled from memory** (WebFetch of
github.com/KellerJordan/Muon's `muon.py`) -- this is precise numerical code where a wrong constant
silently produces a worse optimizer, not a crash, so the "verify correctness against reference"
discipline this project already applies to kernels ([[verify-correctness-against-reference-before-perf]])
applies here too. Confirmed: Newton-Schulz coefficients `(3.4445, -4.7750, 2.0315)`, 5 default steps,
Frobenius-norm+1e-7 normalize, transpose-to-smaller-dimension-first (compute optimization only),
momentum EMA `m = beta*m + (1-beta)*g` (beta=0.95, via `lerp_`, NOT the naive `mul_+add_` a first
guess would produce), Nesterov lookahead `update = (1-beta)*g + beta*m` (nesterov defaults TRUE,
easy to miss -- it's not exposed as a param in the class, buried in `muon_update`'s default), scale
`sqrt(max(1, fan_out/fan_in))` applied post-orthogonalization, decoupled weight decay `p *= (1-lr*wd)`
before the update. Muon's own docs note 5-step Newton-Schulz does NOT converge to exact UV^T --
singular values land ~Uniform(0.5,1.5), not pinned to 1 -- confirmed empirically (see below).

**Axis-convention trap, caught before shipping**: PyTorch's `nn.Linear.weight` is `[out,in]`
(size(-2)=out), but this project's weight matrices are `[rows=in, cols=out]` (`op_linear`: `y = x @
W`, W stored [in,out]) -- the OPPOSITE order. A naive port of `scale = sqrt(max(1,
size(-2)/size(-1)))` as `sqrt(max(1,rows/cols))` would have been silently backwards for every
non-square matrix in this project (most of them -- D_MODEL x D_FF is never square). Correct mapping
here: `sqrt(max(1, cols/rows))`.

**Implementation** (`include/sub0/muon.hpp`, engine-free pure math + `src/backend_cpu.cpp`'s hybrid
`AdamW::step()`):
- `newton_schulz5()` -- transpose-for-compute-efficiency (always works on the smaller of rows/cols as
  the Gram-matrix dimension, matters here since FFN matrices are D_MODEL x D_FF, ~4x non-square).
- `AdamW` (`include/sub0/core.hpp`) gained `use_muon` (constructor), `set_muon_lr()`/`muon_lr()` --
  Muon needs a MUCH larger, separately-scheduled LR (reference default 0.02) than AdamW's ~1e-3-5e-3,
  since orthogonalized updates have completely different magnitude/geometry. Sharing one LR would
  have silently starved whichever optimizer got the wrong scale -- caught this before wiring the
  training loop, not after. `train_stage.cpp` computes it via the SAME `lr_schedule()` warmup/decay
  shape at a separate peak (`MUON_LR_BASE=0.02`, not batch-scaled like AdamW's is -- Muon's own
  convention doesn't use Adam's sqrt-batch heuristic).
- Per-parameter routing uses `PARAM_LAYOUT` (not the pre-existing `W->views`, which only carries
  off/n/decay, not shape/kind) -- `PARAM_LAYOUT[i]` and `W->views[i]` are the SAME sequential slot by
  the pre-existing "lock-step" invariant (layout.hpp's own doc comment), so this was a safe, free
  upgrade to iterate. Muon-eligible = Wq/Wk/Wv/Wo/W1/W2/Wg (hidden 2D matrices); AdamW keeps
  TokEmb/PosEmb/LmHead/LmBias/norms/biases -- Muon's own documented usage pattern, not invented here.
- **Checkpoint format: zero changes.** Muon's momentum buffer reuses AdamW's existing `g_param_m`
  arena for Muon-routed params (g_param_vel goes unused/wasted for those, harmless at this project's
  scale) -- same design principle as the gated-FFN work's checkpoint-safety call
  ([[gguf-import-feasibility-review]]), avoiding any risk to already-written checkpoints. Optimizer
  choice is NOT persisted in the checkpoint (re-specify `--optimizer` each invocation, same as
  `--batch`/`--lr` already work) -- a deliberate scope cut, not an oversight.
- GPU: Muon only wires into the CPU path (`AdamW::step()`); the GPU path has its own separate
  optimizer kernel (backend_cuda.cu's `adam_step_kernel`), untouched. `--optimizer muon` on a run that
  ends up using GPU prints an explicit warning and trains with plain AdamW instead, rather than
  silently no-op'ing (this exact "silent partial application" failure mode was rejected because it's
  exactly the confusing-diagnostic-gap pattern this project already fixed once for the resume-override
  case, see [[train-diagnostics-backlog]]).
- New CLI: `sub0llm-train --optimizer adamw|muon` (`include/sub0/cli_stages.hpp`), threaded through
  `sub0_train_stage`'s signature (gained one `int optimizer` param).

**Verification**:
1. Pure math: an ad-hoc scratch harness (not committed) confirmed the DEFINING property directly --
   after 5 steps, off-diagonal of `X @ X^T` collapses ~1000x vs the raw random input (e.g. 180->0.05
   at the 448x1792 FFN shape) while the diagonal lands in a bounded [0.57,0.98]-ish range, exactly
   matching the "US'V^T, S'~Uniform(0.5,1.5)" documented behavior -- NOT exact orthonormality, which
   was my first (wrong) test expectation before checking the docs.
2. 6 new Catch2 cases (`tests/muon_tests.cpp`, `sub0_frontend_tests`, engine-free): the same
   off-diagonal-collapse + bounded-diagonal property across square/wide/tall/both FFN orientations,
   determinism, in-place aliasing safety, all-zero-matrix (no NaN/Inf, the +1e-7 epsilon holds), 1-row
   and 1-column degenerate shapes, and `scale_factor`'s exact values at known ratios. 11,353 assertions,
   all pass.
3. Zero regressions: full production-scale (`smoke_gpu`) non-CUDA suite still 124,694,438/124,694,439
   (the one failure is the SAME pre-existing engine_tests gradient-check gap, unchanged -- see
   [[engine-tests-gradient-check-dims-dependent]]); tiny gated-FFN CPU build still 363,450/363,450.
4. Real end-to-end training comparison (tiny d32L2H4 model, `data/english.txt`, 60 steps each): AdamW
   val_nelbo 6.42->5.82, Muon 6.42->5.81 (Muon slightly ahead here, NOT a claim of general superiority
   from one 60-step toy run -- just confirms it converges sanely). Resumed the Muon checkpoint for 20
   more steps (5.8075->5.6675, smooth continuation, no discontinuity) -- confirms the momentum-buffer
   reuse round-trips correctly through save/load.
5. **A/B on the REAL `data/tinystories.txt` corpus** (2026-07-04, per user request -- "run a/b against
   the tinystories corpus as a quick way to check"), d128 L4 H4 seq128 vocab4229 (deliberately NOT
   gated-FFN, to isolate the Muon variable), CPU-only, same seed (123) so both runs see the IDENTICAL
   window sequence, 300 steps each (still deep in warmup, no eval fires yet at this step count -- this
   is a "does it run/converge sanely" check, not a quality verdict): AdamW train loss 8.1091 in 26.7s
   wall-clock (~58.7k tok/s); Muon train loss 8.2305 in 4m51s wall-clock (~5.0k tok/s) -- **~11x
   slower**, notably worse than the throughput gap seen at the tiny d32 scale. Loss trajectories stayed
   close (both barely below the ln(vocab)~8.35 random-guess floor at 300 steps, as expected this early).
   **Root cause of the 11x, not yet fixed**: `newton_schulz5` (`include/sub0/muon.hpp`) is a
   deliberately correctness-first, NOT performance-tuned implementation -- plain triple-nested loops,
   no `#pragma omp simd`, no threading, a fresh `std::vector` heap allocation every call -- vs. this
   project's forward/backward GEMMs (`op_linear` etc.) which are all restrict-pointer,
   vectorization-friendly, tuned code. The gap widens with matrix size (Newton-Schulz cost scales
   ~m²n per iteration), which is why d128's 11x looks worse than d32's earlier, milder gap. **This is
   very likely an implementation-optimization gap, not an inherent Muon cost** -- worth a dedicated
   optimization pass (SIMD the inner loops, reuse scratch buffers instead of per-call heap allocs,
   maybe thread the independent per-matrix Newton-Schulz calls within one step()) before drawing any
   conclusion about whether Muon is "worth it" at this project's scale wall-clock-wise. A good
   candidate for the Fable-subagent delegation the user flagged (see
   [[fable-subagent-for-big-risky-tasks]]) once it's scoped as its own task -- not attempted this pass.

**newton_schulz5 optimization (2026-07-04, same day, follow-up): CODE DONE, NOT YET REBUILT/VERIFIED
this session (paused for quota).** Per [[agents-md-established]] rule #1 (no per-call heap alloc),
rewrote `include/sub0/muon.hpp`'s `newton_schulz5`: the X/A/AA/BX working buffers are now
`static thread_local std::vector<float>`, grown via `resize()` (a no-op once already large enough) --
so after the first call at each thread's largest-seen matrix shape, every subsequent call allocates
nothing. Fixed a real correctness trap this reuse introduces: every loop is now explicitly bounded by
the CURRENT call's `mn`/`mm`, not by the (possibly larger, stale-tailed) buffer's own `.size()` -- the
original code had two "whole-container" operations (`for (float v : X)`, `std::copy(X.begin(),
X.end(), out)`) that would have silently read/written past the intended bounds once a later call
reused a buffer grown by an earlier, bigger call. Also added `#pragma omp simd` + `__restrict`
pointers to the hot loops, and changed the `BX = B @ X` matmul from a dot-product-per-output-element
reduction to a contiguous-axpy accumulation pattern (matching `op_linear`'s own established forward-GEMM
loop order exactly -- same reasoning: keep the innermost loop contiguous for vectorization). Verified
via the SAME standalone orthogonality harness used originally (off-diagonal collapse + diagonal range
checks) -- produced numbers IDENTICAL to the pre-optimization run at every tested shape, strong evidence
the rewrite is behaviorally equivalent. Added a new Catch2 case (`tests/muon_tests.cpp`) specifically
targeting the buffer-reuse trap: call at a large shape (256x512) to grow the buffers, then at a small
shape (16x24), check the small result is still a valid orthogonalization (not corrupted by the
larger call's leftover buffer contents).

**Verification completed (2026-07-04, same day, after the quota pause)**: all three follow-up steps
ran clean. (1) `sub0_frontend_tests` `[muon]`: 11,356/11,356 assertions (7 cases, the new buffer-reuse
test included). (2) Full production-scale `smoke_gpu` non-CUDA suite: 124,694,438/124,694,439 --
IDENTICAL pattern to every prior run this session, same one pre-existing unrelated failure (see
[[engine-tests-gradient-check-dims-dependent]]), zero regressions from the optimization. (3) Re-ran the
EXACT same tinystories A/B (d128 L4 H4 seq128 vocab4229, seed 123, 300 steps, `--optimizer muon`):
**train loss 8.2305 at step 300 -- bit-identical to the pre-optimization run's 8.2305**, confirming
the rewrite is not just "probably" but ACTUALLY numerically equivalent, not merely close. Wall-clock:
1m25s (85.2s, ~17.7k tok/s), vs the pre-optimization run's 4m52s (291.6s, ~5.0k tok/s) -- a **~3.4x
speedup** from eliminating the per-call heap allocation and adding SIMD hints alone (no threading
attempted). Muon is now ~3.2x slower than AdamW at this shape (85.2s vs AdamW's 26.7s baseline), down
from the pre-optimization ~11x gap -- much more consistent with Newton-Schulz's INHERENT extra compute
(5 iterations of several matmuls per matrix vs AdamW's one elementwise pass) than with an
implementation inefficiency, which is the expected, accepted shape of this specific cost/benefit
tradeoff now that the accidental (non-inherent) overhead is gone.

**Not done / deferred**: no CUDA kernel (CPU-only for now, `TODO(muon-gpu)` at the warning site); no
real (non-toy) production-scale training comparison yet -- worth running once the production line
picks a corpus/config, to see whether Muon's per-step compute cost is worth it at THIS project's
actual scale vs. a longer AdamW run in the same wall-clock budget.
