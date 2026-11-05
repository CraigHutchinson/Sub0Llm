---
name: muon-cuda-parity-in-progress
description: DONE 2026-07-08 (40f96a1) -- Muon GPU implementation complete, independently verified, committed. Closes the entire CUDA-parity backlog (tied-embeddings/QK-norm/gated-FFN/Muon all done). Audit found 2 real hazards (TF32 leaking via shared cuBLAS handle; naive grad-zeroing would corrupt Muon's momentum) plus a real bug caught by testing during implementation
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Phase 1 (audit+plan) complete and reviewed (2026-07-08), via Fable, phased dispatch per the
[[cuda-parity-backlog-deferred]] template.** More novel than gated-FFN (which this backlog closed
first) since Newton-Schulz is matrix-level iterative math (GEMMs + reductions across 5 iterations),
not elementwise kernels, and raises a real precision question the prior 3 GPU-parity features never
faced.

**Precision recommendation (approved, first-principles reasoning not just CPU-mirroring):** double
accumulation ONLY for the Frobenius-norm reduction (reuses the existing `grad_normsq_kernel` verbatim
-- already double, DRY), `CUBLAS_COMPUTE_32F_PEDANTIC` for all 3 Gram/apply GEMMs per NS iteration, NO
FP64 GEMMs anywhere (quantified as ~100+ms/step at sm_120's ~1/64 FP64 throughput -- arithmetically
untenable, not a vibes call). Reasoning: FP32 accumulation error ~2.5e-6 relative at k≤1792 (this
project's max FFN width), and the algorithm's own reference implementation (Keller Jordan's) reportedly
runs the WHOLE NS iteration in bf16 in production -- CPU's double Gram accumulation is conservative
belt-and-braces, not a numerical necessity. Parity gate will be set from MEASURED evidence across all 6
test shapes (hypothesis ≤1e-4, hard ceiling 1e-3, matching this project's existing FP32-probe precedent
of <1e-2, not the tighter <1e-5 bar used for elementwise kernels) -- paired with a parity-INDEPENDENT
Gram-property check (off-diagonal <0.3, diagonal ∈(0.3,2.0), mirrors `muon_tests.cpp`'s own CPU bounds)
as the TRUE correctness authority, since orthogonalization correctness doesn't require exact parity to
a specific double-precision CPU run.

**Real hazard #1, independently verified before greenlighting: TF32 can silently leak into the
optimizer via the SHARED cuBLAS handle.** `run_fwd_bwd` flips the shared handle's math mode every
step via `set_handle_tf32`/`cublasSetMathMode` (confirmed real at `backend_cuda.cu:1502`) -- a Muon
GEMM naively issued on that same handle would silently degrade precision with no crash, no obvious
test failure at toy scale. Mitigated by a dedicated `gemm_muon()` using `CUBLAS_COMPUTE_32F_PEDANTIC`
(compute type is authoritative over handle math mode) instead of the general `gemm()`/`gemm_t()`. A
required test forces the TF32 knob ON and confirms Muon output is unaffected -- the direct proof case.

**Real hazard #2: the "just zero the Muon-eligible gradients before running the flat Adam kernel"
shortcut would be SUBTLY DESTRUCTIVE, not just wasteful.** `m[i] = b1*m[i] + (1-b1)*0` still DECAYS
`g_dev_m` toward zero every step -- but for Muon-eligible slices, `g_dev_m` holds MUON's momentum
(reused per the CPU convention), so this would silently corrupt Muon's actual momentum state while
LOOKING like a correct, simpler implementation. Correctly rejected in favor of a constant-memory
range-scan skip table (`g_muon_ranges`, mirrors the existing `g_decay_ranges` mechanism exactly) so the
flat `adam_step_kernel` skips Muon-eligible ranges entirely rather than zero-updating them. A required
test asserts Muon-eligible `vel` slices stay EXACTLY 0.0f as a tripwire for this exact bug class.

**Host-side dispatch restructuring**: `device_adam_step` gains a `muon_lr` param (`<=0` = pure AdamW,
bit-identical to today); when set, a new host loop over `PARAM_LAYOUT` dispatches one Newton-Schulz
"call" (~30 launches: 1 EMA/Nesterov kernel, 2 for the norm reduction+memset, 1 normalize, 5×(3 GEMMs +
2 combine kernels), 1 apply kernel) per Muon-eligible matrix, THEN the existing flat `adam_step_kernel`
runs over everything else via the new skip-range table. `is_muon_kind()`/the eligible-PKind set
(currently inlined separately in `backend_cpu.cpp`) gets hoisted to `layout.hpp` as ONE shared
definition for CPU+GPU (boy-scout DRY, matches this session's established pattern).

**Launch-overhead cost, quantified (not hand-waved) and explicitly DEFERRED to a later measured pass**:
production d448/L11/H7 has 66 Muon-eligible matrices (77 with gated-FFN) -> ~1,980 extra launches/step
(a ~5-6x multiplication of the ~390-430 launches/step baseline from the earlier fineweb_edu nsys
finding), ~300 GFLOP/step of (batch-independent) NS compute, estimated ~20-40ms/step added in the naive
form. Explicitly NOT optimized in this pass (correctness first, feature is opt-in) -- flagged mitigation
options for later: CUDA-graph-capturing the whole optimizer step (shapes are 100% compile-time static;
`g_dev_gs`'s existing device-scalar-indirection pattern is the precedent for keeping per-step scalars
graph-friendly) or `cublasGemmStridedBatchedEx` batching same-shape matrices to actually fill the SMs
(this project's own d448 GEMMs are already flagged elsewhere as SM-underutilizing at this scale).

**No transpose kernel needed at all** (a genuine improvement over the CPU reference, which physically
transposes): a row-major `[rows,cols]` buffer is a column-major `[cols,rows]` matrix, so both
orientations are expressible purely via cuBLAS op-flags on the SAME buffer layout -- eliminated an
entire class of buffer/copy code the CPU version's own comments (muon.hpp lines 65/131) explicitly
flag as a TODO/wish-it-away.

**Checkpoint compatibility**: traced end-to-end (not just assumed) -- `g_param_vel`/`g_dev_vel` already
stay unused/zero for Muon-eligible params on CPU today, and checkpoints are opaque `PARAM_FLOATS` blobs
regardless of what wrote them, so zero format change, zero `CKPT_VERSION` bump needed. Phase 3 will
still verify this empirically via a real save->resume round-trip, not just trust the trace.

Phase 2 greenlit with all of the above confirmed/approved, plus: standard multi-scale test bar (tiny +
production d448/HD64 + gated-FFN-combined for Wg routing + default-build byte-identical regression),
and the `train_stage.cpp:1065-1071` silent-fallback-to-AdamW warning gets deleted (the request becomes
honored for real once this lands).

**DONE (2026-07-08, `40f96a1`), independently verified before commit -- not just trusting the
agent's report.** One real bug caught by the agent's OWN testing during Phase 2/3 (not by
inspection): `adam_step_kernel`'s Muon skip-check was gated on parameter KIND alone (always uploaded,
independent of whether Muon is active), so a plain AdamW-only GPU run (`muon_lr<=0`) silently stopped
updating every Muon-eligible-kind matrix (Wq/Wk/Wv/Wo/W1/W2) entirely -- caught immediately by the
PRE-EXISTING "CUDA AdamW step matches the CPU optimizer update" regression test (cos 0.32 instead of
&gt;0.7). Fixed by gating the skip on `muon_active` (`muon_lr>0.f`), not kind-eligibility alone --
exactly the kind of "genuinely load-bearing regression test, not a hypothetical" this project's own
testing philosophy is built on.

**My own independent review, matching the gated-FFN precedent, not a rubber stamp:**
- Hand-verified all 4 new elementwise kernels (`muon_ema_nesterov_kernel`, `muon_normalize_kernel`,
  `axpby_kernel`, `muon_apply_kernel`) character-for-character against `muon.hpp`'s CPU reference --
  exact formula matches, correct in-place-aliasing safety (each thread reads its own index before
  writing).
- **Fully hand-derived and verified the cuBLAS row-major GEMM op-flag algebra from first
  principles** -- the single riskiest, most novel piece of this port (no physical transpose anywhere;
  both matrix orientations reachable via op-flags alone on the same native buffer). Did NOT just trust
  the code comment's claimed derivation: independently re-derived the base row-major-via-column-major
  identity from scratch, cross-grounded it against TWO already-tested, known-correct GEMM patterns
  already in this file (`launch_linear`'s forward GEMM, `launch_linear_bwd_t`'s transposed backward
  GEMM), then walked all 4 of the new Muon GEMM calls (both `rows<=cols` and `rows>cols` branches for
  both the Gram-matrix and the `BX` computation) through the verified rule by hand and confirmed every
  one produces exactly the mathematically-required product. No errors found.
- Independently checked the TF32-handle hazard is real (`cublasSetMathMode` confirmed at
  `backend_cuda.cu:1502`, a genuine global mutable per-handle setting flipped every step) before
  accepting the `gemm_muon()`/PEDANTIC mitigation.
- Confirmed the `train_stage.cpp` GPU-side `muon_lr_t` computation is EXACT formula-parity with the
  CPU branch's own `opt.set_muon_lr(lr_schedule(step, MUON_LR_BASE, lr_warmup_steps))`.
- Confirmed the required `vel==0.0f` tripwire test and the checkpoint round-trip test both exist in the
  diff exactly as required (the round-trip test's own comment explicitly cites my review guidance
  verbatim -- confirms the agent internalized the instruction, not just acknowledged it).
- **Live-rebuilt from scratch myself** (disposable `plain_small` d32/L2 CUDA build, MSVC env via
  `vcvars64.bat` + PowerShell -- Bash's `cmd.exe` invocation mangles arguments, established this
  session) and ran the full `[cuda]` suite independently: 33 test cases / 1,101,424 assertions, exit
  0 -- test COUNT exactly matches baseline-plus-4-new-tests (independent cross-check that all 4 new
  tests genuinely ran, not just 2 that happened to print `WARN` diagnostics visibly vs 2 using
  `INFO`-only, which Catch2 only shows on failure). Newton-Schulz self-test: rel-L2 ~5e-7 to 1.1e-6 at
  all 6 shapes (both orientations) -- far tighter than the ≤1e-4 hypothesis, FP32-PEDANTIC is
  essentially exact at this scale. TF32-forced-on immunity test: **max abs diff = 0, max rel diff = 0
  exactly** -- perfect proof the PEDANTIC compute type is fully immune to the shared handle's math mode.
- Confirmed git state was clean throughout (a pre-existing, unrelated `winfix` stash from 2026-05-31
  was the only stash entry both before and after -- the agent's own "confirmed via baseline comparison"
  claims did not leave any new stash artifact behind).

**Launch-overhead mitigation (CUDA-graph-capturing the optimizer step, or batched GEMM across
same-shape matrices) remains an explicit, deliberate, NOT-yet-done follow-up** -- correctness landed
first per instruction, feature is opt-in. Folded into [[hardware-efficiency-iteration-standing-task]]'s
backlog scope, not tracked as a separate blocking item.

**This closes the entire CUDA-parity backlog** -- see [[cuda-parity-backlog-deferred]] for the full
history (tied-embeddings, QK-norm, gated-FFN, Muon, all four now GPU-complete). The multi-size
TinyStories Chinchilla-ratio investigation's blocker (see [[checkpoint-tools-and-best-retention]]) is
now CLEARED -- next natural step, pending the user's own scheduling.
