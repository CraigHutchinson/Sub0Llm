---
name: qk-norm
description: Per-head RMSNorm on Q/K before RoPE -- DONE CPU (2026-07-04) + CUDA (2026-07-06 via Fable), item 4 of the architecture backlog
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Implemented `--qk-norm` (item 4 of [[architecture-ideas-backlog]]): RMSNorm applied independently to
each attention head's `D_HEAD`-length slice of Q/K (a single `[1,D_HEAD]` gamma shared across heads,
Gemma2's query/key-norm convention), applied right after the Q/K linear projection and BEFORE RoPE.

**Why a new op, not a generalized `op_rmsnorm`:** `op_rmsnorm` normalizes the WHOLE row (one
statistic over all `D_MODEL` elements). Calling it on a per-head RESHAPED view would still mix every
head into one norm statistic unless the reduction itself is grouped -- there is no way to get
per-head independence by just changing the call site, the reduction has to be group-aware. Rather
than risk regressing the heavily-used, well-tested `op_rmsnorm`/`Op::RMSNorm` path by generalizing its
math in place, added a fully separate `op_qknorm`/`Op::QKNorm` (backend_cpu.cpp) with its own
per-(t,head) `rinv` scratch -- mirrors `op_rmsnorm`'s structure exactly, just grouped by `D_HEAD`
instead of the whole row. Same idea as why SwiGLU got its own op instead of generalizing GELU.

**Checkpoint-format-safe** with no new header field: `layout.hpp` adds 2 params per layer
(`PKind::QNorm`, `PKind::KNorm`, each `[1,D_HEAD]`) only when `USE_QK_NORM`, inserted between `Wo` and
the FFN block. `NUM_PARAMS`/`PARAM_FLOATS` change accordingly, and `param_floats`/`trainable_floats()`
already discriminates a mismatch -- same reasoning as gated-FFN/tied-embeddings.

**A real aliasing bug caught before it shipped:** the KV-cache decode path's `qknorm_row` helper was
first written with separate `x`/`y` `__restrict` pointers (mirroring `rmsnorm_row`'s forward-with-
separate-output style), then called with the SAME buffer for both (`qknorm_row(qn, gamma, qn, ...)`)
to normalize in place -- passing aliasing pointers to `__restrict` parameters is UB regardless of
whether the specific access pattern happens to be safe. Fixed by making `qknorm_row` genuinely
in-place (single `__restrict` pointer, no second operand), matching `rope_row`'s existing convention
for the exact same in-place-on-Q/K use case.

**Verification performed:**
- Default (disabled) `smoke_gpu` config (d448 production dims): rebuilt clean, full non-CUDA suite
  124,694,441/124,694,441 (one more than the prior 124,694,440/124,694,440 baseline, exactly matching
  the new `STATIC_REQUIRE(!USE_QK_NORM || COMPUTE_MODE==Cpu)` added to layout_tests.cpp), same single
  pre-existing unrelated gradient-check failure (see [[engine-tests-gradient-check-dims-dependent]]) --
  zero regressions to the default build.
- `--qk-norm 1` alone at d32/l2/h4/seq32: full suite green, 339,962/339,962 assertions, including the
  finite-difference gradient check -- direct numerical proof `op_qknorm`'s forward+backward are correct.
- `--qk-norm 1` COMBINED with `--gated-ffn 1` at d192/l2/h4/seq32 (interoperability check between two
  independent CPU-only axes): full suite green, 3,663,472/3,663,472 assertions, gradient check clean.
- Real end-to-end smoke test (isolated `--qk-norm 1`, d32): trained 60 steps (val_nelbo ~6.4 -> 5.8445),
  ran `sub0llm gen` (confirms the KV-cache `qknorm_row` in-place decode path), confirmed
  `qk_norm=1` in meta.txt and the `...v756rq...` registry tag, resumed training 20 more steps
  (5.8445 -> 5.6907, smooth continuation, proving checkpoint round-trip correctness).

Two build dirs (`muon_ab`, `gated_test`) had a PRE-EXISTING, unrelated inconsistency in their cached
build facts (`HAS_CUDA=false` but a stale nonzero `CUDA_ARCH`), tripping the
`STATIC_REQUIRE(HAS_CUDA || CUDA_ARCH==0)` self-consistency check and blocking compilation entirely.
Not caused by this change -- worked around per-invocation with explicit `--has-cuda 0 --cuda-arch 0`
on the configure command for those two dirs; not worth a full re-cmake-configure for scratch/test dirs.

CUDA support NOT started -- `backend_cuda.cu` has a `static_assert(!USE_QK_NORM, ...)` guard. Of the 4
CPU-only features now needing GPU parity, this is flagged as the SIMPLEST (plain elementwise per-head
RMSNorm kernel + backward, no GEMM/transposed-operand complications like tied-embeddings, no new
weight-shape like gated-FFN) -- see [[cuda-parity-backlog-deferred]].

**CUDA DONE (2026-07-06, via Fable, committed `c05c6ff`).** Three kernels
(`qknorm_act_kernel`/`qknorm_save_act_kernel`/`qknorm_backward_act_kernel`, unified across Q/K via a
`blockIdx.z` selector) wired into all four forward/backward call sites. Named constexpr per-layer
offset constants (`kLn1..kB2`, `kFinalBase`) plus a `consteval check_layer_offsets()` cross-verified via
`static_assert` against `PARAM_LAYOUT` -- a misindexed slot is a compile error, not a silent wrong-tensor
read. New `TrainScratch::qk_pre` (bf16, zero bytes when disabled) stashes pre-norm x for the
backward-pass checkpoint recompute (RoPE overwrites the same buffer immediately after qknorm runs).
Verified: dedicated self-test at 4 H/DH/row combos (~1e-7 rel-L2 both fwd/bwd); full `[cuda]`-tagged
suite 30/30 (41.4M assertions) at real production dims (D448/L11/H7/DH64); memplan VRAM prediction
within ~2% of measured. ncu-profiled `qknorm_backward_act_kernel`: 411 us, 3x its sibling kernels,
75.8% of stall cycles attributed to L1TEX long-scoreboard from the `atomicAdd` dgamma fan-in (gamma is
`[1,DH]` shared across every row AND every head, so ~28,672 blocks contend per address at production
scale) -- real (~6-8% of the whole backward pass) but not dominant (GEMM-heavy FFN/attention backward
still bigger), documented in the kernel's own comment, no speculative fix per "only if ncu shows a
problem" policy.

**Independently re-verified test-isolation finding (2026-07-06, my own follow-up, not Fable's):** the 6
failures Fable saw mid-run (`cuda_tests.cpp:136/167/249/294/700`, `engine_tests.cpp:225`) are confirmed
PRE-EXISTING and unrelated to qk-norm -- reproduced identically on the OLD pre-session binary. I went
one step further and found these are ALSO test-order/test-isolation flakiness, not deterministic: the
same `smoke_gpu` (qk-norm off) binary passes 30/30 clean every time when run with `[cuda]` tag alone
(isolated from `engine_tests.cpp`/`config_tests.cpp`/etc.), and the single failing test case in
isolation ("CUDA attention-only gradient stays aligned with the CPU") passes cleanly on repeated solo
reruns with fresh RNG seeds -- but the SAME 6 failures reproduce consistently in the FULL mixed-file
suite across multiple different RNG seeds. This points to cross-test-file state contamination (some
earlier non-CUDA test leaving global/GPU state dirty for a later CUDA gradient check), a recurrence of
the same CLASS of bug as [[cuda-padded-window-gradient-mismatch]] (which was resolved via an RAII
CudaGuard fix and had re-verified clean on 2026-07-03) -- worth a fresh investigation, NOT a QK-norm
regression. Doesn't block committing qk-norm CUDA since the isolated `[cuda]`-tag signal is clean at
both scales and is the more reliable one; flagged as open follow-up work.
