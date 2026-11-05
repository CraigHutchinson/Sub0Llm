---
name: cuda-kernel-optimization-waves
description: 5(+)-wave CUDA kernel optimization sweep on backend_cuda.cu -- warp-cooperative reductions + constexpr dims; waves 1-4 done, real-world-verified 21% production throughput gain
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Started 2026-07-02**, user directive: "optimize all the cuda kernels... iterate on measure, improve, test,
commit for 5 times... keep making best judgement." Follows on from [[attention-kernel-throughput-bottleneck]]
(the earlier nsys finding that flagged rmsnorm/ce_backward as one-thread-per-row anti-patterns) and
[[lm-head-tensor-core-fix-and-graceful-stop]] (established the A/B methodology used here).

**Infrastructure built first:**
- A "medium" GPU profiling reference model (d448/L11/H7/seq256, `fineweb_smoke.txt` corpus, build dir
  `smoke_gpu`) -- big enough for representative per-kernel shapes/occupancy, small enough (~1.1GB resident)
  that ncu's kernel-replay backup buffer never trips the VBS-blocked system-memory-backed fallback that
  makes ncu unusable against the full d768 production model (see [[ncu-vbs-memory-backup-blocker]]). This is
  now the default target for iterating on kernel work -- NOT the production model.
- `SingleInstanceGuard` (train_stage.cpp) -- named Win32 mutex scoped to the model dir's canonicalized path,
  refuses a second concurrent `sub0llm-train` against the same checkpoint. Direct response to repeated
  duplicate/zombie-process incidents this session. Functionally verified (two concurrent launches: first
  proceeds, second exits 1 immediately with a clear message, no side effects).

**The recurring kernel-level fix (waves 1-4), all in `src/backend_cuda.cu`:**
One-thread-per-ROW reduction kernels (`rmsnorm_backward_act_kernel`, `rmsnorm_train_act_kernel`,
`ce_backward_kernel`, `rmsnorm_kernel`) rewritten to one-BLOCK-per-row with threads striding coalesced
across the row, reduced via a new `block_reduce_sum`/`block_reduce_max<BLOCK>` helper (warp shuffle +
shared memory, added near `to_f32`/`st_act`). The old layout put a warp's simultaneous loads C floats
apart (adjacent threads = adjacent ROWS at a fixed column) -- zero coalescing, 77-90% of warp cycles lost
to L1TEX scoreboard stalls per ncu.

**Second, independent lever, same waves — leverage baked constexpr config:** D_MODEL/N_HEADS/D_FF/VOCAB
are genuine compile-time constants (one build = one architecture, generated `sub0_config.hpp`), but several
kernels took them as RUNTIME parameters anyway. Dropping the parameter and using the constant directly
(a) lets strided loops fully unroll where the bound is small (D_MODEL: 192/448/768) and (b) folds `/C`
divisions into compile-time multiplies. **Caveat, learned the hard way twice:** don't assume a parameter is
constant without grepping EVERY call site -- `T` in `ce_backward_kernel` looks like it should be SEQ_LEN but
genuinely varies (benchmark/self-test paths use shorter windows), and `bias_grad_kernel`'s `N` varies across
its call sites even though the sibling `bias_grad_act_kernel`'s `N` doesn't (single call site). Also:
VOCAB-length loops (tens of thousands of iterations) must NOT be fully `#pragma unroll`ed even once
constexpr -- code-size/I-cache disaster; only D_MODEL-scale loops (hundreds) got the unroll treatment.

**Results (ncu, `--set full`, medium/d448 model unless noted):**
| Kernel | Duration | Occupancy | Compute thpt |
|---|---|---|---|
| rmsnorm_backward_act_kernel | 5.64ms -> 0.317ms (17.8x) | 27.7% -> 94.1% | 3.2% -> 31.3% |
| rmsnorm_train_act_kernel (fwd) | 885.7us -> 93.5us (9.5x) | 22.7% -> 91.9% | 4.5% -> 59.1% |
| ce_backward_kernel | 30.33ms -> 6.44ms (4.7x) | 28.0% -> 100% | -- (now 88.6% DRAM-bound) |
| rmsnorm_kernel (decode, M=1) | 49.89us -> 10.37us (4.8x) | ~2-4% both (M=1 caps this) | -- |

Correctness held throughout: `sub0_tests [cuda]` stayed at 20/21 test cases across all 4 waves (the one
failure -- attn dq/dk register spill at HD=64 -- is pre-existing, confirmed unrelated by grep: my changes
never touch the attention kernels). `rmsnorm_kernel`'s before/after needed `sub0llm-gen.exe`, not
`sub0llm-train.exe` -- that kernel is decode/eval-only (GPU mode routes training through
`rmsnorm_train_act_kernel` exclusively; `sub0llm-train.exe`-wrapped ncu captures never reach it). Caught via
git-stash before/after since I'd already written the fix before realizing this.

**Real-world verification (2026-07-02, production d768 model, resumed checkpoint step 3834):** single
A/B trial each (git-checked-out pre-Wave-1 backend_cuda.cu vs current HEAD, same checkpoint, cool-GPU start
each time) -- **A (baseline): 40, 40, 41 win/s steady. B (waves 1-4): 49, 49 win/s steady. ~21% real
wall-clock training throughput improvement**, both trials internally consistent (no thermal-drift red flag
like the lm_head investigation hit -- see [[thermal-confounds-ab-wallclock-testing]]). User accepted this as
sufficient (not the full interleaved A/B/A/B protocol) given the consistency and the independent ncu
evidence per kernel.

**Review agent findings (Sonnet subagent, `backend_cuda.cu` full-file audit) -- feeding waves 5+:**
High-confidence extras beyond the 4 kernels above: RoPE kernels (`rope_kernel`/`rope_act_kernel`/
`rope_bwd_act_kernel`/`rope_one_kernel`) compute `d = C/H` at runtime per thread when `D_HEAD` is already
baked; `build_qkv_kernel`/`split_dqkv_kernel` same div/mod-fold opportunity; `attn_decode_kernel` has a
**dead, unused `H` parameter** (grep-verified, just delete it); flash-attention kernels (already good on
HD/TILE_K/TILE_Q) still take `C`/`in_stride` at runtime despite both being constant at every call site
(low-risk stride-math cleanup on the already-optimized kernels, not a new unlock); embed kernels similarly;
`grad_normsq_kernel`/`adam_step_kernel`'s `n` is genuinely `PARAM_FLOATS` constexpr, zero-risk drop.
Vectorized (float4) loads flagged as possible but call-site-specific (VOCAB-width buffers aren't always
divisible by 4) -- not a blanket win, needs care per kernel.

**Plan for remaining waves (not yet done):** Wave 5 = batch the review-agent's low-risk mechanical cleanups
above. Before that, an audit-checklist pass confirming every agent finding was actually actioned or
explicitly deferred with a reason (user's explicit ask, to stop things silently falling through). Wave 6 =
a Fable subagent doing a "helicopter view" of the WHOLE CUDA orchestration end-to-end (execution-graph
shape, overlap opportunities, hardware utilization across the full step) PLUS memory usage/layout/stack/
shared-mem review specifically (user's reasoning: a footprint win in one place can unlock bigger batches or
better SM occupancy elsewhere) -- outcome to be triaged with best judgement into further waves.

**Process notes worth repeating:** measure-before-code on the MEDIUM model (ncu works there, doesn't on
production -- see [[ncu-vbs-memory-backup-blocker]]), test correctness (`sub0_tests [cuda]`) after every
single kernel change before moving to the next, commit each wave separately with the measured numbers in
the commit message. User explicitly asked to delegate "big wide-arching analysis" to a Fable subagent
(model: "fable" in the Agent tool) rather than doing broad/exploratory scans inline -- applies to the
review-agent-style audits and Wave 6, not to concrete/already-scoped kernel edits.
