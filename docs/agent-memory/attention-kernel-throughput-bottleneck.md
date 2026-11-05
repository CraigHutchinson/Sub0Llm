---
name: attention-kernel-throughput-bottleneck
description: "CUDA training throughput bottleneck is the naive attention kernel (L1/TEX-cache-bound O(T^2) K/V reads), NOT the GEMMs; measured via ncu; fix = shared-memory/FlashAttention tiling of the fwd + 2 backward attn kernels"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

**Definitive diagnosis (2026-07-01, ncu + sub0-cuda-selftest bench) of why d448/T256 training is slow
(~7k tok/s, ~7s/step):** the ATTENTION kernels, not the GEMMs.

- `sub0-cuda-selftest bench`: d448 forward is 0.44ms at batch64/T64 (~45 TFLOPS, healthy) but **823ms at
  batch184/T256** (~0.24 TFLOPS) -- ~185x less efficient. TF32 barely helps (1.16x) -> NOT GEMM-bound.
- **ncu** (`sub0-cuda-selftest ncu`, GPU perf-counter access must be enabled for all users first):
  `attn_kernel` = **69 ms/layer x 11 = ~92% of the forward**; Memory 90% / **L1-TEX 93%** / Compute 19% /
  DRAM 4% / occupancy 57% (regs cap blocks at 20). => **L1/TEX-cache-bound from REDUNDANT K/V reads.**

**Mechanism:** the attn kernels are one-thread-per-query, streaming ALL keys/values from global each
query with NO shared-memory tiling (the "flash" comment only means no O(T) local scratch). K/V are re-read
O(T) times -> O(T^2) L1/TEX traffic that saturates the cache at T=256. (Blocks already hold 64 queries of
ONE (batch,head) -- idx = b*H*T + h*T + i -- so a shared K/V tile would be reused 64x for free.)

**The fix = shared-memory / FlashAttention-style tiling** of the THREE attention kernels in the training
step: `attn_train_act_kernel` (train forward), `attn_backward_head_act_kernel` +
`attn_backward_query_act_kernel` (backward grads). Load K/V tiles into shared memory, online-softmax
across tiles, causal-skip tiles above i. Verify on the FAST d192 (tinystories) parity + measure via
`sub0-cuda-selftest`. Potentially 10-30x on the step.

**PROGRESS: FORWARD DONE (committed, on top of 24c73a5).** `attn_fwd_tiled_kernel<A,HD,TILE_K>` in
backend_cuda.cu: block owns TILE_Q=64 queries of one (b,h), grid=(T/64,H,batch); stages K/V tiles to
shared, q+acc in registers (HD=D_HEAD compile-time -> unrolled, no spill); keys still consumed in
increasing j so it is **BIT-IDENTICAL** to naive (max rel diff 0.00e+00). Now serves BOTH inference
(`launch_attn`) and training (`launch_attn_train_t`); old float `attn_kernel` removed; naive
`attn_train_act_kernel` kept as the on-device parity reference. MEASURED d448/T256/bf16: attn fwd
38.7ms->4.6ms = **8.5x**; full inference forward 823ms->338ms. GUARDS (load-independent, NOT
wall-clock): `sub0_cuda_attn_check` runs naive-vs-tiled on same q/k/v (parity) + a dimensionless
speedup RATIO -> `sub0-cuda-selftest attn` + a Catch2 test (REQUIRE parity + ratio>2x). Note: fwd
attention runs TWICE/step (forward_train + backward checkpoint-recompute) so this saved ~2 passes.

**BACKWARD DONE (committed).** Replaced the naive backward (head-per-thread = only batch*H~1288
threads, re-streams K/V/Q, ACCUMULATES dq/dk/dv IN BF16 rmw) with a flash backward = 3 atomic-free
kernels, each parallel over the full (b,h,T) grid + shared-tiled: (1) attn_bwd_stats (per-query m_i,
1/Z_i, dot_i=sum_j p_ij<dout_i,v_j> in FP32 -- two tiled key-passes; computes dot in fp32 not from
bf16 att, so NO att arg needed); (2) attn_bwd_dq query-parallel tiled over keys (dq_i owned); (3)
attn_bwd_dkv key-parallel tiled over queries (dk_j/dv_j owned). Accumulate in FP32 registers, round to
bf16 ONCE => MORE accurate than naive. Per-query stats in a grow-on-demand scratch (g_bwd_*, freed in
train_free). dqkv no longer pre-zeroed (each output written once). MEASURED d448/T256/bf16: backward
attn 1590ms->35ms = **45x**; full train step batch64 17272ms->480ms = **36x** (949->34160 tok/s).
Correctness: ALL 18 [cuda] tests pass at d192 (4.37M assertions) incl CPU-fp32 reduced-grad +
attention-only-grad + padded + AdamW. Guard: sub0_cuda_attn_bwd_check (naive-vs-tiled dq|dk|dv within
~sqrt(T)*bf16eps=6.3% -- the naive is the NOISIER one, bf16-accumulate) + Catch2 test (ratio>3x).

**REAL-WORLD: fineweb batch184 train ~6400 tok/s -> 66234 tok/s (~10x)** on the fully-tiled build
(the selftest 36x is at batch64 where the naive backward's low parallelism dominates more; at 184 the
GEMMs are a bigger share). NOTE a benchmark-harness VRAM artifact: sub0-cuda-selftest train bench at
batch128 spilled to WDDM sysmem (holds fwd+train+opt scratch + the check's transient 1.4GB) -> 20s
fluke; the REAL train at 184 is clean (no fwd scratch alongside). Both attention forward AND backward
now flash-tiled -- attention is no longer the bottleneck; next levers = GEMM/MMQ (see
[[llama-cuda-shootout]], [[optimization-target-is-hardware-ceiling]]).

**Fineweb resume LIVE (2026-07-01):** resumed from step 5024 (val 3.1452) on the FULLY-tiled build at
66k tok/s, GPU resident step, `--steps 0` (plateau backstop 30ep/188340), detached (survives session).
model dir sub0llm_fineweb_smoke_d448l11h7sq256v16488rj_ae8f45b. Earlier: 3143->5024 dropped val
3.2594->3.1452.

**DKV register-split DONE (2026-07-02, commit 3779fd7), measured with ptxas -v at native (d768/HD=96,
the config that first exposed this -- also needed a shared-mem tile fix, attn_tile_q<HD>, to even
COMPILE at HD=96; see [[large-model-windows-build-run-gotchas]]):** attn_bwd_dkv_kernel held
kr+vr+dka+dva = 4*HD registers/thread (384 at HD=96), over the 255-register architectural cap ->
ptxas spilled to local memory (255 regs used, 984B stack frame, 1172B spill store/load -- an
occupancy AND per-launch-latency cost). Split into attn_bwd_dv_kernel (dv_j depends only on p, hence
kr, NOT v at all -> 2*HD regs) and attn_bwd_dk_kernel (needs kr+vr together, no further split -> 3*HD
regs). Measured: **dv = 241 regs, ZERO spill** (clean win); **dk = 255 regs (still capped) but spill
1172B->560B (-52%)**. dv also drops stat_dot/v/Sd (not needed), lighter on shared mem + bandwidth too.
Correctness at native: "backward matches CPU reduced gradient" + "...with short padded windows" +
"attention-only gradient" all pass; "flash-attention backward matches naive" passes at **33.6x
speedup** (vs naive, at d768/T512/batch128 -- not directly comparable to the d448/T256 45x figure
above, different scale, but confirms no regression from the split).

**RESOLVED (2026-07-02, commit 528c32a): attn_bwd_dq_kernel's independent 255-register-cap spill**
(628B, unrelated to the dkv work above). Its 3 resident arrays (qr[HD], dr[HD], dqa[HD] = 3*HD = 288
at HD=96) have NO natural 2-way split like dkv's dk/dv (dq is a SINGLE output; qr+dr both needed
together in the same s/dp computation every iteration). Two cheap fixes were tried and ruled out
first (both zero effect, confirmed 255 is a hard per-thread architectural cap): removing #pragma
unroll, and a smaller `__launch_bounds__` hint. The actual fix: a WARP-COOPERATIVE channel split (the
technique production FlashAttention kernels use) -- 2 threads share a query, each owning HD/2
channels (144 registers), combining partial s/dp dot products via `__shfl_xor_sync`. LANES=1 (small
HD) makes every formula reduce algebraically to the pre-split code, so ts_gpu/smaller builds are
unaffected. Verified 226 regs, ZERO spill (was 255/628B); gradient parity holds at native d768/HD=96
(CPU-fp32 reduced gradient, attention-only gradient, short-padded-windows); measured backward
speedup vs naive jumped **33.6x -> 61.1x** as a direct result. Also added `sub0_cuda_attn_regcheck` +
a Catch2 test using `cudaFuncGetAttributes` (not ptxas -v text parsing) to guard stats/dq/dv/fwd at
zero spill and bound dk's small by-design spill -- closing a real gap (the existing speedup-ratio
test would NOT have caught a return-to-spilling regression, since a spilling kernel still clears
>3x against the much-slower naive reference).

**Also discovered while auditing: cuda_tests.cpp has a PRE-EXISTING, order-dependent test flake**
(confirmed NOT caused by this session -- reproduces identically at commit 77b2acb, before any of
today's gen-wiring/dkv-split changes): "CUDA backward matches the CPU reduced gradient" and "CUDA
attention-only gradient stays aligned with the CPU" both pass in isolation (cos~0.9999) but 2/76 tests
fail when the full suite runs together, deterministically (same 2 tests, same order). Root cause
suspected: `sub0::build_model()` only randomizes weights ONCE per process (`g_params_init.exchange`),
so later TEST_CASEs inherit whatever an earlier test left the shared weights as -- a fragile,
undocumented cross-test dependency, not a numerical bug. Not fixed (needs a per-test weight reset,
e.g. a test-only reseed hook); recorded for a future session per the boy-scout rule.

**Quick wins already landed (24c73a5):** cudaDeviceScheduleBlockingSync (frees the pinned host core;
the "one core at 100%" was a busy-wait, GPU is genuinely 100% util). TF32 bake = ~0% on this model
(attention-bound), so NOT worth a committed default. Tooling ready: ncu-mode selftest + Catch2 benchmarks
([[frontend-layer-testability-and-tooling]] extended). Relates to [[llama-cuda-shootout]] (MMQ gap) and
[[optimization-target-is-hardware-ceiling]] -- but attention tiling is the FIRST lever, before GEMM/MMQ.
