---
name: hardware-efficiency-iteration-standing-task
description: Standing task (2026-07-06) -- keep iterating toward the hardware's 100% efficiency ceiling, big refactors authorized in advance
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's explicit instruction (2026-07-06): "add a task to iterate on optimizing as close to the 100%
hardware (no unnecessary micro idles etc) efficiency goals making big refactors as needed to achieve the
goal. test your work commit and continue."

**What this is:** a standing, recurring task -- not a one-shot request. Between other work, keep coming
back to this: hunt for any remaining micro-idle (launch-overhead gaps, unnecessary syncs, serialized
stages that could overlap, under-occupied kernels, host/device round-trips that could be pipelined) and
close it. Continues [[optimization-target-is-hardware-ceiling]] (theoretical-ceiling framing) and
[[overlap-memcpy-compute-standing-reminder]] (the open-ground list: decode loop, checkpoint I/O, B1
chunk-to-chunk pipelining) but is broader -- ANY source of hardware idling, not just memcpy/compute
overlap specifically.

**Why:** the user wants the engine to approach the physical ceiling of the hardware, not just be
"pretty fast" -- this is a standing goal to revisit repeatedly as new levers get exposed by each
feature landing (e.g. B1 chunking created a new chunk-boundary pipelining opportunity; QK-norm CUDA
landing will create its own profiling surface once verified).

**How to apply:**
- Big refactors are pre-authorized when they're in service of this goal -- don't hold back a
  structural change just because it's large, if it's the right fix for an idle/inefficiency found via
  real profiling.
- Still gated by [[ncu-depth-vs-nsys-busy-time]]: never call something "optimized" off nsys busy% alone
  -- pair every claim with an ncu per-kernel depth pass (occupancy, achieved-vs-theoretical throughput,
  stall reasons).
- Still gated by [[verify-correctness-against-reference-before-perf]]: any refactor must hold the
  existing correctness gates (CPU/GPU parity, greedy-decode parity) before its speedup counts.
- Test, then commit, after each concrete round of this work -- don't let it accumulate into a large
  uncommitted backlog (see [[resolve-preexisting-issues-blocking-our-work]] and the general "commit
  after every task" standing instruction).
- Do NOT commit currently-broken work to chase this goal faster -- correctness regressions block
  everything else in the pipeline, including this task.

**Phase 1 audit done (2026-07-07, via Fable), Phase 2 greenlit and in progress.** Full writeup of
findings lives in [[ncu-depth-vs-nsys-busy-time]] and [[overlap-memcpy-compute-standing-reminder]]
(both updated with real numbers). Summary of the prioritized plan:
- **Greenlit for Phase 2 (in progress):** (1) trivial -- hoist `set_handle_tf32(false)` out of decode's
  per-token hot path into `kv_reset`; (2) shared atomicAdd-fan-in fix for `rmsnorm_backward_act_kernel`
  + `qknorm_backward_act_kernel` (both near-full occupancy already, both L1TEX-long-scoreboard-stall
  dominated from the same dgamma-atomic-contention root cause); (3) decode-loop CUDA-graph capture to
  collapse ~147 launches/token (currently 31% GPU busy, 68.6% of wall-time is `cudaLaunchKernel`
  dispatch overhead) -- the "big refactor" of this wave, has an existing parity gate
  (`sub0_cuda_forward_one_check`) to validate against.
- **Deferred, explicitly NOT bundled into this wave:** attention-backward register-pressure redesign
  (`attn_bwd_dk/dq/stats`, `attn_fwd_tiled` -- all near their own register-limited occupancy ceiling,
  8-12% theoretical occupancy from 168-182 regs/thread) -- this is attention's current 37.1%-of-step
  cost floor, the single largest remaining lever, but payoff is uncertain and it's the highest-risk
  change in the file. Needs its own scoped investigation/prototype before any implementation commitment.
- **Deferred, low priority:** checkpoint-save/compute overlap via a background writer thread (~3s hidden
  per boundary, occurs every 1000-8000 steps -- real but small).
- **Confirmed dead end, don't revisit:** B1 chunk-to-chunk multi-stream pipelining (GEMMs already
  near-peak SM throughput and occupancy, no idle bubble to fill).

**Phase 2 done (2026-07-07, via Fable), independently re-verified by me, committed `d4663b3`.** All
three greenlit items landed in `src/backend_cuda.cu` only (414/91 lines):
- `set_handle_tf32(false)` hoisted from per-token decode into `sub0_cuda_kv_reset` (once/session).
- Shared `flush_dgamma_partial<WIDTH,BLOCK>` fix for `rmsnorm_backward_act_kernel` +
  `qknorm_backward_act_kernel`'s atomicAdd fan-in: grid-stride over multiple rows/pairs per block,
  per-thread register accumulator, one atomicAdd-per-channel flush per block instead of per row. Grid
  size capped via `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (queried+cached) at a whole number of
  full-occupancy waves -- Fable's own first cut used a guessed constant, ncu caught a partial trailing
  wave costing up to 50% of runtime, fixed before reporting. rmsnorm: 619-625us -> 568.6us (~9-10%
  faster), occupancy held at 95.7%. qknorm: L1TEX-long-scoreboard stall-cycles/instruction dropped
  ~5x (91.6 -> 17.8). I independently re-verified: full line-by-line code read (including confirming
  `block_reduce_sum`'s pre-existing trailing `__syncthreads()` makes the new per-block loop-calling
  pattern safe -- a real question worth checking, not just trusting the comment), rebuilt `qknorm_prod`
  myself from source and reran its `[cuda]` suite (41,436,308/30, exact match) + QK-norm gradient check
  + decode parity, all green.
- Decode-loop CUDA-graph capture: `forward_one_device_graphed()` + `g_decode_state` (persistent 2-int
  device buffer for id/pos) + `_g` kernel variants (`embed_one_kernel_g`/`rope_one_kernel_g`/
  `attn_decode_kernel_g`) + new `kv_append_kernel` (needed since a captured memcpy node's destination
  address is fixed at capture time, but the KV-cache append position changes every token -- verified
  its indexing exactly matches the two raw `cudaMemcpyAsync` D2D calls it replaces). Deliberately
  deviated from the literal "cudaGraphExecKernelNodeSetParams per-node patching" instruction in favor of
  a fixed-pointer/changing-contents approach (same idiom the existing batched-forward
  `capture_graph()` already uses) -- reviewed and accepted, avoids ~45 manual per-node-handle
  bookkeeping sites for the same outcome. `invalidate_decode_graph()` wired to every existing
  `invalidate_graph()` trigger (verified complete via grep -- `kv_alloc`/`kv_free` have exactly one
  call site each, both covered). Real result: decode GPU-busy 31.16% -> 50.29%, `cudaLaunchKernel`
  API-time share 68.6% -> 1.5%, median inter-token interval ~1.86x faster, ~1.27x end-to-end wall-clock
  (interleaved A/B, this GPU's thermal confounds noted). I independently reran the decode parity gate
  (`sub0_cuda_forward_one_check`, 21/21) on two rebuilt configs.

**Attention-backward register-pressure investigation done (2026-07-07, via Fable), implementation
greenlit and in progress -- NOT yet committed.** Key reframe: the real binding constraint was
**shared memory occupancy, not registers** (registers were near their own ceiling, as expected, but
`Block Limit Shared Mem` was AT OR BELOW `Block Limit Registers` for every attention kernel -- the
32-33.5KB/block Ks/Vs tiles at `TILE_K=64` were the actual cap). Shrinking `attn_tile_k<HD>()`/
`attn_tile_q<HD>()`'s HD<=64 branch from 64->32 and widening the block size from 64->128 threads
(pure tile/grid-shape change, ZERO arithmetic changes, verified bit-exact in an isolated harness)
roughly doubles achieved occupancy and gives 1.6-2.3x wall-clock speedup on `attn_fwd_tiled_kernel`/
`attn_bwd_stats/dq/dv/dk_kernel` at production HD=64, with smaller-but-real gains at HD=32 -- measured
across 3 independent process launches to rule out this GPU's thermal drift (absolute times varied
4.7-19.5ms run-to-run, but speedup RATIOS held within 2%).

**Real negative results found and reported honestly (don't repeat these):** widening the existing
DQ/DK lane-split from LANES=2->4 at HD=64 REDUCED registers further (96) and raised occupancy (41.7%)
but made wall-clock WORSE (extra `__shfl_xor_sync` overhead outweighs the gain once the tile/block fix
is in) -- don't revisit without new evidence. ILP-splitting the dot-product accumulator: flat to
negative everywhere tested -- FMA-chain latency isn't a real stall cause here.

**One real risk I caught before greenlighting implementation:** `attn_tile_k<HD>()`/`attn_tile_q<HD>()`
are already HD-templated (safe to shrink, naturally scoped to HD<=64 by the existing branch structure)
but `kAttnTileQ` (the block-size constant) is a FLAT non-templated constexpr used at ~10 call sites --
a blanket 64->128 change would silently touch HD=96/128 configs too, which were only "verified by
inspection" in the investigation, not actually tested. Directed the implementation to make the widened
block size HD-templated too (mirroring `attn_tile_k<HD>()`'s own pattern), so HD>64 is bit-for-bit
provably unchanged rather than assumed fine -- exactly the dims-transfer trap this project has been
burned by before (see the d768-vs-d448 stale-baseline catch earlier in this same task).

**Not fully vetted, flagged as a possible future increment, NOT implemented:** extending the LANES
warp-cooperative split to `attn_fwd_tiled_kernel`/`attn_bwd_stats_kernel`/`attn_bwd_dv_kernel` (which
have no split today) showed a real additional increment at HD=32 in the isolated harness, but needs
the same correctness rigor (bf16-tolerance, gradient-check, HD=96 check) as the original DQ/DK split
before it's trustworthy -- don't fold into the current change, revisit separately if picked up.

**Implementation DONE (2026-07-07, via Fable), independently re-verified by me, committed `4dba86b`.**
`attn_tile_k<HD>()`/`attn_tile_q<HD>()`'s HD<=64 branch shrunk 64->32; `kAttnTileQ` (was a flat
constant) replaced with HD-templated `attn_block_q<HD>()` (128 for HD<=64, 64 -- unchanged -- above),
exactly the scoping I required before greenlighting. **A real bug self-caught by the verification
bar, not by inspection:** Fable's first pass only edited `attn_tile_k` (fwd/stats/dq), missing
`attn_tile_q` (dv/dk) -- fresh ncu on the REAL kernels (not the isolated harness, which was built
symmetrically and would never have caught this) showed dv/dk's shared memory and occupancy hadn't
moved, exposing the gap immediately. Fixed, rebuilt, reverified. This is the concrete payoff of
insisting on "verify the real kernels, not just the harness" as a standing rule.

Verified at 3 scales: HD=64 production (41.4M assertions), HD=32 (3.0M), and a NEW HD=96 build where
`sub0_cuda_attn_regcheck` proved byte-identical register/spill counts before/after (not inferred --
measured). I independently rebuilt `qknorm_prod` myself and reran: exact assertion-count match, plus
spot-checked the gradient-parity/register-budget/flash-attention-self-test cases directly (flash-attn
forward 13.1x vs naive, backward 175x -- both up from the pre-fix ~8.3x/~88x, both at unchanged
tolerance). Real result: occupancy roughly doubles across all five kernels, per-kernel time drops
1.5-2.2x, a real training step's total GPU kernel time -19.8%, full step 1.10-1.28x faster on
interleaved wall-clock. This closes out the attention-backward register-pressure investigation for
now -- the deferred LANES-widening ideas above remain open follow-ups, not blocking.
