---
name: depth-attention-stage0-done
description: "Depth attention: Stage 0 + Stage 1 (CPU forward+backward) are DONE and green at two shapes; Stage 2 (CUDA) is NOT started and CUDA currently REFUSES to build with a non-zero stride. docs/DEPTH_ATTENTION.md is the handover"
metadata:
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-29T23:52:06.508Z
---

Arm D of the LoopSplit investigation. **`docs/DEPTH_ATTENTION.md` is the handover document** -- forward
quoted from the verified reference, backward hand-derived, memory analysis, staging, per-stage status.
Read it before writing any math.

## Settled by reading the real source (not recall)

1. **Zero new learnable parameters.** It rewrites only V (a softmax-weighted mixture across depth, per
   batch/kv-head/position) and passes K through untouched. `PARAM_FLOATS` and the checkpoint are
   unchanged; there is no weight gradient. Much cheaper than LoopSplit.
2. **It DOES participate in the loop.** The append runs on every EXECUTION, so a looped middle layer
   appends once per pass and later passes attend over earlier ones -- arm D is a real test of the
   mechanism ([[depth-attention-rescues-loopsplit-hypothesis]]).
3. The cache stores the **already-mixed** value, so the mixture is recursive across participating layers.

## DECISION 2026-07-29: BUILD arm D anyway

Two probes ([[fixed-point-hypothesis-FALSIFIED]], [[loop-repeat-extrapolation-probe]]) refuted depth
attention's *stated rationale* and I recommended stopping. **The user overrode that, correctly**: both
probes measured a model trained WITHOUT depth attention, so they are indirect evidence about depth
attention itself. Build it and get hard data before the later arms.

## DONE -- Stage 0 (356ac10) + Stage 1 (2cf2db3)

Stage 0: `--depth-attn-stride`, `DEPTH_ATTN_STRIDE` constexpr, `RunConfig` X-macro, `MODEL_ARCH_ID`,
and into `ARCH_FINGERPRINT`'s spare high byte so **stride 0 reproduces every fingerprint ever produced
BIT-IDENTICALLY** (pinned in `layout_tests.cpp` against a literal transcription of the old packing).
It must be in the fingerprint because it adds no parameters, so `nfloat` cannot catch a cross-load.

Stage 1: `Op::DepthAttn` + `op_depth_attn` (batched) and `depth_mix_row` (decode); `USE_DEPTH_ATTN` /
`DEPTH_CACHE_MAX` in layout.hpp; arena + node-pool sizing grown per execution.

## The two things worth remembering from Stage 1

1. **A gradient check cannot validate this op.** A no-op `return v;` has a perfectly correct backward
   and passes the finite-difference check -- verified by mutation, not assumed. What does work, and it
   takes both: `forward_one`-vs-`forward` parity catches ONE implementation dead (1.5 rel divergence),
   and a new "the mixed V depends on Q" test catches BOTH dead (movement exactly 0). V cannot depend on
   the query at all unless the mix is live; it observes through `kv_vrow_ptr` (the sequence cache holds
   the MIXED V), so it needs no test-only API. Generalizes: for any op whose no-op form is
   differentiable, add a PRESENCE test. See [[regression-test-on-reproducible-bug]] for the sibling rule.
2. **`Node`'s fixed a/b/w/bias fanout could not express the S+1 (K,V) inputs** -- the resolved 5a
   blocker. Fix: a `thread_local` side table keyed by cache slot, plus one `int Node::depth_s` recording
   the cache depth that node saw (unrecoverable at backward time, because the cache keeps growing after
   the node runs). Widening Node was rejected (hot arena object); pairwise decomposition was rejected as
   mathematically WRONG (the depth softmax is jointly normalised and does not factor).
   The cross-execution dK/dV needs no scheduling work on CPU -- not because "the tape handles it", but
   because `backward()` walks the pool in REVERSE and every write is `+=`.

Verified at two shapes, full suite green. The load-bearing one is **d128/L6/H4/kv2, middle 2 x3,
stride 2** -- the only config exercising `GQA_GROUP > 1` (the q-bar mean-reduce and the 1/G scatter) AND
`stride > 1`. Directional FD 1.45116 vs analytic 1.45148. Identity gate at stride 0: 44,153,269
assertions unchanged to the assertion, +1 test case (the presence test, 0 assertions when off).
Test build dir: `out/build/depthattn` (CPU, tests ON).

Deliberate divergence from the reference, documented in the doc's 3a: the stride gates on the
EXECUTION index, not the layer index. Makes `DEPTH_CACHE_MAX` a compile-time constant and spreads
entries evenly across passes; **identical at stride 1**, which is what arm D runs.

## Stage 2 (CUDA): DONE (b0a8bf9), designed in 813d8bf

Training (`forward_train` + `backward_device`) and batched inference (`forward_device`) run depth
attention on device. Gated on CPU/CUDA reduced-gradient parity in a build that is looped AND GQA AND
depth-attention, so the cross-execution path is really exercised (d128/L6/H4/kv2, middle 2 x3, stride 2,
5 slots; executions 1..9 all read earlier owners' slots):

| | grad rel-L2 | cos |
|---|---|---|
| depth ON  | 0.0127474 | 0.999919 |
| depth OFF (control, same shape) | 0.0102665 | 0.999948 |

The delta is one extra bf16 round-trip (the cache is `act_t`; dK/dQ accumulate through a bf16
read-modify-write), not a correctness gap. **Always run that control** -- without it the ON number alone
says nothing about whether the new op degraded parity.

NOT covered: `forward_one_device` (single-token CUDA decode) aborts loudly. Use the CPU backend to `gen`
from a depth-attention model. Train + report are fine.

`docs/DEPTH_ATTENTION.md` 5b is the design; four findings from tracing the real backend, none visible
from the CPU port, all of which survived implementation:

1. **No atomics needed.** Cache slot `s` is owned by execution `s*stride`; every reader has a HIGHER
   execution index; `backward_device` walks in REVERSE -- so readers run first and the accumulators are
   complete when the owner needs them. Plain `+=`. The attention backward's write-once invariant holds.
2. **The in-place mix destroys the own pre-mix V** that `dL/dp_S` needs (`launch_attn_train_t` takes ONE
   `in_stride` for Q/K/V, so the mixed V must land in `qkv`). Algebraic recovery is unstable; per-execution
   saving costs ~704 MB. Free fix: the backward recomputes `qkv` one execution at a time, so ONE reusable
   `[M,D_KV]` temp suffices, emitted as a second output of the mix kernel.
3. **No P storage** -- recompute the depth softmax in the backward, as the flash path is already P-free.
4. **Including the gradient accumulators, ~1.5x the doc's old forward-only estimate**: 264 MB per cache
   slot at the arm shape, so stride 1 = 4.2 GB (does not fit), stride 4 = 1.06 GB. `memplan.hpp`'s
   `train_scratch_bytes` must gain the term or the VRAM-fit batch over-commits.

One more thing implementation added, worth reusing: **`sub0::DEPTH_SCHEDULE`** (layout.hpp) gives
`live[e]` / `own[e]` per execution as compile-time constants, replacing an open-coded `e % stride` at
four sites (CPU forward, CPU decode, CUDA forward, CUDA backward) that each needed the same two numbers.
It carries the invariant that `own[e] == live[e]` -- an execution appends after mixing, so it never reads
its own slot -- and a `static_assert` ties it to `DEPTH_CACHE_MAX`, because a slot array short by one is a
silent device out-of-bounds write, not an allocation failure. Related: the cache VIEW helpers hand out
EVERY allocated slot rather than the first S, because the owner slot sits exactly one past the readable
range; a truncated view was an off-by-one waiting to happen. Compare the accumulate-vs-overwrite hazard
in [[cuda-padded-window-gradient-mismatch]].

## Memory is the binding constraint at the arm shape, and `stride` is what relieves it

Cache is `S` entries of `[B, N_KV_HEADS, T, D_HEAD]` for K and V, retained for the backward. At batch
448 / T 512 / 3 kv-heads / D_HEAD 32 / fp32 one entry-tensor is ~88 MB: stride 1 at 16 executions is
~2.8 GB (does not fit), stride 4 is ~0.7 GB (plausible). So **arm D will likely have to run stride 4,
not stride 1** -- which is where the exec-vs-layer gating divergence above actually bites. Report
measured VRAM with any quality number ([[three-pillar-shootout-policy]]). CUDA treats `qkv` as a
recomputed CHECKPOINT, so the depth cache cannot borrow those buffers.

## The number to watch

Not mean NELBO -- the **context-length gain** from `sub0llm report`. On the clean arms it tracks
execution count (A 14.2% / B 13.6% / C 13.3%, see [[loopsplit-3arm-clean-result-looping-works]]) rather
than parameter count. A depth-attention loop arm that lifts B's gain toward or past A's is direct
evidence for the mechanism.

## Build-graph gotcha: `sub0_core` does NOT compile backend_cuda.cu

In a GPU build dir the `.cu` belongs to its own **`sub0_backend_cuda`** target. `cmake --build --target
sub0_core` links the DLL without compiling it, and reports success. That let a CUDA compile break
(an unqualified `USE_DEPTH_ATTN`, fixed in 9b3988f) look verified when it was not. **To check a CUDA
change actually compiles, build `sub0_backend_cuda`** -- or the default target, not `sub0_core`.

## Gotcha hit while setting the test build up

Do NOT hardlink `generated/corpus.tok` between build dirs to skip re-tokenizing. The stamp keys on
vocab, so a differing vocab re-tokenizes anyway, and a write-through would corrupt the donor build.
(This time the configurator replaced the file rather than writing through, so nothing was lost.)
