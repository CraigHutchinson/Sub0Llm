# Depth attention — design, derived from the reference

Arm D of the LoopSplit investigation. The hypothesis it tests
(`depth-attention-rescues-loopsplit-hypothesis`): a looped block applies the SAME function repeatedly,
so its passes differ only by their input and converge toward a fixed point. Depth attention breaks that
symmetry by giving each pass a direct channel to what earlier passes computed.

Everything below is derived from the verified reference source
(`Nanbeige/Nanbeige4.2-3B/modeling_nanbeige.py`, fetched 2026-07-29), not from recall — per `AGENTS.md`
§5. The forward is quoted; the backward is hand-derived here and has no counterpart in the reference
(PyTorch gets it from autograd; this project does not).

## 1. What the reference actually does

Three functions: `_depth_attention_mix_value`, `_apply_depth_attention`,
`_apply_depth_attention_then_update_cache`. The load-bearing one:

```python
    keys   = [key   for key, _     in source_kv] + [current_key]
    values = [value for _,   value in source_kv] + [current_value]
    key_stack   = torch.stack(keys,   dim=0)
    value_stack = torch.stack(values, dim=0)
    logits = (query_for_kv.unsqueeze(0).float() * key_stack.float()).sum(dim=-1)
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    depth_probs = torch.softmax(logits * softmax_scale, dim=0).to(value_stack.dtype)
    return (depth_probs.unsqueeze(-1) * value_stack).sum(dim=0).to(current_value.dtype)
```

Four facts that decide the whole design:

1. **It rewrites only V. K is passed through untouched.** Sequence attention then runs normally on
   (unchanged K, mixed V). So this is *not* a second attention over the sequence — it is a per-position
   convex re-mixing of the value vector across depth.
2. **Zero new learnable parameters.** No projections, no gates, no norms. `PARAM_FLOATS` is unchanged,
   the checkpoint is unchanged, and there is no new weight gradient to derive. This is dramatically
   cheaper than LoopSplit was.
3. **The softmax is over the DEPTH axis** (`dim=0`), per (batch, kv-head, position) — a distribution
   over "which layer's value should this position use", scaled by `head_dim ** -0.5`.
4. **The cache stores the ALREADY-MIXED value**, because `value_states` is reassigned before the
   append:
   ```python
   value_states = _depth_attention_mix_value(...)
   if layer_idx % config.depth_attention_stride == 0:
       depth_attention_kv_cache.append((layer_idx, key_states, value_states))
   ```
   The mixture is therefore recursive across participating layers.

Under GQA the query is reduced to KV groups by a **mean over the G query heads of each group**:

```python
    return query.reshape(b, num_kv_groups, num_query_heads // num_kv_groups, t, d).mean(dim=2)
```

Config: `enable_depth_attention` (master flag) and `depth_attention_stride` (only layers with
`layer_idx % stride == 0` append to the cache; every layer still *reads* it).

## 2. It does participate in the loop — the hypothesis is testable

The open question recorded in the hypothesis note was whether the depth cache participates in the
LoopSplit repeated block or only in the un-looped head/tail. **It participates.**
`_apply_depth_attention_then_update_cache` is called from the attention forward, which runs on every
EXECUTION, and the append is gated on `layer_idx` alone. A middle layer executing R times therefore
appends R times — once per pass, each with that pass's own K/V.

So pass 2 attends over pass 1's key/value. The passes are no longer the same function of their input:
later passes have strictly more context. That is precisely the symmetry-breaking the hypothesis
predicted, which makes arm D a real test of it rather than a loosely-related feature.

## 3. Forward, in this project's terms

Per batch item `b`, KV head `h`, position `t`. All vectors are `D_HEAD`-long. Let the participating
depth entries be `d = 0 .. S-1` (cached) plus `d = S` (current), and `s = D_HEAD^(-1/2)`.

```
  q̄      = (1/G) · Σ_{g ∈ group(h)}  q[h·G + g]        (GQA mean-reduce; identity when G == 1)
  ℓ_d     = s · ⟨q̄, k_d⟩                                d = 0..S
  p       = softmax(ℓ)                                  over the DEPTH axis
  v_out   = Σ_d  p_d · v_d
```

`k_S`, `v_S` are the current execution's own key/value; `k_d`, `v_d` for `d < S` come from earlier
executions' caches. Only `v_out` leaves — `k_S` continues to sequence attention unmodified.

## 4. Backward, derived here

Given `ḡ = ∂L/∂v_out` (a `D_HEAD` vector), for each `d = 0..S`:

```
  ∂L/∂v_d = p_d · ḡ                                    ← CROSS-EXECUTION for d < S
  ∂L/∂p_d = ⟨ḡ, v_d⟩
  ∂L/∂ℓ_d = p_d · ( ∂L/∂p_d − Σ_e p_e · ∂L/∂p_e )      standard softmax Jacobian
  ∂L/∂k_d = s · ∂L/∂ℓ_d · q̄                            ← CROSS-EXECUTION for d < S
  ∂L/∂q̄  = s · Σ_d ∂L/∂ℓ_d · k_d
  ∂L/∂q[h·G + g] = (1/G) · ∂L/∂q̄                       scatter back over the group
```

**The structural cost is in the two lines marked CROSS-EXECUTION.** The reverse pass is no longer a
clean per-execution walk: execution `e`'s depth-attention backward must ACCUMULATE into the dK/dV of
every earlier participating execution. On CPU this needed no scheduling work, but not because "the tape
handles it" — it is specifically because `backward()` walks the node pool in REVERSE and every gradient
write on that backend is `+=`. A depth node sits after every node it read, so an earlier execution's K/V
node has received every contribution by the time the walk reaches it and propagates onward. On
CUDA the reverse pass is hand-written and strictly per-execution today, so it needs per-execution dK/dV
accumulation buffers that stay live across the whole backward — this is the single largest piece of
work in the port, and the one most likely to be silently wrong (compare the accumulate-vs-overwrite
hazard LoopSplit already hit: `cuda-padded-window-gradient-mismatch`).

## 5. Memory — the reason `depth_attention_stride` exists

The cache holds `S` entries of `[B, N_KV_HEADS, T, D_HEAD]` for K **and** V, and training must keep
them for the backward. At the arm-A/B shape (batch 448, T 512, N_KV_HEADS 3, D_HEAD 32, fp32):

```
  per entry, per tensor : 448 · 512 · 3 · 32 · 4 B ≈ 88 MB
  stride 1, 16 executions: 16 · 2 · 88 MB ≈ 2.8 GB      ← does not fit alongside training scratch
  stride 4, 16 executions:  4 · 2 · 88 MB ≈ 0.7 GB      ← plausible
```

So **stride is not a tuning nicety, it is what makes this fit at all**, and arm D must report the
measured VRAM alongside quality per the standing three-pillar policy. Note the CUDA training forward
currently treats `qkv` as a CHECKPOINT (recomputed in backward), so the depth cache cannot simply
borrow those buffers — it needs its own retained storage or its own recompute.

## 3a. One deliberate divergence: the stride gates on the EXECUTION, not the layer

The reference gates the cache append on `layer_idx % stride == 0`. This engine gates on the EXECUTION
index instead (`op_depth_attn`'s call site in `Model::forward`). Two reasons, and one check:

- The live entry count becomes exactly `ceil(LOOP_EXEC_COUNT / stride)` (`DEPTH_CACHE_MAX`), which is
  what the arena and node-pool sizing need as a compile-time constant. Gating on layer index makes the
  count depend on how the loop schedule happens to interleave, for no benefit.
- The entries spread evenly across passes rather than clustering wherever the participating layer
  indices land in the schedule.
- **At stride 1 the two are identical** — every execution participates either way — and stride 1 is what
  the arm-D measurement runs. So this cannot affect the result being measured.

## 5a. RESOLVED in Stage 1 — `Node` could not express the depth cache

`sub0::Node` (core.hpp) has a FIXED fanout — `a`, `b`, `w`, `bias`. Depth attention consumes `S+1`
(K, V) pairs, where `S` grows with the execution index. **There is no way to hang a variable-length
input list off a Node**, and the backward pass needs those exact nodes to accumulate `∂L/∂k_d` and
`∂L/∂v_d` into (§4's CROSS-EXECUTION terms). This is the first thing Stage 1 must solve, and it is not
visible from the reference (PyTorch just holds a Python list and autograd tracks it).

Options considered:

1. **Widen `Node`** — add a pointer array or a span of inputs. Rejected: `Node` is the hot arena
   object, `MAX_NODES`/`ACT_CAP` are sized from it (`backend_cpu.cpp:84-102`), and every op pays for a
   field only one op uses.
2. **Decompose into pairwise ops** — rejected as WRONG, not just costly: the depth softmax is jointly
   normalised over all `S+1` entries, so it does not factor into a chain of pairwise mixes.
3. **CHOSEN, and implemented — a per-execution side table.** It landed as a file-scope
   `thread_local DepthCache` in `backend_cpu.cpp` rather than a `Model` member (the free `op_*` functions
   are defined above `Model` and cannot see it), sized `LOOP_EXEC_COUNT` so the array is never
   zero-length. `Node` gained one `int depth_s` — the recorded cache depth, which cannot be recovered at
   backward time because the cache keeps growing after the node runs. The exec count is a compile-time
   constant, so the sketch was:
   ```cpp
   Node* depth_k[sub0::LOOP_EXEC_COUNT] = {};   // K contributed by execution e (nullptr = not cached)
   Node* depth_v[sub0::LOOP_EXEC_COUNT] = {};
   int   depth_n = 0;                            // how many are live in THIS forward
   ```
   `forward()` appends when `e % DEPTH_ATTN_STRIDE == 0`; the `Op::DepthAttn` node records its own
   execution index (reuse the spare `heads` field, or add one `int`), and `backward_node` reads the
   table to find the nodes whose `grad` spans it must accumulate into.

   This is sound because the table is `thread_local` alongside the Model, backward runs on the same
   thread before `graph_reset()`, and the arena keeps those nodes alive for exactly that window — the
   same lifetime assumption `op_embed` already relies on for its `ids` pointer (see `forward()`'s
   `static thread_local pos_ids` comment). **Reset `depth_n = 0` at the top of every `forward()`**, or a
   second forward in the same graph would mix the first one's cache into it.

Two further Stage-1 requirements this exposes:

- **`MAX_NODES` / `ACT_CAP` must grow**: one extra node per execution, each holding a `[T, D_KV]`
  output. Under-sizing the arena is a silent overwrite, not an allocation failure.
- **The gradient check must use `DEPTH_ATTN_STRIDE = 1` and `LOOP_REPEATS >= 2`**, so at least one
  execution really does read a cache entry from an earlier execution. At stride 0 (off) or repeats 1
  the cross-execution path is never taken and the check would pass while proving nothing — the same
  trap the CUDA parity test has to avoid (§6, Stage 2).

## 5b. Stage 2 (CUDA) design, derived from tracing the real backend

Four findings from reading `forward_train` / `backward_device` (not from the CPU port, which they do not
resemble). Each one changes the plan.

**1. The cross-execution accumulation needs NO atomics, and the reason is an ordering argument.**
Cache slot `s` is owned by exactly one execution, `e_own(s) = s * DEPTH_ATTN_STRIDE`. Every execution
that READS slot `s` has `e > e_own(s)`. `backward_device` walks executions in REVERSE, so every reader
is processed BEFORE the owner. Therefore, by the time the walk reaches `e_own(s)`, the accumulators
`ddepth_k[s]` / `ddepth_v[s]` are complete and can simply be added into that execution's `dqkv` K/V
sub-blocks. Within one kernel launch, one block owns one `(row, kv-head)` exclusively; across launches,
same-stream ordering serialises them. So plain `+=`, no atomics, and the write-once invariant the
attention backward protects is untouched.

**2. The in-place mix DESTROYS the own V that the backward needs — and the fix is free.**
`v_out = Σ_d p_d·v_d` must land in `qkv`'s V sub-block, because `launch_attn_train_t` takes ONE
`in_stride` for Q/K/V and so cannot read a mixed V from a separate buffer. But §4's `∂L/∂p_S = ⟨ḡ, v_S⟩`
needs the execution's OWN pre-mix V, which the overwrite has just destroyed. Recovering it by algebra
(`(v_out − Σ_{d<S} p_d v_d) / p_S`) is unstable for small `p_S`. Saving it per execution costs
`LOOP_EXEC_COUNT` buffers (~704 MB at the arm shape) — unaffordable.
The fix: `backward_device` already RECOMPUTES `qkv` per execution and processes ONE execution at a time,
so a **single reusable `[M, D_KV]` temp** suffices, and the mix kernel gets it for free as a second
output — it is already reading the own V. ~44 MB at the arm shape, once.

**3. No softmax-probability storage.** The depth `p` can be recomputed in the backward from the
recomputed `q`/`k` plus the retained cache, exactly as `launch_attn_train_t` is already P-free. This
removes an `[M, N_KV_HEADS, S+1]` per-execution buffer from the plan entirely.

**4. Buffer inventory, and it is what picks arm D's stride.**
Per cache slot, at `[M, D_KV]`: K and V retained as `act_t` (2 B, matching how `qkv` is stored) plus
their two f32 accumulators (4 B) = 12 B per element. At the arm shape (M = 448·512, D_KV = 96):
```
  M · D_KV                     = 22.0 M elements
  per slot (2·2 B + 2·4 B)     = 264 MB
  stride 1 -> 16 slots         = 4.2 GB   <- does not fit
  stride 4 ->  4 slots         = 1.06 GB  <- tight but plausible on 8 GB
```
This supersedes §5's forward-only estimate: including the gradient accumulators is ~1.5x that figure.
`memplan.hpp`'s `train_scratch_bytes` must gain the term, or the VRAM-fit batch estimate silently
over-commits.

**Kernel shape** (both directions): grid `(rows, N_KV_HEADS)`, block 32, each thread striding over
`a < D_HEAD`; `q̄` and the `S+1` probabilities in shared memory; one `block_reduce_sum` per depth entry
for the logit and (in the backward) for `∂L/∂p_d`. The cache pointers travel as a by-value struct of
`DEPTH_CACHE_MAX` pointers, so no device-side pointer table is needed.

**Ordering inside one execution's backward**, which is the part most likely to be silently wrong:
```
  attn_bwd                      -> dqkv (dQ, dK, dV_mixed)
  depth_attn_bwd:
      if owner: dV_mixed += ddepth_v[s];  dK += ddepth_k[s]   (contributions from LATER executions)
      consume dV_mixed -> dV_own (OVERWRITE V), dK += , dQ += (1/G scatter)
      scatter into ddepth_k[d], ddepth_v[d] for every d < S
  rope_bwd  ->  qknorm_bwd  ->  qkv GEMM bwd
```
The `if owner` step must happen before `dV_mixed` is consumed, and both K contributions are additions
into the same sub-block RoPE's backward later rotates — which is consistent, because the depth logits
are formed from the SAME post-RoPE, post-QK-norm q/k that attention uses.

**Still open, and deliberately so**: `forward_one_device` (decode). Its rows=1 cache is trivial in size,
but it lives inside a captured decode graph. Training and eval/report go through `forward_train` /
`forward_device`, so arm D does not need decode — but it must not silently skip the mix either, so the
compile-time refusal stays until decode is covered too.

## 6. Staging

Each stage lands green on its own; the identity gate (`AGENTS.md` §4 — assertion AND test-case counts
unchanged on an unmodified default build) runs after every one.

- **Stage 0 — config axes, off by default.** `DEPTH_ATTENTION` (bool) and `DEPTH_ATTN_STRIDE` (int),
  emitted as `constexpr` by `tools/configurator.cpp`, consumed via `if constexpr`, added to
  `RunConfig`'s X-macro and to `ARCH_FINGERPRINT` — it changes COMPUTATION without changing
  `PARAM_FLOATS`, which is exactly the case the fingerprint exists to catch (`nfloat` is blind to it,
  so without this a depth-attention checkpoint would silently load into a plain build).
- **Stage 1 — CPU forward + backward — DONE.** `Op::DepthAttn` + `op_depth_attn` (batched) and
  `depth_mix_row` (decode), option 3's side table, arena/pool sizing, and a CUDA `static_assert` that
  refuses to build a depth-attention GPU binary until Stage 2 exists (a silent no-op there would train a
  different architecture while every shape check still agreed). Verified at two shapes, both green:
  | shape | executions | stride | GQA_GROUP | directional FD vs ‖g‖ |
  |---|---|---|---|---|
  | d196 L11 H7, middle 3 ×2 | 14 | 1 | 1 | (green, full suite) |
  | d128 L6 H4 kv2, middle 2 ×3 | 10 | 2 | 2 | 1.45116 vs 1.45148 |
  The second shape is the one that matters for coverage: it is the only one exercising `GQA_GROUP > 1`
  (the q̄ mean-reduce and the `1/G` scatter of §4's last line) and `stride > 1`.
  **Mutation-tested, because the gradient check alone proves nothing here**: a no-op `return v;` has a
  perfectly correct backward and passes the finite-difference check. Stubbing each implementation in turn
  showed the guards are `forward_one`-vs-`forward` parity (catches one side dead: 1.5 relative divergence)
  plus the new "mixed V depends on Q" test (catches BOTH dead: movement exactly 0). Neither alone suffices.
  Identity gate: at the default stride 0 the suite is assertion-identical (44,153,269) with +1 test case,
  the new presence test, which runs zero assertions when depth attention is off.
- **Stage 2 — CUDA forward + backward — DONE.** Training (`forward_train` + `backward_device`) and
  batched inference (`forward_device`); §5b's design held up in full. Gated on CPU/CUDA reduced-gradient
  parity in a build that is looped AND GQA AND depth-attention (d128/L6/H4/kv2, middle 2 ×3, stride 2,
  5 slots), so the cross-execution path is genuinely exercised — **with the depth-off control at the
  same shape**, without which the ON number says nothing:
  | | grad rel-L2 | cos |
  |---|---|---|
  | depth ON | 0.0127474 | 0.999919 |
  | depth OFF (control) | 0.0102665 | 0.999948 |
  The delta is one extra bf16 round-trip (the cache is `act_t`; dK/dQ accumulate through a bf16
  read-modify-write), not a correctness gap. `sub0::DEPTH_SCHEDULE` was added so the four sites needing
  per-execution slot bookkeeping share one compile-time source of truth.
  **Not covered**: single-token CUDA decode (`forward_one_device`) aborts loudly rather than skipping the
  mix. `gen` from a depth-attention model on the CPU backend; train and report are fine.
- **Stage 3 — arm D**, run at the SAME pinned batch/lr as arms A/B/C and matched on TOKENS. See
  `loopsplit-3arm-batch-confound` for why that is not optional. **In flight** at stride 4: it fits at
  batch 448 (stride 1 would not — §5b finding 4 was right), costs ~14% throughput against arm B
  (95,398 vs 111,415 tok/s, and arm D's is a warmup figure so the real gap is smaller), and the appends
  land at executions 0/4/8/12 so pass 2 does read slots contributed by the head and by pass 1 — the
  channel under test is exercised rather than accidentally null.

## 7. What arm D is expected to show

If the fixed-point hypothesis is right, depth attention should help the LOOPED arm specifically and
help the plain arms much less — because it is the loop whose passes are otherwise near-identical. The
sharpest single number is not mean NELBO but the **context-length gain** already reported by
`sub0llm report`, which on the current three arms tracks execution count (A 14.2% / B 13.6% /
C 11.8%) rather than parameter count. A depth-attention loop arm that lifts B's gain toward or past
A's would be direct evidence for the mechanism.
