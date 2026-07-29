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
every earlier participating execution. On CPU the tape handles this for free once the op exists. On
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

## 6. Staging

Each stage lands green on its own; the identity gate (`AGENTS.md` §4 — assertion AND test-case counts
unchanged on an unmodified default build) runs after every one.

- **Stage 0 — config axes, off by default.** `DEPTH_ATTENTION` (bool) and `DEPTH_ATTN_STRIDE` (int),
  emitted as `constexpr` by `tools/configurator.cpp`, consumed via `if constexpr`, added to
  `RunConfig`'s X-macro and to `ARCH_FINGERPRINT` — it changes COMPUTATION without changing
  `PARAM_FLOATS`, which is exactly the case the fingerprint exists to catch (`nfloat` is blind to it,
  so without this a depth-attention checkpoint would silently load into a plain build).
- **Stage 1 — CPU forward + backward** as a new tape op, gated by the finite-difference gradient check
  at BOTH a toy and a production-shaped config (`engine-tests-gradient-check-dims-dependent`).
- **Stage 2 — CUDA forward + backward**, with the cross-execution dK/dV accumulation of §4, gated by
  the existing CPU/CUDA gradient-parity harness plus a LOOPED parity case (a non-looped test cannot
  exercise the cross-execution path at all).
- **Stage 3 — arm D**, run at the SAME pinned batch/lr as arms A/B/C and matched on TOKENS. See
  `loopsplit-3arm-batch-confound` for why that is not optional.

## 7. What arm D is expected to show

If the fixed-point hypothesis is right, depth attention should help the LOOPED arm specifically and
help the plain arms much less — because it is the loop whose passes are otherwise near-identical. The
sharpest single number is not mean NELBO but the **context-length gain** already reported by
`sub0llm report`, which on the current three arms tracks execution count (A 14.2% / B 13.6% /
C 11.8%) rather than parameter count. A depth-attention loop arm that lifts B's gain toward or past
A's would be direct evidence for the mechanism.
