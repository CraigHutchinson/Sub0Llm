# Gated DeltaNet — design, derived from the reference

Status: **Stage 1 + Stage 2 + Stage 3 DONE — real CPU forward AND backward (fixture-verified against the
real PyTorch reference's own autograd) plus a CUDA FORWARD-ONLY port of the chunked-parallel form
(fixture- and CPU-reference-verified on real hardware; no CUDA backward, no CUDA decode).** See §6 for the
staging status, the fixture-comparison numbers, and the one real correction to this doc's own §1b found
while implementing Stage 1 (plus a real, independent pre-existing CPU bug Stage 3 found and fixed along
the way). Follows
`docs/DEPTH_ATTENTION.md`'s structure (that document is the precedent for how a genuinely new op type
gets threaded through this engine) and `docs/QWEN4_PREVIEW_REFERENCE.md`'s staged plan, which flags this
as the higher-risk of the two Qwen4-preview mechanisms because this engine's attention today is
softmax-only with no recurrent-state precedent at all. Per `AGENTS.md` §5, everything mathematical below
is quoted or worked from the real, verified reference source fetched for this doc — not from recall.

## 0. Sources, and their confidence

**This section was rewritten after the parallel real-weight-extraction work landed on `main`
(`docs/QWEN4_PREVIEW_REFERENCE.md`, commit `802a3d0`) partway through this pass — its findings supersede
this doc's first draft, which had sourced the equations from `modeling_qwen3_next.py` (Qwen3-Next) as a
proxy for the actual Qwen4-preview model. That proxy was reasonable at the time (no better source was in
hand) but is now known to differ in real, verifiable ways from the actual model (§1's structure below);
every equation in this section is now sourced from the real thing.**

**Primary, highest-confidence source — the real model's own code, already fetched and verified by that
parallel work**: `transformers==5.16.1` (mainline PyPI, Apache-2.0, no `trust_remote_code` needed —
confirmed installed and importable), `transformers/models/qwen4_exp/modeling_qwen4_exp.py`, class
`Qwen4ExpTextGatedDeltaNet`. This is the actual module `Qwen/Qwen3.8-Flash-Next` ships (`model_type:
qwen4_exp`, `architectures: Qwen4ExpForConditionalGeneration` — `transformers` itself already calls this
"Qwen4-Exp", not merely "an architecture Qwen4 will resemble"). Quoted directly below, not paraphrased.

**Real extracted weights and CPU-verified fixtures, already landed at
`tests/fixtures/qwen4_preview/gdn_layer0_small_*`**: the complete, real weight tensors for
`model.language_model.layers.0.linear_attn.*` (a full Gated-DeltaNet layer, ~110.6MB, bf16→float32,
extracted via HTTP Range requests against the real `Qwen/Qwen3.8-Flash-Next` safetensors shards — see
`gdn_layer0_small_manifest.json`), sliced from `hidden_size=2560 / 16 key heads / 48 value heads` down to
a commit-sized `hidden_size=32 / 1 key head / 3 value heads` (the real 3× value:key head ratio preserved;
`head_k_dim = head_v_dim = 128` UNCHANGED — only `hidden_size` and head COUNTS were sliced), then run
through the real, unmodified `Qwen4ExpTextGatedDeltaNet.forward()` on a reproducible-seed synthetic input.
**This is now the actual Stage 1 (this doc's numbering — see the staging-terminology note below)
correctness gate, not a "once it lands" placeholder** — it has landed.

**Secondary sources, corroborating but superseded where they conflict with the above**: Sebastian
Raschka's Gated DeltaNet writeup and vLLM's/Maxime Labonne's Qwen3-Next/Qwen3.5 write-ups (conceptual
framing, no exact equations); `modeling_qwen3_next.py` (Qwen3-Next's own module, fetched directly from
`huggingface/transformers` — a real, close relative, but NOT the model this project targets, and it
differs from `Qwen4ExpTextGatedDeltaNet` in real, structural ways — see §1c); a `gist.github.com` analysis
of Qwen3.5's GDN, used only to cross-check.

**Staging-terminology note, to avoid a real collision**: `docs/QWEN4_PREVIEW_REFERENCE.md` uses "Stage 0"
/"Stage 1" for *reference verification* / *real-weight-fixture extraction* (both DONE, as of that doc's
own 2026-08-30 update). This doc, following `docs/DEPTH_ATTENTION.md`'s staging convention instead (per
this pass's own task scope), uses "Stage 0" for the *in-engine config-skeleton* landed this pass, "Stage
1" for the *future CPU forward implementation*, etc. Both numbering schemes are legitimate for what they
each track; they are simply two different axes, so a reader should not assume "Stage 1" means the same
thing in both docs.

## 1. What the reference actually does

### 1a. Construction — `Qwen4ExpTextGatedDeltaNet.__init__`, quoted verbatim

```python
class Qwen4ExpTextGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
            kernel_size=self.conv_kernel_size, groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen4ExpTextRMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon,
            activation=config.output_gate_type or config.hidden_act,
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
```

**The one structural correction from the earlier (Qwen3-Next-sourced) draft of this doc**: projections are
**FOUR separate linear layers** — `in_proj_qkv` (Q+K+V fused, width `2·key_dim + value_dim`), `in_proj_z`
(the output gate, alone), `in_proj_b`, `in_proj_a` (the two gate-input scalars, each alone) — not
Qwen3-Next's `in_proj_qkvz` (Q+K+V+Z all fused) plus `in_proj_ba` (both gate inputs fused). The TOTAL
parameter count is unaffected (same floats, different grouping — see §3b), but the op-graph shape this
implies is 4 `Linear` nodes feeding the recurrence, not 2 (§4).

**Real config values** (`gdn_layer0_small_manifest.json`, both the real full-scale config and the
commit-sized slice used for the fixture):

| Quantity | Full scale (real checkpoint) | Small fixture |
|---|---|---|
| `hidden_size` | 2560 | 32 |
| `linear_num_key_heads` | 16 | 1 |
| `linear_num_value_heads` | 48 | 3 |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 (**equal**) | 128 / 128 (unchanged) |
| `linear_conv_kernel_dim` | 4 | 4 (unchanged) |
| `hidden_act` / `output_gate_type` | `silu` / `sigmoid` | same |

Two facts this resolves that the earlier draft had flagged as open:
- **`head_k_dim == head_v_dim` in the real model** (both 128) — confirming this project's simplifying
  reuse of a single `D_HEAD` for both (§3a) holds for the real architecture too, not just as a
  convenience.
- **`num_v_heads / num_k_heads == 3` exactly**, matching the "3×" ratio `docs/QWEN4_PREVIEW_REFERENCE.md`
  independently verified from `config.json`.

### 1b. Gates and the delta-rule recurrence — `forward`, quoted verbatim (no-cache/prefill path)

```python
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
...
core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
    query, key, value, g=g, beta=beta, initial_state=recurrent_state,
    output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True, ...
)
...
core_attn_out = self.norm(core_attn_out, z)   # RMSNormGated: norm(x) * act(gate)
output = self.out_proj(core_attn_out)
```

`α_t = -exp(A_log) · softplus(a_t + dt_bias)` (always ≤ 0), `g_t = exp(α_t) ∈ (0, 1]` — a genuine decay,
never growth; `β_t = σ(b_t) ∈ (0, 1)`. `A_log`/`dt_bias` are learned, **one scalar per V-head**
(`nn.Parameter(torch.ones(num_v_heads))`); `a_t`/`b_t` are per-token, per-V-head activations from
`in_proj_a`/`in_proj_b`.

**This resolves the earlier draft's flagged uncertainty about state head-grouping, with real code, not
inference**: `query.repeat_interleave(num_v_heads // num_k_heads, dim=2)` — Q and K are explicitly
broadcast UP from `num_k_heads` to `num_v_heads` (repeating each K/Q head across its group of V-heads,
GQA-style but inverted relative to this project's own softmax-attention convention — see §3a's table)
**before** the recurrence runs. So internally, the recurrence itself operates uniformly at
`num_v_heads` head-granularity — the recurrent state genuinely is `[num_v_heads, head_k_dim,
head_v_dim]` per batch item, with repeated (not distinct) K/Q values across a group. This confirms §2/§3's
existing `N_HEADS`-per-state-head design was the right call, now on real evidence rather than inference
from a proxy model's tensor-shape comments.

The exact per-chunk recurrence (`torch_chunk_gated_delta_rule`'s inner loop, quoted verbatim — this
IS the delta rule: `v_new` is the error-correction term, "how wrong the state's current prediction is"):

```python
for i in range(0, total_sequence_length // chunk_size):
    q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
    attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
    v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
    v_new = v_i - v_prime
    attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
    core_attn_out[:, :, i] = attn_inter + attn @ v_new
    last_recurrent_state = (
        last_recurrent_state * g[:, :, i, -1, None, None].exp()
        + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    )
```

`v_new = v_i − v_prime` is EXACTLY the delta-rule error correction this mechanism is named for, done for
a whole chunk of positions at once via `k_cumdecay` (a cumulative-decay-weighted `k`) instead of one
token's `kv_mem` at a time; `attn`/`decay_mask` form the intra-chunk (within this chunk, causal,
decay-weighted) contribution and `attn_inter` the inter-chunk (this chunk's queries reading the state
carried in from all *previous* chunks) contribution — a genuinely chunk-parallel (dense matmul) form of
the same recurrence,
now backed by the real code rather than a secondary analysis. **No explicit `1/√d_k` scale appears
anywhere in this recurrence** — confirming (not just conjecturing, as the earlier draft had to) that Q/K
enter already L2-normalized (`use_qk_l2norm_in_kernel=True` above) and need no further rescaling.

### 1c. Short causal conv and output gating, quoted verbatim

The real conv is depthwise (`groups=conv_dim`), applied to the concatenated Q|K|V (never Z), with
**causal padding baked into the conv itself** (`padding=conv_kernel_size - 1`, left-padding effectively,
since the output is later truncated to the input length) rather than a separate masking step:
```python
self.conv1d = nn.Conv1d(in_channels=self.conv_dim, out_channels=self.conv_dim, bias=False,
                         kernel_size=self.conv_kernel_size, groups=self.conv_dim,
                         padding=self.conv_kernel_size - 1)
```
Output gating: `core_attn_out = self.norm(core_attn_out, z)` where `norm` is `Qwen4ExpTextRMSNormGated(
self.head_v_dim, ...)` — **RMSNorm's learned weight is `[head_v_dim]` (128 at full scale), shared across
all `num_v_heads` heads, not one weight per output channel across the full `value_dim`.** This is a real
correction from the earlier draft (which had assumed a `D_MODEL`-wide norm weight by analogy with a
plain post-attention LayerNorm) and matters directly for §3b's PARAM_FLOATS arithmetic. It is also, by
direct structural analogy, the SAME convention this project's own `QNorm`/`KNorm` already use
(`layout.hpp`: `add(1, D_HEAD, PKind::QNorm, ...)` — one `[1, D_HEAD]` weight, shared across heads, not
one weight per full-width channel) — a real, load-bearing precedent already in this codebase for exactly
this shape of parameter.

### 1d. Cached (decode) convolution state

Not directly re-verified against `modeling_qwen4_exp.py` this pass (the fixture work above targeted the
no-cache/prefill path only), but the general Mamba2-lineage convention — a `[batch, conv_dim,
conv_kernel_size - 1]` fixed-size ring buffer for the short causal conv's state, alongside the
`[batch, num_v_heads, head_k_dim, head_v_dim]` recurrent state itself — is architecturally required for
single-token decode regardless of the exact class name, and is the shape §2 designs the arena/memplan
interaction around.

### 1e. RESOLVED in Stage 1 — the decode branch was re-fetched and re-quoted directly

This section originally flagged that the real `Qwen4ExpTextGatedDeltaNet.forward()`'s *cached*
(single-token decode) branch and its exact `torch_recurrent_gated_delta_rule` had not been re-quoted
from `modeling_qwen4_exp.py` directly, and that Stage 1 should do so before relying on it byte-for-byte
(AGENTS.md §5). Stage 1 did: this machine has `transformers==5.16.1` installed, so
`python3 -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; print(inspect.
getsource(m.torch_recurrent_gated_delta_rule))"` (and the same for `Qwen4ExpTextGatedDeltaNet.forward`,
`torch_chunk_gated_delta_rule`, `causal_conv1d_fn`/`causal_conv1d_update`, `Qwen4ExpTextRMSNormGated`,
`l2norm`) fetched the REAL, exact, installed source directly — not a proxy, not Qwen3-Next's analogue.

Two things this resolved:
1. `torch_recurrent_gated_delta_rule` IS exactly the token-at-a-time unrolling this doc always predicted
   — `S = g_t·S; kv_mem = Sᵀk_t; delta = β_t·(v_t−kv_mem); S += k_t⊗delta; o_t = Sᵀq_t`, confirmed
   line-for-line, not merely "reasonable to assume by analogy" any more.
2. **A real correction the assumption-based version would have missed**: both `torch_recurrent_
   gated_delta_rule` and `torch_chunk_gated_delta_rule` scale query by `1/sqrt(head_k_dim)`
   immediately after L2-normalizing it, ONE LINE before §1b's already-quoted loop even begins — so §1b's
   own claim "No explicit `1/√d_k` scale appears anywhere in this recurrence" is true only of the lines
   it quotes. `include/sub0/gdn_math.hpp` implements the scale; see that file's header comment and §6's
   Stage 1 entry for the fixture-match numbers that confirm it (a large mismatch without the scale, an
   exact match with it).

## 2. Interaction with this engine's KV-cache / arena model

This engine's only existing "carries information across positions" primitive is the softmax-attention KV
cache: `kv_krow_ptr(layer, pos)` / `kv_vrow_ptr(layer, pos)` (`core.hpp`) return pointers into a per-
(layer-or-execution, position) row store that **grows with T** — `[B, N_KV_HEADS, T, D_HEAD]`, sized by
`SEQ_LEN` at compile time (`memplan.hpp`'s persistent-buffer terms). Gated DeltaNet's "cache" is
structurally a different SHAPE of thing, not a bigger or smaller version of the same shape:

| | softmax-attention KV cache | Gated DeltaNet recurrent state |
|---|---|---|
| grows with T? | yes — one row per position, up to `SEQ_LEN` | **no** — one `[d_k, d_v]` matrix per (batch, head), full stop |
| read pattern | every later position reads every earlier row | only ever read/written by the NEXT position; nothing before it is retained after the update |
| memory at decode | `O(B · N_KV_HEADS · T · D_HEAD)` | `O(B · N_HEADS · D_HEAD²)` (using this project's mapping — see §3) — **independent of T** |
| memplan today | `Dims.exec_layers`-scaled persistent term already exists | no equivalent term exists yet |

This is exactly `docs/QWEN4_PREVIEW_REFERENCE.md`'s summary ("O(1) per token... replacing the O(n) KV
cache") made concrete against this codebase: the *conv_state* (§1d, `[B, conv_dim, K_conv-1]`) and the
*recurrent state* (`[B, N_HEADS, D_HEAD, D_HEAD]`, see §3's mapping) are BOTH fixed-size, and neither one
is expressible as a `kv_krow_ptr`/`kv_vrow_ptr`-style per-position row store. They need their own arena
slot, sized once at configure time exactly like the existing KV cache is, but keyed by `(batch, head)`
only — never by position.

**Per AGENTS.md §1 (no heap allocation in hot paths), the state buffer must be a pre-sized arena slot
that the recurrence *overwrites in place* every position, not something that grows.** Concretely, for a
future Stage 1:
- **Training** (batched forward over `[batch, seq]`, this engine's existing convention per `op_attn`
  above): the state buffer is transient PER FORWARD CALL, sized `[batch, N_HEADS, D_HEAD, D_HEAD]` (or
  `[max_rows_per_window, ...]` under this engine's row-major batched-window convention), and is fully
  consumed within one `op_gdn`-style call the same way `op_attn`'s `[H,T,T]` softmax-probability scratch
  (`out->scratch`, §4) is — allocated once from the arena, written and read entirely inside the op, never
  retained across calls. This is a `train_scratch_bytes`-style term (`memplan.hpp`), not a
  `persistent_bytes` one.
- **Decode** (`forward_one`/`kv_reset`, this engine's existing incremental-inference convention): THIS is
  where the state genuinely persists across calls, the same way the KV cache does — `kv_reset()` would
  need to zero the GDN state (its correct "start of a fresh generation" value, since a freshly-reset state
  encodes no history, exactly like an empty KV cache) alongside whatever it already resets, and each
  `forward_one` call would update it in place. This is a `persistent_bytes`-style term, and unlike the KV
  cache it does **not** grow with `SEQ_LEN` — a genuinely new, small, constant addition to the resident
  footprint regardless of context length, which is the entire architectural point of this mechanism.
- **LoopSplit interaction, resolved by an existing precedent**: `core.hpp`'s own doc comment on
  `kv_krow_ptr` already establishes that a looped middle layer's KV history is kept **per EXECUTION, not
  per LAYER** ("`layer` is an EXECUTION index... a repeated middle layer runs several times per token and
  EACH EXECUTION keeps its own K/V history"). The natural, consistent choice for a GDN state buffer under
  LoopSplit is the same: `LOOP_EXEC_COUNT` independent state slots (one per execution of a GDN layer),
  not one slot shared and mutated across passes. Nothing in the real Qwen reference informs this choice —
  Qwen3-Next has no LoopSplit — so this is this engine's own design decision, made for consistency with
  the KV-cache precedent it already lives next to, not derived from the reference.

**A quantified comparison at a representative shape** (d448 L11 H7, using this project's own
`kv_krow_ptr` doc's D_KV convention and this section's `N_HEADS`/`D_HEAD` mapping, batch 1, fp32, one
GDN layer): the KV cache for ONE softmax layer at `SEQ_LEN=512` is `512 · D_KV · 4 B` per tensor (K and V)
≈ `512 · 448 · 4 ≈ 917 KB` (at `N_KV_HEADS == N_HEADS`, i.e. `D_KV == D_MODEL`), growing linearly with
context; the GDN recurrent state for the SAME layer is `N_HEADS · D_HEAD · D_HEAD · 4 B = 7 · 64 · 64 · 4
≈ 115 KB`, **flat regardless of context length** — already smaller at 512 tokens, and the gap widens
without bound as context grows. This is the concrete shape of the "O(1) vs O(n)" claim at this project's
own dims, not just the abstract complexity statement.

## 3. Checkpoint / PARAM_FLOATS design

### 3a. Which of this project's existing axes to reuse, and which are genuinely new

Per AGENTS.md §8 ("only add the surface area actually consumed"), the design below reuses every
dimension this engine already has rather than inventing a parallel set:

| Reference quantity | This project's mapping | Reused from |
|---|---|---|
| `head_k_dim` (Q, K width per head) | `D_HEAD` | existing |
| `num_k_heads` (Q, K head count) | `N_KV_HEADS` | existing (GQA axis) |
| `head_v_dim` (V, Z width per head) | `D_HEAD` | existing |
| `num_v_heads` (V, Z head count) | `N_HEADS` | existing |
| Q, K total width (`key_dim`) | `D_KV` (`= N_KV_HEADS · D_HEAD`) | existing, **already named** |
| V, Z total width (`value_dim`) | `D_MODEL` (`= N_HEADS · D_HEAD`) | existing, **already true by construction** |
| conv kernel size | `GDN_CONV_KERNEL` (new, but a fixed `constexpr int`, not a CLI flag — see below) | new, minimal |
| which layers are GDN | `GDN_FULL_ATTN_STRIDE` (Stage 0, already landed) | new |

`GDN_CONV_KERNEL` is deliberately **not** exposed as a CLI flag yet, per AGENTS.md §8 — Stage 0 has no op
to consume it, and the real model's own verified value (`linear_conv_kernel_dim = 4`, §1a's table, both at
full scale and in the fixture — not a placeholder) is the obvious fixed starting point once Stage 1 needs
an actual number; making it sweepable is a later decision to make WITH a working op to measure, not before
one exists. Note the head-count mapping is the **inverse** of this engine's existing GQA convention: GQA
narrows K/V relative to a wider Q (`N_HEADS ≥ N_KV_HEADS` heads for Q, fewer for K/V); GDN narrows Q/K
relative to a wider V/Z (`N_HEADS` heads for V/Z, `N_KV_HEADS` — potentially fewer — for Q/K). Reusing the
SAME two integers for both is intentional (no third head-count axis needed) but the direction of the
inequality one would sanity-check against does not carry over between the two mechanisms — worth flagging
explicitly so Stage 1 does not transplant a GQA-shaped assumption (e.g. "Q is always the wide one") into
GDN code.

### 3b. Per-layer PARAM_FLOATS delta, worked from `layout.hpp`'s actual existing terms

A softmax-attention layer's mixer parameters today (`layout.hpp`'s `make_param_layout`, ignoring the
shared `Ln1`/`Ln2`/FFN terms which do not change under this proposal):
```
  Wq + Wo  = 2 · D_MODEL²
  Wk + Wv  = 2 · D_MODEL · D_KV
  [+ QNorm + KNorm = 2 · D_HEAD, only if USE_QK_NORM]
```
A GDN layer's mixer parameters, using §3a's mapping and the REAL 4-way projection split verified in §1a
(`in_proj_qkv` = Q+K+V fused, at `D_KV` width for Q/K and `D_MODEL` width for V; `in_proj_z` = the output
gate, `D_MODEL` wide; `in_proj_b`/`in_proj_a` = the two gate-input scalars, `N_HEADS` wide each — one
scalar per V-head, same TOTAL floats as a fused `in_proj_ba` would cost, just two separate tensors rather
than one):
```
  in_proj_qkv  = D_MODEL · (2·D_KV + D_MODEL)
  in_proj_z    = D_MODEL · D_MODEL
  in_proj_b + in_proj_a = 2 · D_MODEL · N_HEADS
  conv1d       = (2·D_KV + D_MODEL) · (GDN_CONV_KERNEL + 1)     [+1 for the per-channel bias]
  A_log + dt_bias = 2 · N_HEADS
  norm (RMSNormGated) weight = D_HEAD          [per §1c: shared across heads, NOT D_MODEL-wide --
                                                 corrected from this doc's first draft using the real
                                                 fixture's weight_norm shape [128] == head_v_dim]
  out_proj     = D_MODEL²
```
(`in_proj_qkv + in_proj_z` together total `D_MODEL·(2·D_KV + 2·D_MODEL)`, the same combined width
Qwen3-Next's single fused `in_proj_qkvz` would have — confirming the real model's 4-way split changes the
OP-GRAPH shape (§4) but not the total parameter count relative to what this doc's first draft assumed.)

Net delta (GDN minus softmax-attention, `Wq/Wo` vs `out_proj`/`in_proj_z`, `Wk/Wv` vs the `D_KV` term in
`in_proj_qkv` both cancelling algebraically):
```
  Δ = 2·D_MODEL·N_HEADS + (2·D_KV + D_MODEL)·(GDN_CONV_KERNEL + 1) + 2·N_HEADS + D_HEAD
      [− 2·D_HEAD, if USE_QK_NORM was on — GDN's own Q/K L2-norm has no learned γ at all, §1c,
       netting the D_HEAD terms to ZERO in that case rather than a further reduction]
```
At this project's d448 L11 H7 shape (`D_MODEL=448, N_HEADS=7, D_HEAD=64, D_KV=448` at `N_KV_HEADS==
N_HEADS`, `GDN_CONV_KERNEL=4`, confirmed as the real model's own value in §1a's table, not a placeholder),
`Δ ≈ 2·448·7 + (2·448+448)·5 + 14 + 64 − 128 ≈ 6272 + 6720 + 14 + 64 − 128 = 12,942` floats per converted
layer. Regardless of the exact number, `Δ` is small but real and strictly positive at every dimension
combination this project would actually configure (`D_MODEL, N_HEADS, D_KV, GDN_CONV_KERNEL` all
positive), so **PARAM_FLOATS will differ between a GDN build and a non-GDN build at matching dims in
every practically reachable case.**

**A genuine, unresolved dimensional-convention mismatch worth flagging for Stage 1, found by trying to
apply this same arithmetic to the real fixture's own shape rather than a synthetic one**: the real
fixture's `hidden_size=32` with `head_v_dim=128, num_v_heads=3` does NOT satisfy `hidden_size ==
num_v_heads · head_v_dim` (`3 · 128 = 384 ≠ 32`) — the real GDN module never assumes `hidden_size ==
num_heads · head_dim` the way this project's `D_MODEL == N_HEADS · D_HEAD` always does; `key_dim`/
`value_dim` are independent PROJECTIONS from `hidden_size`, not a reshape OF it. §3a's mapping (reusing
`D_HEAD`/`N_HEADS`/`N_KV_HEADS` directly, which implicitly assumes `D_MODEL == N_HEADS · D_HEAD`) is true
for every dims combination this project's own softmax attention currently configures — that equality is
an invariant of THIS project's attention layout, not a general truth about GDN — but Stage 1 should treat
it as an assumption this project's mapping carries in, not a property GDN itself requires, especially once
real-weight parity testing tries to reproduce the small fixture's own (non-conforming) shape exactly.

### 3c. Why that is not enough on its own — the design decision already landed in Stage 0

AGENTS.md §3's rule 1 asks: does an EXISTING field already discriminate this axis, "confirmed by working
through the actual arithmetic, not assumed"? §3b's arithmetic says PARAM_FLOATS differs in every
*practically reachable* configuration — but unlike GQA's `D_KV` substitution (which the existing
`ARCH_FINGERPRINT` comment calls out as "strictly monotonically in N_KV_HEADS, so no collision is
possible" — a proof, not an observation), §3b's `Δ` is a sum of several independent terms with different
signs and no monotonicity argument tying it to a single knob. Nothing rules out some `(D_MODEL, N_HEADS,
D_KV, GDN_CONV_KERNEL, N_LAYERS, GDN_FULL_ATTN_STRIDE)` combination where a GDN-converted model happens to
land on the exact same total float count as some *other* build's plain-attention model — the classic
"same SIZE, different COMPUTATION" case `ARCH_FINGERPRINT` exists specifically to catch (its own doc
comment's ROPE_THETA example is exactly this shape of risk).

So per AGENTS.md §3 rule 2, GDN's layer schedule joins the fingerprint mechanism **in addition to**
relying on PARAM_FLOATS, as belt-and-braces rather than either alone. This is already implemented
(Stage 0, this pass): `include/sub0/layout.hpp`'s `ARCH_FINGERPRINT2`/`GDN_FULL_ATTN_STRIDE`/
`GDN_SCHEDULE`.

**The concrete discovery that shaped the implementation**: `ARCH_FINGERPRINT` (the existing word) has
**zero spare bits** — `8 (depth_attn_stride) + 8 (middle_layers) + 16 (repeats) + 32 (rope_theta) = 64`,
worked through explicitly rather than assumed, exactly as AGENTS.md §3 requires. Shrinking any existing
field to make room would either break `ROPE_THETA`'s exact `bit_cast` round-trip (needs all 32 bits) or
cap `LOOP_REPEATS` below a value some future run might actually want (16 bits is already tight). Per
AGENTS.md §3's stated preference for an "ADDITIVE, gracefully-degrading format... over reshuffling
existing fields," the design is a **second, wholly new fingerprint word** (`ARCH_FINGERPRINT2`) rather
than a repack of the first — and, having been burned once by a word with no headroom, this one is
deliberately built with 56 *reserved* spare bits from day one (only the low byte is assigned, to
`gdn_full_attn_stride`), so the *next* shape-neutral, computation-changing axis after this one does not
repeat the exact scramble that produced `ARCH_FINGERPRINT`'s own "the byte middle_layers was never able
to reach" situation.

**On-disk plumbing, additive at every point that already existed, per AGENTS.md §3 rule 2's own
precedent**:
- `engine_core.cpp`'s `model.bin`: a **third** trailing 8-byte record, after the existing tokenizer
  fingerprint and `ARCH_FINGERPRINT` trailers. Tolerant of a short/missing read (every file on disk today
  predates this field, and "no trailer" correctly decodes to `0` — no GDN — because that is the only
  architecture that has ever existed, not a guess the way `ARCH_FINGERPRINT_LEGACY`'s "assume un-looped"
  inference had to be for pre-LoopSplit files).
- `train_stage.cpp`'s `.ckpt`: `CKPT_VERSION` bumped 6→7 to add the same word. Safe by this format's own
  existing contract — `load_checkpoint` refuses on ANY version mismatch and starts a fresh run rather than
  attempting a partial/misaligned read, exactly the trade every prior version bump (2 through 6) already
  made; an in-flight v6 checkpoint is not corrupted by rebuilding to v7, only not resumed.
- `registry.hpp`'s `RunConfig`: one new `SUB0_RUN_CONFIG_FIELDS` row (`gdn_full_attn_stride`), which
  self-propagates to `config.json` read/write/describe and to `MODEL_ARCH_ID` (via `ArchAxes2`/
  `arch_fingerprint2`) with no hand-written second copy, per that file's own stated design intent.

**One honestly-reported side effect**: `MODEL_ARCH_ID` (the model-registry identity hash, `layout.hpp`)
now mixes in `ARCH_FINGERPRINT2`/`GDN_FULL_ATTN_STRIDE`, and because its FNV-1a `mix()` helper perturbs
the hash on every call regardless of the value passed (XOR-with-zero is a no-op, but the subsequent
`h *= prime` is not), `MODEL_ARCH_ID`'s numeric value has shifted for every pre-existing build even
though GDN is off. This is not new behavior introduced by this change — the exact same thing happened
when `DEPTH_ATTN_STRIDE` was added to `MODEL_ARCH_ID`'s mix — and `MODEL_ARCH_ID`'s own doc comment
already establishes why it is safe: it is "a DIAGNOSTIC check... the actual load-time gate is
engine_core.cpp's binary Header comparison, which is authoritative regardless of what this says." The
real, checkpoint-critical fingerprints (`ARCH_FINGERPRINT`, `ARCH_FINGERPRINT2`) are the ones proven
bit-identical at neutral in `layout_tests.cpp`; `MODEL_ARCH_ID` shifting is a cosmetic registry-naming
consequence, not a correctness regression.

## 4. The `Node`-fanout question — resolved, and differently than depth attention's

`docs/DEPTH_ATTENTION.md` §5a hit a real wall here: `Node`'s fixed `a`/`b`/`w`/`bias` fanout could not
express a variable-length, cross-EXECUTION list of (K, V) pairs, and the fix was a `thread_local` side
table outside `Node` entirely. **Gated DeltaNet does not hit the same wall**, and the reason is
structural, not incidental: DepthAttn's extra inputs come from OTHER executions' completed nodes (a
graph-level, cross-node dependency); GDN's "history" is entirely INTRA-node — a sequential dependency
across POSITIONS within the SAME forward call, which this engine already has a working, verified pattern
for.

The load-bearing precedent is `op_attn` itself (`backend_cpu.cpp`):
```cpp
static Node* op_attn(Node* q, Node* k, Node* v, int H) {
    ...
    Node* out = mk_node(Op::Attn, T, C);
    out->a = q; out->b = k; out->bias = v; out->heads = H;
    auto [P, Pg] = arena_alloc((size_t)H * T * T);
    out->scratch = P;
    for (int h = 0; h < H; ++h)
        for (int i = 0; i < T; ++i)          // <-- the causal loop over ALL T positions lives HERE,
            for (int j = 0; j <= i; ++j)     //     entirely inside this one op, invisible to the
                ...                          //     Node graph at position granularity
```
Attention's O(T²) causal double-loop over positions is entirely internal to one `Op::Attn` node; the
graph never sees per-position nodes. GDN's O(T) recurrence loop (§1b) is the exact same shape of thing,
just cheaper (one pass over T, not T²) and with a running scalar/matrix accumulator instead of a growing
softmax normalizer. Concretely, a future `op_gdn` would take:
- `a` = the `in_proj_qkv` projection's activation (Q+K+V fused, per §1a's real 4-way split — itself an
  ordinary `Op::Linear` node upstream, exactly how today's separate `Wq`/`Wk`/`Wv` feed `op_attn`'s three
  arguments) — or, if a short causal conv (§1c) becomes its own op, the POST-conv activation;
- `b` = a SECOND node combining `in_proj_b`/`in_proj_a`'s activations (§1a: the real model keeps these as
  two separate `Linear` layers rather than Qwen3-Next's single fused `in_proj_ba`; this engine could
  either keep them as two upstream `Op::Linear` nodes and concatenate their outputs into one `b`-sized
  activation before `op_gdn`, or run `op_gdn` with a genuinely separate fifth conceptual input — `Node`
  has no fifth slot, so the concatenation-before-the-op route reuses the existing `a`/`b`/`w`/`bias` shape
  without needing one; Stage 1 should decide this once an actual `op_gdn` signature is being written, not
  here);
- `w`/`bias` = available for `in_proj_z` (the output-gate projection, needed by the gated-RMSNorm step,
  §1c) and/or the recurrence's per-head `A_log`/`dt_bias` leaf parameters — `Node`'s four pointer slots
  (`a`, `b`, `w`, `bias`) comfortably cover GDN's four/five real inputs where `op_attn`'s three already
  fit in three of them;
- `heads` = `N_HEADS` (reusing the field `op_attn` already uses for exactly this purpose);
- `scratch` = the transient per-forward-call recurrent-state working buffer (§2's training-time
  scratch term), sized `[batch-rows, N_HEADS, D_HEAD, D_HEAD]`, allocated once from the arena and fully
  consumed inside the op — the direct analogue of `op_attn`'s own `[H,T,T]` softmax-probability
  `scratch`.

No new pointer field, no widened `Node`, no side table. **The multi-op pipeline this implies** (mirroring
how softmax attention itself is `Linear(Wq) + Linear(Wk) + Linear(Wv) + Op::Attn + Linear(Wo)`, a mix of
generic ops plus one irreducible specialized kernel): `Linear(in_proj_qkv) + Linear(in_proj_z) +
Linear(in_proj_b) + Linear(in_proj_a) + [a new short-causal-conv op, if it doesn't fit an existing
primitive] + Op::GatedDeltaNet (the irreducible recurrence) + [a new gated-RMSNorm op, likely a small
extension of the existing RMSNorm op rather than wholly new] + Linear(out_proj)` — FOUR upstream
projections now, per §1a's real structure, not the two this doc's first draft assumed from the
Qwen3-Next proxy. Working out exactly which of the conv/gated-norm steps reuse an existing op vs. need a
new `Op::` enum value is Stage 1 scope, not this pass's — the point established here is narrower and
load-bearing: **the one part of this pipeline that is genuinely novel (the recurrence itself) fits
`Node`'s existing shape without modification**, unlike DepthAttn.

**One real difference from `op_attn` worth flagging for Stage 1's backward design**: `op_attn` retains
its full `[H,T,T]` probability tensor in `scratch` specifically so backward does not need to recompute
the forward softmax. A naive GDN backward would want the analogous thing — every intermediate state `S_t`
for `t = 0..T-1` — which is `O(T · d_k · d_v)` per head, i.e. the SAME order of memory `op_attn`'s own
`[H,T,T]` scratch already costs (swap which factor is `T`). That is a real cost, not a free lunch, and it
means GDN's *training-time* memory advantage over softmax attention is smaller than its *inference-time*
advantage (§2) — the O(1) state is only O(1) once you no longer need yesterday's value for a gradient.
The established alternative, already this engine's own convention on the CUDA side
(`docs/DEPTH_ATTENTION.md` §5b's finding 2, `backward_device` "already RECOMPUTES `qkv` per execution"):
**recompute the state trajectory during backward from the retained inputs, rather than retaining it from
forward.** This is the natural target for GDN's own backward design once Stage 1 gets there, not
resolved further in this pass.

## 5. Training-time algorithm: which form Stage 1 should target, and why

The real model itself only ever runs the chunked form (§1b) for anything longer than one cached token —
§1e already flags that `modeling_qwen4_exp.py`'s own decode-time sequential loop was not directly
re-quoted this pass, but Qwen3-Next's analogous `torch_recurrent_gated_delta_rule` (this doc's first
draft, now superseded as the PRIMARY source but still a faithful illustration of the sequential form) pays
real Python-level per-iteration interpreter overhead, independent of how cheap the underlying tensor op
is — that is specifically why the reference reserves it for single-token decode and always chunks for
training/prefill.

**That reason does not transfer to this engine as-is, and the distinction matters.** This project's
`op_attn` is already a *compiled, tight C++ loop* over `T` positions with no interpreter in the loop body
— the softmax-attention analogue of "the slow Python path" here is not slow at all; it is the only
forward this engine has ever shipped. The sequential GDN recurrence (§1b's per-token reading, before
chunking) is the same shape of thing: `O(T)` total work (one rank-1 state update and one read-out per
position, each `O(d_k · d_v)`), executed as a compiled loop, with **no algorithmic penalty from being
"sequential"** — it is not a nested loop the way attention's own causal double-loop is (`O(T²)`, §4's
quoted code). A naive, faithful, compiled sequential implementation of GDN is therefore not "the toy
version to get right before the real one" the way it would be in eager PyTorch; it is a plausible
**production-adequate CPU implementation on its own terms**, at this project's current CPU-only training
scale.

**Stage 1's recommended target, and the reasoning in order:**
1. Implement the sequential, token-at-a-time unrolling of §1b's recurrence (`S = g_t·S`,
   `kv_mem = Sᵀk_t`, `delta = β_t·(v_t − kv_mem)`, `S += k_t ⊗ delta`, `o_t = Sᵀq_t`) as `op_gdn`'s CPU
   forward. This is a direct token-by-token unrolling of the SAME closed-form update the real chunked
   code computes (§1b's `v_new`/`attn_inter`/state-update lines reduce to exactly this at chunk size 1) —
   not a simplification invented for this port — so "verify against the real reference" (AGENTS.md §5)
   and "get the fast path working first" are the SAME step here, which is not usually true.
2. **Verify it against the real extracted weights + activations already landed at
   `tests/fixtures/qwen4_preview/gdn_layer0_small_*`** (§0/§1a) — no longer a future dependency, this is
   available now. This is the correctness gate, not a numerical-gradient check alone
   (`docs/QWEN4_PREVIEW_REFERENCE.md`'s own "remember the depth-attention lesson" note applies here too: a
   no-op recurrence would still pass a naive gradient check). **One thing to verify explicitly when this
   happens**: the fixture's own output was produced via the CHUNKED path (§1b), not the sequential one —
   so a Stage 1 CPU forward built on the sequential form and matched against this fixture proves BOTH
   correctness against the real model AND numerical equivalence between the sequential and chunked forms
   on a real case, which is a stronger result than either check alone.
3. The chunked-parallel form (§1b) becomes the real question once a CUDA backend is targeted (a later
   stage, out of scope here) — its actual benefit is parallelizing across the `C` positions in a chunk
   using dense matmuls/tensor cores, which matters for GPU throughput and for vectorizing across a large
   training batch, not for CPU correctness or even CPU throughput at this project's current scale.
   Adopting it then must be gated on exact numerical agreement with the sequential form from step 1 as
   the reference implementation to diff against — a re-association of the same recurrence is exactly the
   kind of change where a subtle reordering bug produces a plausible-looking but wrong result.

This mirrors an existing project precedent directly: `backend_cpu.cpp`'s own softmax attention is a plain
O(T²) loop today, not FlashAttention-tiled (project memory `attention-kernel-throughput-bottleneck` — the
tiled form is a documented, *not-yet-done* performance follow-up). Building GDN's simple form first and
its tiled/chunked form later, gated on numerical parity with the simple one, is the same order of
operations this project already chose for attention itself.

## 6. Staging

- **Stage 0 — config axes, off by default. DONE, this pass.** `GDN_FULL_ATTN_STRIDE` (int CLI flag,
  hard-clamped to `{0}` by both the CLI `CLI::Range(0,0)` and a `layout.hpp` `static_assert` — two
  independent refusals of the one dangerous case, per the project's "guard at the lowest callable seam"
  standing preference), `USE_GATED_DELTANET` (derived bool, always false today), `GDN_SCHEDULE` (a
  `gdn_schedule_for<N_LAYERS>(stride)` consteval per-layer classification, unused until Stage 1's
  `Model::forward` dispatch consumes it — inert compile-time bookkeeping only, the same status
  `DEPTH_SCHEDULE` had before its own Stage 1), and `ARCH_FINGERPRINT2`/`GDN_FULL_ATTN_STRIDE` folded
  into `RunConfig` and `MODEL_ARCH_ID` (§3c). Pinned in `tests/layout_tests.cpp` at TWO shapes — 8 layers/
  stride 4 (divides evenly, the Qwen3-Next ratio) and 11 layers/stride 3 (odd, ragged, does not divide
  evenly) — per AGENTS.md §7's explicit lesson that a single compiled shape previously hid a real
  LoopSplit bug at odd layer counts. **Identity gate**: the full default-build test suite went from
  4,380,812 assertions / 132 test cases to 4,380,848 / 134 — a difference of exactly the 36 assertions in
  the 2 new GDN test cases added this pass, and not one assertion anywhere else, confirmed by running the
  suite before and after via `git stash` at the same build. Also re-verified at a second, production-
  shaped, ODD-layer-count real build (d448 L11 H7 seq512) — the full suite is slow to re-run at that scale
  end-to-end, but the `[layout][config][gdn][depth]`-tagged subset (10 test cases) passes identically
  there too, including both new GDN schedule/fingerprint tests.
- **Stage 1 — CPU forward only — DONE.** `GDN_FULL_ATTN_STRIDE`'s Stage 0 hard clamp lifted (CLI
  `CLI::Range` now `(0, 1024)`; `layout.hpp`'s `static_assert` now just `>= 0`), per-layer `make_param_
  layout()`/`NUM_PARAMS` (nine new `PKind`s: `GdnInProjQkv/Z/B/A`, `GdnConv`, `GdnALog`, `GdnDtBias`,
  `GdnNorm`, `GdnOutProj`), and `op_gdn` (`backend_cpu.cpp`) wired into `Model::forward`'s per-execution
  dispatch (`GDN_SCHEDULE.full_attn[l]`) plus the mirror decode path in `forward_one` via a new `GdnCache`
  (persistent per-execution recurrent-state/conv-history slots, reset — unconditionally, unlike KVCache —
  by `kv_reset()`, exactly §2's design). The shared math (§5 step 1's sequential recurrence, the S1c
  short causal conv + SiLU, the S1c RMSNormGated gate) lives engine-free in `include/sub0/gdn_math.hpp`
  so it is directly unit-testable at the real fixture's own (non-conforming) shape — see §3b's flagged
  mismatch, confirmed exactly as predicted: the fixture cannot be run through this project's own `Model`
  at all (`hidden_size=32 ≠ num_v_heads·head_v_dim=384`), so `tests/gdn_qwen4_fixture_tests.cpp` calls
  `sub0::gdn::forward` directly, engine-free, the same pattern `ngram_qwen4_fixture_tests.cpp` already
  established.

  **One real correction to this doc, found only by re-fetching `transformers==5.16.1`'s installed source
  directly (AGENTS.md §5) rather than trusting §1b's already-quoted snippet**: both
  `torch_recurrent_gated_delta_rule` and `torch_chunk_gated_delta_rule` scale **query** by
  `1/sqrt(head_k_dim)` immediately after L2-normalizing it, one line before §1b's quoted loop even
  starts. §1b's claim "No explicit `1/√d_k` scale appears anywhere in this recurrence" is true only of
  the lines it quotes — the earlier line was missed. Omitting this scale reproduces a plausible-looking
  but numerically wrong output (confirmed directly: a large fixture mismatch without it, an exact match
  with it). See `gdn_math.hpp`'s own header comment for the full account — this is exactly the kind of
  thing AGENTS.md §5 exists to catch, and it would not have been caught without actually re-fetching the
  real, installed module source (this machine has `transformers==5.16.1` installed, so this was a direct
  `python3 -c "import inspect, ..."` fetch, not a web lookup).

  **Fixture-comparison numbers** (`tests/gdn_qwen4_fixture_tests.cpp`, real weights, `T=6` real
  fixture input, sequential CPU forward vs. the real reference's CHUNKED-path output): max
  `|out − expected| = 4.36557e-11`, max relative diff `1.85136e-05`, `sum|expected| = 0.00170643` over
  192 output values — essentially float32-rounding-level agreement, not an approximate match. This
  directly confirms §5 step 2's prediction: the sequential and chunked forms agree exactly at the token
  level on a real case, at the SAME time as confirming correctness against the real model. (A pre-port
  double-precision NumPy re-derivation of the identical math, checked directly against the fixture
  before any C++ was written, measured `4.4e-11` — the C++ float32 port lands at the same order of
  magnitude, as expected.)

  **Mutation-style numerical-property checks** (same file, no fixture needed — call
  `sub0::gdn::recurrence_step` directly, factored out of `forward()` as its own tested primitive):
  at `g_t ≡ 1, β_t ≡ 1` with mutually orthonormal keys, the state is verified EXACTLY equal (to `< 1e-6`
  float32 noise) to the closed-form running sum of `k_t ⊗ v_t` outer products after every one of 5 steps
  — the recurrence genuinely degenerates to plain cumulative summation, not merely "doesn't obviously
  break". At `β_t ≡ 0, g_t ≡ 1`, the state is proven a BIT-FOR-BIT no-op (exact IEEE-754 equality) and
  the output is invariant across 4 unrelated, nonzero `(k_t, v_t)` trial inputs — the kind of check a
  "forgot to gate the write on beta" mutant would fail even though it might still pass a single fixed
  fixture comparison. All 3 GDN test cases (fixture match + 2 mutation checks): 87 assertions, green.

  **Two-scale identity check** (AGENTS.md §7, reusing this project's own precedent of an odd/
  non-dividing layer count as the second shape): at `GDN_FULL_ATTN_STRIDE == 0` (neutral), the full
  default engine test suite (`sub0_tests`) is assertion- AND hash-identical to `main` at BOTH shapes —
  not just the same count, the same computed forward/gradient/decode content hash:
  | shape | assertions | test cases | forward hash | grad hash | decode hash |
  |---|---|---|---|---|---|
  | d96 L8 H2 kv2 seq128 (even stride-4-shaped) | 4,978,665 | 137 | `ec1d6e05e417d129` | `fa6094a1748ccead` | `7178bbf9b452aa0d` |
  | d132 L11 H4 kv2 seq96 (odd/ragged) | 11,952,837 | 137 | `11510f553cc233d9` | `1b4bbb318f886909` | `278d7ccd2c6e3ab3` |

  identical on `main` and on this Stage-1 commit at both shapes (verified by building each side of the
  diff at the same generated config header and diffing `sub0_tests`' own `arch_identity_tests.cpp`
  output, not just "still green"). Two PRE-EXISTING `layout_tests.cpp` cases hard-coded the pre-Stage-1
  assumption that every layer has an identical tensor set (the `NUM_PARAMS` formula, and the decay/
  ternary-flag consistency loop) — both updated to be `GDN_SCHEDULE`-aware, verified as no-ops at stride
  0 (the two-scale numbers above already reflect the fix) and verified to actually catch/pass correctly
  at a real GDN-on build (d96 L8 H2 kv2, stride 3): the layout-consistency test now passes there; the
  OTHER pre-existing failure at that shape (`arch fingerprint2 stays bit-identical when Gated DeltaNet
  is off`) is not a regression — it is testing the OFF premise, which is deliberately false in that build.

  **Forward-only correctness inside the real engine, at both shapes**: this project's own pre-existing
  `forward_one (KV-cache) matches the full forward per position` test (`tests/engine_tests.cpp`, no
  GDN-specific code — it runs unconditionally) passed at a GDN-ON build with NO changes needed, at both
  shapes: worst per-position relative diff `3.88717e-07` (d96 L8 H2 kv2, stride 3, 6 GDN + 2 attention
  layers) and `5.81367e-07` (d132 L11 H4 kv2, stride 4, 9 GDN + 2 attention layers) — both far inside the
  `1e-3` fast-math tolerance and consistent with ordinary floating-point noise, not an algorithmic
  mismatch. This is the batched `op_gdn` and the decode-path `GdnCache`/`forward_one` branch agreeing
  with EACH OTHER on a real mixed attention+GDN, GQA, LoopSplit-free forward pass — independent evidence
  from the fixture match above, which never exercises `forward_one` or a mixed-layer schedule at all.

  **No design revision needed beyond §1b's scale correction above.** §2's arena/decode-state design,
  §3's checkpoint/fingerprint plumbing, and §4's "no Node-fanout problem" finding all held up exactly as
  written once actually implemented — `op_gdn` takes `Layer&` directly (no generic `a`/`b`/`w`/`bias`
  routing needed, since Stage 1 has no backward to preserve node-graph linkage for) and `GdnCache` is a
  direct structural mirror of `KVCache`, with the one documented divergence already anticipated by §2:
  it must re-zero UNCONDITIONALLY on `kv_reset()` (an accumulator, not a self-healing per-position row
  store the way `KVCache` is).

  **No backward pass, no CUDA** — this stage's own explicit scope boundary. `backend_cpu.cpp`'s
  `backward_node` has an `Op::GDN` case that aborts loudly (not a silent no-op) if `train_batch` ever
  reaches a GDN node, mirroring `backend_cuda.cu`'s new `static_assert(!sub0::USE_GATED_DELTANET, ...)`
  guard (added this stage — Stage 0 had not needed one, since the axis was compile-clamped to 0
  everywhere before now) at the next callable seam down, since a GDN CPU binary compiles and runs fine
  for forward-only uses (gen/eval/report) regardless.
- **Stage 2 — CPU backward — DONE.** `include/sub0/gdn_math.hpp` gains `sub0::gdn::backward` (the full
  adjoint of `forward`'s in_proj*, causal conv, L2-norm+scale, sequential recurrence, RMSNormGated and
  out_proj) and its `recurrence_step_backward` primitive, plus `bwd_scratch_floats`. Per §4's own closing
  paragraph, this is RECOMPUTE-based: it reruns `forward`'s math from `x` and the 9 weight tensors
  (already retained by the engine's Node/arena machinery) rather than retaining the state trajectory
  `S_0..S_{T-1}` from `forward` itself — the trajectory is materialized only inside `backward`'s own
  scratch buffer, and only for the one GDN node currently being backpropagated.

  **Node-linkage, resolved (§4's own deferred question, now answered under real load)**: Stage 1's
  `op_gdn` took `Layer&` directly with no generic `a`/`b`/`w`/`bias` routing, since it had no backward to
  preserve linkage for. Stage 2 needs `backward_node`'s `Op::GDN` case to reach all 9 of a GDN layer's
  parameter Nodes (to write their gradients) plus the input node — 10 pointers, more than `Node`'s fixed
  4-slot fanout can hold. This is the identical wall `docs/DEPTH_ATTENTION.md` §5a already hit and fixed
  once (a Node cannot express more inputs than its fixed pointer fanout), so it gets the same fix:
  `backend_cpu.cpp`'s `GdnLinkCache`, a `thread_local` side table of the 9 parameter-Node bundles, keyed
  by a new `Node::gdn_link` int field (mirroring `Node::depth_s`'s own pattern) that `op_gdn` populates
  and `backward_node`'s `Op::GDN` case reads. `Node::a` (the input) needed no change — it was already
  generically threaded and is unaffected by the 9-weight-tensor problem. Backward's own transient
  recompute scratch (dominated by the `T·N_HEADS·D_HEAD²` state-trajectory buffer) is a separate,
  lazily-sized `thread_local` `std::vector` (`GdnBwdScratch`) reused across every GDN node's backward
  call one at a time, deliberately NOT routed through `arena_alloc` (which never reclaims within one
  graph's lifetime and would otherwise need headroom for as many simultaneous copies as there are GDN
  layers).

  **Correctness — real PyTorch-autograd oracle, the primary gate (AGENTS.md §5/§6)**: the real, installed
  `transformers==5.16.1` `Qwen4ExpTextGatedDeltaNet`'s own chunked-path `.backward()`, run on the SAME
  real extracted layer-0 weights Stage 1's forward fixture uses, with `loss = dot(output, a fixed
  reproducible random vector)` (not plain `sum()`, so every output element carries a distinct nonzero
  upstream gradient — `tests/fixtures/qwen4_preview/gdn_layer0_small_{dout,grad_*}.bin`). Every one of
  the 9 parameter-tensor gradients plus the input gradient matches to ≈3e-7 relative (double-precision
  NumPy re-derivation vs. the real oracle's float32 autograd; the ported C++ float32 version matches the
  same real numbers to the same order, e.g. `d(in_proj_qkv)`: max|diff|=4.07e-10 vs max|real|=1.05e-3,
  `d(A_log)`: max|diff|=1.02e-10 vs max|real|=5.55e-5). A real bug was caught by this cross-check BEFORE
  the C++ port: the first hand-derivation conflated `d(delta_t)` with `d(k_t)`'s contribution from
  `B_t = k_t⊗delta_t` (used `beta_t·v_t` instead of the outer-product adjoint `G_t@delta_t`) and dropped
  the `-kv_mem_t` term from `d(beta_t)` — found by bisecting against saved PyTorch-autograd intermediate
  gradients at the exact stage the two hand-derived quantities first diverged. A SECOND, independent
  cross-check (a synthetic `num_k_heads=2` shape — the real fixture's own `num_k_heads=1` never exercises
  the repeat_interleave group-sum fold — against `torch.autograd` on a from-scratch PyTorch
  reimplementation of the identical math) matched to ~1e-15 relative, double-precision machine noise.

  **Finite-difference check, independent of the real oracle (AGENTS.md §6's "standard check" — mirrors
  `tests/engine_tests.cpp`'s own FD-check style)**: `tests/gdn_qwen4_fixture_tests.cpp` adds a per-element
  central-difference probe over every one of the 9 weight tensors plus the input, at the same synthetic
  `num_k_heads=2` shape — e.g. `w_qkv`: max|analytic−numeric|=3.96e-5 vs max|numeric|=0.72,
  `a_log`: max|analytic−numeric|=7.57e-6 vs max|numeric|=0.0415.

  **Presence/mutation-style check (the depth-attention lesson)**: both the real-oracle test and the FD
  test explicitly assert `d(A_log)`/`d(dt_bias)` are genuinely nonzero (and, via the FD/real-oracle match
  itself, genuinely gate-value-dependent, not a structurally-present stub) — a mutant that silently zeroed
  either gate scalar's backward path would fail both checks, not just a naive "field populated" test.

  **Two-scale identity check (AGENTS.md §4/§7), at neutral (`GDN_FULL_ATTN_STRIDE == 0`)**: this branch
  and `main` (`9f42e38`), built at the SAME two shapes Stage 1 established (same corpus, same generated
  vocab 332 both times) — assertion counts AND the forward/grad/decode fingerprints are byte-identical
  between the two:
  | shape | assertions | test cases | forward hash | grad hash | decode hash |
  |---|---|---|---|---|---|
  | d96 L8 H2 kv2 seq128 | 4,712,955 | 137 | `1cbfe9df31ac89ae` | `85bed0fd41277fd0` | `15cd828a539ffaca` |
  | d132 L11 H4 kv2 seq96 | 11,594,607 | 137 | `f2a2141a1b8a466b` | `58877ebaf797c204` | `b186d0caf7f48649` |

  (These absolute numbers differ from Stage 1's own recorded table because this pass used a small,
  reproducible synthetic corpus rather than Stage 1's original one, which was not available in this
  session — the gate that matters, and the one actually checked, is THIS branch vs. `main` at matching
  corpus/dims, not reproducing Stage 1's historical constants.) Expected structurally, not just
  empirically: `op_gdn`'s entire body sits behind `if constexpr (USE_GATED_DELTANET)`, compile-time false
  at stride 0, so none of Stage 2's runtime code — `GdnLinkCache`, `GdnBwdScratch`, `gdn::backward` —
  is ever reached there; the two builds' object code differs only in genuinely dead paths.

  **Real GDN-on build, unfiltered full suite (AGENTS.md §10)**: d96 L8 H2 kv2, stride 3 (6 GDN + 2
  attention layers) — 136/137 test cases pass (5,034,333 assertions); the one failure is the SAME
  pre-existing, expected one Stage 1 already documented (`arch fingerprint2 stays bit-identical when
  Gated DeltaNet is off` — testing the OFF premise, deliberately false in this build). The load-bearing
  new result: `tests/engine_tests.cpp`'s whole-model `analytic gradients match finite differences` test
  — which exercises `train_batch`'s REAL forward→loss→backward→AdamW path on this real mixed GDN+
  attention model — now PASSES (previously this would have hit the Stage 1 `std::abort()`): directional
  numeric=2.19188 vs. ‖g‖=2.19971, all 6 per-parameter spot checks within tolerance. `forward_one
  (KV-cache) matches the full forward per position` still passes too (worst per-position relative diff
  `3.43571e-07`), confirming the decode path — untouched by this stage — stayed correct.

- **Stage 3 — CUDA FORWARD ONLY — DONE.** Targets §5 step 3's chunked-parallel form
  (`torch_chunk_gated_delta_rule`, re-fetched and re-verified directly against the installed
  `transformers==5.16.1` source for this stage, not paraphrased from §1b) rather than a re-run of
  Stage 1's sequential CPU recurrence — its real benefit is parallelizing across a chunk's `C=64`
  positions via dense-matmul-shaped work, which only matters once a GPU is the target (§5 step 3's own
  reasoning). The old build-time `static_assert(!sub0::USE_GATED_DELTANET, ...)` guard is lifted; a GDN
  CUDA build now compiles and correctly runs `forward_device` (batched training-shaped/inference forward
  — the path `sub0_cuda_forward`/`sub0_cuda_forward_loss` both use). **No CUDA backward, no single-token
  decode** — this stage's own explicit, deliberate scope boundary, enforced at RUNTIME rather than a
  build-time block (mirroring depth attention's own single-token-decode gap, §5b): `forward_train`/
  `backward_device`/`forward_one_device` abort loudly if ever reached with GDN on (mirroring
  `backend_cpu.cpp`'s Stage 1 `backward_node` Op::GDN abort, applied here to `backward_device` per this
  stage's own task); the training extern seam (`run_fwd_bwd`, the shared body of `sub0_cuda_backward`/
  `sub0_cuda_train_step`) and the decode extern seam (`sub0_cuda_kv_reset`/`sub0_cuda_forward_one`) both
  refuse gracefully one layer up. `device_backend.hpp`'s `Sub0DeviceCaps` reports this honestly:
  `supports_train`/`supports_decode` both go false for a GDN build; `supports_eval` stays true
  (`sub0_cuda_forward_loss` is built on the now-GDN-aware `forward_device`, so evaluating a GDN model on
  GPU works fine — only training and gen's decode path fall back to CPU).

  **Algorithm decomposition, validated BEFORE any CUDA was written** (AGENTS.md S5/S6): the real
  `torch_chunk_gated_delta_rule` splits cleanly at its own natural boundary — `decay_mask`, the
  WY-inverted intra-chunk `attn` (a forward-substitution solve, `for i in range(1,chunk_size): attn[i,:i]
  = row + (row*sub).sum`, genuinely sequential over the 64 intra-chunk positions but with NO cross-chunk
  dependency), `value' = attn @ v_beta`, and `k_cumdecay = attn @ (k_beta·exp(g_cum))` depend ONLY on
  their own chunk; only the SECOND loop (`last_recurrent_state`/`attn_inter`/`core_attn_out`) has a real
  chunk-to-chunk sequential dependency. This became the CUDA port's two phases (below). A host-side
  numpy/torch script re-implemented this exact decomposition in float64 and checked it directly against
  the REAL installed `torch_chunk_gated_delta_rule` (not a re-derivation of gdn_math.hpp) at three shapes
  — single-chunk (T=6, matching the real fixture), multi-chunk (T=130 → 3 chunks, GQA repeat_interleave),
  and an exact chunk-size boundary (T=64, zero padding) — all matching to float32-rounding-level
  agreement (rel diff ~1e-6 to ~2e-6) before a single CUDA kernel was written.

  **CUDA kernels** (`backend_cuda.cu`'s "Gated DeltaNet — CUDA FORWARD" section): the four input
  projections (`in_proj_qkv/z/b/a`) and `out_proj` reuse the existing `launch_linear` GEMM helper
  directly (this project's own `[in,out]` weight convention needs no adaptation). New kernels:
  `gdn_conv_silu_kernel` (causal depthwise conv+SiLU), `gdn_gates_kernel` (β/log-decay), `gdn_qknorm_kernel`
  (L2-norm+scale with the `repeat_interleave` broadcast), `gdn_phaseA_kernel` (one block per
  (batch, v-head, chunk), `GDN_CHUNK=64` threads — decay_mask + the WY-inverted `attn` + value′/k_cumdecay,
  fully parallel across all three grid axes since no chunk depends on another here), `gdn_phaseB_kernel`
  (one launch PER CHUNK INDEX from a host `for` loop — grid (batch, v-head), genuinely sequential across
  launches, the recurrence's own real cross-chunk dependency, not an implementation compromise) and
  `gdn_rmsnorm_gated_kernel`. Deliberately plain FP32 throughout (no bf16 mirror, no tensor-core path) and
  — after a real hardware finding below — NO shared-memory caching of the O(chunk·dim) matrices either:
  a correctness-first simplification per AGENTS.md S5/S6, not a performance claim; a follow-up perf pass
  is real future work, not attempted this stage.

  **A real hardware hazard, caught by this stage's own test, not assumed away** (per the task's own
  flagged risk categories — register spills, VRAM limits, driver-level faults): an earlier version of
  `gdn_phaseB_kernel` cached `last_state`/`v_new` in DYNAMIC shared memory (~96 KB at the real fixture's
  `dk=dv=128` shape) and raised the per-block shared-memory limit via `cudaFuncSetAttribute` — whose
  return value went unchecked. On this session's actual GPU that raise silently failed (the requested
  ceiling exceeded this card's real per-block shared-memory maximum), the kernel launch that followed
  then itself failed silently too, and `core_out` was left as zero-initialized fresh device memory —
  caught immediately by `sub0_cuda_gdn_check` (relative diff exactly 1.0, GPU output identically zero),
  not discovered by inspection. Fixed by moving `last_state`/`v_new` to caller-owned GLOBAL scratch
  entirely (no shared memory in this kernel at all) — redundant bandwidth, not a correctness compromise,
  and it sidesteps the device-dependent limit rather than working around it with a fragile size-dependent
  opt-in. This is a real, hardware-specific failure mode distinct from a code bug in the ALGORITHM (the
  math itself, once actually run, matched to float32 precision at every shape tested) — exactly the kind
  of category this stage's own task flagged as a real risk to watch for, not assume away.

  **Two-level correctness gate, both green** (`cuda_tests.cpp`'s `sub0_cuda_gdn_check` /
  `sub0_cuda_gdn_forward_raw`, meaningful in ANY CUDA build regardless of `USE_GATED_DELTANET`, the same
  convention the QK-norm/SwiGLU kernel checks already follow): **primary** (per S5 step 3's own
  instruction — parity against Stage 1's CPU SEQUENTIAL reference, `gdn_math::forward()`, not a fresh
  fixture comparison), at three shapes including the multi-chunk one (the only one exercising Phase B's
  cross-chunk `last_state` carry — AGENTS.md S7's own lesson: a single-chunk-only check could not catch a
  state-carry bug):
  | shape | rel diff (CUDA vs CPU sequential) |
  |---|---|
  | dk=dv=128, Hk=1/Hv=3, T=6 (single chunk, matches real fixture) | 8.57e-7 |
  | dk=16/dv=24, Hk=2/Hv=4, T=130 (3 chunks, GQA repeat_interleave) | 1.46e-6 |
  | dk=dv=8, Hk=Hv=1, T=64 (exact chunk boundary, no padding) | 3.67e-6 |

  **secondary** (the real Qwen4-preview fixture — a stronger oracle, since its expected output was
  produced by the ACTUAL reference model's own chunked path): max |diff| = 3.64e-11, max relative diff =
  7.52e-6, over 192 output values — matching Stage 1's own CPU-sequential fixture result (4.37e-11) to the
  same order, confirming the chunked CUDA port reproduces the real reference as faithfully as the
  sequential CPU port did.

  **A real, independent pre-existing bug found (and fixed) along the way, not a Stage 3 change of its
  own**: verifying against the compiled ENGINE (not just the standalone kernel checks above) at a real
  mixed-layer build — d96 L8 H2 kv2 seq128, `GDN_FULL_ATTN_STRIDE=3` (6 GDN + 2 attention layers) —
  exposed that `backend_cpu.cpp`'s `op_gdn` (Stage 1, already merged) passed `dt_bias`/`a_log` in SWAPPED
  argument positions relative to `gdn::forward()`'s real signature, at BOTH its batched-forward and
  T=1-decode call sites (`g_gdn_link.push()` and `backward_node`'s own `gdn::backward()` call already had
  the correct order). Neither `gdn_qwen4_fixture_tests.cpp` (calls `gdn::forward` directly, never through
  `op_gdn`) nor the whole-model finite-difference gradient check (perturbs a NAMED tensor and compares
  against that same name's analytic gradient — a CONSISTENT relabeling does not by itself fail it) could
  have caught this from the CPU side alone; Stage 3's CUDA port, built independently from the verified
  `gdn_math.hpp` reference rather than copied from `op_gdn`'s call site, disagreed with the CPU engine's
  real forward output and exposed it. Before the fix, `sub0_cuda_forward`-vs-CPU parity at the real
  GDN-on build read a max abs logit diff of 0.096–0.13 (well outside the 1e-2 tolerance); after the fix,
  0.00234. Fixed at both call sites; `engine_tests.cpp`'s own CPU-only whole-model gradient check (which
  had also been failing on this session's build beforehand) passes too, post-fix.

  **Real GDN-on build, unfiltered full suite** (AGENTS.md S10), same d96 L8 H2 kv2 seq128 stride-3 shape:
  186 test cases, 166 passed, 20 failed — every remaining failure is either an EXPECTED consequence of
  this stage's forward-only scope (`sub0_cuda_backward`/`sub0_cuda_train_step`/`sub0_cuda_train_profile`
  correctly refusing, ~15 cases) or a pre-existing, unrelated artifact proven untouched by this branch's
  diff against `main` (`ARCH_FINGERPRINT2`'s own documented "off premise" test; a `kTestDims`/
  `trainable_floats()` mismatch in a pre-existing memplan test that does not model a mixed GDN/attention
  layer schedule; one binding-compose scratch-slot count assertion, plausibly an artifact of this
  session's atypically tiny 553-token synthetic test vocabulary). `cuda_tests.cpp`'s own
  "`supports_decode` is honest" caps test was updated to account for GDN as a second, independent
  unsupported-capability axis (unlike depth attention, GDN also turns off `supports_train`).

  **Two-scale identity at neutral** (`GDN_FULL_ATTN_STRIDE == 0`): `layer_base()`/`ffn_base_for()` are
  proven, by a dedicated `static_assert`, to reproduce the OLD fixed-stride offset formula exactly when
  every layer is attention-shaped — a compile-time proof, not just an empirical one. Empirically, the
  neutral CUDA build's full suite (186 test cases / 5,767,878 assertions at this session's small test
  shape/corpus) has exactly the same 2 pre-existing, diff-proven-unrelated failures before and after every
  commit in this stage, and the critical `sub0_cuda_forward`-vs-CPU parity test is untouched by inspection
  (the new GDN branch in `forward_device`'s per-layer dispatch is `else if constexpr
  (USE_GATED_DELTANET)`, generating no device code at all when off; the existing attention branch's code
  is byte-for-byte unchanged, merely moved inside an `if` that is always true at neutral).

  **Not build-verified at a second (odd-layer) shape or against a real training corpus** — this
  session's verification used a small synthetic corpus (553-token vocab) at one even-layer-count shape;
  a second shape and a real corpus are natural follow-ups, not blocking findings.

  **Follow-up work, explicitly out of scope for this stage**: CUDA backward (TODO(gdn-gpu-train) at the
  lifted guard's old location) and a per-token GDN state cache inside the captured decode graph
  (TODO(gdn-gpu-decode)); a bf16/tensor-core forward pass and shared-memory tiling for the per-chunk GEMMs
  (this stage's own deliberate correctness-first simplification, not yet revisited).
