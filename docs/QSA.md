# Qwen Sparse Attention (QSA) + lightning indexer — design, derived from the reference

Status: **Stage 0 (config skeleton, hard-gated off) + Stage 1 (CPU forward only) — this pass.** Follows
`docs/GATED_DELTANET.md`'s, `docs/GATED_RESIDUAL.md`'s and `docs/MOE.md`'s own structure and staging
discipline (real source quoted verbatim, engine-interaction analysis, checkpoint design, two-scale identity
checks). `docs/MOE.md` in particular, as the most recently landed worked example, is this doc's closest
precedent. Per AGENTS.md S5, every equation below is quoted or worked from the real, installed
`transformers==5.16.1` source fetched for this doc -- not from recall, and not from
`docs/QWEN4_PREVIEW_REFERENCE.md`'s own partial quote alone (that doc quotes the indexer's `__init__` and
part of its `forward`; this pass re-fetched BOTH classes in full and found the attention side --
`Qwen4ExpTextAttention`'s doubled `q_proj` and its `sigmoid(gate)` output gate -- was never quoted there
at all, and is load-bearing, see S1b).

## 0. Sources, and their confidence

**Primary, highest-confidence source -- the real model's own code**, fetched directly on this machine
(`transformers==5.16.1`, mainline PyPI, no `trust_remote_code`), via
`python -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; print(inspect.getsource(m.Qwen4ExpTextQSAIndexer))"`
and the same for `Qwen4ExpTextAttention`, `apply_rotary_pos_emb`, `rotate_half`, `eager_attention_forward`,
`repeat_kv` and `Qwen4ExpTextRMSNorm` -- the identical technique every prior stage in this thread used,
confirmed still installed and importable on this machine.

**Real config values**, re-fetched this pass from the real `Qwen/Qwen3.8-Flash-Next` `config.json` rather
than trusted from `docs/QWEN4_PREVIEW_REFERENCE.md`'s table:

| key | value |
|---|---|
| `hidden_size` | 2560 |
| `num_attention_heads` | 24 |
| `head_dim` | **256** (NOT `hidden_size / num_attention_heads` = 106.7 -- see S3a) |
| `num_key_value_heads` | 2 (12:1 GQA) |
| `indexer_n_heads` | 4 |
| `indexer_kv_heads` | 1 |
| `indexer_head_dim` | 128 |
| `indexer_budget` | 2048 |
| `indexer_compress_ratio` | 4 |
| `partial_rotary_factor` | 0.25 (=> rotary_dim = 64 of head_dim 256) |
| `rms_norm_eps` | 1e-06 |
| `attention_bias` | false |
| `num_hidden_layers` | 48; `layer_types` = 12 x `full_attention`, 36 x `linear_attention` |

**A load-bearing fact this pass verified rather than assumed, and which decides S2's whole integration
story**: `layer_types` was read out of the real `config.json` element by element and is EXACTLY
`full_attention` iff `l % 4 == 3` for all 48 layers (12 of 48). That is bit-for-bit the rule
`layout.hpp`'s existing `gdn_schedule_for<LAYERS>(stride)` already implements at `stride == 4`
(`is_gdn = (l % stride != stride - 1)`). QSA therefore needs **no new per-layer schedule axis at all** --
see S2.

**No real-weight fixture existed before this pass.** S7/S8 extract this stage's own fixture (one real QSA
layer's indexer + attention weights) following the exact HTTP-Range-against-the-real-safetensors technique
validated five times now in this thread (GDN, n-gram embeddings, Gated Residual, MoE, and now this).

## 1. What the reference actually does

### 1a. The indexer -- `Qwen4ExpTextQSAIndexer`, quoted verbatim

```python
class Qwen4ExpTextQSAIndexer(nn.Module):
    """Select QSA token indices from compressed key blocks."""

    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.indexer_n_heads
        self.index_kv_heads = config.indexer_kv_heads
        self.index_head_dim = config.indexer_head_dim
        self.token_budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.token_budget // self.compress_ratio
        self.index_qk_proj = nn.Linear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
        )
        self.q_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpTextRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        # The cos/sin here are the full positions for the keys, so we need to slice to get only the current positions for the queries
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]

        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        # Note that the mask is never None here as we only allow eager and sdpa, and we do not allow sdpa's mask skip
        # It's always 4D with either bool (sdpa) or float (eager) and already gives us the valid indices
        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1, dtype=torch.int32, device=hidden_states.device,
        )
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                # Compute selected tokens
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )

                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    # Remap the indices of the blocks to the indices of individual tokens
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        # Create the additive mask to be added to the main causal mask
        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
        )
        # We absorb all the -1 by scaterring them to the last index that we will drop
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
        # if using eager, convert to float mask
        if attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

        return selected_token_mask
```

Facts this resolves, verified rather than assumed:

- **One fused projection, asymmetrically split.** `index_qk_proj` is a single
  `Linear(hidden_size, (n_heads + kv_heads) * head_dim, bias=False)` whose output is `torch.split` into a
  `n_heads * head_dim` query part and a `kv_heads * head_dim` key part -- NOT two separate projections and
  NOT an even split. With the real `indexer_kv_heads == 1`, `.squeeze(2)` on the key reshape is only valid
  at exactly 1 kv head; this port therefore **requires `idx_kv_heads == 1`** (asserted, not assumed -- the
  reference would raise on any other value).
- **The query is normalized and rotated; the raw key is NEITHER, at projection time.** `q_layernorm` +
  `apply_rotary_pos_emb` run on `q` immediately. `raw_keys` is stored, cached and pooled COMPLETELY RAW --
  `k_layernorm` and RoPE are applied only AFTER the mean-pool, to the POOLED block key, at the block's
  FIRST token's position (`group_starts`). Getting this order wrong (normalizing per token then pooling,
  or rotating per token then pooling) is a silently-plausible mutant this pass's fixture check is
  specifically designed to catch (S7).
- **Block pooling is a MEAN in float32**, over exactly `compress_ratio` consecutive VISIBLE tokens
  (`key_groups.float().mean(dim=1)`), and only over COMPLETE blocks -- `num_complete_blocks =
  n_visible // compress_ratio`.
- **The score is `relu`-then-SUM over indexer heads, then scaled** -- `relu(q_h . k_b)` summed over the 4
  indexer heads, divided by `sqrt(index_head_dim)`. Note the ReLU is applied BEFORE the head-sum, so a
  head that scores a block negatively contributes exactly 0, not a negative pull. This is a real
  divergence from an ordinary attention logit (no softmax anywhere in the indexer) and would be easy to
  get backwards from recall.
- **The block-level top-k is `min(block_topk, num_complete_blocks)`** where
  `block_topk = token_budget // compress_ratio` (2048 // 4 = **512 blocks**, i.e. 2048 tokens' worth). When
  the visible prefix is shorter than the budget, EVERY complete block is selected and the mechanism
  degenerates to dense attention -- which is exactly what happens at this project's own small test scales
  and is worked around deliberately in S7's fixture design.
- **The incomplete TAIL is ALWAYS visible**, unconditionally, regardless of any score:
  `tail = local_visible_indices[num_complete_blocks * compress_ratio :]` is concatenated onto the selection
  after the top-k. So the most recent `n_visible % compress_ratio` tokens are never droppable. (This is why
  the output buffer is sized `token_budget + compress_ratio - 1`.)
- **The result is a per-query visibility mask that is INTERSECTED with the causal mask**, not a
  replacement for it (S1b's `attention_mask & selected_token_mask` / `+ selected_token_mask`).

### 1b. The attention branch -- `Qwen4ExpTextAttention`, quoted verbatim

```python
class Qwen4ExpTextAttention(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, layer_idx: int):
        ...
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen4ExpTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.indexer = Qwen4ExpTextQSAIndexer(config, layer_idx)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        selected_token_mask = self.indexer(hidden_states, position_embeddings, attention_mask, past_key_values)
        # Combine both masks (they are never None, and are always 4D with either bool for sdpa, or float for eager)
        if attention_mask.is_floating_point():
            attention_mask = attention_mask + selected_token_mask
        else:
            attention_mask = attention_mask & selected_token_mask

        # The cos/sin are the full positions here due to the indexer, so we need to slice to get current positions
        position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        attn_output, attn_weights = attention_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
```

Facts this resolves, none of which appear in `docs/QWEN4_PREVIEW_REFERENCE.md`'s existing partial quote:

- **`q_proj` is DOUBLE width** (`num_attention_heads * head_dim * 2`) and is chunked PER HEAD -- the view is
  `(*input_shape, -1, head_dim * 2)` and the chunk is on the LAST axis, so within each head's own
  `2*head_dim` slab the FIRST `head_dim` is the query and the SECOND `head_dim` is the gate. It is **not**
  a `[all queries | all gates]` split of the flat row. Getting this wrong produces a subtly wrong, still
  plausible-looking output at every head but head 0 -- a real identity-swap-class hazard of exactly the
  kind `[[independent-reimplementation-catches-identity-swap-bugs]]` exists to catch, so S7's fixture check
  runs at `n_heads >= 2` specifically to make the two layouts distinguishable.
- **The attention output is element-wise gated by `sigmoid(gate)` BEFORE `o_proj`** -- a per-(head,channel)
  sigmoid gate, the same shape as the attention output itself. This is the "gated attention" variant, not
  plain MHA, and nothing in the existing engine's `op_attn` path has it.
- **`q_norm`/`k_norm` are per-head RMSNorm over `head_dim`, applied BEFORE RoPE**, using the same
  zero-centered `(1.0 + weight)` gain convention `docs/GATED_RESIDUAL.md` S1a already documented for this
  model family (`Qwen4ExpTextRMSNorm`, re-fetched and re-read this pass, quoted there). This project's own
  existing `op_qknorm` uses gain = `w` directly, NOT `1 + w` -- a real convention divergence, handled in S2.
- **RoPE is `apply_rotary_pos_emb`'s HALF-SPLIT convention over a `rotary_dim` PREFIX**, with
  `rotary_dim = cos.shape[-1]` and the remaining `head_dim - rotary_dim` channels passed through
  unrotated. Quoted verbatim:
  ```python
  def rotate_half(x):
      x1 = x[..., : x.shape[-1] // 2]
      x2 = x[..., x.shape[-1] // 2 :]
      return torch.cat((-x2, x1), dim=-1)

  def apply_rotary_pos_emb(q, k=None, cos=None, sin=None, unsqueeze_dim=1):
      cos = cos.unsqueeze(unsqueeze_dim); sin = sin.unsqueeze(unsqueeze_dim)
      rotary_dim = cos.shape[-1]
      q_rope, q_nope = q[..., :rotary_dim], q[..., rotary_dim:]
      q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
      q_rotated = torch.cat([q_rope, q_nope], dim=-1)
      ...
  ```
  At `partial_rotary_factor=0.25` and `head_dim=256`, `rotary_dim` is 64 -- and the SAME 64-wide cos/sin
  are reused by the indexer on its own `indexer_head_dim=128` vectors (S1a), rotating only their first 64
  channels. One `rotary_dim`, two different vector widths. This project's own RoPE is INTERLEAVED-pair and
  full-width (`[[rope-convention-mismatch-forward-vs-forward-one]]`), a real divergence -- handled in S2/S4b.
- **Attention itself is ordinary eager softmax attention with GQA `repeat_kv`**, `scaling = head_dim**-0.5`,
  softmax in float32, and the (causal AND selected) mask added as a `min_dtype` additive term. Quoted:
  ```python
  attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
  if attention_mask is not None: attn_weights = attn_weights + attention_mask
  attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
  attn_output = torch.matmul(attn_weights, value_states)
  ```
  So the ONLY thing "sparse" about QSA at the arithmetic level is which mask entries are `-inf`; there is
  no separate sparse kernel to port. That is a genuinely important scoping fact: the sparsity is a
  PERFORMANCE property of the real implementation, and a CORRECTNESS-neutral one for a reference port.

## 2. How QSA interacts with the existing `op_attn` path, and the three-way-schedule question

The task named this as the open design question ("whether `GDN_SCHEDULE`'s per-layer classification extends
to a three-way choice (attention/GDN/QSA) or needs its own schedule"). Worked through from the real
`config.json` and the real current code, not decided by analogy:

**Answer: a three-way per-layer classification, but derived from the EXISTING axes -- QSA gets no new
schedule axis of its own.** The reasons, in order of weight:

1. **In the real model, "full attention" and "QSA" are the same thing.** Every `Qwen4ExpTextAttention`
   instance unconditionally constructs `self.indexer = Qwen4ExpTextQSAIndexer(...)` and unconditionally
   calls it at the top of `forward` (S1b). There is no `Qwen4ExpTextAttention` WITHOUT an indexer anywhere
   in the model. So the real per-layer choice is binary (`linear_attention` = GDN vs `full_attention` =
   QSA), not ternary -- the ternary appearance comes only from THIS engine also supporting a plain
   (non-QSA) softmax-attention layer, which the real model does not have.
2. **The real `layer_types` array IS `gdn_schedule_for(4)`.** Verified element by element this pass (S0).
   A second, independent QSA stride would be able to express layer sets the real model cannot have (a QSA
   layer that is also a GDN layer is meaningless; a full-attention layer that is NOT QSA is not a Qwen4
   layer), i.e. it is speculative surface in the precise sense AGENTS.md S8 forbids -- and MoE's own
   `D_FF` reuse (docs/MOE.md S3a) is this thread's established precedent for declining a redundant axis.
3. **Nothing is lost.** `USE_QSA` is a single model-wide gate; the per-layer selection composes it with the
   existing `GDN_SCHEDULE`. At `GDN_FULL_ATTN_STRIDE == 0` (this project's default) EVERY layer is
   full-attention, so a QSA-on/GDN-off build makes every layer QSA -- the cheapest useful test shape. At
   `GDN_FULL_ATTN_STRIDE == 4` with QSA on, the schedule is bit-for-bit the real model's 12-of-48.

Concretely, `layout.hpp` gains a derived, independently-testable three-way classification (NOT a new CLI
axis):

```cpp
enum class LayerMixer : unsigned char { Attn, Gdn, Qsa };
template <int LAYERS> consteval std::array<LayerMixer, LAYERS> qsa_schedule_for(int gdn_stride, bool qsa_on);
inline constexpr auto MIXER_SCHEDULE = qsa_schedule_for<N_LAYERS>(GDN_FULL_ATTN_STRIDE, USE_QSA);
```

parameterized on `(layers, gdn_stride, qsa_on)` rather than closed over this build's constants -- the same
reasoning `gdn_schedule_for<LAYERS>`/`depth_schedule_for<EXECS>` were given, and the same reason AGENTS.md
S7 exists: the modulo rule's behaviour at an ODD layer count and at a stride that does not divide it must
be assertable without compiling that build (S9's own two-scale test does exactly this).

**`GDN_SCHEDULE` stays exactly as it is and keeps its existing meaning** (`full_attn[l]`), so every existing
consumer -- `make_param_layout()`, `Model::build_layout`, `Model::forward`, `Model::forward_one` -- is
unchanged in a QSA-off build. `MIXER_SCHEDULE[l] == Qsa` is definitionally
`GDN_SCHEDULE.full_attn[l] && USE_QSA`, asserted as an invariant so the two can never drift.

### 2a. How the QSA sublayer sits next to the existing `op_attn`

`op_attn`/`op_rope`/`op_qknorm` are **not modified, and not reused**. A QSA layer's whole mixer sublayer is
one new op, `op_qsa(a, L)`, exactly the `op_gdn(a, L)` precedent -- indexer, q/k/v/gate projection,
per-head RMSNorm, RoPE, masked softmax attention, the sigmoid output gate and `o_proj`, all inside one op
reading `Layer&` directly. Three reasons this is right rather than threading QSA through `op_attn`:

- **`op_attn` has no mask input at all** (it is hard-coded causal), and adding a per-query
  `[T, T]` boolean mask parameter to a shared op used by every existing build is exactly the
  "changing a shared surface" hazard AGENTS.md S10 is about, for zero benefit to a forward-only stage.
- **The norm and RoPE conventions genuinely differ** (S1b: `(1 + w)` gain vs this engine's `w`; half-split
  over a rotary prefix vs this engine's interleaved full-width). Reusing `op_qknorm`/`op_rope` would make
  the QSA layer numerically NOT the real model's layer, defeating the fixture gate's entire purpose.
  Keeping the real conventions inside `qsa_math.hpp` and out of the shared ops means a QSA-off build is
  byte-identical by construction, not by a runtime branch that happens to be false.
- **The output gate has no home in the existing graph** -- `attn_output * sigmoid(gate)` sits between
  `op_attn` and `Wo`, and there is no existing op there.

A QSA layer therefore uses NONE of `Wq`/`Wk`/`Wv`/`Wo`/`QNorm`/`KNorm` in their existing form. Instead it
gets its own tensor set (S3b), the same way a GDN layer replaces those six with its own nine.

### 2b. Deliberate, documented Stage-1 divergences from the real model

Recorded explicitly (not silently), the same way `docs/GATED_RESIDUAL.md` S2 recorded its `Ln1`/`Ln2` one:

1. **`head_dim` stays `D_MODEL / N_HEADS` in the ENGINE.** The real model's `head_dim=256` with
   `num_attention_heads=24` and `hidden_size=2560` means `n_heads * head_dim = 6144 != hidden_size` --
   the real `q_proj` is `[2560, 12288]` and `o_proj` is `[6144, 2560]`. This engine's `D_HEAD = D_MODEL /
   N_HEADS` invariant is baked into `sub0_config.hpp` itself and into the existing attention/GDN/KV-cache
   code; breaking it is an independently-scoped, shape-changing refactor with a much larger blast radius
   than this stage. `include/sub0/qsa_math.hpp` DOES take `head_dim` as an independent `Dims` field (so
   the fixture can exercise the real, non-dividing relationship), and the ENGINE instantiates it at
   `head_dim = D_HEAD`. Named here as WP4's problem, not silently skipped.
2. **Full-width rotary in the engine (`rotary_dim = D_HEAD`), partial in the math core.** `qsa_math.hpp`
   takes `rotary_dim` explicitly and caller-supplied `cos`/`sin` tables, so the fixture exercises the real
   `partial_rotary_factor=0.25`. The engine passes `rotary_dim = D_HEAD` (full) because
   `partial_rotary_factor` is not a config axis this project has, and adding one is unconsumed surface
   (AGENTS.md S8) until WP4 actually needs it.
3. **Causal-prefix visibility only.** The reference's `torch.nonzero(visible_token_indices[...])` supports
   an ARBITRARY per-query visibility set (padding masks, packed sequences). This engine has exactly one
   mask -- plain causal -- so the visible set for query `t` is always the contiguous prefix `[0, t]`, and
   `qsa_math.hpp` takes `kv_len` and derives the blocks from it directly rather than from a general index
   list. This is a restriction of the reference to the only case this engine can express, not a different
   formula: at a contiguous prefix the two are identical by inspection (`local_visible_indices == arange(kv_len)`).
4. **`q_proj`'s doubled width is stored as TWO tensors, not one.** `PARAM_LAYOUT` gets `Wq`-shaped
   `QsaQProj` and `QsaGateProj` separately rather than one `[D_MODEL, 2*D_MODEL]` tensor. A chunked linear
   IS exactly two linears (`chunk` on the output axis of a bias-free `nn.Linear` is a partition of its
   weight rows), so this is a storage-layout re-derivation of the same arithmetic -- the same class of
   convention re-mapping as this project's `[rows=in, cols=out]` transpose, NOT an approximation. Recorded
   because a future WP4 weight-transplant must split the real `q_proj` per-head (S1b's per-head chunk
   order), not down the middle of the flat row.

**LoopSplit / depth-attention / Gated Residual / MoE interaction**: none. QSA is a mixer sublayer, exactly
where `op_gdn` sits, so Gated Residual's `gr_read`/`gr_write` wrap it identically and MoE's FFN replacement
is downstream and untouched. Depth attention is defined in terms of softmax attention's own K/V and is out
of scope for a QSA layer for the same reason `docs/GATED_DELTANET.md` gave for a GDN layer -- a deliberate,
documented Stage-1 simplification. Under LoopSplit a repeated middle layer re-runs the SAME `Layer`'s QSA
tensors, exactly as it already re-runs the same `Ln1` -- QSA has no cross-EXECUTION state (its only
cross-call state is the decode-path indexer key cache, which is per-execution-slot like `g_kv`, S6).

## 3. Checkpoint / `PARAM_LAYOUT` impact

### 3a. Classification of every axis the task named, per the `ARCH_FINGERPRINT2` precedent

`layout.hpp`'s own three-way rule ("1. changes a tensor shape -> `PARAM_FLOATS` already discriminates it;
2. changes computation, not shape -> ADD to `ARCH_FINGERPRINT`/`ARCH_FINGERPRINT2`; 3. deliberately
variable between train and inference -> exclude and say why"), applied to each of the eight quantities:

| reference quantity | this project's mapping | new CLI axis? | classification |
|---|---|---|---|
| `num_attention_heads` | `N_HEADS` (**existing**) | no | **rule 1** -- already shape-changing via `D_KV`/`D_HEAD`; already in `MODEL_ARCH_ID`. QSA adds a second shape dependence (`QsaGateProj` is `[D_MODEL, N_HEADS*D_HEAD]`), which only strengthens it. |
| `head_dim` | `D_HEAD` = `D_MODEL / N_HEADS` (**existing, derived**) | no | **rule 1**, derived -- not independently settable (S2b.1). |
| `num_key_value_heads` | `N_KV_HEADS` (**existing**) | no | **rule 1** -- `D_KV` narrows `Wk`/`Wv`; `layout.hpp`'s own GQA comment already classifies it. |
| `indexer_n_heads` | `QSA_INDEXER_N_HEADS` | **new** | **rule 1** -- widens `QsaIdxQkProj` to `[D_MODEL, (n+kv)*hd]`, strictly monotonically. |
| `indexer_kv_heads` | `QSA_INDEXER_KV_HEADS` | **new** | **rule 1** -- same tensor, same monotonic widening. (Constrained to exactly 1 when on, per S1a.) |
| `indexer_head_dim` | `QSA_INDEXER_HEAD_DIM` | **new** | **rule 1** -- widens `QsaIdxQkProj` AND both `[1, hd]` indexer norms, strictly monotonically. |
| `indexer_budget` | `QSA_INDEXER_BUDGET` | **new** | **rule 2** -- changes NO tensor shape. It sets `block_topk = budget / ratio`, i.e. HOW MANY blocks a query may attend to. Two builds identical but for the budget have BYTE-IDENTICAL checkpoints and compute different attention. Joins `ARCH_FINGERPRINT2`. |
| `indexer_compress_ratio` | `QSA_INDEXER_COMPRESS_RATIO` | **new** | **rule 2** -- changes NO tensor shape either (pooling is over activations, not weights). It changes both the block size AND `block_topk`. Joins `ARCH_FINGERPRINT2`. |

Reasoning for the two rule-2 members, spelled out because this is the exact failure mode
`ARCH_FINGERPRINT2` exists for: `indexer_budget` and `indexer_compress_ratio` participate only in
`block_topk = budget // ratio` and in the *pooling arity*. Neither appears in ANY `nn.Linear`/`nn.Parameter`
shape in either quoted class (checked exhaustively against S1a/S1b's `__init__`s, not assumed) -- the
indexer's only parameters are `index_qk_proj`, `q_layernorm` and `k_layernorm`, none of which mentions
either. So a checkpoint trained at `budget=2048, ratio=4` loads with zero complaint into a `budget=256,
ratio=8` build and silently attends to a different token set: precisely the `LOOP_MIDDLE_LAYERS`/
`ROPE_THETA` hazard `ARCH_FINGERPRINT` was created for, and the identical shape of `GDN_FULL_ATTN_STRIDE`'s
and `EXPERTS_PER_TOK`'s own membership.

**Bit placement in `ARCH_FINGERPRINT2`** (which has 48 spare bits, `[63:16]`, deliberately reserved from
day one; bytes 0 and 1 are already `gdn_full_attn_stride` and `experts_per_tok`):

```
  [63:40] reserved (always 0 today)   |   [39:32] indexer_compress_ratio (8 bits)
  [31:16] indexer_budget (16 bits)    |   [15:8]  experts_per_tok        |  [7:0] gdn_full_attn_stride
```

`indexer_budget` gets **16** bits, not 8, worked through explicitly rather than assumed: the real value is
2048, which does not fit 8 bits, and a token budget is naturally in the thousands. `compress_ratio` gets 8
(real value 4; a ratio above 255 is not a mechanism anyone builds). Both are `static_assert`ed against
their field widths, matching how `LOOP_REPEATS`/`DEPTH_ATTN_STRIDE` are. At the neutral setting (both 0)
`ARCH_FINGERPRINT2` is bit-identical to every value it has ever produced, so no existing checkpoint is
invalidated -- the same "additive, gracefully-degrading" property AGENTS.md S3 rule 2 requires and MoE's
own byte-1 addition already established. No new trailer record and no `CKPT_VERSION` bump: this rides the
`ARCH_FINGERPRINT2` trailer already on disk.

`MODEL_ARCH_ID` mixes in the three shape-changing indexer axes unconditionally (it covers every axis, by
its own stated design intent); the two fingerprint-2 members ride the existing `mix(ARCH_FINGERPRINT2)`
call and are NOT folded in twice, exactly as `EXPERTS_PER_TOK` already is not.

### 3b. Per-layer `PARAM_FLOATS` delta, worked from the real shapes

Using this project's `[rows=in, cols=out]` convention throughout (the transpose of `nn.Linear`'s
`[out_features, in_features]`, re-derived per AGENTS.md S5). Let `IDX_QK_OUT = (QSA_INDEXER_N_HEADS +
QSA_INDEXER_KV_HEADS) * QSA_INDEXER_HEAD_DIM`.

A QSA layer's own tensors (REPLACING the softmax-attention layer's `Wq/Wk/Wv/Wo[+QNorm,KNorm]`):

```
  QsaQProj      = D_MODEL * D_MODEL                (= N_HEADS*D_HEAD; the query half of the real q_proj)
  QsaGateProj   = D_MODEL * D_MODEL                (the gate half -- S2b.4)
  QsaKProj      = D_MODEL * D_KV
  QsaVProj      = D_MODEL * D_KV
  QsaOProj      = D_MODEL * D_MODEL
  QsaQNorm      = D_HEAD                           ([1, D_HEAD], per-head, (1+w) gain)
  QsaKNorm      = D_HEAD
  QsaIdxQkProj  = D_MODEL * IDX_QK_OUT
  QsaIdxQNorm   = QSA_INDEXER_HEAD_DIM             ([1, hd])
  QsaIdxKNorm   = QSA_INDEXER_HEAD_DIM
```

against what a softmax-attention layer costs today:

```
  attn_layer_floats = 2*D_MODEL*D_MODEL + 2*D_MODEL*D_KV + (USE_QK_NORM ? 2*D_HEAD : 0)
  qsa_layer_floats  = 3*D_MODEL*D_MODEL + 2*D_MODEL*D_KV + 2*D_HEAD
                    + D_MODEL*IDX_QK_OUT + 2*QSA_INDEXER_HEAD_DIM
  delta_per_layer   = qsa_layer_floats - attn_layer_floats
                    = D_MODEL*D_MODEL + D_MODEL*IDX_QK_OUT + 2*QSA_INDEXER_HEAD_DIM
                      + (USE_QK_NORM ? 0 : 2*D_HEAD)
  delta_total       = (number of FULL-ATTENTION layers) * delta_per_layer
```

**Strictly positive whenever QSA is on**, worked through rather than asserted: every term is a product of
strictly positive quantities and none can cancel (`D_MODEL*D_MODEL` alone -- the gate projection QSA adds
and plain attention does not have -- already dominates), so `delta_total` is monotonically increasing in
`D_MODEL`, `IDX_QK_OUT` and the full-attention layer count, and exactly 0 when QSA is off. This is the
same strength of guarantee `docs/GATED_RESIDUAL.md` S3b established, i.e. the STRONGEST form of
"shape-changing": `PARAM_FLOATS` alone discriminates any QSA-on build from any QSA-off build at identical
dims, and `engine_core.cpp`'s existing `Header` byte-count comparison already refuses the cross-load.

**The number of full-attention layers is itself `GDN_SCHEDULE`-dependent**, so `qsa_param_delta()` takes
`(n_layers, gdn_stride, d_model, d_kv, d_head, qk_norm, idx_n, idx_kv, idx_hd)` explicitly and recomputes
the schedule internally -- deliberately NOT closed over this build's constants, so `layout_tests.cpp` can
hand-check it at hypothetical shapes including an odd layer count with a non-dividing stride (S9).

### 3c. On-disk plumbing

Purely additive at every point `PARAM_LAYOUT` already handles additively: new `PKind`s appended to the
enum (enumerator order is irrelevant -- `PARAM_LAYOUT`'s APPEND order is what matters), new tensors placed
at the natural position in `make_param_layout()`'s existing per-layer loop as a THIRD branch of the
existing `full_attn` dispatch (`if (qsa) {...} else if (full_attn) {...} else {GDN}`), so a QSA-off build's
byte layout for every other tensor is completely unchanged. No new `Header` field (S3b: `PARAM_FLOATS`
discriminates), no new trailer record and no `CKPT_VERSION` bump (S3a: the two shape-neutral axes ride the
existing `ARCH_FINGERPRINT2` trailer).

## 4. Compile-time specialization

### 4a. New `RunConfig`/generated-header constants

Five new int flags, mirroring `--num-experts`/`--hc-count`'s own shape exactly:

- `--qsa-indexer-n-heads`   -> `constexpr int QSA_INDEXER_N_HEADS` (0 = off, else >= 1)
- `--qsa-indexer-kv-heads`  -> `constexpr int QSA_INDEXER_KV_HEADS` (0 = off, else exactly 1 -- S1a)
- `--qsa-indexer-head-dim`  -> `constexpr int QSA_INDEXER_HEAD_DIM` (0 = off, else even and >= 2, since
  the half-split rotary needs an even rotary prefix)
- `--qsa-indexer-budget`    -> `constexpr int QSA_INDEXER_BUDGET` (0 = off, else >= compress_ratio)
- `--qsa-indexer-compress-ratio` -> `constexpr int QSA_INDEXER_COMPRESS_RATIO` (0 = off, else >= 1)

plus the derived gate and never-degenerate buffer forms:

- `inline constexpr bool USE_QSA = (QSA_INDEXER_N_HEADS >= 1 && QSA_INDEXER_BUDGET >= 1);`
- all five must be on or off TOGETHER -- a single `static_assert` on the count of nonzero flags, so a
  half-configured QSA build cannot compile (the same "guard at the lowest callable seam" pattern
  `HC_COUNT`/`HC_LOWRANK` and `NUM_EXPERTS`/`EXPERTS_PER_TOK` already use, generalized to five).
- `inline constexpr qsa::Dims QSA_DIMS{...}` -- always a valid, never-divide-by-zero shape even when off
  (the "describes a shape nothing builds" idiom `GDN_DIMS`/`GR_DIMS`/`MOE_DIMS` already use), plus
  `QSA_DIMS_BUF` (all widths clamped to >= 1) purely for sizing decode-path scratch arrays.

**No new CLI flag for `num_attention_heads`/`head_dim`/`num_key_value_heads`**: `--heads`/`--kv-heads`
already exist and `D_HEAD` is derived (S3a). Adding duplicates would be unconsumed surface (AGENTS.md S8)
AND would create two sources of truth for the same axis -- exactly the class of bug AGENTS.md S10's
`current_build_dims()` note exists to prevent. This is the same call `docs/MOE.md` S3a made for `D_FF`.

**Stage 0 (this pass's first commit) hard-clamps all five to 0**: `->check(CLI::Range(0, 0))` in
`configurator.cpp` AND `static_assert(QSA_INDEXER_* == 0, ...)` (the strict single-value form) in
`layout.hpp`, so a config-skeleton commit that compiles genuinely cannot express anything but the current
architecture. Stage 1 (this pass's second commit) relaxes both layers to the real ranges once `op_qsa`
exists to compute against -- the exact two-refusal staging GDN/GR/MoE each used.

### 4b. `include/sub0/qsa_math.hpp` -- the engine-free math core

A `sub0::qsa::Dims{hidden_size, n_heads, head_dim, n_kv_heads, idx_n_heads, idx_kv_heads, idx_head_dim,
budget, compress_ratio, rotary_dim}` struct and a small set of free functions, all taking this project's
`[rows=in, cols=out]` weight convention and explicit caller-owned scratch (AGENTS.md S1 -- no heap
allocation anywhere), and all parameterized on `Dims` rather than the build's constants so a standalone
test can exercise the real fixture's own (non-dividing, partial-rotary) shape:

- `rms_norm_row(x, w, n, eps, out)` -- the `(1 + w)` zero-centered gain convention (S1b).
- `rope_apply_row(x, cos, sin, rotary_dim, head_dim)` -- half-split over the rotary PREFIX (S1b).
- `indexer_project_row(d, x, qk_proj_w, q_ln_w, cos_pos, sin_pos, eps, out_q, out_raw_k)` -- the fused
  split, q normalized+rotated, the key left RAW (S1a).
- `indexer_select_row(d, q, raw_keys, kv_len, k_ln_w, cos, sin, eps, out_mask, scratch)` -- block the
  causal prefix, mean-pool, `k_layernorm`, rotate at the block start, `relu`-sum-scale score, top-k
  blocks, ALWAYS keep the tail; writes a `[kv_len]` visibility byte-mask.
- `attn_project_row(d, x, q_w, gate_w, k_w, v_w, q_norm_w, k_norm_w, cos_pos, sin_pos, eps, out_q,
  out_gate, out_k, out_v)`.
- `attn_row(d, q, gate, k_cache, v_cache, kv_len, mask, o_proj_w, out, scratch)` -- masked softmax
  attention with GQA broadcast, `head_dim^-0.5` scaling, `* sigmoid(gate)`, then `o_proj`.
- `forward(d, T, hidden, ...all weights..., cos, sin, eps, out, scratch)` -- the whole prefill sublayer, a
  thin loop over the row functions, so `Model::forward` and `Model::forward_one` provably run the SAME
  arithmetic (the forward/forward_one parity test is what gates that, S9).

The top-k over blocks is host-side scalar code inside `indexer_select_row`, NOT a separate differentiable
`Node` -- the identical decision, for the identical reasons, that `docs/MOE.md` S4c made for the router's
own top-k (bounded tiny compute; no Stage-1 backward consumer that would need it split out; splitting it
would create an order-sensitive cross-op scratch dependency for no forward-pass benefit).

## 5. The `Node`-fanout question -- one op, no side table (the `op_gdn`/`op_moe` precedent)

One new `Op::Qsa` enumerator: `op_qsa(Node* a, Layer& L)` -> output `[T, D_MODEL]`, `out->a = a`. It reads
its ten tensors off `Layer&` directly. No side table: Stage 1 has no backward walking the node pool yet
(S6), so nothing needs to recover those tensors from a bare `Node*` -- the identical resolution
`op_gdn`'s Stage 1 and `op_moe` both reached (`GdnLinkCache` was a Stage-2, BACKWARD-only need).

## 6. Backward -- explicitly out of scope, loud abort

`backward_node`'s `switch` gains one `case Op::Qsa:` -> `std::println(stderr, "fatal: ...")` +
`std::abort()`, refusing to silently train a different architecture than requested -- the exact
`Op::GDN`/`Op::GatedResidual`/`Op::Moe` Stage-1 precedent.

**The real subtleties a future Stage 2 must not get wrong, named so it is not designed blind**:
(1) the block-level `topk` is non-differentiable at the selection boundary, and the SELECTED SET must be
treated as a constant for the step (straight-through on the selection) -- but unlike MoE's router there is
no softmax coupling the unselected blocks, because the indexer's score path is `relu`-then-SUM with NO
normalizer, so an unselected block's `index_qk_proj` columns genuinely DO receive zero gradient, and a
block whose every head scored negative receives zero gradient through the ReLU even when selected. That is
the OPPOSITE conclusion to `docs/MOE.md` S6's router, and assuming the MoE answer transfers here would be
wrong in both directions. (2) The indexer produces a MASK, and gradient does not flow through a mask at
all -- so in the reference the indexer's own weights receive gradient ONLY via the pooled-key/query score
path, which the forward pass discards after the top-k. A naive Stage 2 that backpropagates only through
the attention output would train `index_qk_proj` not at all, silently. (3) `raw_keys` is pooled ACROSS
tokens, so an indexer key gradient at block `b` must be split evenly (`1/compress_ratio`) across the
`compress_ratio` tokens in that block, at the block start's rotation -- a cross-token coupling this
engine's per-row backward machinery has no existing analogue for.

## 7. Correctness gate

`tests/qsa_qwen4_fixture_tests.cpp` (engine-free, mirroring `gdn_qwen4_fixture_tests.cpp`'s /
`gated_residual_qwen4_fixture_tests.cpp`'s / `moe_qwen4_fixture_tests.cpp`'s pattern): reads
`tests/fixtures/qwen4_preview/qsa_layer3_small_*` -- the real
`model.language_model.layers.3.self_attn.*` tensors (layer 3 is the FIRST `full_attention` layer in the
real 48-layer stack, verified from `layer_types`, so it is the first layer that HAS an indexer at all),
extracted by HTTP Range against the real safetensors shards and column-sliced from the real
`hidden_size=2560` down to a commit-sized width -- transposes the raw PyTorch `[out,in]` weight files into
this project's `[in,out]` convention, and calls `sub0::qsa::*` directly. Compares against the real,
unmodified `transformers==5.16.1` `Qwen4ExpTextAttention.forward()` output on a reproducible-seed synthetic
input (exact numbers in S10, Stage 1 execution output, not fixed here).

**Which axes are sliced and which are kept real** (the same distinction `docs/GATED_RESIDUAL.md` S3a drew
for `HC_COUNT` and `docs/MOE.md` for `num_experts`): `hidden_size` is an ordinary projection width, freely
sliced. `num_attention_heads` (2, not 24) and `num_key_value_heads` (1) are reduced but kept `>= 2` /
GQA-nontrivial specifically so S1b's per-head query/gate chunk order is DISTINGUISHABLE from a flat-row
split. `indexer_n_heads` is kept `>= 2` so the `relu`-then-SUM-over-heads is distinguishable from a single
head. `indexer_kv_heads` is kept at its real 1 (the reference requires it). **`indexer_budget` and
`indexer_compress_ratio` are the axes this mechanism exists to test and are deliberately set so the
selection is NOT degenerate**: at the real 2048/4 with a short test sequence every block is selected and
the mask is all-ones, i.e. the fixture would pass identically for an implementation that never ran the
indexer at all. The fixture instead uses a small budget so `block_topk < num_complete_blocks` and real
blocks are genuinely DROPPED -- verified by asserting the reference's own mask has zeros in it before the
comparison is trusted.

**Presence/mutation checks**, per the depth-attention lesson this thread keeps citing:
1. A second REAL reference output is computed by re-running the real PyTorch module with the indexer's
   `index_qk_proj` perturbed so it selects a genuinely DIFFERENT block set; our port's own output under the
   same perturbation must match that second real reference, AND the two real outputs must differ by a real,
   nonzero amount -- proving the output depends on WHICH tokens were selected, not merely that a plausible
   number comes out.
2. A fixture-free property check that the selected set tracks the score: a synthetic indexer that scores
   one block highest selects that block, and the incomplete TAIL is present in the selection regardless of
   any score (S1a) -- the one rule a "just take the top-k blocks" mutant would drop.
3. A fixture-free check that the per-head query/gate chunk order (S1b) is the per-head one: at
   `n_heads = 2` a flat-row split and a per-head split give provably different outputs, asserted directly.

## 8. Fixture extraction

Same surgical HTTP-Range technique validated four times before this: read
`model.safetensors.index.json` to find which shard holds each `model.language_model.layers.3.self_attn.*`
tensor, read that shard's 8-byte header length + JSON header via one small Range request to get each
tensor's `data_offsets`, then Range-fetch ONLY the byte spans of the row slices actually needed. Never a
bulk shard download. Provenance, slicing convention and the exact tensor list are recorded in
`qsa_layer3_small_manifest.json`, as every prior fixture does.

## 9. Two-scale identity + real-build verification

Per AGENTS.md S7 and this thread's own repeated lesson (LoopSplit's odd-layer-count `static_assert` was
invisible at one scale and instant at another): `layout_tests.cpp` pins QSA's neutral-setting identity at
TWO shapes -- this thread's now-standard pair, d96 L8 H2 seq128 (even) and d132 L11 H4 kv2 seq96
(odd/ragged) -- asserting `PARAM_FLOATS`/`NUM_PARAMS` are UNCHANGED from a QSA-unaware calculation at both,
plus a standalone `qsa_param_delta()` check against hand-computed values at each.

**The odd/non-dividing layer-count case is mandatory and is tested directly on the SCHEDULE, not only on
the delta**: `qsa_schedule_for<LAYERS>(gdn_stride, qsa_on)` is asserted at `LAYERS = 11` with
`gdn_stride = 4` (4 does not divide 11: layers 3 and 7 are full-attention/QSA, layer 10 is NOT, and the
final partial group 8..10 must not be misclassified), at `LAYERS = 11, stride = 3`, and at
`LAYERS = 8, stride = 4` -- the exact class of scale-dependent bug AGENTS.md S7 was written about, made
assertable without compiling those builds because the function takes its parameters explicitly.

A genuinely QSA-ON small build is compiled once to confirm `forward()`-vs-`forward_one()` parity holds with
the mechanism actually active (the standard `engine_tests.cpp` parity test, which is QSA-agnostic -- it
exercises whatever the compiled `Model` computes). Exact assertion counts, hashes and parity numbers are in
S10 (Stage 1 execution output), not fixed here.

## 10. Results (Stage 1 execution)

**A real OUT-OF-BOUNDS WRITE found and fixed by actually building a QSA-ON config, not by reasoning
alone.** `rope_apply_row()` rotated `rotary_dim` channels of whatever vector it was given, but the
indexer's own per-head query and pooled block key are `indexer_head_dim` wide — so at any config with
`indexer_head_dim < rotary_dim` it wrote `(rotary_dim - indexer_head_dim)` floats past the end of that
head's slice, into whatever the caller's scratch layout happened to put next. It surfaced as a
`forward()`-vs-`forward_one()` parity FAILURE (worst per-position relative diff **0.11375**, against a
`1e-3` gate) at `D_MODEL=16, N_HEADS=2, indexer_head_dim=4, D_HEAD=8`. Root-caused with a standalone
batched-vs-incremental harness that isolated it precisely: **PREFIX-only batched calls
(`qsa::forward(d, t+1, ...)`) matched the incremental row-by-row composition EXACTLY at every row, while
the full-`T` batched call diverged from both** — i.e. cross-row scratch contamination whose presence
depended on the buffer layout, not a formula error, and not (as first suspected) discrete top-k
selection amplifying float noise: the measured block-score margins were healthy (`0.08`-`0.28`, nowhere
near a tie). Fixed by giving `rope_apply_row` the vector's own width and clamping the rotary prefix to
it, AND by adding `static_assert(!USE_QSA || QSA_INDEXER_HEAD_DIM >= D_HEAD)` in `layout.hpp` — a
precondition the real model satisfies for free (`rotary_dim 64 <= indexer_head_dim 128 <= head_dim 256`)
but this engine, whose `rotary_dim` is `D_HEAD` and whose indexer width is an independent axis, does not.
The batched-vs-incremental check is now a permanent regression test
(`tests/qsa_qwen4_fixture_tests.cpp`, per `[[regression-test-on-reproducible-bug]]`) asserting BITWISE
equality, which it now achieves (`max abs diff = 0.0`).

**Fixture correctness gate** (`tests/qsa_qwen4_fixture_tests.cpp`, real weights extracted from
`Qwen/Qwen3.8-Flash-Next` layer 3's real `model.language_model.layers.3.self_attn.*` tensors — **198,976
bytes total fetched by HTTP Range**, never a bulk shard download): **`max|out - expected| = 1.86265e-09`**
over 384 values (`sum|expected| = 0.511787`) against the real, unmodified `transformers==5.16.1`
`Qwen4ExpTextAttention.forward()`. **The fixture is genuinely non-degenerate**: the real indexer drops
**112 of the 300 causally-visible mask entries** — asserted by both the extraction script (before writing)
and the test (before trusting any match), because at the real `budget=2048, ratio=4` on a 24-token
sequence every block would be selected and this whole file would pass identically for an implementation
that never ran the indexer at all.

**Presence/mutation checks** (same file): (1) a SECOND real reference computed by re-running the real
module with ONLY the indexer's key-half `index_qk_proj` rows negated and scaled — our port's own output
under the same perturbation matches that second real reference to `1.39698e-09`, AND the two real outputs
genuinely differ by `max|out(real) - out(mutant)| = 0.0079712` with **128 mask entries changed**, proving
the output depends on WHICH tokens were selected; (2) fixture-free property checks that block selection
tracks the score and FLIPS when the query flips, that the incomplete TAIL survives regardless of score,
and that `min(block_topk, num_blocks)` keeps everything when there are fewer blocks than the budget;
(3) the bitwise batched-vs-incremental check above, itself guarded by `REQUIRE(total_dropped > 0)` so it
cannot pass vacuously on a dense mutant. `[qsa]` tag: **4 test cases / 349 assertions**, green. Full
`sub0_frontend_tests`: **196 test cases / 115,097 assertions** (192 / 114,748 before — exactly +4 cases
and +349 assertions, nothing else).

**Two-scale identity check** (AGENTS.md §7): at the neutral setting the full default engine test suite
(`sub0_tests`) is HASH-identical to pre-QSA `main` at BOTH shapes, through every commit in this pass:

| shape | assertions (before → after) | test cases | forward hash | grad hash | decode hash |
|---|---|---|---|---|---|
| d96 L8 H2 seq128 vocab26260 (even) | 18,118,452 → 18,119,453 | 142 → 145 | `d782c2f22a8f8470` | `b9e6ef80b96c7987` | `da8778997d09402e` |
| d132 L11 H4 kv2 seq96 vocab26260 (odd/ragged) | 29,667,408 → 29,668,745 | 142 → 145 | `44fcf318125286ff` | `b0299e22d5ea4df7` | `e136194ed6db704d` |
| d196 L11 H7 seq256 vocab26260 (production-shaped, `d196check`) | 53,918,298 → 53,919,635 | 142 → 145 | `acfb10b7d0fa03ad` | `5f16f677739cecc9` | `470870e4c0867de5` |

Every "before" column above is a REAL measurement, not an assumed number: pre-QSA `main` (`b3ae26d`) was
checked out and rebuilt at each shape in the same build directory with the same compiler and flags, then
re-run, so the comparison is genuinely before/after on this machine. (The plan document's own recorded
production baseline — 53,917,287 / 137 cases — predates the Gated Residual and MoE test cases that have
landed since, which is exactly why it was re-derived rather than cited.) The d196 and d132 deltas are
identical (+1337) because both have `N_LAYERS == 11` and therefore the same `NUM_PARAMS` (123), which is
what the new per-`PARAM_LAYOUT`-entry loop scales with.

All three hashes are UNCHANGED at both shapes. The assertion deltas (+1001 and +1337) are fully accounted
for by the three new `[layout][qsa]` test cases and nothing else: both add the same fixed set, plus a
per-`PARAM_LAYOUT`-entry loop (10 `REQUIRE`s per tensor: 90 tensors at d96 → 900, 123 at d132 → 1230,
difference 330) and a per-layer loop (2 `REQUIRE`s per layer, difference 6) — 330 + 6 = 336, exactly the
observed 1337 − 1001.

**Real QSA-ON builds** (this doc's own scope: correctness-fixture-scale configs, not production dims):

- `D_MODEL=16, N_LAYERS=2, N_HEADS=2, N_KV_HEADS=2, SEQ_LEN=32`, indexer `2` heads / `1` kv / `head_dim 8`,
  `budget=8, ratio=4` (⇒ `block_topk = 2` of up to 6 complete blocks — genuinely sparse): the standard,
  QSA-agnostic `forward_one`-vs-`forward` parity test passes the `1e-3` gate; `[layout]` 430,527
  assertions / 16 test cases green.
- **`N_LAYERS=4, GDN_FULL_ATTN_STRIDE=4` with QSA also on** — the real model's own layer arrangement in
  miniature (layers 0-2 GDN, layer 3 QSA): `[layout]` 440,102 assertions / 16 cases and the same parity
  test both green. This build exists specifically to exercise the OTHER dispatch branch, because "wired
  into only one of two call sites" is exactly how MoE's own Stage 1 SIGSEGV happened (`docs/MOE.md` §9);
  `forward_one`'s QSA path is deliberately reached through a single named `do_full_attn_mixer()` lambda
  that both the GDN-on and GDN-off dispatch paths call, rather than duplicated in each.

**Scope confirmed, not merely assumed**: the full, untagged `sub0_tests` suite at a QSA-ON build reaches
`backward_node`'s loud `abort()` ("fatal: Qwen Sparse Attention has no backward pass yet...") on the first
`Op::Qsa` node a training-path test tries to differentiate through — exactly Stage 1's declared scope
boundary (§6), confirmed by hitting it rather than only documenting it.

**CUDA guard**: `static_assert(!sub0::USE_QSA, ...)` added to `backend_cuda.cu`, mirroring the GR/MoE
guards. It is unconditionally satisfied at the default (QSA off) so it cannot regress any existing CUDA
build. The consumer sweep (AGENTS.md §10) found `backend_cuda.cu` does mirror `GDN_SCHEDULE.full_attn` in
its own `layer_slots()`/`build_qkv_weights()` param-layout arithmetic, which a QSA layer would break —
the guard is what makes that unreachable rather than silently wrong, and lifting it must update those
sites too.

**Divergence from the task brief, recorded explicitly**: the brief asked for "a QSA layer-schedule knob
analogous to `GDN_SCHEDULE`" as a new `RunConfig` constant. §2 concluded against a new *knob* and
delivered the *derived* three-way `MIXER_SCHEDULE` instead, because the real `config.json`'s `layer_types`
array was verified element-by-element to be exactly `gdn_schedule_for(4)` and no `full_attention` layer in
the real model lacks an indexer — so an independent QSA stride could only express layer sets the real
model cannot have (AGENTS.md §8). The testable substance the brief wanted is preserved: `qsa_schedule_for
<LAYERS>(gdn_stride, qsa_on)` is a free function of explicit parameters, asserted at an odd, non-dividing
layer count (11 layers at stride 4, where the ragged tail 8/9/10 contains no full-attention layer at all)
exactly as §9 required.
