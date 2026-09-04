# Gated Residual (hyper-connections) — design, derived from the reference

Status: **Stage 0 (config skeleton, hard-gated off) + Stage 1 (CPU forward only) — this pass.** Follows
`docs/GATED_DELTANET.md`'s own structure and staging discipline (real source quoted verbatim, engine-
interaction analysis, checkpoint design, two-scale identity checks) since that document is this project's
established template for threading a genuinely new, off-by-default mechanism through the engine. Per
AGENTS.md §5, every equation below is quoted or worked from the real, installed `transformers==5.16.1`
source fetched for this doc — not from recall, and not from `docs/QWEN4_PREVIEW_REFERENCE.md`'s own
facts-table paraphrase alone.

## 0. Sources, and their confidence

**Primary, highest-confidence source — the real model's own code**, fetched directly on this machine
(`transformers==5.16.1`, mainline PyPI, no `trust_remote_code`):
`transformers.models.qwen4_exp.modeling_qwen4_exp`, classes `Qwen4ExpTextGatedResidual` (the module
itself), `Qwen4ExpTextRMSNorm` (its grouped-norm primitive), and `Qwen4ExpTextDecoderLayer`/
`Qwen4ExpTextModel` (how the module is actually wired into a real forward pass — load-bearing, since the
module alone does not show how its two return-value shapes get consumed). Fetched via
`python3 -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; print(inspect.
getsource(m.Qwen4ExpTextGatedResidual))"` and the same for the other three classes — the identical
technique every prior stage in this thread used, confirmed still installed and importable on this machine.

**Real extracted weights and CPU-verified fixture, landed this pass at
`tests/fixtures/qwen4_preview/gated_residual_layer0_small_*`**: the real `model.language_model.layers.0.
attn_hyper_connection.*` tensors (shard `model-00001-of-00131.safetensors` — the SAME shard layer 0's GDN
weights already live in, since both belong to layer 0), extracted via HTTP Range requests (never
downloading the shard in full — at real scale this module's own weights are ~13.2MB contiguous, already
small enough that no further "download-then-slice-a-window" step was needed the way GDN's 110MB layer
did), sliced from the real `hidden_size=2560 / hc_count=4 / hc_lowrank=320` down to a commit-sized
`hidden_size=8` (with `hc_count` kept at its real value 4 — see §3a for why that axis is NOT sliced like
the others — and `hc_lowrank` sliced to 6), then run through the real, unmodified
`Qwen4ExpTextGatedResidual.forward()` (`use_combine=True`) on a reproducible-seed synthetic input. See
`gated_residual_layer0_small_manifest.json` for the exact column-slice convention and provenance.

**Secondary source**: `docs/QWEN4_PREVIEW_REFERENCE.md`'s own table (`hc_count=4`, `hc_lowrank=320`,
paper name "Gated Residual (GR)", tech_report.pdf §2.2 — NOT literally "mHC"), already independently
verified there against `config.json` + code + the paper. Cross-checked, not assumed: this pass's own
`config.json`-derived header fetch (§1a's real-config table below) reproduces the same two numbers from
the actual installed tensor shapes, not by trusting the table.

**Cross-reference, explicitly NOT the same design**: project memory `[[nanbeige-architecture-reference]]`
documents a verified `NanbeigeHyperConnectionModule` (Sinkhorn-Knopp-projected mHC) — useful for the
GENERAL hyper-connections shape (multiple parallel streams, per-layer read/write gates, a mixing
mechanism) but its actual formulas (a doubly-stochastic Sinkhorn-normalized mixing MATRIX) do not appear
anywhere in the real GR module quoted below, which uses a low-rank sigmoid gate and a plain mean instead.
Verified independently rather than assumed to match.

## 1. What the reference actually does

### 1a. The module itself — `Qwen4ExpTextGatedResidual`, quoted verbatim

```python
class Qwen4ExpTextGatedResidual(nn.Module):
    def __init__(self, config: Qwen4ExpTextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=self.hidden_size, eps=config.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden_size, config.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank, hc_hidden_size, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden_size, self.hc_count, bias=False) if use_combine else None

    def forward(
        self, hyper_input: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(...)
        hyper_input_normed = self.hc_norm(hyper_input)
        input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
        input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
        input_mix_weight = input_mix_weight.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed_input = (input_mix_weight * hyper_input_normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2
        )
        if self.block_inject_weight is None:
            return mixed_input
        injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
        return mixed_input, hyper_input, injection_weights
```

And the grouped-norm primitive it uses, `Qwen4ExpTextRMSNorm`, quoted verbatim (this is the SAME class
`Qwen4ExpTextGatedDeltaNet`/`Qwen4ExpTextAttention` use for their own q/k norms — a shared engine-wide
primitive, not something GR invented):

```python
class Qwen4ExpTextRMSNorm(nn.Module):
    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        self.group_size = group_size
        ...
    def _norm(self, x):
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.flatten(-2) if self.group_size is not None else out
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())   # note: (1 + weight), NOT weight directly
        return output.type_as(x)
```

Three facts this resolves, verified rather than assumed from the facts-table paraphrase:

- **`hc_norm` is GROUPED**: `group_size=hidden_size` on a `dim=hc_count*hidden_size` tensor means each of
  the `hc_count` streams is RMS-normalized independently (its own mean-square over only its own
  `hidden_size` slice), but the LEARNED WEIGHT is the full `hc_count*hidden_size` width — one gain value
  per (stream, channel) pair, not one shared across streams the way this project's own `QNorm`/`KNorm`
  (`[1,D_HEAD]`, shared across heads) are. This matters directly for §3b's PARAM_FLOATS arithmetic.
- **Zero-centered weight convention**: `weight = nn.Parameter(torch.zeros(dim))` and `output * (1.0 +
  weight)` — the gain is `1 + w`, so a freshly-initialized (`w=0`) `hc_norm` is the identity RMS-norm
  (gain exactly 1), not the "some learned gain, typically initialized to 1" convention this project's own
  `Ln1`/`Ln2`/`QNorm`/`KNorm` currently use (`ones(t)` in `init_weights()`, i.e. gain = w directly). GR's
  own `GrHcNorm` parameter therefore needs its OWN, different init (§4c) — this is a real, verified
  divergence from every existing norm this engine has, not a detail to gloss over.
- **`eps=config.rms_norm_eps`**, the real config's own value (`1e-6`, confirmed in the fixture manifest),
  not this engine's existing `op_rmsnorm`'s hardcoded `1e-5f`. Per `gdn_math.hpp`'s own precedent (RMS_EPS/
  L2_EPS hardcoded as the real model's verified values, not exposed as a knob), GR's own math uses its own
  `1e-6f` constant rather than reusing `op_rmsnorm`'s.

**No separate pre-block LayerNorm exists in the real model at all.** `Qwen4ExpTextAttention.__init__` and
`Qwen4ExpTextSparseMoeBlock.__init__` were checked directly (this pass) for a `self.input_layernorm`/
`self.post_attention_layernorm` and found to have neither — only per-head `q_norm`/`k_norm`. `hc_norm`
inside GatedResidual is therefore doing double duty in the real architecture: it is BOTH the "read" gate's
own normalization AND the mixer's entire pre-block norm (the wide-to-narrow `mixed_input` GatedResidual
hands the mixer IS the value the real model's attention/GDN reads directly, un-normalized again). This is
a real, load-bearing architectural fact and it is NOT reproduced by this pass's engine integration — see
§2's explicit, documented simplification for why.

### 1b. How the module is actually wired — `Qwen4ExpTextDecoderLayer.forward`, quoted verbatim

```python
def forward(self, hidden_states, position_embeddings, attention_mask=None, conv_mask=None,
            past_key_values=None, ple_input_ids=None, **kwargs):
    if self.ple is not None:
        hidden_states = hidden_states + self.ple(hidden_states, ple_input_ids, past_key_values, conv_mask=conv_mask)

    hidden_states, hyper_input, injection_weights = self.attn_hyper_connection(hidden_states)
    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(hidden_states, cache_params=past_key_values, attention_mask=conv_mask, **kwargs)
    else:
        hidden_states, _ = self.self_attn(hidden_states, position_embeddings, attention_mask=attention_mask,
                                           past_key_values=past_key_values, **kwargs)

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)

    hidden_states, hyper_input, injection_weights = self.mlp_hyper_connection(hidden_states)
    hidden_states = self.mlp(hidden_states)

    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    hidden_states = hyper_input + injection.flatten(-2)
    return hidden_states
```

Each decoder layer owns **two independent `Qwen4ExpTextGatedResidual` instances** — `attn_hyper_connection`
(wraps whichever mixer this layer uses: GDN or softmax attention, decided per layer by `layer_type`) and
`mlp_hyper_connection` (wraps the MoE block) — each with its OWN `hc_norm`/`input_mix_weight_down`/
`input_mix_weight_up`/`block_inject_weight` (four independent learned tensors, not shared).

The READ/WRITE split, worked through explicitly:
- **READ** (`GatedResidual.forward`, `use_combine=True`): normalizes the wide `[hc_count*hidden_size]`
  stream per-group, computes a per-(stream,channel) sigmoid gate via a low-rank bottleneck
  (`down` → `/hc_count` → `silu` → `up` → `sigmoid`), and MEAN-pools (not sum) the gated, normalized
  streams down to one `[hidden_size]` vector — this is what the mixer (attention/GDN/MoE) actually reads
  as its input, replacing what a plain single-stream architecture would feed it directly. Also computes,
  from the SAME normalized wide input, a small `[hc_count]` per-stream **injection weight** vector via a
  second, un-related linear + `2·sigmoid(·/hc_count)` (range `(0, 2)`, not `(0, 1)` — the `2×` factor lets
  a stream receive UP TO double the raw mixer output, not merely attenuate it).
- **WRITE** (done by the decoder layer itself, not inside the module): the mixer's OWN output (still
  `[hidden_size]`, one shared value across all `hc_count` streams) is broadcast to every stream, scaled
  per-stream by that stream's own injection weight, and added onto `hyper_input` — the ORIGINAL, un-normed
  wide stream from before this GatedResidual instance ran (`self.attn_hyper_connection` returns it
  unchanged as its second value specifically so the write step has it) — producing the next wide stream.

### 1c. Model-level entry/exit — `Qwen4ExpTextModel.__init__`/`.forward`, quoted verbatim (relevant lines)

```python
# __init__:
self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(config, use_combine=False)

# forward:
hidden_states = inputs_embeds
...
hidden_states = hidden_states.repeat(1, 1, self.config.hc_count)     # ENTRY: tile, not zero-pad
for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    hidden_states = decoder_layer(hidden_states, ...)
hidden_states = self.hyper_connection_mixer(hidden_states)            # EXIT: collapse, no gate/combine
return Qwen4ExpModelOutputWithPast(last_hidden_state=hidden_states, ...)
```

**Entry**: the token embedding (ordinary `[hidden_size]`-wide) is TILED `hc_count` times (`torch.repeat`,
literally duplicated, not zero-padded or independently initialized) to seed all `hc_count` streams
identically at layer 0. **Exit**: after every decoder layer, ONE MORE `GatedResidual` instance
(`use_combine=False`, so `block_inject_weight` is `None` and `forward()` returns just `mixed_input`, per
§1a's `if self.block_inject_weight is None: return mixed_input` branch) collapses the wide stream back
down to `[hidden_size]` before the LM head.

Total real instances per model: `2 * num_hidden_layers + 1` (two per layer, one at the very end) — each an
independent set of `hc_norm`/`down`/`up`[/`block_inject`] tensors.

> **CORRECTION (2026-09-04) — there is NO final norm after the exit collapse, and this section used to
> imply otherwise.** The paragraph above originally read "...back down to `[hidden_size]` before the
> final norm/LM head (not shown above, but structurally the next step in any decoder stack)", and the
> §2 note below stated outright that `LnF`'s "real counterpart `Qwen4ExpTextModel.norm` is also `1e-6`".
> **That claim was never checked against the real model and is wrong.** What it actually was: an
> inference from "structurally the next step in any decoder stack" — i.e. from what other decoder stacks
> do — presented as if it were a fact read out of this one. Exactly the failure mode `AGENTS.md` §5
> exists to prevent, and the same shape as the `Ln1`/`Ln2` assumption blocker D had to undo.
>
> Checked now, against the real checkpoint from two independent directions, both of which agree:
>
> - **GGUF** (`UD-IQ1_S`, WP4c's own full tensor census): there is no `output_norm.weight` under that or
>   any other name. The ONLY non-`blk.` tensors in the whole 1,224-tensor set are `output.weight`,
>   `output_hc_{down,norm,up}.weight`, `per_layer_token_embd.weight` and `token_embd.weight`.
> - **safetensors** (`model.safetensors.index.json`, fetched directly from
>   `huggingface.co/Qwen/Qwen3.8-Flash-Next`, 170,726 bytes, searched for every "norm"-bearing tensor
>   name that is not per-layer): the only model-level language-model norm is
>   `model.language_model.hyper_connection_mixer.hc_norm.weight`.
>
> That last name is the answer, not a near-miss: it is **this section's own exit instance's `hc_norm`** —
> the `use_combine=False` `Qwen4ExpTextGatedResidual` built in `Qwen4ExpTextModel.__init__`, already a
> real `PARAM_LAYOUT` entry here (`GrHcNorm`) and already correctly transplanted by WP4c. So **the GR
> exit's own grouped `hc_norm` (at the real `rms_norm_eps = 1e-6`) IS this model's final normalization**,
> applied to the wide stream on the way into the collapse, and the collapsed `mixed_input` feeds
> `lm_head` un-normed again. A separate `LnF` tensor/op does not exist in the real model at all.
>
> `make_param_layout()` therefore emits no `LnF` under `USE_GATED_RESIDUAL`, and `Model::forward` /
> `forward_one` feed `lm_head` straight from the exit instance's `mixed_input` — see the §2 note below
> for the measured cost of having got this wrong.

## 2. How 4 parallel residual streams fit this engine's current single-residual `Node` graph

Traced from the real current code (`src/backend_cpu.cpp`'s `Model::forward`), NOT assumed: today, `h`
(the residual) is a single `Node*` of shape `[T, D_MODEL]`, threaded through the per-layer loop exactly
once per layer as
```cpp
Node* a = op_rmsnorm(h, L.ln1);
... mixer on a, producing mixer_out (D_MODEL-wide) ...
h = op_add(h, mixer_out);
Node* f = op_rmsnorm(h, L.ln2);
... FFN on f ...
h = op_add(h, f);
```
identically whether the mixer is softmax attention or GDN (GDN's Stage 1 landed exactly this shape:
`h = op_add(h, op_gdn(a, L))`, no structural difference from the attention branch's `h = op_add(h,
op_linear(att, L.Wo, ...))`). This IS the biggest structural question this task raises, per its own
framing: GDN never touched how residuals compose (it stayed a drop-in mixer inside an unchanged single
stream); GR replaces the very shape of `h` itself, for EVERY layer regardless of mixer type.

**Decision: GR is a WRAPPING transform around `h`'s entry/exit at each sub-block, not a restructuring of
the sub-block's own internals.** Concretely, `h` becomes `[T, HC_COUNT*D_MODEL]` when GR is on (still a
single `Node*`, just wider), and the existing `ln1`/mixer/`ln2`/FFN code is **completely untouched** —
it still operates on an ordinary `[T,D_MODEL]` activation, produced by a new "read" step and consumed by a
new "write" step that sit strictly OUTSIDE the part of the pipeline GDN/attention/FFN already own:

```
h  (wide when ON, [T, HC_COUNT*D_MODEL]; == today's [T,D_MODEL] when OFF)
mixed, inj = GR_READ(h, <this sub-block's own hc_norm/down/up[/block_inject]>)   -- NEW, wraps entry
attn_in = op_rmsnorm(mixed, L.ln1)                                                -- UNCHANGED
mixer_out = <existing attention/GDN code, unchanged, reading attn_in>             -- UNCHANGED
h = GR_WRITE(h, mixer_out, inj)                                                   -- NEW, wraps exit
                                                                                   -- (replaces op_add(h, mixer_out))
... identical shape again for ln2/FFN ...
```

This is a **strictly weaker** structural change than GDN's own op_gdn integration needed for the mixer
itself — GR never needs to know whether the sub-block it wraps is attention, GDN, or FFN; it only ever
sees that sub-block's `D_MODEL`-wide input and output. `if constexpr (USE_GATED_RESIDUAL)` gates the
wrapping calls; when off, the code that runs is BYTE-IDENTICAL to today's `op_add(h, mixer_out)` path (not
merely equivalent — the exact same function calls, no new code compiled in at all), satisfying AGENTS.md
§4's "zero effect on existing builds" requirement by construction rather than by a runtime branch that
happens to be false.

> **RESOLVED in WP4b (blocker D, `docs/WP4_SCOPE.md` §2)** — the simplification described in the next
> paragraph no longer applies. `make_param_layout()` now emits **no `Ln1`/`Ln2` at all** when
> `USE_GATED_RESIDUAL`, and `Model::forward`'s `gr_read` / `forward_one`'s `gr_read_row` return GR's
> `mixed_input` **directly**, un-normed — matching the real `Qwen4ExpTextDecoderLayer`. The pipeline
> sketch above therefore reads `attn_in = mixed` under GR, not `op_rmsnorm(mixed, ln1)`. Two consequences
> worth naming: `gr_param_delta()` became an addition **minus** `2*d_model` per layer (still strictly
> positive at every legal setting, so `PARAM_FLOATS` still discriminates GR-on from GR-off and no
> fingerprint bit is needed), and the wrong-eps path this simplification carried is gone — GR's own
> `hc_norm` uses the real `rms_norm_eps = 1e-6`, whereas the `op_rmsnorm` it used to route through
> hardcodes `1e-5`.
>
> ~~**Still open, deliberately out of blocker D's scope**: `op_rmsnorm`'s `1e-5` remains for `LnF` (the
> model-level final norm, whose real counterpart `Qwen4ExpTextModel.norm` is also `1e-6`)...~~
>
> **RESOLVED 2026-09-04, and the parenthetical above was factually wrong — see §1c's correction.**
> There is no `Qwen4ExpTextModel.norm` in the real checkpoint, at `1e-6` or at any eps: the model applies
> no separate RMSNorm after the GR exit collapse at all. So `LnF` is now removed under
> `USE_GATED_RESIDUAL` the same way `Ln1`/`Ln2` were, for the same reason, in the same file — and the
> eps question at that site evaporates with it rather than being answered.
>
> **This was a correctness bug, not a tidiness one, and it is measured.** WP4c synthesized the missing
> tensor to `1.0`, the RMSNorm identity GAIN — but an identity gain is not an identity OPERATION:
> RMSNorm still divides by the row's own RMS. WP4d then measured, on the real weights, that the engine
> was therefore normalizing a second time where the real model normalizes once, moving the final logits
> by **0.174 against a logit rms of 0.605 — ~28.8% of scale** (and by 4.90 at the first position, whose
> pre-collapse RMS is furthest from 1). Removing the site removes exactly that perturbation; the run
> after the fix reproduces the same 0.174 as the size of what was removed.
>
> `gr_param_delta()` gains a further `- d_model`, once, at the model level (on top of blocker D's
> `- 2*d_model` per layer). Still strictly positive at every legal setting, so `PARAM_FLOATS` alone
> still discriminates GR-on from GR-off and no fingerprint bit is needed.
>
> **Genuinely still open, and narrowed by this**: `op_rmsnorm`/`op_qknorm`'s `1e-5` vs the real `1e-6`.
> `LnF` was the LAST site a GR build could reach either op through (WP4d verified that by reading the
> real-axes forward path), so under `USE_GATED_RESIDUAL` the discrepancy is now unreachable. It remains
> a real, unaddressed difference for any non-GR build, where those are shared always-present ops and
> changing their eps is its own scoped change with its own neutral-identity gate.

**A deliberate, documented simplification vs. the real model, made for this stage's own scope reasons**:
per §1a's finding that the real model has NO separate `ln1`/`ln2` (GR's own `hc_norm` replaces them
entirely), a maximally faithful engine port would remove this project's existing `Ln1`/`Ln2` PKinds when
GR is on and feed the mixer directly from GR's `mixed_input`. This pass does NOT do that: `Ln1`/`Ln2`
stay exactly where they are today (applied to the now-mixed, single-stream activation), for three reasons.
(1) `Ln1`/`Ln2` are shared infrastructure serving GDN and softmax attention uniformly today; removing them
is a separate, independently-scoped, shape-changing refactor unrelated to proving GR's OWN math is
correctly ported. (2) This pass's correctness gate (§5) is a fixture test of the ISOLATED
`Qwen4ExpTextGatedResidual` module, not a full-decoder-layer numerical parity test against the real model
— nothing in this stage's scope depends on removing them. (3) Per AGENTS.md §8, don't build the surface a
later stage needs before that stage exists: if a future pass targets full real-weight decoder-layer parity
(the stated end goal of this whole staged plan), removing `Ln1`/`Ln2` under GR becomes that pass's own,
explicitly-scoped shape-changing checkpoint change — not bundled unannounced into this one. This is
recorded here explicitly (not silently) so it is not mistaken for an oversight.

**`Model::forward`'s LOOP_EXEC_COUNT/LAYER_EXEC_ORDER (LoopSplit) interaction**: unaffected structurally —
GR wraps each EXECUTION's own sub-blocks exactly like `Ln1`/`Ln2` already do (a repeated LoopSplit middle
layer re-runs the SAME `Layer`'s GR tensors on each pass, exactly as it already re-runs the SAME `Ln1`).
No new per-execution state is needed (unlike GDN's recurrent state or the depth-attention cache) — GR has
no cross-position or cross-execution memory at all, only a per-call wide/narrow reshape.

## 3. Checkpoint / `PARAM_LAYOUT` impact

### 3a. Which axes are genuinely new, and which of the real model's axes are NOT sliced independently

| Reference quantity | This project's mapping | Reused / new |
|---|---|---|
| `hidden_size` | `D_MODEL` | existing |
| `hc_count` | `HC_COUNT` (new CLI flag) | new |
| `hc_lowrank` | `HC_LOWRANK` (new CLI flag) | new |
| `hc_norm`/`down`/`up`/`block_inject` weight shapes | derived from `D_MODEL`, `HC_COUNT`, `HC_LOWRANK` | new (§3b) |

Unlike GDN's `head_k_dim`/`head_v_dim` (fixed real values this project reuses without exposing as a CLI
knob, since Stage 1 had no op to tune them against), the task's own instructions call for BOTH `hc_count`
and `hc_lowrank` as real, independent CLI flags mirroring `--gdn-full-attn-stride`/`--ngram-max-n`'s
pattern — because unlike `GDN_CONV_KERNEL` (an architectural constant this project's engine has no reason
to ever vary once GDN exists), `hc_count`/`hc_lowrank` are the two knobs an eventual real Qwen4 config
would need to reproduce exactly (`4` and `320`), so exposing them now, off by default, costs nothing and
avoids a second flag-plumbing pass later.

**`HC_COUNT` is NOT reduced from its real value in the fixture** (kept at 4), while `HC_LOWRANK` IS sliced
(320→6) alongside `D_MODEL` (2560→8) — the same distinction GDN's own fixture drew between `head_k_dim`/
`head_v_dim` (fixed, unsliced — a per-head shape with real architectural meaning) and `hidden_size`/
`key_dim`/`value_dim` (freely sliced — ordinary projection widths). `HC_COUNT` is the one axis this whole
mechanism exists to test (§5's presence check depends on all 4 streams genuinely differing), so shrinking
it below its real value would test a qualitatively different regime (e.g. `hc_count=2` degenerates the
mean-pool and the injection-weight vector to trivial cases a bug could hide behind); `HC_LOWRANK` is an
ordinary bottleneck-rank hyperparameter with no such structural floor, so it is sliced like any other
Linear-projection width (the same way GDN sliced `key_dim`/`value_dim`, which are also ordinary
projections, unlike `head_k_dim`/`head_v_dim`).

### 3b. Per-instance and per-layer `PARAM_FLOATS` delta, worked from the real shapes

One `GatedResidual` instance's parameters (`WIDE = HC_COUNT * D_MODEL`), using this project's own
`[rows=in, cols=out]` weight-layout convention throughout (re-derived, not assumed, from the real
`nn.Linear` `[out_features, in_features]` convention — same axis-order flip `gdn_math.hpp`'s header
comment already documents for GDN):
```
  hc_norm            = WIDE                                  (a [1, WIDE] gain, NOT [1, D_MODEL] --
                                                                per §1a, one gain per (stream, channel))
  input_mix_weight_down = WIDE * HC_LOWRANK                   (in=WIDE, out=HC_LOWRANK)
  input_mix_weight_up   = HC_LOWRANK * WIDE                   (in=HC_LOWRANK, out=WIDE)
  block_inject_weight   = WIDE * HC_COUNT                     (in=WIDE, out=HC_COUNT; ABSENT when
                                                                use_combine=False, i.e. the top-level
                                                                model instance)
```
Per layer: **two** full instances (`attn_hyper_connection`, `mlp_hyper_connection`), each with
`block_inject_weight`. Once, model-level: **one** instance WITHOUT `block_inject_weight`
(`hyper_connection_mixer`).

```
Δ_per_layer = 2 * (WIDE + 2*WIDE*HC_LOWRANK + WIDE*HC_COUNT)
Δ_top       =      WIDE + 2*WIDE*HC_LOWRANK
Δ_total     = N_LAYERS * Δ_per_layer + Δ_top
```

At this pass's own fixture-adjacent test shape (`D_MODEL=8, HC_COUNT=4, HC_LOWRANK=6` ⇒ `WIDE=32`):
`Δ_per_layer = 2*(32 + 2*32*6 + 32*4) = 2*(32+384+128) = 1088`, `Δ_top = 32 + 384 = 416` — small, real,
and strictly positive for any `HC_COUNT >= 2, HC_LOWRANK >= 1, D_MODEL >= 1`: every term in `Δ_per_layer`/
`Δ_top` is a PRODUCT of strictly-positive quantities, so the sum can never be zero or negative. This is
the strongest possible form of "shape-changing" — not merely "differs in most configurations" (GDN's own
`Δ`, a signed sum of several terms with no such guarantee — hence GDN's own belt-and-braces
`ARCH_FINGERPRINT2`) but a pure, unconditional ADDITION of brand-new tensors on top of an entirely
UNCHANGED existing set (recall §2: `Ln1`/`Ln2`/`Wq`/`Wk`/`Wv`/`Wo`/GDN's own nine tensors/the FFN weights
are ALL untouched by this design — GR adds tensors, it never resizes or removes one).

### 3c. Classification: shape-changing, `PARAM_FLOATS` already discriminates it — no fingerprint bits needed

Per AGENTS.md §3 rule 1 / layout.hpp's own `ARCH_FINGERPRINT` classification rule #1 ("changes a tensor
shape → `PARAM_FLOATS` already discriminates it, nothing to do"): §3b's `Δ_total` is monotonically
increasing in `HC_COUNT`, `HC_LOWRANK`, and `N_LAYERS` (every term positive, no cancellation possible with
any OTHER axis, unlike GDN's mixed-sign `Δ`), and exactly `0` when `HC_COUNT == 0` (§4a's neutral
setting). This is the exact shape of the n-gram-embeddings precedent (`NGRAM_EMBED`'s own classification
in `layout.hpp`), not GDN's — a genuinely new tensor is added, so `Header`/`PARAM_FLOATS`'s existing
byte-count comparison at load time already refuses a mismatched checkpoint; no new `ARCH_FINGERPRINT`-
family word is needed. `MODEL_ARCH_ID` still mixes in `HC_COUNT`/`HC_LOWRANK` unconditionally regardless
of this (per that function's own stated "covers every axis, shape-changing and computation-changing
alike" design intent — the same treatment `NGRAM_MAX_N`/`NGRAM_TABLES_PER_ORDER`/`NGRAM_TABLE_SIZE`
already get there).

### 3d. On-disk plumbing

Purely additive at every point PARAM_LAYOUT already handles additively: new `PKind`s appended to the
existing enum (order among enumerators is irrelevant — `PARAM_LAYOUT`'s APPEND order is what matters, per
`layout.hpp`'s own header comment), new tensors appended at the natural position in `make_param_layout()`'s
existing per-layer loop (right before each sub-block's own weights, so a GR-off build's byte layout for
every EXISTING tensor is completely unchanged — GR only ever ADDS entries, never reorders one). No new
`Header`/`.ckpt` trailer record and no `CKPT_VERSION` bump are needed (unlike GDN's `ARCH_FINGERPRINT2`
trailer) — `PARAM_FLOATS` alone is the existing, sufficient gate, and it already fails loudly on a byte-
count mismatch (`engine_core.cpp`'s existing `Header` comparison).

## 4. Compile-time specialization

### 4a. New `RunConfig`/generated-header constants, mirroring `GDN_FULL_ATTN_STRIDE`/`NGRAM_MAX_N` exactly

- `--hc-count` (CLI int, default `0`) → generated `constexpr int HC_COUNT`. `0` = off. Per §3a, `1` is
  disallowed (a single "hyper-connection" stream is not the mechanism this exists to test, and its own
  math is not obviously well-defined for `hc_count=1` under the mean-pool — the real config never sets
  it below `2` either): `static_assert(HC_COUNT == 0 || HC_COUNT >= 2, ...)`, mirroring `NGRAM_MAX_N`'s
  own "`0` or `>= 2`" pattern exactly (both a CLI-time diagnostic in `configurator.cpp` and the
  compile-time `static_assert` in `layout.hpp`).
- `--hc-lowrank` (CLI int, default `0`) → generated `constexpr int HC_LOWRANK`. Must be `0` exactly when
  `HC_COUNT == 0`, and `>= 1` exactly when `HC_COUNT >= 2` — `static_assert((HC_COUNT >= 2) ==
  (HC_LOWRANK >= 1), ...)`, so the two flags cannot be set inconsistently (one on, the other off) at
  either the CLI or the compile-time layer.
- `inline constexpr bool USE_GATED_RESIDUAL = (HC_COUNT >= 2);` — the single gate every `if constexpr`
  in `backend_cpu.cpp` keys off, mirroring `USE_GATED_DELTANET`/`NGRAM_EMBED`'s own naming and shape.
- `inline constexpr int HC_WIDE = USE_GATED_RESIDUAL ? HC_COUNT * D_MODEL : D_MODEL;` — the width `h`
  actually has. At the neutral setting this collapses to exactly `D_MODEL`, so every existing
  `[T, D_MODEL]`-shaped expression in `Model::forward`/`forward_one` that is NOT explicitly wrapped in a
  GR branch continues to typecheck and compute identically (this is what makes "leave the untouched code
  literally untouched" in §2 possible, rather than merely "equivalent").
- `sub0::gr::Dims GR_DIMS{D_MODEL, HC_COUNT, HC_LOWRANK}` (§4b) — always a valid, never-divide-by-zero
  shape even when `USE_GATED_RESIDUAL` is false (same "describes a shape nothing builds" idiom
  `GDN_DIMS`/`DEPTH_CACHE_MAX` already use), so downstream code never needs a SEPARATE guard just to
  construct it.

**Stage 0 (this pass's own first commit) hard-clamps both flags to their neutral value**, per the task's
explicit instruction to mirror GDN's own Stage 0 exactly: `--hc-count`/`--hc-lowrank` get
`->check(CLI::Range(0, 0))` in `configurator.cpp` AND `layout.hpp` keeps `static_assert(HC_COUNT == 0, ...)`
/`static_assert(HC_LOWRANK == 0, ...)` (the strict, single-value form, not yet the `0 || >= 2` form) until
Stage 1 (this pass's own SECOND commit) relaxes both — the same two-refusal "guard at the lowest callable
seam" pattern `GDN_FULL_ATTN_STRIDE` used, so a config skeleton commit that types-checks and compiles
genuinely cannot express anything but the current (GR-off) architecture, independent of whether Stage 1's
op exists yet.

### 4b. `include/sub0/gated_residual_math.hpp` — the engine-free math core, mirroring `gdn_math.hpp`'s role

A `sub0::gr::Dims{hidden_size, hc_count, hc_lowrank}` struct (own `wide()`/`down_scratch_floats()`/
`normed_scratch_floats()` helpers, the same explicit-parameters style `gdn_math.hpp`'s own `Dims` uses so
a standalone test can exercise the real fixture's own non-default shape regardless of what this project's
compiled build happens to be configured for) and four free functions, ALL taking this project's own
`[rows=in, cols=out]` weight convention and explicit caller-owned scratch (AGENTS.md §1 — no heap
allocation):
- `hc_norm(d, T, wide_in, norm_w, out_normed)` — the grouped, zero-centered-weight RMSNorm (§1a),
  factored out on its own (unlike GDN, where norm was folled into one monolithic `forward()`) specifically
  so `mix()`/`gate()` below can each call it independently without a shared side buffer, at the cost of
  recomputing it twice per sub-block wrap (§4c explains why that duplication is the right trade here).
- `mix(d, T, normed, down_w, up_w, out_mixed, scratch)` — `silu(down(normed)/hc_count)` → `sigmoid(up(·))`
  → reshape to `[hc_count, hidden_size]` → elementwise-multiply by `normed`'s own same reshape → MEAN over
  the `hc_count` axis. `scratch` needs `T*hc_lowrank` floats (the down-projection's pre-activation).
- `gate(d, T, normed, block_inject_w, out_inj)` — `2*sigmoid(block_inject(normed)/hc_count)`, `[T,
  hc_count]`. No scratch needed (a single linear + elementwise activation).
- `combine(d, T, wide_in, mixer_out, inj, out_wide)` — `out_wide[t, s*hidden+j] = wide_in[t,s*hidden+j] +
  inj[t,s] * mixer_out[t,j]`, the WRITE step (§1b/§2). Pure elementwise/broadcast, no scratch.
- `tile(d, T, h, out_wide)` — `out_wide[t, s*hidden+j] = h[t,j]` for every stream `s` (§1c's entry). No
  scratch.

### 4c. Why `mix`/`gate` each independently call `hc_norm` rather than sharing one precomputed buffer

The real model computes `hyper_input_normed` ONCE per `GatedResidual.forward()` call and reuses it for
both `mixed_input` and `injection_weights`. This design recomputes it independently inside `mix()` and
`gate()` (i.e. the ENGINE calls `hc_norm` twice per sub-block wrap — once inside the op that produces
`mixed_input`, once inside the op that produces `inj`) rather than threading a shared normalized buffer
between two ops. This is a deliberate simplification, not an oversight: `hc_norm` costs `O(T*WIDE)`, while
`mix`'s own down/up matmuls cost `O(T*WIDE*HC_LOWRANK)` and `gate`'s own matmul costs `O(T*WIDE*HC_COUNT)`
— at any `HC_LOWRANK` or `HC_COUNT` `>= 1` the duplicated norm is a strictly smaller term than either
matmul it sits next to, so the redundant compute is bounded and cheap, and avoiding a second, order-
sensitive cross-op scratch dependency (which would need its own lifetime contract, the same kind of
bookkeeping GDN's `GdnLinkCache` side table exists to manage) keeps this Stage 1 pass's op graph the
simplest correct thing rather than the most compute-efficient one — a legitimate Stage 1 trade per this
project's own "get the simple form working and correct first" precedent (`docs/GATED_DELTANET.md` §5).

## 5. The `Node`-fanout question — resolved WITHOUT any side table, unlike GDN's own backward

Per the real current code (`op_gdn`, `backend_cpu.cpp`), Stage 1 GDN's forward did NOT need a side table
either — `op_gdn(Node* a, Layer& L)` takes `Layer&` directly and reads its nine weight tensors' `.data()`
straight off it, with no side table at all (`GdnLinkCache` was added later, in GDN's own Stage 2, purely
to let `backward_node` reach those nine tensors from a bare `Node*` it walks the pool with — a BACKWARD-
only need). Since this pass is explicitly forward-only (§7), GR's new ops follow the exact same pattern:
they take `Layer&` (extended with GR's own weight fields) directly, not a generic `a`/`b`/`w`/`bias` Node
fanout, and store only the ONE input activation Node on `out->a` (mirroring `op_gdn`'s own
`out->a = a`) — enough for a future backward's abort-placeholder case (§6) to at least name the right
input node, exactly as `op_gdn`'s Stage 1 form did before Stage 2 needed more.

Four new `Op` enumerators, each a small, independently-testable function:
- `Op::GrTile` — `op_gr_tile(Node* h)`: `out->a = h`. Output `[T, HC_WIDE]`.
- `Op::GrMix` — `op_gr_mix(Node* wide, Layer& L, bool is_attn)`: `out->a = wide`. Output `[T, D_MODEL]`.
  Reads `L.gr_attn_hc_norm`/`L.gr_attn_mix_down`/`L.gr_attn_mix_up` (or the `gr_mlp_*` triple) directly.
- `Op::GrGate` — `op_gr_gate(Node* wide, Layer& L, bool is_attn)`: `out->a = wide`. Output
  `[T, HC_COUNT]`.
- `Op::GrCombine` — `op_gr_combine(Node* wide, Node* mixer_out, Node* inj)`: `out->a = wide; out->b =
  mixer_out; out->w = inj;` — this one genuinely IS a plain 3-activation combine with no weight tensors at
  all, so it fits Node's native fanout exactly, no `Layer&` needed.
The MODEL-level top instance (§1c's exit collapse) reuses `op_gr_mix` with `L` replaced by the `Model`'s
own three top-level weight fields and no `Op::GrGate`/`Op::GrCombine` call at all (mirroring
`use_combine=False`'s own real-code branch precisely — `block_inject_weight` genuinely does not exist for
this instance, so no engine call tries to read it).

This is a *simpler* fanout story than GDN's own Stage 1 needed for op_gdn's single call (nine tensors, one
call) only in the sense that GR splits into four SMALLER calls rather than one large one; the reason
neither needs a side table for their OWN forward is the same reason in both cases — Stage 1 has no
backward walking the node pool yet, so nothing needs to recover weight tensors from a bare `Node*`.

## 6. Backward — explicitly out of scope, loud abort per the `Op::GDN` Stage 1 precedent

Per the task's explicit scope boundary and the exact `Op::GDN` Stage-1 precedent (`backend_cpu.cpp`,
commit `bac8bfd`, before GDN's own Stage 2 replaced it): `backward_node`'s `switch` gains four new `case`
labels (`Op::GrTile`, `Op::GrMix`, `Op::GrGate`, `Op::GrCombine`), each `std::println(stderr, "fatal: ...")`
+ `std::abort()`, explaining that Gated Residual has no backward pass yet and refusing to silently train a
different architecture than the one requested — the same "guard at the lowest callable seam" reasoning
GDN's own placeholder used, since a CPU binary with `USE_GATED_RESIDUAL` on compiles and runs fine for
gen/eval/report (this pass's own scope) but must never reach `backward()` on a graph containing one of
these nodes undetected.

## 7. Correctness gate

`tests/gated_residual_qwen4_fixture_tests.cpp` (engine-free, mirroring `gdn_qwen4_fixture_tests.cpp`'s own
pattern exactly): reads `tests/fixtures/qwen4_preview/gated_residual_layer0_small_*`, transposes the raw
PyTorch `[out,in]` weight files into this project's `[in,out]` convention, and calls `sub0::gr::hc_norm`/
`mix`/`gate` directly (this fixture's own shape — `D_MODEL=8, HC_COUNT=4, HC_LOWRANK=6` — happens to be a
perfectly ordinary shape this project's own compiled `Model` COULD in principle be built at, unlike GDN's
fixture; engine-free is still the right test boundary here, matching this whole thread's established
precedent for a math-core correctness gate rather than a full-`Model` one, and keeping this test
independent of whatever `HC_COUNT`/`HC_LOWRANK` the currently-compiled binary happens to have). Compares
against the real `output_mixed`/`output_injection_weights` fixture files (exact numbers in the final
report, not reproduced here since they are Stage 1 execution output, not a design decision).

**Presence/mutation check, per the depth-attention lesson this whole thread keeps citing**: a subtly wrong
implementation that silently ignored 3 of the 4 streams (e.g. read only stream 0, or summed instead of
meant) could still pass a loose fixture-tolerance check if the real weights happen to weight one stream
heavily. The test therefore ALSO constructs a synthetic `HC_COUNT=4` input where each stream is a
distinguishable, orthogonal signal (stream `s` nonzero only in its own channel range) and asserts that
`mixed_input` genuinely reflects a nontrivial contribution from every stream (each stream's own
contribution, isolated by zeroing every OTHER stream and re-running, changes `mixed_input` by a nonzero,
independently-verified amount) — a mutant that reads only stream 0 fails this even on an input where the
real fixture-tolerance check might not catch it.

## 8. Two-scale identity + real-build verification

Per AGENTS.md §7 (never trust a single compiled shape) and this thread's own repeated lesson (LoopSplit's
odd-layer-count `static_assert` bug was invisible at one scale and instant at another): `layout_tests.cpp`
pins `HC_COUNT`/`HC_LOWRANK`'s neutral-setting identity at TWO shapes — reusing GDN's own precedent pair,
8 layers (even) and 11 layers (odd/ragged) — asserting `PARAM_FLOATS`/`NUM_PARAMS` are UNCHANGED from a
GR-unaware calculation at both. A genuinely GR-ON small build (this pass's own commit-sized config, not a
production scale — explicitly out of scope per the task's own "do NOT attempt a real 2560-wide/48-layer
scale" instruction) is built once to confirm `forward()`-vs-`forward_one()` parity holds with the
mechanism actually active, not merely compiling. Exact assertion counts, hashes, and the parity numbers
are in the final report (Stage 1 execution output), not fixed in this design doc ahead of running them.

## 9. Results (Stage 1 execution)

**Fixture correctness gate** (`tests/gated_residual_qwen4_fixture_tests.cpp`, real weights, `T=6` real
fixture input): EXACT bit-for-bit match against the real `transformers==5.16.1` output --
`max|out-expected| = 0.0` for both `mixed_input` (`sum|expected| = 10.3492` over 48 values) and
`injection_weights` (`sum|expected| = 24.015` over 24 values). Stronger than GDN's own Stage 1 fixture
result (which matched to `~4e-11`, a float32-rounding-level match against a chunked-vs-sequential
algorithmic reformulation) -- GR's CPU port is a direct, unreformulated translation of the real forward
math, so an exact match is the correct outcome, not merely a good one.

**Presence/mutation checks** (same file, no fixture needed): (1) zeroing any ONE of the real fixture's
`hc_count=4` streams in isolation (a synthetic orthogonal-signal input) changes both `mixed_input` and
`injection_weights` by a real, nonzero amount for every one of the 4 streams -- a mutant reading only a
subset would fail this even on inputs where a loose fixture-tolerance check might not catch it; (2) a
degenerate identical-stream construction distinguishes the real MEAN reduction from a plausible SUM
mutant by the resulting per-channel ratio's numeric range (mean landed at `0.504`, inside `(0,1)` as
predicted; a sum mutant would land at `~2.0`, outside it, since `hc_count=4`). All 3 test cases: 73
assertions, green.

**Two-scale identity check** (AGENTS.md S7): at `HC_COUNT == 0` (neutral), the full default engine test
suite (`sub0_tests`) is assertion- AND hash-identical to `main` at BOTH shapes:

| shape | assertions | test cases | forward hash | grad hash | decode hash |
|---|---|---|---|---|---|
| d96 L8 H2 seq128 (even) | 4,504,583 | 139 | `f9c33312fa7cc1c3` | `088c3d9b8da37b62` | `9092eda27c35fea1` |
| d132 L11 H4 kv2 seq96 (odd/ragged) | 11,810,675 | 139 | `544a0a50fa57782e` | `11a6c926c3840ae4` | `a3e4ad39c3088013` |

identical before Stage 0's own two new test cases (137 cases, 4,504,571 / 11,810,663 assertions
respectively -- exactly 12 fewer, matching the 2 new test cases' own 12 assertions and nothing else)
through every subsequent commit in this pass, verified by rebuilding each side at the same generated
config header and diffing `sub0_tests`' own summary line.

**Real GR-ON build, forward/forward_one parity** (this doc's own scope: a small correctness-fixture-scale
config, not production dims): `D_MODEL=16, N_LAYERS=2, N_HEADS=2, SEQ_LEN=32, HC_COUNT=4, HC_LOWRANK=6` --
a genuinely GR-active build, PARAM_LAYOUT carrying the real 4-tensor-per-instance/3-instances-per-layer/
+1-top-level shape. `engine_tests.cpp`'s existing, GR-agnostic "`forward_one` (KV-cache) matches the full
forward per position" test (no GR-specific code needed -- it exercises whatever the compiled `Model`
actually computes) passes with worst per-position relative diff `1.93151e-07` -- float32-rounding-level
agreement between the Node-graph training path (`op_gr_tile`/`op_gr_mix`/`op_gr_gate`/`op_gr_combine`)
and the raw-pointer decode path (`gr::tile`/`gr::hc_norm`/`gr::mix`/`gr::gate`/`gr::combine` called
directly), confirming both independently-written implementations of the same math agree on a real
forward pass, not just compile. The `[layout][config][gr]`-tagged subset (13 test cases, 23,779
assertions) also passes at this build, including a real fix this pass found by actually compiling a
nonzero-`HC_COUNT` build rather than only reasoning about it: two pre-existing generic
`layout_tests.cpp` checks (the `NUM_PARAMS` closed-form count and the per-`PKind` decay/ternary
consistency check) had no GR term at all and failed a REAL assertion at `HC_COUNT=4`, not a hypothetical
one -- fixed the same pass, per AGENTS.md S10's own lesson about a diff not showing you every consumer.

**Scope confirmed, not merely assumed**: the full, untagged default `sub0_tests` suite at this same
GR-ON build reaches `backward_node`'s loud `abort()` on the first GR node a training-path test tries to
differentiate through (many of that suite's test cases call `train_batch`/`backward`) -- exactly Stage
1's own declared scope boundary (S6), confirmed by actually hitting it rather than only documenting it.
gen/eval/report-shaped tests (forward-only) all pass; train/tune-shaped tests correctly, loudly refuse.
