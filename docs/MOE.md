# Mixture of Experts — design, derived from the reference

Status: **Stage 0 (config skeleton, hard-gated off) + Stage 1 (CPU forward only) — this pass.** Follows
`docs/GATED_DELTANET.md`'s and `docs/GATED_RESIDUAL.md`'s own structure and staging discipline (real
source quoted verbatim, engine-interaction analysis, checkpoint design, two-scale identity checks) --
`docs/GATED_RESIDUAL.md` in particular, as the most recently landed worked example, is this doc's closest
precedent. Per AGENTS.md S5, every equation below is quoted or worked from the real, installed
`transformers==5.16.1` source fetched for this doc -- not from recall, and not from
`docs/QWEN4_PREVIEW_REFERENCE.md`'s own facts-table paraphrase alone (that table's `num_experts=512`/
`num_experts_per_tok=10`/`moe_intermediate_size=640`/`shared_expert_intermediate_size=640` are reused, not
re-derived -- already independently verified there and in `docs/QWEN4_MEMORY_ORCHESTRATION.md` S2a).

## 0. Sources, and their confidence

**Primary, highest-confidence source -- the real model's own code**, fetched directly on this machine
(`transformers==5.16.1`, mainline PyPI, no `trust_remote_code`):
`transformers.models.qwen4_exp.modeling_qwen4_exp`, classes `Qwen4ExpTextSparseMoeBlock` (the block
itself), `Qwen4ExpTextTopKRouter` (the router), `Qwen4ExpTextExperts` (the routed experts, stored as 3D
tensors) and `Qwen4ExpTextMLP` (the plain SwiGLU FFN class the shared expert reuses). Fetched via
`python3 -c "import inspect, transformers.models.qwen4_exp.modeling_qwen4_exp as m; print(inspect.
getsource(m.Qwen4ExpTextSparseMoeBlock))"` and the same for the other three classes -- the identical
technique every prior stage in this thread used, confirmed still installed and importable on this machine.

**Real config values**, fetched directly from the real, installed `Qwen/Qwen3.8-Flash-Next`
`config.json` on this pass (`num_experts=512`, `num_experts_per_tok=10`, `moe_intermediate_size=640`,
`shared_expert_intermediate_size=640`, `hidden_act=silu`, all matching `docs/QWEN4_PREVIEW_REFERENCE.md`'s
own table) -- and, load-bearing for S1a's weighting scheme, `norm_topk_prob`, which does **not** appear
anywhere in the real `config.json` (checked explicitly, recursively, not assumed absent) and therefore
takes the dataclass default `norm_topk_prob: bool = True` from
`transformers.models.qwen4_exp.configuration_qwen4_exp`.

**No real-weight fixture existed before this pass.** Unlike GDN/GR, whose fixtures had already landed on
`main` from earlier passes, S5/S6 below extract this stage's own fixture (router + only the top-k experts
a hand-picked input actually selects + the shared expert, from the real
`model.language_model.layers.0.mlp.*` tensors), following the exact HTTP-Range-against-the-real-safetensors
technique validated four times now in this thread (GDN, GR, n-gram embeddings, and now this).

## 1. What the reference actually does

### 1a. The block itself — `Qwen4ExpTextSparseMoeBlock`, quoted verbatim

```python
class Qwen4ExpTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = Qwen4ExpTextTopKRouter(config)
        self.experts = Qwen4ExpTextExperts(config)
        self.shared_expert = Qwen4ExpTextMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output
```

The router, quoted verbatim (`Qwen4ExpTextTopKRouter`):

```python
class Qwen4ExpTextTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

The routed experts, quoted verbatim (`Qwen4ExpTextExperts`):

```python
@use_experts_implementation
class Qwen4ExpTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states
```

And the shared expert's own class, `Qwen4ExpTextMLP` (an ordinary, non-routed SwiGLU FFN, reused with
`intermediate_size=shared_expert_intermediate_size`):

```python
class Qwen4ExpTextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```

Facts this resolves, verified rather than assumed from the task's own framing:

- **Softmax over ALL `num_experts` logits, THEN top-k** -- not top-k-then-softmax (a real, common
  alternative in other MoE families that would change which experts get selected at all: softmax-then-topk
  and topk-then-softmax select the SAME top-k set here only because softmax is monotonic, but the
  resulting WEIGHTS differ whenever `norm_topk_prob` is false; with it true, as this model has, the two are
  numerically identical after renormalization -- worked through explicitly, not assumed, because the task
  named this exact ambiguity to resolve).
- **`norm_topk_prob=True`** (S0): the selected top-k weights are renormalized to sum to 1 AFTER selection,
  so `expert_output` is a proper convex combination of the routed experts' own outputs.
- **The shared expert is NOT re-weighted by the router's own output at all.** It is a completely separate,
  always-on SwiGLU FFN, scaled by its own independent sigmoid gate (`shared_expert_gate`, a
  `Linear(hidden_size, 1, bias=False)` reading the SAME input `x`, not the router's logits) and simply
  ADDED to the routed experts' weighted sum. This is a real, verified divergence from a naive "shared
  expert is expert #0 with a fixed weight" reading -- the task explicitly asked this to be resolved rather
  than assumed, and the answer is: two structurally independent contributions, summed.
- **Per-expert FFN is 3-matrix SwiGLU** (`gate_up_proj` chunked into gate/up halves, `down_proj`), exactly
  this project's own existing `USE_GATED_FFN` convention (Wg/W1/W2, no bias) -- confirmed from the real
  class construction (`Qwen4ExpTextExperts.__init__`'s own tensor shapes), not assumed from the
  arithmetic `docs/QWEN4_MEMORY_ORCHESTRATION.md` S2a already worked out.
- **The router has NO bias** (`F.linear(hidden_states, self.weight)`, no `+ bias`) and no
  aux-loss-free bias term of the kind some other MoE families (e.g. DeepSeek-V3) add to the routing
  logits before top-k -- this is a plain softmax router, verified by reading `Qwen4ExpTextTopKRouter`'s
  own `forward()` in full, not assumed from familiarity with other MoE designs.

## 2. Row-independence — why this needs no `Node`-graph/execution-order interaction at all

Traced from the real current code, NOT assumed: unlike Gated DeltaNet (a genuinely NEW per-layer schedule,
some layers softmax attention, some GDN -- `GDN_SCHEDULE`) or Gated Residual (a wrapping transform around
EVERY sub-block's entry/exit, still per-layer), MoE in the real model replaces the FFN block for **every**
decoder layer uniformly when the model has MoE at all -- there is no per-layer on/off schedule to derive,
only a single model-wide gate. `USE_MOE` is therefore a plain `if constexpr` swap of the FFN computation
inside the existing per-layer loop (`backend_cpu.cpp`'s `Model::forward`/`forward_one`), touching nothing
about `LAYER_EXEC_ORDER`, `LoopSplit`, or any cross-execution state -- the SIMPLEST of the three
mechanisms' own engine-interaction stories, precisely because the real model's own design has no per-layer
variation to express.

**Row-independence, the second reason this integrates simply**: unlike GDN's chunked recurrent state
(cross-TOKEN memory within a call) or GR's cross-STREAM mixing, MoE's router/expert computation for
token `t` depends ONLY on token `t`'s own hidden state -- no cross-position state at all. This is why
`moe_math.hpp`'s own `scratch_floats()` is NOT scaled by `T` (unlike `gdn_math.hpp`'s `scratch_floats(d,
T)`): one scratch buffer serves an entire `[T, hidden_size]` batch, reused row by row.

## 3. Checkpoint / `PARAM_LAYOUT` impact

### 3a. Which axes are genuinely new

| Reference quantity | This project's mapping | Reused / new |
|---|---|---|
| `num_experts` | `NUM_EXPERTS` (new CLI flag) | new |
| `num_experts_per_tok` | `EXPERTS_PER_TOK` (new CLI flag) | new |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | `D_FF` (existing) | **reused, not a new axis** -- see below |
| router/expert/shared-expert weight shapes | derived from `D_MODEL`, `D_FF`, `NUM_EXPERTS` | new (S3b) |

`D_FF` (this project's existing FFN-width knob) is reused for BOTH the routed-expert and shared-expert
SwiGLU width, rather than adding a third CLI axis: the real model's own `moe_intermediate_size` (640) and
`shared_expert_intermediate_size` (640) are already equal, so one shared width is real-model-faithful, not
an approximation -- and AGENTS.md S8 disfavors adding a knob nothing yet needs to vary independently.

`EXPERTS_PER_TOK` is genuinely bounded by `[1, NUM_EXPERTS]` when on (checked at both the CLI layer and,
implicitly, by the router's own `torch.topk` semantics -- selecting more experts than exist is meaningless)
-- mirroring `--hc-lowrank`'s own "must be >= 1 exactly when the paired axis is on" pattern.

### 3b. Per-layer `PARAM_FLOATS` delta, worked from the real shapes

Using this project's own `[rows=in, cols=out]` weight-layout convention throughout (the transpose of the
real `nn.Linear`/`gate_up_proj`/`down_proj` conventions, re-derived per AGENTS.md S5, not assumed):

```
  MoeRouter          = D_MODEL * NUM_EXPERTS                          (no bias)
  one routed expert  = D_MODEL*D_FF (gate) + D_MODEL*D_FF (up) + D_FF*D_MODEL (down)
                      = 3 * D_MODEL * D_FF
  shared expert      = 3 * D_MODEL * D_FF                              (same SwiGLU shape, own weights)
  MoeSharedGateProj  = D_MODEL * 1                                     (no bias)
```

Per layer, MoE **replaces** the existing FFN block's own tensors (3 for gated SwiGLU: Wg/W1/W2; 4 for
plain: W1/B1/W2/B2) -- unlike Gated Residual's own `gr_param_delta` (a pure ADDITION on top of an
unchanged existing tensor set), this is `moe_layer_floats - dense_ffn_floats`:

```
moe_layer_floats  = D_MODEL*NUM_EXPERTS + NUM_EXPERTS*3*D_MODEL*D_FF + 3*D_MODEL*D_FF + D_MODEL
dense_ffn_floats  = 3*D_MODEL*D_FF                        (gated FFN this project defaults to)
                  | 2*D_MODEL*D_FF + D_FF + D_MODEL        (plain FFN)
Delta_per_layer   = moe_layer_floats - dense_ffn_floats
Delta_total       = N_LAYERS * Delta_per_layer
```

**Worked through explicitly, not assumed, that this is strictly positive for every `NUM_EXPERTS >= 2`**
(the minimum "on" value, per S4a): at the cheapest possible case (`num_experts=2`, replacing a GATED dense
FFN, the larger of the two dense forms), `moe_layer_floats - dense_ffn_floats = (2*D_MODEL + 2*3*D_MODEL*
D_FF + 3*D_MODEL*D_FF + D_MODEL) - 3*D_MODEL*D_FF = 3*D_MODEL + 6*D_MODEL*D_FF`, strictly positive for any
`D_MODEL, D_FF >= 1`. Every larger `NUM_EXPERTS` only adds more strictly-positive `per_expert` terms, so
`Delta_per_layer` is monotonically increasing in `NUM_EXPERTS` and always positive once MoE is "on" -- the
same strength of guarantee `docs/GATED_RESIDUAL.md` S3b established for its own `gr_param_delta`, even
though this delta is a REPLACEMENT, not a pure addition.

### 3c. Classification: `NUM_EXPERTS` is shape-changing (`PARAM_FLOATS` alone); `EXPERTS_PER_TOK` is NOT

This is the one place this doc's own arithmetic genuinely diverges from `docs/GATED_RESIDUAL.md`'s
precedent, and the task explicitly asked it to be worked out rather than assumed:

- **`NUM_EXPERTS`** changes the actual TENSOR COUNT/SHAPE (S3b's `Delta_total`, strictly monotonic) --
  per `layout.hpp`'s own `ARCH_FINGERPRINT` classification rule #1, `PARAM_FLOATS` alone already
  discriminates any two builds that differ only in `NUM_EXPERTS`. No fingerprint bits needed; it still
  joins `MODEL_ARCH_ID` unconditionally, the same treatment `HC_COUNT`/`HC_LOWRANK` already get.
- **`EXPERTS_PER_TOK` changes NOTHING about any tensor's shape.** All `NUM_EXPERTS` experts are stored
  regardless of how many are selected per token (S2: MoE stores every expert as an ordinary resident
  parameter, per this task's own scope boundary -- no offload/residency mechanism exists here). Two builds
  at identical `NUM_EXPERTS`/`D_MODEL`/`D_FF` but different `EXPERTS_PER_TOK` produce **byte-identical**
  checkpoints that compute genuinely different things (a different top-k cutoff every forward pass) --
  exactly `ARCH_FINGERPRINT2`'s own rule #2 ("changes computation, not shape -> ADD IT HERE, or it loads
  silently and computes the wrong thing"), the identical shape of problem `GDN_FULL_ATTN_STRIDE` already
  solved there. `EXPERTS_PER_TOK` therefore joins `ARCH_FINGERPRINT2` (byte 1, alongside GDN's stride in
  byte 0 -- using bits `ARCH_FINGERPRINT2`'s own comment already reserved for exactly this purpose), NOT
  `PARAM_FLOATS`/`moe_param_delta`.

### 3d. On-disk plumbing

Purely additive at every point `PARAM_LAYOUT` already handles additively (same as GR S3d): new `PKind`s
appended to the existing enum, new tensors appended at the natural position in `make_param_layout()`'s
existing per-layer loop (replacing, not appending after, the FFN block -- `if constexpr (USE_MOE) {...}
else if constexpr (USE_GATED_FFN) {...} else {...}`, so a MoE-off build's byte layout for every OTHER
tensor is completely unchanged). No new `Header`/`.ckpt` trailer record is needed for `NUM_EXPERTS` (S3c);
`EXPERTS_PER_TOK` rides the EXISTING `ARCH_FINGERPRINT2` trailer record already on disk for GDN -- no
new trailer, no `CKPT_VERSION` bump, purely a new bit-field inside an already-additive word.

## 4. Compile-time specialization

### 4a. New `RunConfig`/generated-header constants, mirroring `GDN_FULL_ATTN_STRIDE`/`HC_COUNT` exactly

- `--num-experts` (CLI int, default `0`) -> generated `constexpr int NUM_EXPERTS`. `0` = off. `1` is
  disallowed (a single "routed expert" is not a Mixture; also degenerates several of S5's presence checks)
  -- `static_assert(NUM_EXPERTS == 0 || NUM_EXPERTS >= 2, ...)`, the same "`0` or `>= 2`" shape
  `HC_COUNT`/`NGRAM_MAX_N` already use.
- `--experts-per-tok` (CLI int, default `0`) -> generated `constexpr int EXPERTS_PER_TOK`. Must be `0`
  exactly when `NUM_EXPERTS == 0`, `>= 1` and `<= NUM_EXPERTS` exactly when `NUM_EXPERTS >= 2` -- checked
  at both the configurator (a named diagnostic) and `layout.hpp` (a `static_assert`).
- `inline constexpr bool USE_MOE = (NUM_EXPERTS >= 2);` -- the single gate every `if constexpr` in
  `backend_cpu.cpp` keys off.
- `inline constexpr moe::Dims MOE_DIMS{D_MODEL, D_FF, NUM_EXPERTS, EXPERTS_PER_TOK};` -- always a valid,
  never-divide-by-zero shape even when `USE_MOE` is false, same "describes a shape nothing builds" idiom
  `GDN_DIMS`/`GR_DIMS` already use.
- `NUM_EXPERTS_BUF` -- the never-zero array-bound form (same idiom as `HC_COUNT_BUF`), for the
  `Layer::moe_gate[NUM_EXPERTS_BUF]`-style per-expert `Node*` arrays in `backend_cpu.cpp` (S5 below).

**Stage 0 (this pass's own first commit) hard-clamps both flags to their neutral value**, mirroring
GDN's/GR's own Stage 0 exactly: `--num-experts`/`--experts-per-tok` get `->check(CLI::Range(0, 0))` in
`configurator.cpp` AND `layout.hpp` keeps `static_assert(NUM_EXPERTS == 0, ...)`/
`static_assert(EXPERTS_PER_TOK == 0, ...)` (the strict, single-value form) until Stage 1 (this pass's own
SECOND commit) relaxes both to real ranges once the CPU forward op exists to compute against.

### 4b. `include/sub0/moe_math.hpp` — the engine-free math core, mirroring `gdn_math.hpp`'s/`gated_residual_math.hpp`'s role

A `sub0::moe::Dims{hidden_size, d_ff, num_experts, experts_per_tok}` struct and four free functions, all
taking this project's own `[rows=in, cols=out]` weight convention and explicit caller-owned scratch
(AGENTS.md S1 -- no heap allocation): `expert_ffn_row` (one expert's SwiGLU for one token),
`router_topk_row` (softmax-then-topk-then-optional-renormalize, S1a), `forward_row` (the full block for
one token: route, run only the SELECTED experts, weighted-sum, add the gated shared expert) and `forward`
(a thin T-row loop over `forward_row`, scratch reused across rows per S2). See that file's own header
comment for the full derivation and the `TOPK_MAX` design note below.

### 4c. Top-k selection: host-side scalar code inside the math core, NOT a new differentiable `Node`

The task named this as an open design question: does top-k selection need a genuinely new `Op::` to
participate in the (backward-free-for-now) `Node` graph, or can it be host-side scalar code outside it
entirely? **Decision: it is folded INTO `Op::Moe`'s own forward function** (mirroring `op_gdn`'s own
precedent: one op, `Layer&`-parameterized, does routing + all the expert math internally) rather than
factored into a SEPARATE selection op with its own `Node` fanout. Reasoning: (1) per AGENTS.md's own
framing, "a top-10-of-8-or-512 selection is tiny, bounded compute" -- `O(NUM_EXPERTS * EXPERTS_PER_TOK)`,
bounded by `TOPK_MAX=16` stack arrays, genuinely negligible next to any of the matmuls around it; (2)
Stage 1 is explicitly forward-only (S7 below) -- there is no backward consumer that would need top-k's
OWN node in the graph to attach a gradient to, the same reasoning `docs/GATED_RESIDUAL.md` S5 gave for why
its four ops needed no side table either, since nothing walks the pool looking for them yet; (3) splitting
it into a separate op would need to communicate `EXPERTS_PER_TOK` indices/weights to a SUBSEQUENT op
picking which expert weights to read -- exactly the kind of "order-sensitive cross-op scratch dependency"
`docs/GATED_RESIDUAL.md` S4c already argued against for a cheaper, simpler win. One `Op::Moe` node,
taking `Layer&` directly (reading `NUM_EXPERTS_BUF` router/expert/shared-expert tensors off it, mirroring
`op_gdn(Node* a, Layer& L)`'s own signature) is the simplest correct Stage 1 shape.

## 5. The `Node`-fanout question — resolved the same way GR's Stage 1 was

One new `Op::Moe` enumerator: `op_moe(Node* x, Layer& L)` -> output `[T, D_MODEL]`, `out->a = x` (mirroring
`op_gdn`'s own `out->a = a` -- enough for a future backward's abort-placeholder case to at least name the
right input node). No side table: Stage 1 has no backward walking the node pool yet (S6), so nothing needs
to recover the router/expert/shared-expert tensors from a bare `Node*` -- `Layer&` is read directly inside
`op_moe`'s own forward body, the identical precedent `op_gdn`/`op_gr_mix` already established.

## 6. Backward through discrete top-k selection — out of scope this stage, but a real subtlety to name

Per the task's explicit scope boundary (CPU forward only) and the exact `Op::GDN`/`Op::GatedResidual`
Stage-1 precedent: `backward_node`'s `switch` gains one new `case Op::Moe:` label,
`std::println(stderr, "fatal: ...")` + `std::abort()`, refusing to silently train a different architecture
than the one requested.

**The real subtlety for a future Stage 2, named explicitly so it is not designed blind**: standard MoE
practice routes gradient through the SELECTED experts' own FFN weights only (an unselected expert's
`gate`/`up`/`down` matrices get zero gradient this step -- they were never read). But the ROUTER's own
weight gradient is NOT similarly sparse: `router_probs = softmax(router_logits)` is a full-width softmax
over ALL `NUM_EXPERTS` logits, and softmax's own Jacobian `d(probs_i)/d(logits_j)` is nonzero for every
`(i,j)` pair (the normalization term `sum_e exp(logits_e)` couples every logit to every other). So even
though `torch.topk` only SELECTS `EXPERTS_PER_TOK` of the `router_probs` entries to actually weight an
expert's output, the loss gradient flowing back INTO `router_logits` at every one of the `EXPERTS_PER_TOK`
selected positions still has nonzero partial derivatives with respect to EVERY router logit, selected or
not -- meaning `MoeRouter`'s weight matrix genuinely needs a full-width backward pass touching every
column, not merely the columns corresponding to experts that happened to be selected for a given token.
(The `torch.topk` operation itself is non-differentiable at the selection boundary -- standard practice,
and this reference implementation's own, is to treat the selected INDICES as constants for a step, i.e.
`torch.topk`'s "value" gradient flows, its "index" choice does not -- straight-through on the selection,
full-gradient through the softmax that produced the values being selected from.) A Stage 2 implementation
that naively zeroed the router's gradient outside the selected columns (an easy mistake by analogy with
"only selected experts' FFN weights get gradient") would silently under-train the router's ability to
learn NOT to route to an expert, which is exactly half of what a router needs to learn.

## 7. Correctness gate

`tests/moe_qwen4_fixture_tests.cpp` (engine-free, mirroring `gdn_qwen4_fixture_tests.cpp`'s/
`gated_residual_qwen4_fixture_tests.cpp`'s own pattern): reads
`tests/fixtures/qwen4_preview/moe_layer0_small_*` -- the real router weights (sliced `D_MODEL`, full
`NUM_EXPERTS` kept at a small-but-real value per S8's own reasoning below) plus ONLY the real weight
tensors for whichever experts a hand-picked, reproducible-seed synthetic input actually routes to, plus
the real shared-expert tensors -- transposes the raw PyTorch `[out,in]`/`[num_experts,out,in]` weight
files into this project's `[in,out]` convention, and calls `sub0::moe::forward_row` directly. Compares
against the real `Qwen4ExpTextSparseMoeBlock.forward()` output on the same input (exact fixture-comparison
numbers in the final report, Stage 1 execution output, not fixed ahead of running it).

**Presence/mutation check, per the depth-attention lesson this whole thread keeps citing**: a mutant that
always used the SAME fixed expert(s) regardless of the router's actual output could still look plausible on
one fixture if that fixture happens to route to those experts. The test therefore ALSO re-runs
`forward_row` on the SAME input with the router weights REPLACED by a synthetic router that deterministically
selects a DIFFERENT expert subset, and asserts the output changes by a real, nonzero amount -- proving the
implementation genuinely reads the router's selection rather than hard-coding which experts to consult.

## 8. Two-scale identity + real-build verification

Per AGENTS.md S7 and this thread's own repeated lesson: `layout_tests.cpp` pins `NUM_EXPERTS`/
`EXPERTS_PER_TOK`'s neutral-setting identity at TWO shapes -- reusing this thread's now-standard pair, d96
L8 H2 seq128 (even) and d132 L11 H4 kv2 seq96 (odd/ragged) -- asserting `PARAM_FLOATS`/`NUM_PARAMS` are
UNCHANGED from a MoE-unaware calculation at both, and a standalone `moe_param_delta()` check against
hand-computed expected values at each shape (mirroring GR's own Stage 0 test precedent). A genuinely
MoE-ON small build (`NUM_EXPERTS=8, EXPERTS_PER_TOK=2` -- this stage's own commit-sized test scale, real
512/10 explicitly out of scope per the task) is built once to confirm `forward()`-vs-`forward_one()` parity
holds with the mechanism actually active. Exact assertion counts, hashes, and parity numbers are in the
final report (Stage 1 execution output), not fixed in this design doc ahead of running them.

## 9. Results (Stage 1 execution)

**A real bug found and fixed by actually building at a real MoE-ON config, not by reasoning alone**: the
first Stage 1 pass wired `op_moe` into the FFN block inside `Op::GDN`'s own branch (unreachable at the
default `GDN_FULL_ATTN_STRIDE=0`) but NOT into the softmax-attention branch's own copy of that same FFN
block -- the one every layer actually runs at the default stride. A real `NUM_EXPERTS=8, EXPERTS_PER_TOK=2`
build (`D_MODEL=16, N_LAYERS=2, N_HEADS=2, SEQ_LEN=32`) reproduced this as a SIGSEGV in the standard
`forward_one`-vs-`forward` parity test: `L.Wg`/`L.W1`/`L.W2` are never allocated when `USE_MOE` is on
(`build_layout()`'s own `if constexpr` branch skips them), so the untouched branch dereferenced a null
weight pointer. Root-caused by instrumenting `op_moe` with stderr prints and observing NONE fired before
the crash -- proof the reachable code path never called it at all. Fixed by wiring the same
`if constexpr (USE_MOE) { f = op_moe(f, L); } else if constexpr (USE_GATED_FFN) {...} else {...}` into
BOTH branches. A second, smaller consumer gap (AGENTS.md S10's own lesson) surfaced the same way:
`layout_tests.cpp`'s pre-existing "arch fingerprint2 stays bit-identical when Gated DeltaNet is off" test
asserted `ARCH_FINGERPRINT2 == 0` unconditionally, which stopped holding once `EXPERTS_PER_TOK` also lives
in that word (S3c) -- fixed by generalizing the assertion to an implication on `EXPERTS_PER_TOK`'s own
value and adding MoE-specific byte-1 assertions mirroring the existing byte-0 (GDN) ones.

**Fixture correctness gate** (`tests/moe_qwen4_fixture_tests.cpp`, real weights extracted from
`Qwen/Qwen3.8-Flash-Next` layer 0's real `model.language_model.layers.0.mlp.*` tensors -- full real
512-expert router sliced `hidden_size` 2560->16, the real router's own top-10 selection on a real
hand-picked input, only those 10 (+2 decoy) experts' real weight slabs fetched, `moe_intermediate_size`
640->8, one hand-picked real input): **`max|out - expected| = 1.45519e-11`** over `sum|expected| =
0.000384256` across 16 values -- essentially an exact, float32-rounding-level match against the real,
unmodified `transformers==5.16.1` `Qwen4ExpTextSparseMoeBlock.forward()`.

**Presence/mutation checks** (same file): (1) a SECOND real reference (`output_mutant`) was computed by
forcing the real router's own top-10 selection's first two slots to two different, also-real "decoy"
experts and re-running the same real PyTorch module -- our CPU port's own forced-mutant recombination
matches that second real reference to `max|out_mutant - expected_mutant| = 1.79974e-07`, AND the real vs.
mutant outputs genuinely differ by `max|out(real) - out_mutant(forced decoys)| = 2.91125e-05` -- large and
real, proving the output genuinely depends on WHICH experts were consulted, not merely that some plausible
output comes out; (2) a second, fixture-free test confirms `router_topk_row`'s selected SET tracks the
input (an input favoring low expert indices selects `{0,1}`; the same router favoring high indices selects
`{2,3}`), and that `norm_topk_prob=true` renormalizes the selected weights to sum to exactly 1. All 2 test
cases: 44 assertions, green. Full `sub0_frontend_tests` suite (which these fixture tests join, the same
placement `gdn`/`gr` fixture tests use): 192 test cases / 114,748 assertions, all green (190 pre-existing +
2 new).

**Two-scale identity check** (AGENTS.md S7): at `NUM_EXPERTS == 0` (neutral), the full default engine test
suite (`sub0_tests`) is assertion- AND hash-identical across every commit in this pass, at BOTH shapes:

| shape | assertions | test cases | forward hash | grad hash | decode hash |
|---|---|---|---|---|---|
| d96 L8 H2 seq128 vocab26260 (even) | 18,118,452 | 142 | `5c89301ea110f2ce` | `40ec687c2de8cc1d` | `e77acd5f0e790c87` |
| d132 L11 H4 kv2 seq96 vocab26260 (odd/ragged) | 29,667,408 | 142 | `865d6ce187efacff` | `ee590d25a4f2c865` | `95a33863710c3cc8` |

Both shapes' hashes are unchanged from the very first Stage 0 commit through the softmax-attention-branch
bug fix (S9's own finding above) -- confirming that fix, despite touching the reachable code path, remains
byte-identical when `USE_MOE` is false (it only ever added a new `if constexpr` branch that the neutral
build never takes). Assertion counts moved twice, both fully accounted for: Stage 0 added 3 new
`[layout][moe]` test cases (728 assertions); the `ARCH_FINGERPRINT2` consumer-gap fix (S9) added 7 more
assertions to one pre-existing test case (both deltas nothing else).

**Real MoE-ON build, `forward`-vs-`forward_one` parity** (this doc's own scope: a small correctness-
fixture-scale config, not production dims): `D_MODEL=16, N_LAYERS=2, N_HEADS=2, SEQ_LEN=32, NUM_EXPERTS=8,
EXPERTS_PER_TOK=2` -- a genuinely MoE-active build, `PARAM_LAYOUT` carrying the real
router+8-expert-triple+shared-expert-triple+shared-gate shape. `engine_tests.cpp`'s existing, MoE-agnostic
"`forward_one` (KV-cache) matches the full forward per position" test (no MoE-specific code needed -- it
exercises whatever the compiled `Model` actually computes) passes with worst per-position relative diff
**`2.92655e-07`** -- float32-rounding-level agreement between the Node-graph training path (`op_moe`
calling `sub0::moe::forward`) and the raw-pointer decode path (`sub0::moe::forward_row` called directly),
confirming both call sites agree on a real forward pass. `[layout]` at this same MoE-ON build: 478,614
assertions / 13 test cases, all green.

**Scope confirmed, not merely assumed**: the full, untagged default `sub0_tests` suite at this same MoE-ON
build reaches `backward_node`'s loud `abort()` ("fatal: Mixture of Experts has no backward pass yet...") on
the first `Op::Moe` node a training-path test tries to differentiate through -- exactly Stage 1's own
declared scope boundary (S6), confirmed by actually hitting it rather than only documenting it, the same
outcome `docs/GATED_RESIDUAL.md` S9 recorded for its own mechanism. `[layout]`/`[moe]`-tagged and
forward-only tests all pass; train/tune-shaped tests correctly, loudly refuse.

**CUDA guard**: `static_assert(!sub0::USE_MOE, ...)` added to `backend_cuda.cu`, mirroring the Gated
Residual guard exactly. Not build-verified in this environment (no `cl.exe` on `PATH` for `nvcc` in this
sandbox), but the assertion is unconditionally false at the default `NUM_EXPERTS=0`, so it cannot regress
any existing CUDA build.
