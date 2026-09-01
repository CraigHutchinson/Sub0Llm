# N-gram embeddings — design, derived from the reference

Next item on the Nanbeige-features backlog (`nanbeige-features-progress`), and item 1 of
`docs/QWEN4_PREVIEW_REFERENCE.md`'s staged plan ("N-gram embeddings — start here"). The hypothesis this
tests is unrelated to LoopSplit/depth-attention's fixed-point question: rolling token n-gram hashes give
the model a direct, position-local signal about *which specific tokens recently co-occurred*, cheaper
than asking attention to rediscover the same local pattern from scratch every layer.

Everything below is derived from the verified reference source (`Nanbeige/Nanbeige4.2-3B/modeling_nanbeige.py`,
re-fetched 2026-08-30 directly from Hugging Face — NOT the paraphrase in project memory
`nanbeige-architecture-reference`, which predates this implementation and is reproduced in the task
brief only as a starting map), per `AGENTS.md` §5. The class names are `NanbeigeNgramEmbedding`,
`NanbeigeNgramLayerFusion`, `NgramCache`. Config field defaults (`ngram_fused_mode="average"`,
`ngram_mod_force_prime=False`, `insert_ngram_layer_idx=[]`, `ngram_insert_all_layers=False`, sizing
fields `ngram_vocab_size_ratio`/`emb_split_num`/`emb_neighbor_num`/`emb_tp_num` all `None` until a model
card sets them) were fetched from `configuration_nanbeige.py` in the same repo.

## 1. What the reference actually does

Two classes matter for this stage; a third (`NgramCache`) only for decode.

**`NanbeigeNgramEmbedding.compute_ngram_embeddings`** — the hashing core, quoted (trimmed to the
"concat" branch and the non-force-prime path, which is what this project implements — see §7):

```python
def _precompute_vocab_mods(self):
    vocab_mods = {}
    for i in range(2, self.n + 1):            # n-gram order: bigram .. n-gram
        for j in range(self.k):                # k independent hash tables per order
            index = (i - 2) * self.k + j
            emb_vocab_dim = self._ngram_vocab_dims[index]
            mods = []
            power_mod = 1
            for _ in range(i - 1):
                power_mod = (power_mod * self.ngram_hash_base) % emb_vocab_dim
                mods.append(power_mod)
            vocab_mods[(i, j)] = mods
    return vocab_mods

def _get_ngram_ids(self, input_ids, shifted_ids, vocab_mods, ngram):
    ngram_ids = input_ids.clone()
    for k in range(2, ngram + 1):
        ngram_ids = ngram_ids + shifted_ids[k] * vocab_mods[k - 2]
    return ngram_ids

# forward, per (order i, table j):
new_ids = (ngram_ids % emb_vocab_dim)[..., -seq_len:]
x_ngram = self.embedders[index](new_ids)                       # [T, emb_dim]
ngram_embedding_parts.append(x_ngram)                           # "concat" mode
...
x_concat = torch.cat(ngram_embedding_parts, dim=-1)              # [T, emb_dim * num_embedders]
return self.concat_proj(x_concat)                                # -> [T, hidden_size]
```

`shifted_ids[k]` is `input_ids` shifted right by `k - 1` positions, computed by
`_shift_right_ignore_eos`, whose entire point is that a context token **before the start of the current
document is ZERO** — not "no term", literally token id 0 — rather than reaching into a previous,
unrelated document.

`_ngram_embedding_vocab_sizes` (non-force-prime branch, `ngram_mod_force_prime=False` is the config
default): `vocab_sizes[index] = m + index * 2 + 1`, where `m = ngram_vocab_size_ratio * vocab_size`.
`_ngram_hash_base(vocab_size, force_prime=False)` is simply `vocab_size` — the polynomial hash's base is
the real token vocabulary size.

`num_embedders = k * (n - 1)`. In **"concat" mode** (the one this project implements — see §2), each
table's embedding is `hidden_size / num_embedders` wide (`ngram_hidden_size` defaults to `hidden_size`
when `ngram_embedding_hidden_size` is unset), concatenated end-to-end with **no interference between
tables**, then passed through **one** learned linear (`concat_proj`, no bias) down to `hidden_size`. The
other mode, `"average"`, gives every table its own up-projection and sums — not implemented here (§7).

At the top level, `NanbeigeNgramEmbedding.forward` does:

```python
x = self.word_embeddings(input_ids)
if not self.config.skip_ngram_for_input:
    x = x + ngram_embeddings          # concat mode: a plain additive residual
```

**`NanbeigeNgramLayerFusion`** — re-injects the n-gram signal at *multiple* configured layers (not just
once at the input embedding), via a sigmoid-gated additive residual:

```python
key    = self.key_proj(ngram_embeddings)
gate   = sigmoid(<hidden_norm, key_norm> / sqrt(fusion_size))
fused  = gate * self.value_proj(ngram_embeddings)
hidden = hidden + fused
```

**Not implemented in this stage — deferred, not silently dropped.** See §7.

## 2. Scope for this pass: "concat" fusion, single-layer (input-embedding) injection

Per the task brief: **concat** fusion (simpler, no averaging-projection ambiguity) with injection **only
at the input embedding**, matching `NanbeigeNgramEmbedding.forward`'s plain residual above.
`NanbeigeNgramLayerFusion`'s multi-layer gated re-injection is explicitly **deferred to a later stage**
(§7) — this stage adds no gating machinery and no per-layer fusion modules.

## 3. Forward, in this project's terms

Let `E = NGRAM_TABLES_PER_ORDER * (NGRAM_MAX_N - 1)` (`num_embedders`), `d = D_MODEL / E`
(`NGRAM_EMB_DIM` — the concat-mode invariant `d * E == D_MODEL`, so `ngram_hidden_size` is pinned to
`D_MODEL` rather than exposed as a separate knob: one fewer axis, and it is what makes `concat_proj`
square). Table `e` (`0 <= e < E`) belongs to order `i = 2 + e / K` (`K = NGRAM_TABLES_PER_ORDER`) and has
vocab size `v_e = NGRAM_TABLE_SIZE + e*2 + 1`. `B = VOCAB` (the hash base). Per table, the shift
multiplier for context `s` positions back (`s = 1 .. i-1`):

```
  mod_e[s] = B^s mod v_e            (accumulated: mod_e[s] = (mod_e[s-1] * B) mod v_e)
```

Per position `t` in a window of length `T` (window-local index, 0-based):

```
  ctx(t, s)  = ids[t - s]  if t - s >= 0  else  0        (see §5 for why "window start" stands in for
                                                            "document start")
  hash_e(t)  = ( ids[t] + sum_{s=1}^{order(e)-1} ctx(t, s) * mod_e[s] )  mod  v_e
  y_e(t)     = TableEmbed_e[ hash_e(t) ]                   [d]
  ngram(t)   = concat_proj( concat_e( y_e(t) ) )           [D_MODEL]
  h(t)       = tok_emb(ids[t]) + ngram(t)                  (RoPE path; + pos_emb(t) under absolute)
```

## 4. Implementation: composition, not a new op

Stage 1 does **not** add a new `Op` enumerator. "Concatenate then one linear" is exactly the block-matmul
identity

```
  [x_0 | x_1 | ... | x_{E-1}] @ W  ==  sum_e  x_e @ W[e*d : (e+1)*d, :]
```

so the whole pipeline is built from three **already-verified** ops (`op_embed`, `op_linear`, `op_add`):

```cpp
Node* ng = op_linear(op_embed(ngram_tab[0], ngram_ids[0], T), &ngram_wblock[0], nullptr, false);
for (e = 1..E-1)
    ng = op_add(ng, op_linear(op_embed(ngram_tab[e], ngram_ids[e], T), &ngram_wblock[e], nullptr, false));
h = op_add(h, ng);
```

`ngram_wblock[e]` is a **non-owning view** Node — same `op = Op::Leaf` shape as a real parameter leaf,
but its `data`/`grad` spans are a `subspan` of the ONE real `concat_proj` parameter (`ngram_proj`),
covering rows `[e*d, (e+1)*d)`. Rows are the outer (contiguous) dimension in this project's
`[rows=in, cols=out]` weight convention, so that subspan is a genuinely contiguous sub-tensor, not a
strided view. `op_linear`'s existing backward writes `W->grad[p][o] += x[t][p] * dY[t][o]` for
`p` in the LOCAL row range — which is the CORRECT absolute row range of `concat_proj`'s own grad buffer,
since the views partition it disjointly. `AdamW::step` sees `concat_proj` exactly once, as one ordinary
`PARAM_LAYOUT` entry; the views are never separately registered or stepped.

Why this beats a dedicated `Op::NgramConcat`: `docs/DEPTH_ATTENTION.md` §5a needed a side table specifically
because its fan-in was **variable and cross-execution** (`S+1` cached entries, `S` growing with the loop
schedule) — widening `Node` for that would cost every op a field only one op uses. Here the fan-in is
**fixed at compile time** (`E`, baked from `NGRAM_MAX_N`/`NGRAM_TABLES_PER_ORDER`) and the partition is a
static fact about one already-existing parameter, so a plain array of `E` view-Nodes, built once in
`build_layout()` alongside the real parameter leaves, is sufficient — no `Node` fanout change, no new
backward code, and the three composed ops keep their own, already-tested, gradient-verified backward.
The "hashed n-gram lookup" itself (computing which table row each position addresses) is **not**
differentiable input to any op — same status as `op_embed`'s existing `ids` argument (token ids, or
here, hash-derived ids) — so it is plain index arithmetic in `forward()`, not a graph node.

## 5. One deliberate divergence: "before window start" stands in for "before document start"

The reference's `_shift_right_ignore_eos` zero-fills context that would reach across a real document
boundary (an `<|endoftext|>` marker mid-corpus). This project's training windows are already guaranteed
to live inside exactly one document — `window.hpp`'s whole reason to exist ("Training draws random
windows from the token stream... this samples a window that stays inside ONE document"). So **the
window's own local start (`t=0`) already coincides with either the document's true start, or a point
strictly inside it that the model has no other access to anyway** (RoPE positions, KV-cache, everything
else is window-local too). Treating "before window start" as "no context, id 0" is therefore not a loss
of information relative to what the rest of the architecture can see — it is the same convention every
other position-dependent mechanism in this engine already uses. This is documented explicitly (per the
depth-attention precedent of naming deliberate divergences) rather than silently narrower than the
reference.

**A second, real interaction bug this exposed and fixed**: this engine's persistent/scratch-slot content
embeddings (`scratch_slots.hpp`) hand `op_embed` ids **outside** `[0, VOCAB)` — a persistent-slot id is
`>= VOCAB`, dynamically composed per context rather than a real recurring vocabulary token, and its raw
integer value is unbounded. The n-gram hash originally read `ids[t]` unconditionally, so a persistent-
slot id produced an essentially arbitrary hash bucket, and — because it also poisons every LATER
position that reads it as context — made a persistent-slot sequence's ngram contribution diverge from an
equivalent plain-token reference sequence. `persistent_slots_engine_tests.cpp`'s forward differential
test (a persistent-slot sequence must match a plain-token reference bit-for-bit outside the composed
column) caught this immediately once n-gram embeddings were enabled in that test build. Fix: any id
`>= VOCAB` hashes as "no signal" (id 0) — the exact same convention `_shift_right_ignore_eos` already
uses for "no real token here" — applied in both `forward()` (`ngram_tok`) and `forward_one()`
(`id_tok`/the history buffer, which stores the GUARDED value so it stays consistent on later steps too).
This is exactly the class of bug `AGENTS.md` §10 exists to catch: a new consumer of the shared `ids`
array (op_embed's existing raw-id argument) that didn't check how OTHER existing consumers of that same
array (op_embed's own persistent/scratch-slot dispatch) already handle ids outside the ordinary range.

## 6. Cost accounting

**Parameters** (the axis this feature actually changes, unlike depth attention): `E` tables of
`v_e * d` floats each, plus `concat_proj`'s `D_MODEL * D_MODEL`. At a representative shape
(`D_MODEL=448`, `NGRAM_MAX_N=3`, `K=2` → `E=4`, `d=112`, `NGRAM_TABLE_SIZE=VOCAB≈8000`):

```
  4 tables x ~8000 x 112           ≈ 3.58M floats
  concat_proj  448 x 448           ≈ 0.20M floats
  total                            ≈ 3.78M floats  (≈ 15.1 MB at fp32)
```
— non-trivial next to a d448 model's own ~5-10M trainable floats, dominated by the table vocab size, not
`D_MODEL`. `NGRAM_TABLE_SIZE` is therefore the knob to tune first if this needs to shrink.

**Activations** (CPU, per forward — see `backend_cpu.cpp`'s `calc_act_cap()`): the ngram block runs
**once per forward**, not once per execution/layer like depth attention — `E` embed nodes (`[T, d]`
each, summing to `[T, D_MODEL]`) + `E` linear nodes (`[T, D_MODEL]` each) + `E` add nodes (the
accumulator chain plus the final residual add), i.e. `(1 + 2E) * T * D_MODEL` floats and `3E` extra
graph nodes, added once to `ACT_CAP`/`MAX_NODES` rather than multiplied by `LOOP_EXEC_COUNT`.

**Classification against `ARCH_FINGERPRINT`/`MODEL_ARCH_ID`** (`layout.hpp`'s own required call, per its
"RULE FOR ANY NEW COMPILE-TIME AXIS"): n-gram embeddings **change a tensor shape** (real new parameter
tensors), so — per rule 1, the exact precedent GQA's `D_KV` narrowing already established — `PARAM_FLOATS`
already discriminates a cross-load, and the axis does **not** join `ARCH_FINGERPRINT`. It **does** join
`MODEL_ARCH_ID` (which covers every axis, shape-changing and computation-changing alike,
unconditionally) — the same way adding `DEPTH_ATTN_STRIDE` there already changed `MODEL_ARCH_ID` for
every build regardless of its value; an accepted, precedented cost, since `MODEL_ARCH_ID` is a
diagnostic/directory-naming identity, not the checkpoint-format gate (`ARCH_FINGERPRINT` + the `Header`
+ `nfloat` are).

## 7. Explicitly deferred (not silently dropped)

- **`NanbeigeNgramLayerFusion`'s multi-layer gated re-injection.** This stage injects only at the input
  embedding. Re-injecting at configured middle layers via a learned sigmoid gate is a real, separately
  scoped follow-up (project memory already flags it as "top candidate for the next axis-9 spike" for an
  unrelated reason — scratch-token compression — so there may be two independent motivations to build it).
- **`"average"` fusion mode.** Only `"concat"` is implemented (task brief's explicit initial-target
  choice: simpler, no averaging-projection ambiguity).
- **`ngram_mod_force_prime` (prime-forced table vocab sizes).** Only the non-force-prime branch
  (`m + index*2 + 1`) is implemented — it is the config default and needs no extra "next prime after m"
  search logic. `NGRAM_HASH_BASE` is fixed to `VOCAB` (the non-force-prime branch of `_ngram_hash_base`
  too), for the same reason.
- **Tensor-parallel table padding (`emb_tp_num`).** A multi-GPU sharding concern (padding each table's
  vocab dim to a TP-shard multiple); irrelevant to this project's single-device training.
- **CUDA (Stage 2).** Out of scope for this pass entirely — see the `static_assert(!sub0::NGRAM_EMBED, ...)`
  build-time guard in `backend_cuda.cu`, mirroring the ternary-GPU and (historical) GQA/depth-attention
  CUDA guards.
- **`NgramCache`-equivalent correctness beyond a single generation session.** Stage 1's `forward_one`
  decode path *does* carry a rolling context history across sequential single-token calls (this
  project's analogue of `NgramCache.update_ngram_context`), reset exactly when `forward_one`'s existing
  sentinel-pair "fresh generation or non-sequential jump" signal (`prev < 0`) fires — but it has not been
  cross-checked against the reference's own `NgramCache` beam-reordering semantics (`reorder_cache`),
  which this project's single-sequence decode has no equivalent of anyway.

## 8. Staging

- **Stage 0 — config axes, off by default.** `NGRAM_MAX_N` (0 = off, else >= 2), `NGRAM_TABLES_PER_ORDER`
  (k), `NGRAM_TABLE_SIZE` (m; 0 = auto-resolves to the real `VOCAB` once the tokenizer is learned) —
  `--ngram-max-n`/`--ngram-tables`/`--ngram-table-size` in `sub0llm-configure`, emitted as `constexpr`
  into `sub0_corpus.hpp` (these are per-corpus/per-model architecture facts, alongside
  `LOOP_MIDDLE_LAYERS`/`DEPTH_ATTN_STRIDE`). `layout.hpp` derives `NGRAM_EMBED`/`NGRAM_NUM_EMBEDDERS`/
  `NGRAM_EMB_DIM`/`NGRAM_VOCAB_DIMS`/`NGRAM_VOCAB_MODS`, extends `PKind`/`NUM_PARAMS`/
  `make_param_layout()` (additive: appended at the very end, contributing 0 params when off — see §6's
  fingerprint classification for why no `Header`/`.ckpt` field or `ARCH_FINGERPRINT` bit is needed,
  unlike depth attention), and folds the three axes into `MODEL_ARCH_ID`. Pinned in `layout_tests.cpp`:
  the hashing math at hand-verified hypothetical configs (parameterised free functions, same pattern as
  `DepthScheduleT`'s `depth_schedule_for()`), and the neutral-setting contract (zero new `PARAM_LAYOUT`
  entries, `NUM_PARAMS` unaffected).
- **Stage 1 — CPU forward + backward.** `Model::build_layout()`/`init_weights()` grow the `E` table
  leaves + `concat_proj` + the `E` view-Nodes (§4); `forward()` computes the hash and wires the
  embed/linear/add composition; `forward_one()` implements the decode-path equivalent with a rolling
  history buffer (§7's caveat). No new `Op`, no new backward code (§4). Verified: the engine's existing
  generic finite-difference gradient check (`engine_tests.cpp`, config-agnostic — it iterates whatever
  `trainable_floats()` the current build has) at an n-gram-enabled build; a dedicated presence test
  proving the LIVE contribution depends on context, not just the current token (§4's "no dedicated op"
  choice does not weaken this guard — the double-difference isolates the mechanism regardless of how
  many ops it is built from); the existing generic `forward_one`-vs-`forward` parity test, likewise
  config-agnostic, covering the decode path. Bit-identical-at-neutral-setting verified at two model
  scales (`AGENTS.md` §7) — see the final report for the exact assertion-count deltas.
- **Stage 2 — CUDA.** Not attempted this pass; build-time `static_assert` guard only (§7).
- **Real-Qwen-weight fixture check — DONE.** `tests/fixtures/qwen4_preview/ngram_embedding_*` landed on
  `main` (a parallel extraction effort, merged after this branch was first started) with REAL per-table
  embedding values and their real flattened/concatenated form read directly from the
  `Qwen/Qwen3.8-Flash-Next` checkpoint. `tests/ngram_qwen4_fixture_tests.cpp` (part of
  `sub0_frontend_tests`, engine-free) verifies this project's "concat" ordering convention (table/head
  index MAJOR, in-row index MINOR — see §4) reproduces the real module's own `.flatten(-2)` output
  bit-for-bit when fed that fixture's real per-head embeddings, i.e. the fusion/ordering CONVENTION
  matches even though the hash formula and table sizes deliberately do not (§7's scope boundary; Qwen4's
  real hashing is a different splitmix64-style scheme, and it injects via a hyper-connection stream with
  no `concat_proj`-equivalent stage to compare against). Skips gracefully (WARN, not fail) if the
  fixtures are ever absent from a given checkout.
- **Stage 3 — huge external tables (e.g. importing Qwen's real 51.2B-param n-gram table), not started,
  design lives elsewhere.** Today's `ngram_tab[e]` is a fully-resident `PARAM_LAYOUT` leaf, correct only
  for the small, from-scratch-trained tables this stage targets. Growing this to a huge, externally
  sourced, effectively-frozen table (too big for RAM/VRAM) is a distinct backing-store problem, not a
  variant of Stage 1/2 — see `docs/NGRAM_TABLE_TIERED_STORAGE.md` for the design and
  [github.com/CraigHutchinson/Sub0Firn](https://github.com/CraigHutchinson/Sub0Firn) (spec + prior art +
  a concrete, code-grounded trace of exactly what this file's `forward()`/`forward_one()` would need from
  such a backing store — its README §7) for where the actual implementation is scoped to land. Nothing
  in `Model`'s current structure needs to change to make room for this later: `ngram_tab[e]` would become
  a thin client issuing `resolve_into` calls instead of a raw parameter pointer, per that design.
