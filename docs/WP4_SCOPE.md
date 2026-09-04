# WP4 — real-scale Qwen3.8-Flash-Next run + llama.cpp comparison: scoping proposal

Status: **SCOPING ONLY — no engine code, no new `RunConfig` axis, no new CLI flag lands with this
document.** This mirrors exactly how Phase 0 (`docs/QWEN4_MEMORY_ORCHESTRATION.md`) and each of WP1-3
(`docs/GATED_RESIDUAL.md`, `docs/MOE.md`, `docs/QSA.md`) began: a design document first, reviewed on its
own, then implementation in independently-mergeable stages each with its own correctness gate.

**Read first, not re-derived here**: `AGENTS.md` (every rule below is an application of one),
`docs/QWEN4_PREVIEW_REFERENCE.md` (the architecture facts table — §1's target config is cross-checked
against it, not restated from memory), `docs/QWEN4_MEMORY_ORCHESTRATION.md` **v2** (the byte budgets and,
critically, its new §2h shape-gap finding, which is this document's single biggest input),
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` (real GGUF tier byte totals, the `gguf.hpp` dequant gap, and the
llama.cpp `-ot`/`--n-cpu-moe` prior art), and each mechanism's own doc for the per-mechanism `[in,out]`
transpose conventions §3 aggregates.

**What this document is NOT**: it is not a promise that a real run is achievable on this hardware.
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §4a's hedged verdict ("technically launchable, not comfortably
usable", for `llama.cpp` at `UD-IQ1_S`) is unchanged and applies at least as strongly to this engine,
which has no quantized inference path at all today. §7's open questions include, explicitly, whether the
user wants to attempt a real download at all.

---

## 0. Where WP4 actually starts — the three facts that shape everything below

1. **Two of the four mechanisms are already shape-exact at Qwen4's real config; two are not.** Measured
   this pass (`docs/QWEN4_MEMORY_ORCHESTRATION.md` §2h): evaluating this repo's own `make_param_layout()`
   at Qwen4's real `RunConfig` axes yields `NUM_PARAMS = 74,899` tensors and
   `PARAM_FLOATS = 124,027,786,776`. Gated Residual (640,624,640) and MoE (121,094,922,240) match the
   real model **exactly**. GDN is 1.335B short and QSA is 0.349B short, both for the same reason: the
   engine derives `D_HEAD = D_MODEL / N_HEADS` (= 106 here, real is 256) and aliases GDN's key/value head
   counts and head dims onto `N_KV_HEADS`/`N_HEADS`/`D_HEAD` (`layout.hpp`'s `GDN_DIMS`). **A weight
   transplant is impossible until that is fixed** — the destination tensors are literally the wrong shape.
2. **There is no foreign-weight import path in this engine at all.** `include/sub0/gguf.hpp` has exactly
   one consumer repo-wide: `tests/gguf_tests.cpp`. Nothing in `src/` or `tools/` reads a GGUF or a
   safetensors file; `engine_core.cpp`'s `load_model` reads only this project's own `S0L5` header + flat
   `PARAM_LAYOUT`-ordered float blob. WP4 must build the import path, not extend one.
3. **The mechanisms' math cores can already do the real geometry; only the engine binding cannot.**
   `gdn::Dims`, `qsa::Dims`, `gr::Dims` and `moe::Dims` all take every geometry field explicitly — which
   is precisely why WP1-3's fixture tests already run at the real `head_dim=256`/`head_k_dim=128` values
   and match the real reference to `1.4e-09`/`1.5e-11`/`0.0`. **The correctness of the math is not in
   question at real dims. The plumbing is.** This is the single most encouraging fact in this document
   and it is what makes WP4 a plumbing problem rather than a research problem.

---

## 1. The exact target `RunConfig`, every axis named

Cross-checked field by field against `docs/QWEN4_PREVIEW_REFERENCE.md`'s verified facts table and against
`config.json` values re-fetched independently by `docs/QSA.md` §0 and `docs/MOE.md` §0. **"Expressible
today" is a claim about `tools/configurator.cpp`'s `RunConfig` X-macro + the generated `sub0_corpus.hpp`,
verified against `include/sub0/layout.hpp` this pass — not an assumption.**

| Real `config.json` field | Value | This project's axis | Expressible today? |
|---|---:|---|---|
| `hidden_size` | 2560 | `D_MODEL` | **Yes** |
| `num_hidden_layers` | 48 | `N_LAYERS` | **Yes** |
| `num_attention_heads` | 24 | `N_HEADS` | **Yes** |
| `num_key_value_heads` | 2 | `N_KV_HEADS` | **Yes** |
| `head_dim` | **256** | `D_HEAD` | **NO — derived as `D_MODEL/N_HEADS` = 106** (§2, blocker A) |
| `vocab_size` | 248320 | `VOCAB` | **Yes** |
| `tie_word_embeddings` | false | `USE_TIED_EMBEDDINGS = false` | **Yes** |
| `attention_bias` | false | (this engine's attention is bias-free) | **Yes**, by construction |
| `max_position_embeddings` | 262144 | `SEQ_LEN` | Yes as an axis; **not affordable** (§4) — a first run picks a small `SEQ_LEN` |
| `full_attention_interval` | 4 | `GDN_FULL_ATTN_STRIDE = 4` | **Yes** — and `docs/QSA.md` §0 verified the real 48-entry `layer_types` array IS `gdn_schedule_for(4)`, element by element |
| `linear_num_key_heads` | 16 | (aliased to `N_KV_HEADS` = 2) | **NO** (§2, blocker B) |
| `linear_num_value_heads` | 48 | (aliased to `N_HEADS` = 24) | **NO** (§2, blocker B) |
| `linear_key_head_dim` | 128 | (aliased to `D_HEAD` = 106) | **NO** (§2, blocker B) |
| `linear_value_head_dim` | 128 | (aliased to `D_HEAD` = 106) | **NO** (§2, blocker B) |
| `linear_conv_kernel_dim` | 4 | `GDN_CONV_KERNEL` (fixed `constexpr int = 4`) | **Yes** — the constant already equals the real value |
| `hc_count` | 4 | `HC_COUNT` | **Yes** |
| `hc_lowrank` | 320 | `HC_LOWRANK` | **Yes** |
| `num_experts` | 512 | `NUM_EXPERTS` | **Yes** |
| `num_experts_per_tok` | 10 | `EXPERTS_PER_TOK` | **Yes** (`moe::TOPK_MAX` is 16, so 10 fits without touching it) |
| `moe_intermediate_size` | 640 | `D_FF` | **Yes** — and the real `shared_expert_intermediate_size` is also 640, so `docs/MOE.md` §3a's single-width reuse is real-model-faithful |
| `norm_topk_prob` | true (dataclass default) | hard-coded in `moe_math.hpp` | **Yes**, matches |
| `indexer_n_heads` / `_kv_heads` / `_head_dim` / `_budget` / `_compress_ratio` | 4 / 1 / 128 / 2048 / 4 | the five `QSA_INDEXER_*` axes | **Yes** — all five exist and take these exact values |
| `rope_theta` | 10,000,000 | `ROPE_THETA` | **Yes** |
| `partial_rotary_factor` | 0.25 (⇒ `rotary_dim` 64 of 256) | (engine `rotary_dim = D_HEAD`) | **NO** (§2, blocker C) |
| `rms_norm_eps` | 1e-6 | GR/QSA math cores hard-code 1e-6; `op_rmsnorm` hard-codes 1e-5 | Partially — §2, blocker D |
| `mrope_section` | [11,11,10] | — | **NO**; text-only relevance is an open question (§7 Q6) |
| `ple_layer_ids` | [2] (n-gram at decoder layer 1) | `NGRAM_MAX_N` etc. | **Deliberately OFF** (§5) |
| `mtp` block | 4B head | — | **Deliberately OUT** (§5) |
| `vision_config` | 27-layer ViT | — | **Deliberately OUT** — `docs/QWEN4_PREVIEW_REFERENCE.md`'s own scope decision, unchanged |

**Everything else the target build needs, and already has**: `USE_GATED_FFN` is irrelevant when
`USE_MOE` is on (MoE replaces the FFN block entirely, `docs/MOE.md` §3d); `USE_QK_NORM` must be `true`
(the real model has per-head q/k norms — but note QSA layers use their own `QsaQNorm`/`QsaKNorm`, so this
flag affects only the delta arithmetic, not the QSA path); `POS_ENCODING = Rope`; `USE_TERNARY = false`.

---

## 2. The four expressibility blockers, and what each actually requires

These are the concrete content of WP4b. Each is an `AGENTS.md` §10-class change (a shared width/semantic
whose consumers the diff will not show you), so each gets its own consumer sweep.

**Blocker A — `D_HEAD = D_MODEL / N_HEADS` is baked into `sub0_config.hpp` itself.**
The real model has `n_heads * head_dim = 6144 ≠ hidden_size = 2560`. `docs/QSA.md` §2b.1 already named
this as "WP4's problem" and did the right thing by making `qsa::Dims` carry `head_dim` independently.
What has to change: a new `--head-dim` axis (0 = derive, preserving every existing build byte-identically
— the same "0 means the old behaviour" idiom `N_KV_HEADS`/`DEPTH_ATTN_STRIDE` already use), and then a
repo-wide sweep of everything that assumes `N_HEADS * D_HEAD == D_MODEL`: `layout.hpp`'s `D_KV`/
`QKV_STRIDE`/`QK_PRE_STRIDE`, `memplan.hpp`'s `d_kv()`/`qkv_stride()`/`param_floats()` (which re-derive
the same quantities independently and must stay in lock-step per their own comment),
`backend_cpu.cpp`'s attention/KV-cache code, `backend_cuda.cu`'s `layer_slots()`/`build_qkv_weights()`,
and the `Wo`/`o_proj` shape (`[n_heads*head_dim, d_model]`, no longer square). **Classification**
(`layout.hpp`'s own three-way rule): shape-changing ⇒ `PARAM_FLOATS` discriminates it, no fingerprint bit
needed. **This is the largest single piece of engineering in WP4** and is why §6 gives it its own stage.

**Blocker B — GDN's head geometry is aliased to the attention head geometry.**
`GDN_DIMS{D_MODEL, N_KV_HEADS, N_HEADS, D_HEAD, D_HEAD, GDN_CONV_KERNEL}` (`layout.hpp` ~line 232) forces
`linear_num_key_heads == N_KV_HEADS`, `linear_num_value_heads == N_HEADS`, and both linear head dims to
`D_HEAD`. The real model wants 16 / 48 / 128 / 128 against attention's 2 / 24 / 256. These are genuinely
independent axes in the reference and must become independent here: four new flags
(`--gdn-key-heads`, `--gdn-value-heads`, `--gdn-key-head-dim`, `--gdn-value-head-dim`, each 0 = "alias to
the attention axis", preserving today's builds exactly). `gdn_math.hpp`'s `Dims` already takes all four
explicitly, so **no math changes** — this is purely a configurator/`layout.hpp` binding change plus
`make_param_layout()`'s nine GDN tensor shapes. Note the real ratio `num_v_heads / num_k_heads = 3` is
what drives `repeat_interleave` in the reference; the engine's GDN already implements that, it just
cannot currently be told the ratio is 3 rather than 12.

**Blocker C — `partial_rotary_factor`.** The engine's `rotary_dim` is `D_HEAD`; the real model rotates
only the first 64 of 256 channels. `qsa_math.hpp` already takes `rotary_dim` as an explicit `Dims` field
(and `docs/QSA.md` §10 records a real out-of-bounds bug found by getting this relationship wrong), so
again the math is ready and only the axis is missing: one `--rotary-dim` flag (0 = `D_HEAD`).
**Classification: rule 2 — computation-changing, shape-neutral.** It changes NO tensor shape, so it MUST
join `ARCH_FINGERPRINT2` or a checkpoint loads silently and computes the wrong attention — exactly the
`ROPE_THETA`/`LOOP_MIDDLE_LAYERS` hazard that word exists for. `ARCH_FINGERPRINT2` has bits `[63:40]`
free after QSA's own additions (`docs/QSA.md` §3a's map).

**Blocker D — `Ln1`/`Ln2` do not exist in the real model, and `op_rmsnorm`'s eps is 1e-5, not 1e-6.**
`docs/GATED_RESIDUAL.md` §2 recorded this as a *deliberate, documented* Stage-1 simplification: the real
`Qwen4ExpTextDecoderLayer` has NO `input_layernorm`/`post_attention_layernorm` — GR's own grouped
`hc_norm` is the pre-block norm, and the mixer reads GR's `mixed_input` directly. That doc explicitly
deferred removing `Ln1`/`Ln2` to "a future pass targeting full real-weight decoder-layer parity", i.e.
**this one**. Concretely: under `USE_GATED_RESIDUAL`, `make_param_layout()` must stop emitting `Ln1`/`Ln2`
and `Model::forward`/`forward_one` must feed the mixer from `mixed_input` un-normed. This is
shape-changing (`PARAM_FLOATS` discriminates), off-by-default-safe (it only fires when GR is on, which no
existing build has), and — importantly — **it is a correctness requirement, not a tidiness one**: leaving
`Ln1` in place applies a norm the real model does not, so no amount of correct weight transplanting would
reproduce the real output.

---

## 3. `gguf.hpp` dequant gap — current status, verified this pass

**Verified by reading `include/sub0/gguf.hpp` (not assumed from the feasibility doc's own age):**
`enum class TensorType : std::uint32_t { F32 = 0, F16 = 1, Q8_0 = 8 };` — unchanged since
`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §1d recorded it. `tensor_byte_size()` returns its documented
"unsupported" signal `0` for `BF16` (30), `IQ4_NL` (20) and every K-quant (`Q4_K`/`Q5_K`/`Q6_K` =
12/13/14). `dequantize_tensor()` dispatches F32 (copy) / F16 (widen) / Q8_0 (`dequantize_q8_0`) and
returns false otherwise. **Nothing in `src/` or `tools/` includes this header at all** — its only
repo-wide consumer is `tests/gguf_tests.cpp`.

### 3a. Minimum viable tier for a first real run

`docs/QWEN4_DEPLOYMENT_FEASIBILITY.md`'s own findings drive this, and its numbers are reused, not
re-derived:

- Every GGUF tier exceeds 63GB RAM (§1b). The smallest, `UD-IQ1_S`, is 72.55GB.
- The n-gram/PLE table inside it is **28.80GB of that (39.7%)**, stored as `IQ4_NL` — and §5 keeps that
  table out of the first run entirely.
- **Excluding it leaves ≈43.75GB** for backbone + attention + MoE, which
  `docs/QWEN4_MEMORY_ORCHESTRATION.md` §3b measured against a **corrected realistic RAM headroom of
  ~40-45 GiB** and called "plausible, still tight, not comfortably clear of the ceiling."

So the minimum viable dequant tier is decided by which tier's *non-table* remainder fits, and that is
`UD-IQ1_S` or `UD-IQ1_M`. **Both store the table as `IQ4_NL` in a SHARED shard** (§1d), so excluding it
needs a byte-range-aware skip, not a whole-file skip — computable with zero new code from
`TensorInfo::offset`, but only once `tensor_byte_size()` knows `IQ4_NL`'s length rule.

**Minimum dequant work for a first real run, in dependency order (SUPERSEDED — see §3a-bis below for the
real census, once the file existed locally to read directly instead of guessing from an HTTP header
probe):**

| Format | Needed for | Size rule | Decode |
|---|---|---|---|
| `BF16` (30) | the **safetensors** path (§3b) — the real checkpoint's native dtype | trivial: `n × 2`, same shape as the existing `F16` case | trivial: widen the 16 bits into the high half of an f32 (this project already has `f16_to_f32`; bf16 is strictly simpler) |
| `IQ4_NL` (20) | locating/skipping the n-gram table in a shared shard at every UD tier ≤ Q4_K_XL | 18 bytes per 32 elements (§1c) | **not needed for the first run** if the table is skipped, only the SIZE rule is |
| K-quants (`Q4_K`/`Q5_K`/`Q6_K`) | only if a GGUF path is chosen over safetensors | publicly documented per-format block layouts | the real work; deferrable |

**Recommendation, and it is a real fork in the road**: **prefer the safetensors path over GGUF for the
first run**, and add only `BF16` to `gguf.hpp` (a one-line size rule + a small decode, mirroring `F16`).
Reasons: (a) the checkpoint's native form is bf16 safetensors, so no quantization fidelity question
enters the comparison at all — a logits diff against llama.cpp is then a diff of *this engine's* numerics,
not of two different quantizations; (b) this project has validated surgical HTTP-Range extraction against
those exact safetensors shards **five times** (GDN, n-gram, GR, MoE, QSA) and has zero validated GGUF
*read* path beyond header probes; (c) `model.safetensors.index.json` gives a per-tensor shard map, so
per-tensor selective acquisition is already solved. **The cost of this recommendation**: bf16 is
2 bytes/param, so the backbone+MoE at bf16 is ≈231 GiB — far past RAM, meaning the safetensors path
*requires* the expert-offload machinery of §4 to work before a full-model run is possible, whereas
`UD-IQ1_S` would nearly fit. §7 Q2 puts this trade to the user rather than settling it here.

### 3a-bis. The real census (2026-09-04) — read from the actual downloaded file, not a header probe

**§7 Q2 is now decided: the user chose GGUF/`UD-IQ1_S`** (already fully downloaded, all three shards,
72.55GB, to `D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S\`). With the real file locally available,
a throwaway tool built against this project's own `gguf::Reader` (reading just the header — 4MiB was
enough for every shard) gives the ACTUAL per-tensor type breakdown, which is materially different from
what §3/§3a assumed from the earlier HTTP-header-probe summary:

**Every raw GGML type id actually present, across shards 2+3** (shard 1, 10.9MB, is pure metadata/vocab,
zero tensors): `F32`(0), `Q8_0`(8), `Q4_K`(12), `Q5_K`(13), `Q6_K`(14), `IQ2_XXS`(16), `IQ1_S`(19),
`IQ4_NL`(20), `BF16`(30) — **seven formats beyond the three `gguf.hpp` already supports**, not the one or
two either §3a's table or the earlier deployment-feasibility summary implied.

**Per-tensor-role assignment** (from a full dump of `blk.0.*` and `blk.23.*`, a GDN and a QSA layer
respectively, plus every non-`blk.*` tensor):

| Real tensor role | GGUF name(s) | Type | Notes |
|---|---|---|---|
| Token/output embeddings | `token_embd.weight`, `output.weight` | `Q4_K` | both `[2560, 248320]`, untied confirmed |
| GDN `in_proj_qkv` | `blk.N.attn_qkv.weight` | `Q5_K` | `[2560, 10240]` — GGUF's own naming calls GDN's fused QKV projection `attn_qkv`, not an SSM-prefixed name |
| GDN `in_proj_z` (gate) | `blk.N.attn_gate.weight` | `Q5_K` | `[2560, 6144]` |
| GDN `A_log`/`dt_bias`/conv/norm | `blk.N.ssm_a`, `ssm_dt.bias`, `ssm_alpha.weight`, `ssm_beta.weight`, `ssm_conv1d.weight`, `ssm_norm.weight` | `F32` (all) | full precision — these are small vectors, not a real cost |
| GDN `out_proj` | `blk.N.ssm_out.weight` | `Q6_K` | `[6144, 2560]` |
| QSA `q_proj` (query+gate, fused) | `blk.N.attn_q.weight` | `Q5_K` | `[2560, 12288]` — **confirms the real double-width q_proj** (`24×256×2`) directly from the checkpoint, independent of `docs/QSA.md`'s own HF-source derivation |
| QSA `k_proj`/`v_proj` | `blk.N.attn_k.weight`, `attn_v.weight` | `Q5_K` | `[2560, 512]` each |
| QSA `o_proj` | `blk.N.attn_output.weight` | `Q5_K` | `[6144, 2560]` |
| QSA `q_norm`/`k_norm` | `blk.N.attn_q_norm.weight`/`attn_k_norm.weight` | `F32` | `[256]` |
| **QSA indexer** | `blk.N.indexer.q_proj.weight` **and** `indexer.k_proj.weight` (SEPARATE tensors) | `BF16` | `[2560,512]` and `[2560,128]` — **GGUF stores the indexer's fused `index_qk_proj` as TWO tensors**, not one; a transplant into this engine's single `QsaIdxQkProj` slot must concatenate them, in the same n/kv head order `docs/QSA.md` §1a's asymmetric split already documents |
| Gated Residual (all 4 tensors, both instances) | `blk.N.hc_{attn,ffn}_{norm,down,up,inject}.weight` | `F32` (norm/inject) or `Q8_0` (down/up) | matches `docs/GATED_RESIDUAL.md` §1a's tensor set exactly by name pattern |
| MoE router | `blk.N.ffn_gate_inp.weight` | `F32` | `[2560, 512]` — full precision, consistent with §2a(v2)'s "dense-read, always-resident" finding |
| MoE routed gate/up (SEPARATE, not fused `gate_up_proj`) | `blk.N.ffn_gate_exps.weight`, `ffn_up_exps.weight` | `IQ1_S` **or** `IQ2_XXS`, varying **PER LAYER** (e.g. layers 0/3/5 use `IQ1_S`, layers 1/2/4 use `IQ2_XXS`, observed directly, not inferred) | `[2560,640,512]` each — unsloth's own "Dynamic" (`UD`) per-layer importance-based mixed quantization; a transplant CANNOT assume one format per tensor role, it must read each tensor's own `type_raw` |
| MoE routed down | `blk.N.ffn_down_exps.weight` | `IQ4_NL` | `[640,2560,512]` — consistently this format across every layer checked |
| MoE shared expert gate/up | `blk.N.ffn_gate_shexp.weight`, `ffn_up_shexp.weight` | `Q5_K` | `[2560,640]` |
| MoE shared expert down | `blk.N.ffn_down_shexp.weight` | `Q8_0` | `[640,2560]` |
| MoE shared-expert gate scalar | `blk.N.ffn_gate_inp_shexp.weight` | `F32` | `[2560]` — matches `MoeSharedGateProj` |
| N-gram/PLE table (excluded from the run, §5) | `per_layer_token_embd.weight` | `IQ4_NL` | `[160, 320001536]` — confirmed exactly where §1c/§3a said it would be |

**Two structural findings that change the transplant design, not just the dequant checklist**:

1. **GGUF's tensor granularity is NOT the HF/safetensors granularity `docs/*.md` §4b's subtlety table was
   built against.** The indexer's fused `index_qk_proj` and MoE's fused `gate_up_proj` both arrive as TWO
   separate GGUF tensors each. The transplant pipeline must concatenate (indexer: q-half then k-half,
   matching the asymmetric split `docs/QSA.md` §1a already documents; MoE: gate-half then up-half, per
   `docs/MOE.md` §1a's own chunking order) — a real per-tensor-pair reconstruction step, not just a
   name-lookup + transpose.
2. **Per-layer mixed quantization means the dequant dispatch must be driven by each tensor's own
   `type_raw`, never assumed from its name/role.** `blk.0.ffn_gate_exps.weight` and
   `blk.1.ffn_gate_exps.weight` are the same logical tensor at consecutive layers and are quantized
   DIFFERENTLY. `gguf::Reader::tensor_bytes()`/`dequantize_tensor()`'s existing per-tensor (not
   per-schema) dispatch already has the right shape for this — it just needs the new format cases added.

**Revised minimum dequant work — seven formats, not one or two**: `Q4_K`(12), `Q5_K`(13), `Q6_K`(14),
`IQ1_S`(19), `IQ2_XXS`(16), `IQ4_NL`(20), `BF16`(30). All are publicly documented ggml block formats
(same reference this doc already cites for Q8_0's own block layout). This is a materially larger `gguf.hpp`
extension than either this section or the deployment-feasibility doc's own earlier estimate implied even
just for the 4-layer sub-stack (§7 Q3) — layers 0-3 alone already exercise both `IQ1_S` and `IQ2_XXS`
(layers 0/3 vs layers 1/2), plus every other format via the always-present embeddings/attention/GDN/GR
tensors. **User re-confirmed the GGUF path with this real cost made explicit** (2026-09-04, via
`AskUserQuestion`) rather than the recommendation below being silently acted on.

### 3b. What a safetensors reader would need that `gguf.hpp` does not provide

No such reader exists in `include/` today (the five extractions were all done with throwaway Python).
A minimal one is: 8-byte little-endian header length, a JSON header (simdjson is already this project's
reader — `[[json-reads-forward-ondemand-not-dom-facade]]`), then per-tensor `dtype`/`shape`/
`data_offsets`. This is *less* code than `gguf.hpp` already is. **Scope it as a new header, do not
generalize `gguf.hpp` into a two-format reader** — they share nothing but the concept.

---

## 4. The weight-transplant pipeline

Each mechanism doc already resolved its own axis convention in isolation. **Nothing has ever assembled
them into one end-to-end mapping across all 48 layers, and that assembly is where the bugs will be** —
this project's own `[[independent-reimplementation-catches-identity-swap-bugs]]` lesson, and WP3's real
rope out-of-bounds bug, both argue for treating this as its own gated stage rather than a scripting
afterthought.

### 4a. The one convention that applies everywhere

PyTorch `nn.Linear` stores `[out_features, in_features]`. This project stores `[rows=in, cols=out]`.
**Every 2-D weight transposes.** This is not a new finding — `AGENTS.md` §5 records it as the exact trap
that would have made the Muon port silently backwards — but it is the single highest-volume operation in
the pipeline (74,899 tensors) and the one most likely to be right for 74,898 of them.

### 4b. The per-mechanism subtleties already resolved, gathered in one place

| Mechanism | Subtlety | Source |
|---|---|---|
| GR | `hc_norm` is `[1, hc_count*hidden_size]` (one gain per (stream,channel)), **not** `[1, hidden_size]`; and its gain convention is `(1 + w)` with `w` initialised to **zero**, unlike this engine's `Ln1`/`QNorm` which use `w` directly initialised to one | `docs/GATED_RESIDUAL.md` §1a |
| GR | there are **`2 × 48 + 1 = 97` instances**, not 96 — the model-level exit instance has NO `block_inject_weight` (`use_combine=False`) | `docs/GATED_RESIDUAL.md` §1c |
| MoE | experts are stored as **3-D tensors** `gate_up_proj[num_experts, 2*intermediate, hidden]` and `down_proj[num_experts, hidden, intermediate]`; `gate_up_proj` must be **chunked into gate|up halves** and each transposed | `docs/MOE.md` §1a |
| MoE | the shared expert is a separate `Qwen4ExpTextMLP` with its own `gate_proj`/`up_proj`/`down_proj`, plus a `shared_expert_gate` `Linear(hidden, 1)` — **not** expert #0 | `docs/MOE.md` §1a |
| QSA | `q_proj` is double-width and chunks **per head** (`[..., -1, head_dim*2]` then chunk on the last axis), so slot `h` holds `[query_h | gate_h]`. **Splitting the flat row down the middle is wrong for every head but head 0** and produces plausible-looking output | `docs/QSA.md` §1b, §2b.4 |
| QSA | the indexer's `index_qk_proj` is ONE fused linear split **asymmetrically** into `n_heads*head_dim` query and `kv_heads*head_dim` key parts | `docs/QSA.md` §1a |
| GDN | `in_proj_qkv` rows are `[query(key_dim) | key(key_dim) | value(value_dim)]` in that order; `conv1d.weight` is `[conv_dim, 1, kernel]` depthwise | `docs/GATED_DELTANET.md`, `docs/QWEN4_PREVIEW_REFERENCE.md` fixture table |
| GDN | `dt_bias` and `A_log` are **separate, same-shaped `[num_v_heads]` vectors** — the real bug WP-GDN Stage 3 found was these two passed in swapped argument positions, invisible to both the fixture test and the gradient check | `docs/GATED_DELTANET.md` §6 |
| all | tensor names are `model.language_model.layers.{i}.{linear_attn|self_attn|mlp|attn_hyper_connection|mlp_hyper_connection}.*` | `model.safetensors.index.json` |

### 4c. The correctness gate this stage needs (and it is not "it loaded")

**A shape-only check is insufficient** — an identity-swap of two same-shaped tensors (GDN's own
`dt_bias`/`A_log` precedent) passes every shape assertion. Proposed gate, in increasing strength:

1. **Total-count reconciliation.** The transplant must consume exactly `NUM_PARAMS` destination tensors
   and exactly the real checkpoint's 1,658 source tensors minus the deliberately-excluded sets (n-gram
   shards, MTP, vision), with **zero unmatched on either side**. An unmatched-source list is the cheapest
   detector of a whole mechanism silently skipped.
2. **Per-tensor statistical sanity**, the same technique `AGENTS.md` §9 records for the original GGUF
   validation: per-tensor mean/std/min/max against the source, computed on both sides of the transpose.
   Catches a transpose applied twice or not at all on non-square tensors, and catches a wrong slice.
3. **Layer-0 end-to-end fixture replay.** The strongest available check and the reason WP1-3's fixtures
   were built: run the transplanted layer 0 (GDN + GR + MoE) on the *existing* real-weight fixture inputs
   and compare against the *existing* real outputs in `tests/fixtures/qwen4_preview/`. These fixtures are
   sliced-down, so this requires the sliced config, not the full one — but it validates the *mapping*,
   which is the thing under test, without needing the full model resident.
4. **Layer 3 (first QSA layer) replay**, same technique against `qsa_layer3_small_*`.

---

## 5. What is explicitly OUT of the first real run

**Confirmed out, per the original plan and `docs/QWEN4_MEMORY_ORCHESTRATION.md` §3c — no change proposed:**

- **The n-gram/PLE table (51.2B params, 28.80-102.4GB depending on tier).** It is one additive signal at
  a single decoder layer (`ple_layer_ids=[2]` ⇒ 0-indexed layer 1). Omitting it is a *reduced-scope test
  configuration*, not a correctness compromise for the other four mechanisms — **but it does mean the
  logits comparison in §6 is NOT a comparison of the same model.** llama.cpp will include
  `per_layer_token_embd.weight`; this engine will not. **The comparison harness must therefore either
  (a) disable the PLE contribution on the llama.cpp side too, or (b) compare only up to and including
  decoder layer 0's output, before the injection point.** This is a real methodological constraint the
  original plan did not spell out; §6's WP4d builds around (b), which needs no llama.cpp modification.
- **The interface seam it must NOT block on**: `docs/NGRAM_EMBEDDING.md` §7's `ngram_tab[e]` "thin client
  issuing `resolve_into` calls instead of a raw parameter pointer", against `docs/SUB0FIRN_SPEC.md` §3's
  `resolve_into(table_handle, row_indices[], dest_buffer)` contract. **Nothing in WP4a-e should introduce
  a design that assumes the table is absent** — it is *not present in this build*, which is a different
  claim from *cannot be present*. Concretely: keep the PLE injection point as a real, named,
  `if constexpr`-disabled site in the layer loop rather than eliding it from the layer's structure.
- **The MTP head (4B params)** — same reasoning, per `docs/QWEN4_MEMORY_ORCHESTRATION.md` §2g.
- **The vision tower** — `docs/QWEN4_PREVIEW_REFERENCE.md`'s standing scope decision.
- **Any backward pass.** All four mechanisms `abort()` loudly in `backward_node` today, verified by
  actually hitting it in each of WP1-3. WP4 is forward-only; nothing here changes that.
- **CUDA.** `backend_cuda.cu` carries `static_assert(!USE_GATED_RESIDUAL)`, `static_assert(!USE_MOE)` and
  `static_assert(!USE_QSA)`. WP4 is a **CPU-only** work package. (`docs/QSA.md` §10 also notes
  `backend_cuda.cu`'s `layer_slots()`/`build_qkv_weights()` mirror `GDN_SCHEDULE`'s param-layout
  arithmetic and would break on a QSA layer — lifting any guard must update those sites.)

---

## 6. Staged sub-plan

Each stage is independently mergeable and independently reviewable, in the WP1-3 mould: it lands on its
own branch, has its own correctness gate, and confirms the neutral-setting default build is
assertion-and-hash-identical before merge (`AGENTS.md` §4/§7). **Strict dependency order** — every stage
depends on all previous ones; none can be parallelized, for the same reason WP1-3 could not (they all
touch `layout.hpp`/`backend_cpu.cpp`/`tools/configurator.cpp`).

### WP4a — Design doc + acquisition decision (no code)

**This document is WP4a's first half.** Its second half is the user decisions in §7, which gate
everything downstream. Deliverable: §7 answered, and — if a real acquisition is chosen — the
**`docs/QWEN4_MEMORY_ORCHESTRATION.md` §6c item 7 shard-header census** run first (131 header-only Range
requests, cheap), because it independently confirms or refutes §0's shape-gap arithmetic from the
checkpoint side before any larger commitment.

**Gate**: user sign-off on §7; a tensor-by-tensor manifest of the real checkpoint that reconciles to a
parameter total.

### WP4b — Shape expressibility: the four blockers (§2)

The largest engineering stage. Suggested internal order (each a separate commit, each verified
neutral-identical): **B (GDN head axes)** first — it is the most mechanical and `gdn::Dims` already takes
every field; then **C (`--rotary-dim`)** — small, and its `ARCH_FINGERPRINT2` classification is already
worked out; then **D (`Ln1`/`Ln2` removal under GR)** — self-contained to the GR branch; then **A
(`--head-dim`)** last, because it has the widest blast radius and benefits from the others being settled.

**Gate**: (i) at every neutral setting, the default `sub0_tests` suite is assertion- **and hash**-identical
at all three of this thread's standard shapes (d96 L8 H2, d132 L11 H4 kv2, d196 L11 H7); (ii) all four
existing real-weight fixture tests still pass unchanged; (iii) **a new two-scale test asserting
`make_param_layout()` at Qwen4's real axes produces exactly the real model's per-tensor shapes** — i.e.
the §0/§2h shape gap is closed and provably so, checkable at compile time without allocating anything;
(iv) `memplan.hpp`'s `param_floats()` still equals `PARAM_FLOATS` (its own existing lock-step assertion)
under the new axes.

**Note on (iii)**: this is achievable *today* as a `static_assert`-style consteval check, because
`make_param_layout()` is `consteval` and evaluating it at the real axes costs ~4 seconds of compile time
and zero runtime memory — demonstrated this pass. **It is the cheapest possible high-value gate in the
whole of WP4** and should be written before any of §2's changes, so it fails first and passes last.

#### WP4b — EXECUTED (branch `feature/wp4b-shape-blockers`). Results, recorded rather than summarised

All four blockers landed, one commit each, in the order §6 proposed (B, C, D, A).

**The shape gap is closed exactly.** `tests/qwen4_real_shape_tests.cpp` compiles this engine's own
`make_param_layout()` against `tests/qwen4_real_axes/sub0_config.hpp` (the real axes, hand-written) and
`static_assert`s the result. Measured:

| Quantity | Before WP4b (§0/§2h) | After WP4b | Real model |
|---|---:|---:|---:|
| `PARAM_FLOATS` | 124,027,786,776 | **125,711,064,960** | 125,711,064,960 |
| `NUM_PARAMS` | 74,899 | **74,803** | 74,803 |
| GDN subtotal (36 layers) | 751,723,560 | **2,086,510,464** | 2,086,510,464 |
| QSA subtotal (12 layers) | 268,621,296 | **617,358,336** | 617,358,336 |

**The tensor COUNT is 74,803, not §0's 74,899 — a 96-tensor drop that is blocker D working, not a
regression.** 96 == `2 × 48`: the `Ln1`/`Ln2` pair per layer that the real `Qwen4ExpTextDecoderLayer`
does not have. The pre-WP4b 74,899 was measured with them still present. The float total is unaffected
by which way that is counted only because the removal (245,760 floats) is exactly offset elsewhere —
it is not: `PARAM_FLOATS` now matches the real model *because* those norms are gone, and the census in
that test derives the total independently, tensor by tensor, to prove the match is not coincidental.

**Two things the work order did not anticipate, both real:**

1. **`QSA_INDEXER_HEAD_DIM >= D_HEAD` would have rejected the real model.** `layout.hpp` asserted the
   indexer's head width against `D_HEAD`, but the quantity `rope_apply_row` actually indexes is
   `rotary_dim`. The real model has `rotary_dim 64 <= indexer_head_dim 128 < head_dim 256`, so the old
   form was not merely loose — it refused the real configuration outright. Relaxing it to `ROTARY_DIM`
   is a prerequisite, not a loosening.
2. **`forward_one` spelled the head width as `C / H`.** Caught by the existing forward-vs-`forward_one`
   parity check the first time a non-square `--head-dim` build was run (1.71 relative error, plus an
   all-zero decode trace). Exactly `AGENTS.md` §10's class: a derived width re-spelled locally instead
   of read from its one source of truth. The neutral suite could never have found it.

**Gate results** — neutral (all new axes at their derived defaults) `sub0_tests` is hash-identical at all
three standard shapes; `sub0_frontend_tests` (which carries all four real-weight fixture tests) is
byte-identical at 115,424 / 197; a genuinely non-square build (`--head-dim 64` at d96 H2, i.e. `D_Q` 128
vs `D_MODEL` 96) passes the **entire** suite including the finite-difference gradient check; a GR-ON
build passes `[layout]` and runs real forward passes through `Model::forward`; a neutral CUDA build
passes all 196 cases, and a GDN-ON CUDA build's failure set is byte-identical to its own pre-WP4b
baseline (19 pre-existing "no CUDA training path for GDN" refusals).

**Deliberately still open, named here so WP4c/WP4d do not rediscover them:**

- **CUDA refuses an independent `--head-dim`** (`static_assert(N_HEADS * D_HEAD == D_MODEL)` in
  `backend_cuda.cu`, alongside the existing GR/MoE/QSA refusals). That backend bakes `D_MODEL` as the
  attention-output width in the `att`/`proj`/`datt` scratch, `dwqkv`, `build_qkv_act_kernel` and every
  tiled attention kernel. WP4 is CPU-only (§5), so this is a refusal, not a silent wrong answer.
- **`op_rmsnorm`/`op_qknorm` still use `eps = 1e-5`,** where every real `Qwen4ExpTextRMSNorm` uses
  `1e-6`. Blocker D removed the GR path's exposure to it (`Ln1`/`Ln2` are gone), but `LnF` — the
  model-level final norm, whose real counterpart is also `1e-6` — still routes through it. Changing a
  shared, always-present op's eps is its own scoped change with its own neutral-identity gate.
- **`ARCH_FINGERPRINT2` is now FULL** (all 64 bits assigned, after `gdn_key_heads` at [47:40] and
  `partial_rotary_dim` at [63:48]). The next computation-changing, shape-neutral axis needs a third
  additive word.
- **`memplan::param_floats()` models only the dense softmax-attention architecture** — no GDN/GR/MoE/QSA
  term. Its lock-step with `PARAM_FLOATS` is now a compile-time `static_assert` on every build (it was a
  CUDA-only unit test before, so a CPU build could drift silently), guarded to the configurations it
  actually models; at the real axes its `d_q`/`d_kv`/`qkv_stride`/`qk_pre_stride` are checked against
  `layout.hpp`'s directly, which is the part blocker A moved.
- **A real-axes build needs `-fconstexpr-steps` raised** (~74.8k tensors walked several times at compile
  time). Set on the shape-test target; a real engine build at these axes will need the same.

### WP4c — Acquisition + transplant pipeline (§3, §4)

A safetensors reader (§3b) or a `BF16`-extended `gguf.hpp` (§3a), plus a name-mapping + transpose pass
producing this project's own flat `PARAM_LAYOUT`-ordered blob. Deliberately **offline**: it writes a
model file; it does not stream at inference time.

**Gate**: §4c's four-level check, with level 3/4 (fixture replay through the transplant path) as the
hard requirement. Plus: acquired bytes accounted for exactly (an unmatched-tensor list on both sides).

**Scoping input from `docs/QWEN4_MEMORY_ORCHESTRATION.md` §3d(v2)**: at `T=4096` a single GDN layer's
prefill scratch is **417.5 MiB**, growing linearly (≈1.63 GiB at T=16,384; ≈26 GiB at native 262,144).
**Prefill must be chunked, and the chunk length is a first-class decision for this stage**, not something
to discover at run time.

#### WP4c — EXECUTED (branch `feature/wp4c-transplant`). Results, recorded rather than summarised

**The artifact exists and verifies.** `tools/sub0llm-transplant.cpp`, compiled against
`tests/qwen4_real_axes/sub0_config.hpp` with `-DSUB0_QWEN4_LAYERS=4`, produced
`D:\ModelWeights\Sub0Llm-Qwen4-sub4\qwen4_sub4.bin`: **46,590,469,832 bytes** (48-byte `S0L5` header +
**11,647,617,440 floats = 46,590,469,760 bytes** + the three 8-byte trailers), from **4,418,181,760
bytes** of encoded GGUF read out of the real `UD-IQ1_S` shards. That is a **10.5x expansion**, which is
the honest cost of this engine having no quantized inference path: `UD-IQ1_S`'s whole point is that the
weights are 1-2 bits, and every one of them lands as an f32 here.

| Destination region | Tensors | Floats | Bytes (f32) |
|---|---:|---:|---:|
| `tok_emb` | 1 | 635,699,200 | 2,542,796,800 |
| layers 0-2 (GDN), each | 1,558 | 2,593,979,104 | 10,375,916,416 |
| layer 3 (QSA) | 1,559 | 2,587,467,008 | 10,349,868,032 |
| GR exit + `ln_f` + `lm_head` + `lm_bias` | 6 | 642,513,920 | 2,570,055,680 |
| **total** | **6,240** | **11,647,617,440** | **46,590,469,760** |

MoE dominates completely: 2,522,810,880 of each layer's 2,593,979,104 floats (**97.3%**) are the 512
routed experts plus the shared expert. The four mechanisms' own weights are the remaining 2.7%.

**The four-level gate, with the actual numbers:**

| Level | Check | Result |
|---|---|---|
| 1 | total-count reconciliation | 6,240 / 6,240 destinations filled, 11,647,617,440 / 11,647,617,440 floats; **2 synthesized** (`LnF`, `LmBias` — see below); **7 unmatched in-scope sources**, every one a deliberately-excluded PLE tensor |
| 2 | per-tensor statistics across the reshape | **6,238 tensors checked, 0 mismatches** (count and extrema compared EXACTLY, mean/std to 1e-6 relative) |
| 3 | layer-0 (GDN) fixture replay through the transplant | max \|transplanted − real reference\| = **4.37e-11** |
| 4 | layer-3 (QSA) fixture replay through the transplant | max \|transplanted − real reference\| = **1.86e-09** |

Levels 3 and 4 match what `gdn_qwen4_fixture_tests.cpp` and `qsa_qwen4_fixture_tests.cpp` already
record for the same fixtures, i.e. **the transplant path contributes nothing** — which is the claim.
They are real replays, not shape checks: each re-encodes the fixture as a GGUF file under the REAL
model's own tensor names and in GGUF's byte order, which forces the test to perform the INVERSE of
each granularity change before the transplant undoes it.

**Level 2 provably cannot see an identity swap**, so each replay carries a mutation check. Measured:
an `ssm_alpha`/`ssm_beta` swap moves layer 0's output by **4.1e-8**; a down-the-middle `q|gate` split
moves layer 3's by **8.5e-3**; a swapped indexer concat order moves it by **7.4e-3** while being
*statistically identical* to the correct one. The first of those is worth flagging: 4.1e-8 is ~950x the
reference agreement, so it IS detectable — but it is a weak signal, and A_log/dt_bias/alpha/beta all
feed gates through `softplus`/`sigmoid`, which is exactly why the original dt_bias/A_log swap survived
both a fixture test and a gradient check.

**A fifth check, beyond §4c**: `--verify` re-runs the entire pipeline against the written file and
compares every destination **bit-for-bit**. Result: **0 mismatches of 6,240**. Levels 1-4 validate what
the tool computes; only this validates what landed on disk.

**And the independent shape check** (this stage's item 4): the tool `static_assert`s `NUM_PARAMS`,
`PARAM_FLOATS` and `PARAM_LAYOUT[6234].off` against `tests/qwen4_real_axes/sub4_prefix.hpp`, and
`tests/qwen4_real_shape_tests.cpp` asserts the SAME three literals against the real **48-layer**
layout. The two layouts cannot meet in one translation unit (ODR), so this is how "the 4-layer artifact
is layers 0-3 of the real model" becomes a compile-time claim: both builds agree that those four layers
occupy exactly **11,005,103,520** floats, and the 48-layer build additionally confirms the boundary
tensor is layer 4's leading `GrHcNorm`.

**What the real file said that this document had wrong.** Every item below is a correction to §3a-bis's
own table, found by reading the file rather than trusting the table:

1. **MoE needs no pair reconstruction at all.** §3a-bis called `ffn_gate_exps`/`ffn_up_exps` a split of
   a fused `gate_up_proj` requiring concatenation. That fusion is the *HF/safetensors* granularity —
   **this engine already stores `MoeGate` and `MoeUp` as separate `PARAM_LAYOUT` tensors**, so GGUF's
   granularity and the destination's coincide exactly. What is actually needed is the 3-D **expert
   slice**: one source tensor supplies 512 destinations.
2. **QSA's `attn_q` is a SPLIT, not a concat — and per head.** §3a-bis listed only the indexer as a
   granularity mismatch. `blk.N.attn_q.weight` is `[2560, 12288]`, the fused query+gate, and this
   engine stores `QsaQProj`/`QsaGateProj` separately, so the transplant must *split* it — with head
   *h*'s rows being `[query_h | gate_h]` adjacent, per `docs/QSA.md` §1b's `.view(..., head_dim*2)`
   then `chunk`. A down-the-middle split is right for head 0 and wrong for all 23 others.
3. **There is no final norm in the file.** `output_norm.weight` does not exist under that or any other
   name — the only non-`blk.` tensors in the whole 1,224-tensor set are `output.weight`,
   `output_hc_{down,norm,up}.weight`, `per_layer_token_embd.weight` and `token_embd.weight`. §2 blocker
   D assumed a real `LnF` counterpart existed and merely had the wrong `eps`. It does not exist at all,
   which is architecturally coherent (the same reason blocker D removed `Ln1`/`Ln2`: the Gated Residual
   instance's own `hc_norm` IS the norm at that point). `LnF` is therefore **synthesized to 1.0**, the
   RMSNorm identity gain, and `LmBias` to 0.0 — both reported by name in level 1, not defaulted
   silently. **This is a real open question for WP4d**: an identity `LnF` is a no-op, but it is also not
   what the real model does, because the real model has no such site at all.
4. **§3a-bis did not list the Gated Residual EXIT instance.** `output_hc_norm/down/up.weight` are in the
   file, exactly matching `docs/GATED_RESIDUAL.md` §1c's `use_combine=False` model-level instance (three
   tensors, no inject). Its table only covered `blk.N.hc_*`.
5. **The PLE is six MORE tensors, not just the big table.** Layer 1 (the real `ple_layer_ids=[2]`
   site, 0-indexed) additionally carries `ple_conv1d`, `ple_key`, `ple_norm_conv`, `ple_norm_key`,
   `ple_norm_query` and `ple_value`. §5 excludes "the n-gram/PLE table"; the mechanism is a whole
   sub-block at that layer, and all seven show up in level 1's unmatched list.
6. **Two 2-D tensors do NOT transpose**, against §4a's "every 2-D weight transposes":
   `token_embd.weight` is an `nn.Embedding` table (`[num_embeddings, embedding_dim]`, which IS
   `TokEmb`'s own `[VOCAB, D_MODEL]`), and `blk.N.ssm_conv1d.weight` is a depthwise `Conv1d`
   (`ne = [kernel, channels]`, i.e. row-major `[C][K]`, which IS `GdnConv`). Note that
   `token_embd.weight` and `output.weight` declare **identical** dims and get opposite treatment.
7. **A naming trap worth stating once**: GGUF's `ne` is fastest-varying-first, so a `[out, in]`
   row-major PyTorch weight is *declared* `[in, out]` — which reads like this project's own
   `[rows=in, cols=out]` convention while the bytes are still its transpose. **No shape assertion can
   catch a missed transpose here**, because the declared shape was never wrong.
8. **`ssm_alpha` ← `in_proj_a` and `ssm_beta` ← `in_proj_b`** is taken from llama.cpp's own
   `gguf-py/gguf/tensor_mapping.py` (`MODEL_TENSOR.SSM_ALPHA` → `linear_attn.in_proj_a`), not read off
   the names. They are same-shaped `[hidden, 48]` vectors feeding different gates — the exact
   identity-swap shape of `docs/GATED_DELTANET.md` §6's real bug.

Everything else in §3a-bis's table held: all seven new formats are present and needed, the per-layer
mixed quantization is real (`blk.0`/`blk.3` `ffn_gate_exps` are `IQ1_S`, `blk.1`/`blk.2` are
`IQ2_XXS`), and every declared shape matched.

**`gguf.hpp`'s seven new formats.** Each block-size rule was cross-checked against the real file by
differencing consecutive tensors' data offsets — an oracle independent of the struct definitions the
figures came from — and all seven matched exactly: `Q4_K` 144/256, `Q5_K` 176/256, `Q6_K` 210/256,
`IQ2_XXS` 66/256, `IQ1_S` 50/256, `IQ4_NL` 18/32, `BF16` 2/1. Decoding real tensors of each gives
zero-centred, finite, comparable-scale distributions (std 0.0075-0.042, no NaN), which is
`AGENTS.md` §9's real-file validation rather than fixtures alone.

**Deliberately still open, named here so WP4d does not rediscover them:**

- **`LnF` is synthetic** (see finding 3). WP4d must decide whether an identity gain there reproduces
  the real model or whether the site should be removed the way `Ln1`/`Ln2` were.
- **The artifact is f32 and 43.4 GiB for FOUR layers.** Extrapolated, the full 48 would be ~500 GB.
  The offload machinery of WP4e is not optional at full scale; it is the only way this format works.
- **`op_rmsnorm`'s `eps` is still 1e-5** against the real model's 1e-6 (carried forward from WP4b).
- **Nothing has loaded this file.** The engine cannot be built at these axes yet (`Model::forward` at
  real dims is WP4d), so `load_model`'s own header/fingerprint path has never seen it. The header was
  checked field by field by reading the written bytes back, and `--verify` confirms the blob, but
  "the engine accepts it" is an untested claim.
- **The transplant is single-threaded** and takes ~80s for the write, ~105s for a verify. Fine at this
  scale; at 48 layers it would be ~20 minutes, which is still fine, so no work is proposed.

### WP4d — First real forward pass, reduced scope

The smallest thing that is genuinely a real run. **Proposal: a single real decoder layer, then layers
0-N, before the full 48.** This is not timidity — it is the only configuration where a *per-layer*
divergence can be localized, and §5's PLE exclusion means layer 0's output is the last point at which
this engine and llama.cpp are computing the same function anyway.

**Gate**: layer-0 output matches the existing real fixture through the *full engine path*
(`Model::forward`), not just the math cores — the first time any of these mechanisms runs inside `Model`
at real dims. Then `forward` vs `forward_one` parity at real dims (the check that has caught a real bug
in every one of WP1-3).

### WP4e — Expert residency / offload at real scale (§4 of the orchestration doc)

Implements `docs/QWEN4_MEMORY_ORCHESTRATION.md` §3b's already-named policy: a **static layer-range
CPU/GPU split**, per-build, `-ot`/`--n-cpu-moe`-shaped. **What does not exist yet and must be built:**

1. **A `PARAM_LAYOUT` leaf that is not arena-resident.** Every tensor today is a span into one flat
   arena. An offloaded expert needs a leaf whose `data` span points at an mmap'd region or a staging
   buffer. `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §3b found the genuinely encouraging structural fact
   that makes this cheap: `Node::data` is a **non-owning `std::span<float>`**, so `op_linear`/`op_moe`
   cannot tell the difference. What is missing is the *construction* path, not the consumption path.
2. **An explicit resolve pass ahead of the hot loop** (`AGENTS.md` §1 — no runtime-variable latency in a
   hot path). `docs/QWEN4_MEMORY_ORCHESTRATION.md` §4 already classifies MoE expert weights as
   *mandatory explicit resolve*, and names the real latency-hiding opportunity: layer *L*'s router is
   computable as soon as layer *L−1*'s hidden state exists, so layer *L*'s resolve can be issued while
   *L−1* still computes (Eliseev & Mazur's one-layer lookahead).
3. **A compile-time-sized, fixed-capacity slot table** per §5c of that doc — capacity plausibly **0** on
   this hardware for a first build, which that document already flags as a legitimate honest outcome.
4. **The `MoeRouter` exemption** (§2a v2 of that doc): the router is dense-read per token and must stay
   resident even when its layer's experts are offloaded.

**Gate**: bitwise-identical output between an all-resident small build and the same build with the
offload path forced on — the offload must be a *placement* change, never a numerical one, and only a
bitwise check proves that (the same argument `docs/QSA.md` §11 makes for its block-key cache, where an
output check alone could not distinguish cached from uncached).

### WP4f — The llama.cpp comparison harness

**Simplest correctness-first version, and why it is this and not a generation-quality comparison:**

1. **Same tokens in.** Bypass tokenization entirely on both sides — feed a fixed, hand-written token-ID
   array. llama.cpp's `llama-eval-callback`/`--verbose-prompt` accept explicit tokens; this project's own
   `gen` path already accepts a prompt but a token-array entry point avoids any tokenizer-parity question
   (this engine's tokenizer is its own, and is emphatically NOT Qwen's).
2. **Compare hidden states, not generated text.** `llama-eval-callback` (upstream `examples/`) dumps every
   intermediate tensor for a single forward pass. Diff **layer 0's output hidden state** first, then
   deeper layers, then the final logits. A token-level or text-level comparison is a *much* weaker signal
   — a mismatch tells you nothing about where.
3. **A fixed numeric gate per layer.** WP1-3's fixture matches ran `0.0` (GR), `1.5e-11` (MoE),
   `1.4e-09` (QSA), `4.4e-11` (GDN). A whole-layer composition through the engine will be looser; propose
   a per-layer relative gate around `1e-4` and **record the actual number**, rather than picking a
   threshold that whatever comes out happens to pass.
4. **Acquire llama.cpp as a binary, not a submodule.** It is a comparison oracle, not a dependency;
   nothing in `src/` should ever link it.

**Gate**: a recorded per-layer divergence table, and — if divergence exceeds the gate — a *localized*
layer index, which is the actual deliverable of this stage. **"Sub0Llm and llama.cpp agree" is the
success case; "they disagree at layer K, in mechanism M" is an equally valid and equally valuable
outcome**, and is what this staging is designed to produce rather than a single unhelpful global mismatch.

---

## 7. Open questions — the author's own scoping assumptions, for the user to settle

These are **not** rhetorical. Each one changes the plan materially, and this document deliberately does
not pick for the user.

**Q1-Q4 answered by the user (2026-09-03)**, recorded here verbatim rather than paraphrased:

- **Q1 (fake-weights-first)**: NOT a strict either/or as posed. User's answer: *"Get the real weights
  downloaded - doesn't block work as random weights can also be used while it downloads and for quick
  validation of shape and plumbing. But getting real weights from disk will also drive the how/when/where
  to effectively integrate sooner."* — i.e. **run both in parallel, not gated on each other**: WP4b/WP4d's
  shape/plumbing validation proceeds now with synthetic weights (no download dependency), while the real
  acquisition (already in progress, see below) happens concurrently — and once real bytes are on disk,
  inspecting THEM (not a synthetic stand-in) should inform WP4c's transplant design, rather than treating
  WP4c as a purely abstract design exercise ahead of having the actual file. Revises §6's staging: WP4b
  and WP4a's acquisition do not block each other.
- **Q2 (transplant format)**: **GGUF / `UD-IQ1_S`** — the tier already downloading to
  `D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S\` (all three shards, ~72.55GB total, including the
  n-gram/PLE table for completeness even though §5 keeps it out of the first run's compute). This
  overrides §3a's safetensors/bf16 recommendation. **Consequence, stated plainly since the user chose
  this with the cost already named**: `gguf.hpp` needs real `IQ4_NL` (and likely K-quant) dequantization,
  not just the `BF16` size-rule this document proposed as the minimal safetensors-path addition — §3
  should be read as superseded by this choice, not merely supplemented.
- **Q3 (run scope)**: **4-layer real sub-stack (3 GDN + 1 QSA)** — the real model's own repeating unit, as
  proposed.
- **Q4 (Blocker A)**: **Full faithful fix** — `--head-dim` and its full consumer sweep (§2 Blocker A) are
  in scope for WP4b, not deferred. Real attention weights must be transplantable, not just GDN/GR/MoE.

**Still open, unresolved by the above**: Q5 (llama.cpp availability — partially answered: no built binary
found on this machine, but an unbuilt source checkout exists at `D:\Craig\diffusiongemma\llama.cpp`, so
WP4f is "build it" not "acquire it from scratch"), Q6 (M-RoPE degeneracy, unverified), Q7 (per-layer
mismatch gate, proposed `1e-4` not yet confirmed), Q8 (the 0.71B parameter-total discrepancy — the cheap
131-header-request census that would settle it has not been run).

**Q1 — Real weights at all, or a "real shape, fake weights" harness run first?**
A synthetic run (real 48-layer Qwen4 `RunConfig`, randomly-initialised weights) would validate WP4b's
shape work, WP4d's engine path, WP4c's prefill chunking and WP4e's offload machinery **without any
download and without any acquisition decision** — everything except the transplant mapping itself and the
llama.cpp comparison. It costs one extra stage and de-risks four. **The author's recommendation is yes,
do this first**, but it is explicitly the user's call, and the counter-argument is real: it is one more
thing that is not the actual goal.

**Q2 — Safetensors/bf16 or GGUF/`UD-IQ1_S`?** §3a lays out the trade. bf16 is numerically clean and
already-validated for acquisition, but needs §4's offload machinery working before a full model fits.
`UD-IQ1_S` nearly fits in RAM but requires IQ1/IQ4_NL/K-quant *decode* work this engine has never done,
and injects a quantization difference into the very comparison the run exists to make. **Not settled here.**

**Q3 — Full 48 layers, or a truncated stack?** A 4-layer model (layers 0-3, i.e. 3 GDN + 1 QSA — the real
repeating unit) is a *real* Qwen4 sub-stack that exercises every mechanism, fits comfortably, and can be
compared against llama.cpp with the same `--n-layer`-style truncation. It is not "the model", but it is a
real forward pass of real weights. **Is that an acceptable definition of "a real run"?**

**Q4 — How much of §2's blocker work is acceptable?** Blocker A (`--head-dim`) has the widest blast radius
of any change proposed in this thread, touching `layout.hpp`, `memplan.hpp`, both backends and the
checkpoint arithmetic. It is unavoidable for a *faithful* transplant. **It is avoidable for a run that
accepts the engine's `head_dim = 106` shape and simply does not transplant real attention weights** —
which would still be a real GDN/GR/MoE run. Is the faithful version the goal, or is a partial one
acceptable first?

**Q5 — Is a llama.cpp build on this machine assumed available?** §6's WP4f assumes a working `llama.cpp`
binary with `llama-eval-callback`, and assumes the ~72GB `UD-IQ1_S` (or a truncated equivalent) is
acquired for it to load. If neither exists, WP4f is itself a multi-hour acquisition + build task, not the
lightweight harness the plan implies. **Unverified — no llama.cpp build was located on this machine.**

**Q6 — M-RoPE.** `mrope_section = [11,11,10]` is an interleaved multimodal RoPE. For a text-only forward
pass it *may* degenerate to standard RoPE, which is what this plan assumes throughout. **This assumption
is not verified** — it needs the same `inspect.getsource` treatment every other mechanism got. If it does
not degenerate, it is a fifth expressibility blocker.

**Q7 — Which mismatch counts as failure?** §6's WP4f proposes a `1e-4` per-layer relative gate. If the
engine and llama.cpp diverge at, say, `1e-2` at layer 40 while matching at `1e-6` through layer 10, is
that a WP4 failure, or a WP4 success plus a new investigation? Deciding in advance avoids the very
result-dependent goalpost-moving this project's own three-pillar policy exists to prevent.

**Q8 — Does the 0.71B parameter-total discrepancy block WP4?** `docs/QWEN4_MEMORY_ORCHESTRATION.md` §2f
(v2) measures 125.711B against a published 125B and names three candidate explanations without adopting
one. §6c item 7 settles it cheaply (131 header-only requests). **This plan schedules it in WP4a**, on the
grounds that a transplant that silently mismatches on 0.7B of parameters is exactly the failure a cheap
up-front census prevents — but if the user regards the published figure as authoritative and the
discrepancy as a rounding artifact, WP4a shortens.
