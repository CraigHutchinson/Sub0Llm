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

**Superseded 2026-09-04**: this table records WP4c as executed. The `LnF` removal since dropped the
totals to **6,239 tensors / 11,647,614,880 floats / 46,590,459,520 bytes** and the artifact was
regenerated — see "WP4d's `LnF` item — CLOSED". Everything else here still holds.

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
   D assumed a real `LnF` counterpart existed and merely had the wrong `eps` (so did
   `docs/GATED_RESIDUAL.md` §1c, explicitly — now corrected there). It does not exist at all,
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

- ~~**`LnF` is synthetic** (see finding 3). WP4d must decide whether an identity gain there reproduces
  the real model or whether the site should be removed the way `Ln1`/`Ln2` were.~~ **RESOLVED: removed.**
  WP4d measured the identity gain at ~27% of logit scale; the site is gone under `USE_GATED_RESIDUAL`
  and `Dest::LnF` with it (one synthesized destination now, not two). See "WP4d's `LnF` item — CLOSED".
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

#### WP4d — EXECUTED (branch `feature/wp4d-real-forward`). Results, recorded rather than summarised

**The engine now builds, links and runs at the real axes, and the artifact loads.** The build goes
through the REAL `sub0llm-configure`, and its generated `sub0_corpus.hpp` is field-for-field identical to
`tests/qwen4_real_axes/sub0_config.hpp` with `N_LAYERS = 4`:

```
sub0llm-configure --corpus data/cosmopedia.txt --vocab 248202 --corpus-pretok 0 \
  --dmodel 2560 --layers 4 --heads 24 --kv-heads 2 --head-dim 256 \
  --gdn-full-attn-stride 4 --gdn-key-heads 16 --gdn-value-heads 48 \
  --gdn-key-head-dim 128 --gdn-value-head-dim 128 --hc-count 4 --hc-lowrank 320 \
  --num-experts 512 --experts-per-tok 10 --qsa-indexer-n-heads 4 --qsa-indexer-kv-heads 1 \
  --qsa-indexer-head-dim 128 --qsa-indexer-budget 2048 --qsa-indexer-compress-ratio 4 \
  --rotary-dim 64 --seq 128 --d-ff 640 --tie-embeddings 0 --rope-theta 10000000 --compute 0
```

**`--vocab 248202`, not 248320, and that is not a fudge.** The configurator emits the vocabulary the
LEARNER produced, not the target: `tok::learn` maps the Unigram result onto a fixed 288-symbol base
alphabet, and every single-BYTE piece reuses a base id instead of adding one, so
`VOCAB = 288 + (target - S)` where `S` is the corpus's distinct-single-byte count. One cheap probe run
(`--vocab 5000` → 4830 pieces) fixes `S = 170` for cosmopedia, and `248320 - 288 + 170 = 248202` then
lands the real `vocab_size` **exactly**, first try. Learning it took **94s** (the `.words` scan cache hit,
so passes 1-2 were skipped); a corpus large enough to support the vocabulary is the only requirement —
its CONTENT is irrelevant to a forward-pass test that never uses the tokenizer.

**Four blockers had to be fixed before any of this ran. Every one was invisible to WP4b/WP4c because
their gates were compile-time checks against a HAND-WRITTEN config header with no engine linked** — the
real configurator and `Model::forward` had never seen these axes:

| # | Blocker | Why nothing caught it |
|---|---|---|
| E | **`tools/configurator.cpp` has no `--d-ff` at all** — the FFN/expert width was always derived from `D_MODEL` (`4x` plain, `~8/3x` gated), giving **6848** where the real `moe_intermediate_size` is **640**. Under MoE that width is a per-expert weight SHAPE, so no real-weight transplant was expressible through the real tool | `tests/qwen4_real_axes/sub0_config.hpp` simply *declares* `D_FF = 640`. §1's table never asked whether the configurator could PRODUCE it |
| A′ | **The configurator refused `--head-dim 256` at d2560/24 heads.** The `d_model % n_heads` check ran unconditionally and BEFORE `--head-dim` was resolved — three lines above a comment that already says in words that an explicit head-dim is free of it. WP4b blocker A, half-landed | same: no test runs the tool |
| — | **`core.hpp`'s `static_assert(D_MODEL % N_HEADS == 0)` fails at the real axes** (2560 % 24 == 16). Correct only while `D_HEAD` was derived; the quantity that must divide is `D_Q = N_HEADS * D_HEAD`. Now asserted as that | `qwen4_real_shape_tests.cpp` includes `layout.hpp`, **not** `core.hpp`. Nothing had ever compiled the ENGINE at these axes |
| — | **`gdn_math.hpp`'s `forward()` wrote into `float gated[4096]`**, a stack array commented as a "generous" bound on `value_dim`. The real `value_dim` is `48 x 128 = 6144`: a **2048-float stack overrun and a hard segfault** on the very first real GDN layer | the GDN fixture is a SLICED layer (3 value heads, `value_dim` 384). No build in this repo had ever run GDN at the real head counts. `backward()`'s identical `d_gated_row[4096]` was fixed the same way rather than left as the twin defect |

**A fifth, structural one, which is the real memory finding.** `ensure_shared_params()` eagerly allocated
FOUR `PARAM_FLOATS` arenas (weights + gradient + both AdamW moments) and each `Worker` carried a FIFTH.
At these axes that is **43.4 GiB apiece — ~217 GiB before a single token is embedded**, on a 63.4 GiB
machine. They are also provably DEAD in exactly these builds: Gated Residual, MoE and QSA each `abort()`
in `backward_node`. `backend_cpu.cpp` now derives `FORWARD_ONLY` from those same three `USE_*` flags
(**GDN deliberately excluded** — its Stage 2 backward is real and gradient-checked), skips the three
shared arenas, sizes `Worker::grad` to 1 float, and refuses at the lowest callable seam
(`grad_ptr`/`adam_*_ptr`/`reduce_gradients`/`AdamW::step`) if a training path ever asks. `print_config` /
`print_host_memplan` were reporting `4 x PARAM_FLOATS` unconditionally and now report what is actually
paid — 130 GiB of over-statement removed here.

**Results, with the actual numbers:**

| Check | Result |
|---|---|
| `load_model()` accepts `qwen4_sub4.bin` | **YES**, first try, no mismatch to fix. Header (d_model/n_layers/n_heads/d_ff/vocab/ternary/pos_encoding/`param_floats` = 11,647,617,440), `ARCH_FINGERPRINT` and `ARCH_FINGERPRINT2` all matched a real configurator run's. **41.0s** to read 43.39 GiB |
| the two SYNTHESIZED destinations, read back from what LOADED | `LnF` min = max = **1.0**; `LmBias` min = max = **0.0** — exactly what WP4c's finding 3 said it wrote |
| `Model::forward` at real dims | **[6 x 248320] in 1.72s**, zero non-finite. Per-row logits rms 0.60-0.83, range ≈ ±4 |
| **engine path vs independent math-core replay** (see below) | layer 0 `\|\|h_in\|\|` **1.82e-08**, `\|\|delta\|\|` **2.76e-08**; layer 1 **1.39e-08** / **9.91e-09**; layer 2 **1.84e-08** / **3.26e-08** |
| `forward` vs `forward_one` | max abs **5.96e-06** over all 6 x 248,320 logits (worst at row 4); max relative 0.135, but that is a near-zero logit against a `+1e-6` denominator — against the row's own rms 0.74 the error is ~8e-06 relative. **No structural bug**, unlike WP4b's own `C / H` find |
| peak working set | **45.29 GiB** (43.40 after load, +1.86 for the one Worker) against 49.0 GiB free |

**§6's stated gate is not reachable, and the reason is a property of the fixtures, not of this stage.**
Every `tests/fixtures/qwen4_preview/` fixture is a **sliced** layer — `gdn_layer0_small` is `hidden_size`
**32** / 1 key head / 3 value heads, `qsa_layer3_small` is `hidden_size` **16** / 2 heads / `head_dim` 8
(their own manifests say so, alongside a separate `config_real_full_scale_for_reference` block). Neither
its input nor its reference output exists at `hidden_size` 2560, so "layer-0's output through
`Model::forward` matches the existing real fixture" cannot be evaluated: **there is no real-dims
reference anywhere in this repo, and WP4c's levels 3/4 were replays at the FIXTURE's dims through the
transplant mapping, not runs of this artifact.** §4c item 3's own parenthesis ("these fixtures are
sliced-down, so this requires the sliced config") already said this; §6 restated the gate without
carrying the caveat forward.

**What replaced it is the thing WP4d actually adds over WP4c**: does the ENGINE path — the Node graph,
GR's real wrapping, `MIXER_SCHEDULE`, and the `PARAM_LAYOUT` offsets the weights physically landed at —
agree with the math cores WP4c already validated, driven from the SAME loaded weights?
`tools/sub0llm-qwen4-forward.cpp` rebuilds layers 0-2 straight out of `params_ptr()` with plain
`gr::`/`gdn::`/`moe::` calls and compares against the engine's own per-execution residual-stream norms
(`loop_pass_stats`), walking `PARAM_LAYOUT` with a cursor that **checks each `PKind`** rather than
assuming offsets. The agreement above (≤3.3e-08 relative, float32 accumulation-order noise) is a real
result: it covers the embedding, the GR entry tile, six GR instances, three GDN mixers and three MoE
blocks composed in the engine's own order. Layer 3 (QSA) is excluded only because `op_qsa` needs the
backend's internal precomputed rope table.

### WP4d's two open items, resolved

**1. `op_rmsnorm`'s `eps` is 1e-5 vs the real 1e-6 — is it even reachable? Yes, at EXACTLY ONE site, and
it is a site the real model does not have.** Read from the actual `Model::forward` code path rather than
inferred: under `USE_GATED_RESIDUAL`, `gr_read()` takes the GR branch and never calls `op_rmsnorm`
(blocker D removed `Ln1`/`Ln2`); `op_qknorm` is only in the dense-attention `else` branch, which
`USE_QSA` compiles out; GDN's internal norms are `gdn_math.hpp`'s own `RMS_EPS = 1e-6`; GR's are
`gated_residual_math.hpp`'s `hc_norm` at `1e-6`; QSA's are `qsa::RMS_EPS = 1e-6`; MoE has none. **The
only surviving `op_rmsnorm` call in the entire real-axes forward is `h = op_rmsnorm(h, ln_f)`** — the
final norm before the head. So GR's own `hc_norm` does cover every norm a GDN/QSA layer needs, and the
1e-5/1e-6 discrepancy touches nothing but `ln_f`.

Measured there, on the real hidden state: mean-square **0.898**, so `1/sqrt(ms+eps)` differs by
**5.01e-06** relative between the two eps values, moving the logits by **1.67e-05** with no argmax change.
**The eps is not the problem at this site.**

**2. `LnF` synthesized to 1.0 does NOT reproduce the real model — and the fix is to remove the site, not
to change the gain.** WP4c's finding 3 established that `output_norm.weight` exists under no name in the
real GGUF; the only non-`blk.` tensors are `output.weight`, `output_hc_{down,norm,up}.weight`,
`per_layer_token_embd.weight`, `token_embd.weight`. That is architecturally coherent — `output_hc_*` IS
the model-level Gated Residual exit instance, and `gr::mix` builds its output from `hc_norm`-ed streams
(at 1e-6), so the representation reaching the head is already normalized. **But 1.0 is the identity GAIN,
not an identity OPERATION**: RMSNorm still divides by the row's RMS. The engine therefore normalizes a
second time, and the real model does not.

Measured: same hidden state, same `lm_head`, three readouts differing only in that step —
**max|with `ln_f` − without any final norm| = 0.174** against a logit rms of 0.64, i.e. a **~27%-of-scale**
perturbation. (At the first position, whose pre-`ln_f` rms is 2.64 rather than 0.95, the gap is
**4.90** — the effect scales with how far the stream's RMS is from 1, exactly as it must.) The argmax
happened to survive on this input; nothing guarantees that.

**So the resolution is: `LnF` should be removed under `USE_GATED_RESIDUAL` the same way `Ln1`/`Ln2` were
(blocker D), for the same reason and with the same evidence.** *(DONE 2026-09-04 on branch
`fix/gr-remove-lnf` — see "WP4d's `LnF` item — CLOSED" below for the executed result, including a second
independent confirmation from the real `model.safetensors.index.json` that no such tensor exists.)*
This was deliberately NOT done in WP4d:
it is shape-changing (`PARAM_FLOATS` drops by `D_MODEL` = 2,560 and `NUM_PARAMS` by 1), which
**invalidates the 46,590,469,832-byte artifact** and requires a `sub0llm-transplant` re-run. That is a
WP4c-shaped change with its own gate, not something to fold into the stage that discovered it. Until it
lands, **the logits this build produces are not the real model's logits**, and WP4f's comparison must not
be run against them — the per-layer hidden states (which `ln_f` is downstream of, and which §5 already
names as the right comparison point) are unaffected.

### WP4d's `LnF` item — CLOSED (branch `fix/gr-remove-lnf`, 2026-09-04)

**The claim that a real final norm exists was checked and refuted from a SECOND, independent source.**
WP4c found the GGUF has no `output_norm.weight` under any name. `docs/GATED_RESIDUAL.md` §1c had
nevertheless asserted a real `Qwen4ExpTextModel.norm` at `rms_norm_eps = 1e-6` — an inference from what
decoder stacks generally do, never checked. The real `model.safetensors.index.json` was then fetched
directly (170,726 bytes) and every non-per-layer "norm" tensor name enumerated: the only model-level
language-model norm in the whole checkpoint is
`model.language_model.hyper_connection_mixer.hc_norm.weight` — **the GR model-level exit instance's own
`hc_norm`**, which is already a real `PARAM_LAYOUT` entry (`GrHcNorm`), already transplanted by WP4c, and
present in the GGUF too as `output_hc_norm.weight`. Two real sources, two granularities, one answer:
**the GR exit's own `hc_norm` IS the model's final normalization, and `LnF` does not exist.**

**What landed**, exactly blocker D's precedent one level up:

- `make_param_layout()` emits no `LnF` under `USE_GATED_RESIDUAL`; `Model::forward` / `forward_one` feed
  `lm_head` straight from the GR exit's `mixed_input`. Non-GR builds keep the `LnF`/`op_rmsnorm` path
  untouched — this is a GR-specific correction, not a general one.
- `gr_param_delta()` gains a further `- d_model` (once, model-level). Still strictly positive, so
  `PARAM_FLOATS` still discriminates GR-on from GR-off; **no `ARCH_FINGERPRINT2` bit** (shape-changing,
  rule #1 — which matters, since that word is full).
- `Dest::LnF` is gone from `transplant.hpp` entirely rather than left synthesizing an identity gain
  nothing consumes: synthesized destinations drop from 2 to 1 (`LmBias` alone).

**The numbers, before → after:**

| Quantity | Before | After |
|---|---:|---:|
| `PARAM_FLOATS` at the real 48-layer axes | 125,711,064,960 | **125,711,062,400** |
| `NUM_PARAMS` at the real 48-layer axes | 74,803 | **74,802** |
| `qwen4_sub4` (4-layer) tensors / floats | 6,240 / 11,647,617,440 | **6,239 / 11,647,614,880** |
| artifact bytes | 46,590,469,832 | **46,590,459,592** |

Exactly `D_MODEL` = 2,560 floats and one tensor less, and nothing else moved. **`kRealParamFloats` in
`tests/qwen4_real_shape_tests.cpp` is corrected by the same 2,560**, with its own reason recorded:
`docs/QWEN4_MEMORY_ORCHESTRATION.md` §2f(v2)'s backbone reconciliation carries an `ln_f = 2,560` line for
a tensor the real checkpoint does not have. The test's independent hand-written census drops its `ln_f`
term and still meets `make_param_layout()`'s total.

**Gates re-run:**

- **Neutral identity**: `sub0_tests` is **hash-identical at all three standard shapes** —
  d96 L8 H2 seq128 `forward 4e00b8a7dadafff8 / grad 6909ae0b3afc2caa / decode ab31e5533547f73a`,
  d132 L11 H4 kv2 seq96 `289b86042f02843e / 787ec95304201870 / 27ee1bd6fa0f35eb`,
  d196 L11 H7 seq256 `9c8c0c17cd5043d9 / 50fae4b8922bac0e / 55f09cee05eea34b`. Assertion counts are
  **+1** at each shape (17,827,371→372 / 29,771,943→944 / 54,070,193→194), 147 cases unchanged — the one
  new `REQUIRE` counting `LnF` entries in `layout_tests.cpp`, and nothing else.
- **The four real-weight fixture tests**: `sub0_frontend_tests` green, 216 cases, 116,990 → **116,989**
  assertions — exactly `-1`, from `Dest::Count` losing one enumerator in `transplant_tests.cpp`'s own
  per-destination loop.
- **Four-level transplant gate, re-run against the regenerated artifact**: level 1 — 6,239 / 6,239
  destinations, 11,647,614,880 / 11,647,614,880 floats, **1 synthesized** (`LmBias`), 7 unmatched
  in-scope sources (every one a deliberately-excluded PLE tensor, unchanged); level 2 — 6,238 tensors
  checked, **0 mismatches**; level 3 (GDN layer-0 replay) **5.09e-11**; level 4 (QSA layer-3 replay)
  **1.40e-09**; plus `--verify`: **0 bit-for-bit mismatches of 6,239**. Write 82.0s, verify 107.8s.
  The previous artifact is kept as `qwen4_sub4.bin.pre-lnf-fix` for comparison.

**WP4d's own harness re-run on the new artifact** (`sub0llm-qwen4-forward`), against its own prior run:

| Check | Before (wrong) | After |
|---|---|---|
| `load_model` | accepted, 41.0s | accepted, **42.9s**, 43.39 GiB |
| the synthesized tail | `LnF` min=max=1.0, `LmBias` 0.0 | **`LnF` ABSENT from `PARAM_LAYOUT`** (asserted, not merely omitted); `LmBias` 0.0 |
| `Model::forward` | [6 x 248320] in 1.72s, per-row logit rms **0.60-0.83**, range ≈ ±4 | [6 x 248320] in **1.34s**, per-row rms **0.61-2.31**, range **-10.2 … +11.7**, zero non-finite |
| engine vs math-core replay, layers 0-2 | 1.8e-08 / 2.8e-08 class | **unchanged**: ‖h_in‖ ≤ 2.84e-08, ‖delta‖ ≤ 1.7e-08 (`ln_f` was downstream of these, as §5 predicted) |
| `forward` vs `forward_one` | max abs **5.96e-06**, max rel 0.135 | **0 exactly** — removing the site removed the one step where the two paths differed in accumulation order |
| peak working set | 45.29 GiB | 45.29 GiB |

**The dynamic-range change is the signature that this is the real fix, not just a smaller model.** The
pre-fix build's six rows all landed in a narrow `rms 0.60-0.83` band because `ln_f` rescaled every
position's hidden state to unit RMS before the head, flattening exactly the cross-position magnitude
differences the residual stream had built up. Un-normed, the rows whose hidden RMS is furthest from 1
(rows 0-3) now produce correspondingly larger logits (up to rms 2.31) while rows 4-5 — whose hidden RMS
was already ≈1 — are essentially unchanged (0.61 / 0.60 against the old 0.60-0.83 band). That is what
removing a normalization the real model does not have must look like.

**Two direct confirmations, both in the harness's own check 5:**

1. **The readout really is un-normed**: an independent double-precision `lm_head(hidden_last) + bias`
   reproduces the engine's own logits row to **5.89e-06** (float32 GEMM noise over 2,560 terms). No
   normalization can be sitting between them.
2. **The size of what was removed, measured on the same run**: `max |with the removed ln_f - without it|`
   = **0.174** against a logit rms of **0.605**, i.e. **28.8% of scale** — reproducing WP4d's own 0.174 /
   ~27% figure exactly. The argmax happened to agree either way on this input (109782), which is what
   made the bug survivable and not detectable by a top-1 check.

**Confidence.** High that the LAYOUT and the plumbing are now right — two independent real sources agree
the tensor does not exist, the change is exactly `-1` tensor and `-D_MODEL` floats with every other
number unmoved, and `forward`/`forward_one` now agree bit-exactly. Moderate-to-high that the OUTPUT is
now materially more correct: the removed perturbation was 28.8% of logit scale and its removal restores
the cross-position dynamic range a norm-free readout must have. **Still not verified against the real
model's own outputs** — WP4d's own standing caveat is unchanged: no real-dims reference output exists in
this repo, so this is a structural-correctness argument plus a self-consistency check, not a numerical
match. Settling it needs WP4f's llama.cpp oracle.

### WP4d — what is still open

- ~~**`LnF` removal**~~ — **DONE**, see the section immediately above.
- **No real-dims reference output exists.** Closing that needs either a real-dims fixture extracted the
  way the sliced ones were, or WP4f's llama.cpp oracle. WP4d's engine-vs-math-core agreement is a
  consistency check, **not** a check against the real model.
- **`op_qsa`'s rope table is backend-internal**, so layer 3 is outside the replay above.
- **Memory has no headroom for growth.** 45.29 GiB peak against 49.0 GiB free is a 3.7 GiB margin at
  `SEQ_LEN = 128` and `T = 6`. The activation arenas alone are **1,909 MiB per Worker** and scale with
  `SEQ_LEN`; `DEFAULT_THREADS` is 24, so anything that touches a second Worker slot adds another 1.9 GiB.
  A forward-only run is single-worker by construction, but this is the number WP4e's offload work has to
  start from, and it confirms §3b's "plausible, still tight" framing with a measurement.
- **A real engine build at these axes needs `-fconstexpr-steps` raised**, as WP4b predicted; it is now a
  tree-wide `add_compile_options` rather than a per-target flag.

### WP4e — Expert residency / offload at real scale (§4 of the orchestration doc)

**Rescoped 2026-09-04 — the previous framing below solved the wrong problem.** It described WHERE the
f32 bytes for an expert live (arena vs. an mmap'd/staging region) — a placement question. But WP4c's own
real transplant measurement makes the actual constraint plain: **every tensor this engine computes with
is f32, at a 10.5x expansion over the GGUF file's own quantized bytes** (46.59GB f32 from 4.42GB encoded,
for 4 layers). Extrapolated to 48 layers that is **~500GB of f32** — not "500GB that needs a clever
placement policy," but 500GB that must never be MATERIALIZED at all, because the entire reason a
quantized tier was downloaded in the first place was to keep the resident footprint small. A placement
policy over an already-500GB f32 form has already lost the game before it starts (user's own observation,
2026-09-04: *"that's why we need iq4"*).

**What this actually requires, restated**: the ~512×48 routed experts must stay resident in their
**native quantized byte form** (whichever of `IQ1_S`/`IQ2_XXS`/`IQ4_NL` each one actually is — §3a-bis's
own per-layer-mixed-quantization finding applies here too), and dequantization must happen **per
selected expert, per token, into a small reusable scratch buffer**, immediately consumed and discarded —
never into a persistent f32 array for experts that were not selected. This is exactly what `llama.cpp`
itself does (dequantize-on-the-fly inside its quantized matmul kernels, or dequantize into a small
temporary): the quantized file format and the small in-memory footprint are the SAME design decision, not
two separable ones. WP4c's f32 transplant was a deliberate, correct choice **for its own narrow purpose**
(a byte-exact, quantization-noise-free correctness gate against the existing real-weight fixtures, at a
scale — 4 layers — where 43GB of f32 happens to fit) — it is not, and was never meant to be, a preview of
the full-scale storage design.

**Consequence for the engine's own structure, stated plainly**: `op_linear`/`op_moe` today read a
`Node::data` that is a plain `std::span<float>` — there is no quantized-tensor concept anywhere in
`backend_cpu.cpp`. The non-owning-span structure `docs/QWEN4_DEPLOYMENT_FEASIBILITY.md` §3b found
encouraging (the op cannot tell an owned arena slice from an mmap'd one) does NOT help here, because both
of those are f32. **A real fix needs one of**: (a) a new op variant that takes a quantized `TensorInfo`-
like descriptor plus a raw byte span, and dequantizes internally per call (the`gguf::` decoders WP4c just
built are exactly the primitives this would call); or (b) a resolve-pass that dequantizes ONLY the
`experts_per_tok` (10, real value) selected experts per layer per token into a small, compile-time-sized
scratch pool, immediately before that layer's `op_moe` call, and never keeps more than a handful of
experts' worth of f32 resident at once. (b) is closer to this project's existing "explicit resolve pass
ahead of the hot loop" idiom (`AGENTS.md` §1) and is the recommended starting point — **not decided here,
this is WP4e's own design question**, but the shape of the question has changed: it is "how do we
dequantize just-in-time and reuse a small buffer," not "where do we put a 500GB array."

**What survives from the original framing, still real work**:

1. **An explicit resolve pass ahead of the hot loop**, now understood as a DEQUANTIZE step, not a copy/
   move step. `docs/QWEN4_MEMORY_ORCHESTRATION.md` §4 already classifies MoE expert weights as
   *mandatory explicit resolve*; the real latency-hiding opportunity it names (layer *L*'s router is
   computable as soon as layer *L−1*'s hidden state exists, so layer *L*'s dequantize can be issued while
   *L−1* still computes, Eliseev & Mazur's one-layer lookahead) applies identically to a dequantize-based
   resolve.
2. **The `MoeRouter` exemption** (§2a v2 of that doc) — the router is dense-read per token over all 512
   experts and must stay resident (and, since it is `F32` in the real file per §3a-bis's own census,
   resident in its ALREADY-native format — no dequantization gap here at all).
3. A compile-time-sized, fixed-capacity scratch pool sized for at most a handful of concurrently-live
   dequantized experts — a much smaller and more tractable sizing question than §5c of the orchestration
   doc's own GPU-VRAM-slot-count framing, which was itself built on the f32-per-expert size (§3b's "2.64
   MiB/expert IQ4_NL-class" figure was actually already right — it used the QUANTIZED size — the mismatch
   was only ever in WP4c's own transplant choice, not in that document's own arithmetic).

**Gate, revised**: bitwise-identical output between (a) WP4c/d's own existing all-f32-resident 4-layer
path and (b) the same 4 layers run through a dequantize-on-demand path instead — proving the *representation*
change (quantized-resident + per-use dequant vs. all-f32-resident) is numerically inert, the same argument
`docs/QSA.md` §11 makes for its block-key cache (an output check alone cannot distinguish the two without
this exact bitwise comparison). This is checkable at the SAME 4-layer scale WP4d already validates — the
500GB number only bites at full 48-layer scale, so WP4e's own correctness gate does not need to attempt
that scale to be meaningful.

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
