# Qwen3.8-Flash-Next / Qwen4-preview architecture — reference + validation plan

Status: **Stage 0 + Stage 1 DONE (2026-08-30)** — the real primary sources have been fetched and
cross-checked, real weight tensors extracted, and CPU-verified fixture files built from the model's own
real published forward-pass code. Nothing in `src/`/`tools/` consumes this yet — Stage 2 (C++
implementation, gated against the fixtures below) is unstarted. Follow the `docs/DEPTH_ATTENTION.md`
precedent: this is the handover doc, staged status lives here, updated as each stage lands.

## Why this model, and what "supporting" it means here

Alibaba released `Qwen3.8-Flash-Next` on 2026-08-26. Its own Hugging Face README states plainly:
"This experimental preview of the architecture that will underpin Qwen4 is built around a fundamental
rethinking of how the core components of modern large language models (LLMs) interact at scale" — a
primary-source confirmation (not just third-party write-ups) of the framing used throughout this doc. See
[docs/ROADMAP.md](ROADMAP.md)'s 2026-08-30 status entry for why the project is pivoting toward supporting
existing released models rather than further from-scratch training.

**Scale reality check, stated plainly so no later step forgets it**: 125B total / 6B activated params
(MoE) plus a separate 51B-param n-gram embedding table plus a 4B-param MTP (multi-token-prediction) head
— confirmed against the real `model.safetensors.index.json`, whose `metadata.total_size` is
**359,999,963,128 bytes (335.3 GiB / 360.0 GB decimal)** across **131 shard files** and **1,658 tensors**.
This engine trains dense/GQA models in the tens-to-hundreds-of-millions-of-params range on a single
Windows workstation (see [[host-cpu-arrow-lake-hx]]; 63GB RAM, one laptop-class ~8GB-VRAM GPU). Training
or fully running this model here is not on the table. **What Stage 0/1 actually did**: extracted a
complete Gated-DeltaNet layer's real weights (~116MB) and a small set of real n-gram-embedding table rows
(fetched via HTTP range requests against the safetensors shards — never downloading the ~102GB table),
then ran the model's own real, unmodified `transformers` module code on those real weights on CPU to
produce reference (input, output) fixtures. See "Stage 0/1: DONE" below for exact provenance.

## IMPORTANT correction to the previous version of this doc: this is a vision-language model

The earlier draft of this doc treated Qwen3.8-Flash-Next as text-only. **That was wrong.** The real
`config.json`'s top-level `architectures` is `Qwen4ExpForConditionalGeneration`, `language_model_only` is
`false`, and the repo's own HF tags include `image-text-to-text`. There is a full `vision_config` (a
27-layer, 1152-hidden SigLIP-style ViT: `patch_size=16`, `spatial_merge_size=2`, `temporal_patch_size=2`,
`out_hidden_size=2560` matching the text decoder, `image_token_id=248056`, `video_token_id=248057`,
`vision_start_token_id=248053`, `vision_end_token_id=248054`). This engine's scope stays the text decoder
(GDN / n-gram embeddings / QSA) — the vision tower is out of scope — but the fact that this is a VLM,
not an LLM, was missing from the record entirely and needed correcting.

Also load-bearing: the model's internal type name is **`qwen4_exp`** (`Qwen4ExpConfig` /
`Qwen4ExpForConditionalGeneration` in `transformers`) — i.e. `transformers` itself already ships this as
"Qwen4-Exp", not merely "an architecture Qwen4 will resemble". The "Qwen3.8" name is the public release
branding; the code and config call it Qwen4 directly.

## Verified facts (re-derived from the real primary sources — 2026-08-30)

Primary sources fetched directly (all URLs live and content quoted below, not AI-summarized):
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json` — the real, complete config.
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/model.safetensors.index.json` — real tensor→shard map, 1,658 tensors / 131 shards.
- `https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/README.md` — the model card, in the team's own words.
- `https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf` — **this URL is real and resolves** (the previous doc flagged it as unverified; `github.com/QwenLM/Qwen3.8-Flash-Next` exists and its only two files are `README.md` and `tech_report.pdf`). 28-page PDF, text-extracted for citation below.
- `transformers==5.16.1` (installed from PyPI) ships real, non-remote-code modeling files at
  `transformers/models/qwen4_exp/{configuration_qwen4_exp.py, modeling_qwen4_exp.py, modular_qwen4_exp.py}`.
  No `trust_remote_code` custom modeling file is needed or shipped in the HF repo itself.

| Fact | Value | Confidence |
|---|---|---|
| Release date | 2026-08-26 | High (unchanged) |
| Architecture / model_type | `Qwen4ExpForConditionalGeneration` / `qwen4_exp` (`qwen4_exp_text` for the decoder) | **High — verified, config.json + transformers 5.16.1** |
| Modality | **Vision-language** (`language_model_only: false`, `pipeline_tag: image-text-to-text`) — corrects the previous text-only assumption | **High — verified** |
| Total / activated params | 125B total, 6B activated per token (MoE) | **High — verified twice: tech_report.pdf's own results table AND independently re-derived from config.json's expert-pool arithmetic** |
| N-gram embedding table | **51.2B params exactly** (51,200,245,760), `nn.Embedding(320_001_536, 160)`, ~102.4GB in bf16 | **High — verified: exact re-derivation from config.json's real formula matches tech_report.pdf's own "51B" table entry to the first three digits** |
| MTP (multi-token-prediction) head | 4B params (per README), 1 extra hidden layer, `full_attention` (QSA) type, own `rope_theta` | **High — verified, README + config.json `mtp` block** |
| MoE experts | **512 routed experts, top-10 selected per token, PLUS 1 always-on shared expert** (not part of the 512) | **High — verified, config.json `num_experts=512`/`num_experts_per_tok=10` + `modeling_qwen4_exp.py`'s `Qwen4ExpTextSparseMoeBlock`** |
| Hidden size | **2560** | **High — verified (was not previously stated)** |
| Layers | 48, hybrid pattern: unit of 3×(Gated DeltaNet) + 1×(attention), repeated 12× (`full_attention_interval=4`) = 36 GDN : 12 attention layers, exact 3:1 ratio | **High — verified, config.json `layer_types` (48-entry list) + `full_attention_interval`** |
| **The "full_attention" layers are NOT full softmax attention** | Public `config.json` labels them `"full_attention"`, but `configuration_qwen4_exp.py` remaps that label to `"qwen_sparse_attention"` at load time, with the comment *"The real checkpoint contains 'full_attention' entries for layers that are actually using an indexer"*. Every one of those 12 layers runs a `Qwen4ExpTextQSAIndexer` (DeepSeek-DSA-style lightning indexer / block top-k selection) inside `Qwen4ExpTextAttention`. | **High — verified, direct code + comment** |
| Attention (QSA layers) | `num_attention_heads=24`, `head_dim=256`, `num_key_value_heads=2` (12:1 GQA), `partial_rotary_factor=0.25` → rotary_dim=64 of 256, `rope_theta=10_000_000`, interleaved M-RoPE `mrope_section=[11,11,10]` | **High — verified, config.json** |
| QSA indexer | `indexer_n_heads=4`, `indexer_kv_heads=1` (required exactly 1), `indexer_head_dim=128`, `indexer_budget=2048`, `indexer_compress_ratio=4` → 512 selected key-blocks per query out of the compressed prefix | **High — verified, config.json + `Qwen4ExpTextQSAIndexer`** |
| GDN (linear attention) config | `linear_num_key_heads=16`, `linear_num_value_heads=48` (exact 3× ratio), `linear_key_head_dim=128`, `linear_value_head_dim=128`, `linear_conv_kernel_dim=4` (short causal depthwise conv) | **High — verified, config.json + `Qwen4ExpTextGatedDeltaNet`** |
| Gated Residual (hyper-connections) | `hc_count=4` residual streams, `hc_lowrank=320` low-rank input-mixer rank. Real module: `Qwen4ExpTextGatedResidual`. Real paper name: **"Gated Residual (GR)"**, tech_report.pdf §2.2 — NOT literally "mHC"; the paper cites mHC/xHC (Manifold-/Expanded- hyper-connections) as prior art it improves on, with its own distinct read/write-gate design | **High — verified, config.json + code + tech_report.pdf** |
| N-gram embedding placement | Exactly **one** PLE layer: `ple_layer_ids=[2]` (one-indexed) = decoder layer index **1** (0-indexed). Confirmed by tech_report.pdf's own placement ablation (Table 7): a single layer at position 2 is what shipped; multiple layers gave no consistent benefit. | **High — verified, config.json + tech_report.pdf Table 7** |
| N-gram embedding hashing | `ngram_size=3` → uses **both** bigram (n=2) and trigram (n=3) hashing, `heads_per_ngram=8` → 16 total heads (8+8), `ngram_vocab_size_base=20_000_000` (each head's vocab size is an independently-chosen prime just above 20,000,000, via a deterministic splitmix64-seeded search), `ple_embed_dim=2560` → 160-dim per head | **High — verified, config.json + `Qwen4ExpTextNGramEmbedding`, cross-checked against a live instance of the real module at small scale (see Stage 1 below)** |
| N-gram embedding on-disk sharding | `split_ngram_parts=128` — the ~102GB table is stored as 128 row-contiguous `ngram_embedding.shard_N.weight` tensors and concatenated into one logical `nn.Embedding` at load time; this is a checkpoint/TP convenience only, not a different runtime module | **High — verified, model.safetensors.index.json + `configuration_qwen4_exp.py` docstring** |
| Context length | 262,144 native, extensible to 1,000,000 via YaRN | **High — verified, config.json `max_position_embeddings` + README** |
| License | Qwen Community License 1.0 (`license: other`, `license_name: qwen-community-1.0`) | **High — verified, primary-source README frontmatter (was "one source, medium")** |
| Weights | `Qwen/Qwen3.8-Flash-Next`, safetensors bf16, 131 shards | **High — verified via HF repo API file listing** |
| GGUF quantizations | **Not found in this repo.** The HF repo's file listing contains only safetensors + tokenizer/config files — no `.gguf` files, and the README does not mention any. The earlier "~139 pre-made GGUF quantizations" claim could not be corroborated from the primary source and likely referred to a separate third-party/community mirror repo, not `Qwen/Qwen3.8-Flash-Next` itself. | **Corrected: previously "medium", now "not found — likely wrong repo attribution"** |
| Tech report | `github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf` | **High — verified, URL resolves, PDF fetched and text-extracted (28 pages)** |
| Optimizer | Muon on all 2D "linear map" weights (attention q/k/v/output, GDN in/out projections, expert fc1/fc2, N-gram key/value projections); AdamW on input/output embeddings, MoE router, and GR's low-rank projections; **the n-gram embedding table itself runs on plain Adam with weight decay disabled** | **High — verified, tech_report.pdf §3.1 (direct quote below)** — cross-reference [[muon-optimizer]] |

## Real source code, quoted (not paraphrased)

All from `transformers==5.16.1`, `transformers/models/qwen4_exp/modeling_qwen4_exp.py` (installed via
`pip install transformers`, no `trust_remote_code` needed — this is mainline, generated from
`modular_qwen4_exp.py`, Apache-2.0, Copyright 2026 The Qwen Team and The HuggingFace Inc. team).

### Gated DeltaNet layer (`Qwen4ExpTextGatedDeltaNet`)

Construction (note the real head-count/dim wiring this engine's implementation must match):

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
        # QKV
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

The gates and the delta-rule recurrence itself (`forward`, elided cache-handling; the real
`torch_chunk_gated_delta_rule` is the no-cache prefill path exercised by the Stage 1 fixture below):

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

The recurrence itself (`torch_chunk_gated_delta_rule`'s per-chunk loop — this IS the delta rule: an
error-correction term `v_new = v_i - v_prime`, i.e. "how wrong is the state's current prediction",
scaled by the decay gate `g` and applied via the update gate folded into `beta`):

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

### N-gram embedding (`Qwen4ExpTextNGramEmbedding`) — real hashing formula

```python
def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64

def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)
```

Per-head vocab size is a distinct **prime** just above `ngram_vocab_size_base` (`_find_nth_prime_after`,
literal trial division), one per head, so every head's modulus is coprime-ish/distinct — this is what
keeps the 16 heads from colliding on the same hash. Row lookup, per token position, per n-gram order
(2 and 3), per head:

```python
for ngram in range(2, self.ngram_size + 1):
    start_idx = (ngram - 2) * self.heads_per_ngram
    end_idx = start_idx + self.heads_per_ngram
    mixed_ids = shifted_tokens[0] * self.layer_multipliers[0]
    for position in range(1, ngram):
        mixed_ids = torch.bitwise_xor(mixed_ids, shifted_tokens[position] * self.layer_multipliers[position])
    head_vocab_sizes = self.ngram_heads_vocab_sizes[start_idx:end_idx]
    head_offsets = self.ngram_heads_offsets[start_idx:end_idx]
    ngram_ids = torch.remainder(mixed_ids.unsqueeze(-1), head_vocab_sizes.view(1, 1, -1))
    blocks.append(ngram_ids + head_offsets.view(1, 1, -1))
...
return self.ngram_embedding(ngram_ids...).flatten(-2)
```

i.e.: XOR-mix each of the last `ngram-1` shifted tokens (each multiplied by a distinct per-position
64-bit multiplier), reduce mod that head's prime vocab size, add that head's offset into the flat table,
then look up and concatenate all 16 heads' 160-dim rows into one 2560-dim vector per token position. This
formula was **cross-checked row-for-row against a live instance of the real module** (see Stage 1 below) —
not re-implemented from memory and trusted blind.

### QSA indexer (`Qwen4ExpTextQSAIndexer`) — lower priority, code quoted for completeness

```python
self.token_budget = config.indexer_budget
self.compress_ratio = config.indexer_compress_ratio
self.block_topk = self.token_budget // self.compress_ratio
...
key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)   # compress_ratio keys -> 1 pooled key
pooled_keys = self.k_layernorm(pooled_keys)
...
scores = torch.matmul(q[...], block_key_states...).transpose(-1, -2)
scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)
selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
```

i.e.: average every `compress_ratio` (4) consecutive key vectors into one pooled+RoPE'd key, score all
pooled blocks against the (RoPE'd, L2-normed) query with a ReLU'd dot product, and keep only the top
`block_topk` (512) blocks' worth of individual token positions — this additive mask is then OR'd into the
ordinary causal mask before the real (dense) attention softmax runs. Not extracted/fixtured this round
(deprioritized per the original scope decision — GDN and n-gram embeddings both have existing engine
backlog items and were the priority).

### Optimizer recipe (tech_report.pdf §3.1, direct quote)

> "linear maps: the attention q/k/v and output projections, the GDN (Yang et al., 2024) input and output
> projections, the fc1/fc2 of both routed and shared experts, and the key/value projection in N-gram
> embedding layers [use Muon]. The input embeddings and the output head stay on AdamW. ... Finally, the
> n-gram embedding table runs on Adam with weight decay disabled."

## Stage 0/1: DONE (2026-08-30)

### What was verified (Stage 0)

1. Real `config.json` fetched directly (`curl` raw JSON, not a model-card summary) and every architecture
   number above re-derived from it.
2. Real modeling code located: `transformers==5.16.1` (current PyPI release) already ships mainline
   `qwen4_exp` support — no `trust_remote_code` needed. Confirmed by grepping the installed package for
   `qwen4_exp`/`qwen3_next` model directories.
3. `github.com/QwenLM/Qwen3.8-Flash-Next` confirmed to exist (previously flagged unverified); its
   `tech_report.pdf` fetched and text-extracted (28 pages, via `pypdf`) and quoted above.
4. `model.safetensors.index.json` fetched (1,658 tensors / 131 shards) and used to locate the exact shard
   holding a full Gated-DeltaNet layer, and the 128 on-disk shard tensors making up the n-gram embedding
   table.

### What was extracted (Stage 1) — surgically, via HTTP Range requests, never downloading full shards

- **Gated DeltaNet layer 0** (`model.language_model.layers.0.linear_attn.*`, all 9 tensors: `in_proj_qkv`,
  `in_proj_z`, `in_proj_b`, `in_proj_a`, `conv1d.weight`, `dt_bias`, `A_log`, `norm.weight`, `out_proj`)
  fetched in full from `model-00001-of-00131.safetensors` — **115,917,248 bytes (~110.6MB)**, converted
  bf16→float32. This is the complete real weight set for one GDN layer, not a synthetic approximation.
- **N-gram embedding rows**: for a hand-chosen 6-token test sequence, the real row-index hashing formula
  (above) was evaluated to find the 112 unique absolute rows needed (16 heads × 6 positions, with some
  reuse), then exactly those rows — 320 bytes each — were fetched via HTTP Range requests against the
  specific `ngram_embedding.shard_N.weight` tensor (of 128) that contains each row, locating the byte
  offset from that shard's own safetensors header (also fetched via Range, not downloaded whole).
- **Total bytes transferred across all extraction**: ~116MB (dominated by the GDN weights; the n-gram
  rows totaled well under 1MB). Nowhere near the 20GB check-in threshold.

### What was computed (Stage 1) — real module code, real weights, CPU, no cache

1. **GDN small fixture**: the real, contiguous weight VALUES above were sliced down from
   `hidden_size=2560 / 16 key heads / 48 value heads` to a commit-sized
   `hidden_size=32 / 1 key head / 3 value heads` (the real 3× value:key head ratio is preserved; the real
   per-head dims, 128/128, are UNCHANGED — only hidden_size and head COUNT were truncated). The real
   `Qwen4ExpTextGatedDeltaNet.forward()` was then run, unmodified, on these sliced-but-real weights with a
   reproducible-seed synthetic input (`torch.Generator().manual_seed(20260830)`), `cache_params=None`.
2. **N-gram embedding fixture**: the row-index hashing formula was independently re-implemented in plain
   Python (needed because the real 320-million-row table cannot be instantiated in this machine's 63GB
   RAM as a dense `nn.Embedding` — `320_001_536 × 160 × 4 bytes ≈ 205GB`). This replica was **validated
   by instantiating the real `Qwen4ExpTextNGramEmbedding` class at a small scale** (vocab base 101 instead
   of 20,000,000, `head_dim=1`, embedding weight set to `arange(vocab)` so the module's own float output
   directly reveals which row it picked) and confirming an exact, row-for-row match against the replica
   for a 5-token test sequence. The *row indices* used for the real 2560-dim fixture therefore come from
   code proven equivalent to the real module; the *row values* at those indices are the real checkpoint
   bytes, fetched as described above — no part of the fixture's numeric content is synthetic.

### Fixture files and exact format

Location: `tests/fixtures/qwen4_preview/`. All binary files are **raw flat float32 (or int64 where noted),
little-endian, row-major (C order), no header** — shapes/dtypes live in the sibling `*_manifest.json`
(matching this repo's existing preference for plain binary + a small JSON/struct header, and its
`.gitattributes` policy that fixtures stay ordinary small git files rather than LFS blobs). Total size of
both fixture bundles combined: ~326KB.

**Gated DeltaNet fixture** (`gdn_layer0_small_*`, see `gdn_layer0_small_manifest.json` for the full
config/shape/provenance record):

| File | Shape | Contents |
|---|---|---|
| `gdn_layer0_small_input.bin` | `[1, 6, 32]` | hidden_states input to `forward()` |
| `gdn_layer0_small_weight_in_proj_qkv.bin` | `[640, 32]` | rows = `[query(128) \| key(128) \| value(384)]`, sliced from the real checkpoint's `[query(2048)\|key(2048)\|value(6144)]` row layout at the same offsets |
| `gdn_layer0_small_weight_in_proj_z.bin` | `[384, 32]` | |
| `gdn_layer0_small_weight_in_proj_b.bin` | `[3, 32]` | |
| `gdn_layer0_small_weight_in_proj_a.bin` | `[3, 32]` | |
| `gdn_layer0_small_weight_conv1d.bin` | `[640, 1, 4]` | same row selection as `in_proj_qkv` |
| `gdn_layer0_small_weight_dt_bias.bin` | `[3]` | |
| `gdn_layer0_small_weight_A_log.bin` | `[3]` | |
| `gdn_layer0_small_weight_norm.bin` | `[128]` | **not sliced** — RMSNormGated operates per `head_v_dim` (128), unchanged |
| `gdn_layer0_small_weight_out_proj.bin` | `[32, 384]` | |
| `gdn_layer0_small_output.bin` | `[1, 6, 32]` | the real module's `forward()` output — **this is the correctness gate** |

Small config for this fixture: `hidden_size=32, linear_num_key_heads=1, linear_num_value_heads=3,
linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4, hidden_act=silu,
rms_norm_eps=1e-6, output_gate_type=sigmoid`.

**N-gram embedding fixture** (`ngram_embedding_*`, see `ngram_embedding_manifest.json`):

| File | Shape | Contents |
|---|---|---|
| `ngram_embedding_token_ids.bin` | `[6]` int64 | the 6 hand-chosen test tokens |
| `ngram_embedding_row_indices.bin` | `[6, 16]` int64 | absolute row (into the flat 320,001,536-row table) used per (position, head) |
| `ngram_embedding_per_head.bin` | `[6, 16, 160]` float32 | the real 160-dim row fetched for each (position, head) |
| `ngram_embedding_flattened.bin` | `[6, 2560]` float32 | `per_head.reshape(6, 2560)` — **exactly** `Qwen4ExpTextNGramEmbedding.forward()`'s return value for this input, before `Qwen4ExpTextPLELayer` folds it into the hyper-connection stream |

Real config for this fixture (full scale, unmodified — embedding lookups don't need shrinking):
`ngram_size=3, heads_per_ngram=8, ngram_vocab_size_base=20_000_000, ple_embed_dim=2560, seed=1234,
eos_token_id=248044, unigram_vocab_size=248320`; `head_vocab_sizes` and `head_offsets` for all 16 heads,
and the 3 `layer_multipliers`, are recorded in full in the manifest.

### What remains uncertain / not done

- ~~**QSA (Qwen Sparse Attention)**: no weight extraction or fixture built.~~ **DONE** (2026-09-03,
  `docs/QSA.md` Stage 0+1): the recipe named here was followed exactly —
  `model.language_model.layers.3.self_attn.*` located in the index (layer 3 is the first
  `full_attention` layer), 198,976 bytes range-fetched, sliced down, and `Qwen4ExpTextAttention.forward()`
  run for real. Fixture at `tests/fixtures/qwen4_preview/qsa_layer3_small_*`; CPU forward matches to
  `1.86e-09`. Note the code quoted in this doc covers only the INDEXER; `Qwen4ExpTextAttention`'s own
  doubled `q_proj` (per-head `query|gate` chunk) and its `* sigmoid(gate)` output gate were never quoted
  here and are load-bearing — see `docs/QSA.md` §1b.
- **Exact total-parameter reconciliation**: the 125B figure is verified via the tech report's own table
  (strong) and roughly corroborated by hand (48 layers × ~4.9B raw MoE-expert-pool params ≈ 121-126B,
  before adding attention/GDN/embedding/lm_head weights) but a tensor-by-tensor summation across all 131
  shard headers was not performed — would need ~131 small header-only HTTP requests, not done this round
  as the report's own table is a stronger source than a hand re-derivation would be.
- **GGUF quantizations**: could not be found in the primary repo at all (see facts table) — likely a
  wrong-repo citation in the original doc, not re-confirmed elsewhere.
- The MTP head's own forward pass was not investigated beyond its config block (`mtp.hybrid=true,
  num_hidden_layers=1, layer_types=["full_attention"]`, own `rope_theta`).

## Staged plan (updated)

**Stage 0 — DONE**, see above.

**Stage 1 — DONE**, see above. Fixtures live at `tests/fixtures/qwen4_preview/`.

**Stage 2 — implement, off by default, gated against Stage 1's fixtures.** Follow the
`docs/DEPTH_ATTENTION.md` staged precedent exactly: constexpr toggle + `RunConfig` X-macro entry +
`ARCH_FINGERPRINT` bit (Stage 0 there), CPU op + forward/backward (Stage 1), CUDA (Stage 2), each gated
on bit-identical-when-off at two scales before the feature is exercised. **Remember the depth-attention
lesson**: a gradient check alone cannot validate a new op whose no-op form is still differentiable — the
Stage-1 fixtures from real Qwen weights are the actual correctness gate here.

- **N-gram embeddings** — start here. Already the next queued Nanbeige-features item
  ([[nanbeige-features-progress]]), already has a verified real reference
  (`NanbeigeNgramEmbedding`/`NanbeigeNgramLayerFusion`, see [[nanbeige-architecture-reference]] §1) AND now
  a second, independent real reference (this doc) confirming the same core mechanism (hashed n-gram →
  embedding table lookup) with different exact hyperparameters. Lowest architectural risk of the three.
  Gate against `tests/fixtures/qwen4_preview/ngram_embedding_*`.
- **Gated DeltaNet** — new recurrent-state mechanism, no existing Sub0Llm precedent (attention here is
  currently softmax-only). Higher risk: touches the per-layer state/cache design, likely needs new
  checkpoint fields (additive, per AGENTS.md §3) if the recurrent state is meant to persist across
  generation steps like the KV cache does. Scope Stage 2's CPU forward (no training, no cache) first, gate
  against `tests/fixtures/qwen4_preview/gdn_layer0_small_*`, before committing to a backward-pass or
  cache design. **Design + in-engine config skeleton landed** (branch `feature/gated-deltanet-design`,
  2026-08-31) — see `docs/GATED_DELTANET.md` for the verified recurrence (re-sourced from the real
  `Qwen4ExpTextGatedDeltaNet` once this doc's own Stage 0/1 landed, superseding that design doc's
  Qwen3-Next-proxy first draft), the checkpoint/arena-sizing design (a second, additive
  `ARCH_FINGERPRINT2` word — the first one had zero spare bits left), and why the `Node`-fanout wall depth
  attention hit did NOT reappear for Stage 1's forward (no backward to preserve linkage for yet). That
  doc's own "Stage 1" (CPU forward, no backward) is **DONE** (branch `feature/gated-deltanet-stage1`),
  gated on the `gdn_layer0_small_*` fixture above: max abs diff `4.37e-11` against the real reference's
  output, plus mutation-style property checks and a two-scale identity confirmation. **Stage 2 (CPU
  backward) is now also DONE** (branch `feature/gated-deltanet-stage2`) — the `Node`-fanout wall DOES
  reappear here (9 parameter tensors + the input, more than `Node`'s 4-slot fanout), resolved the same
  way depth attention's own variable-length case was (a `thread_local` side table keyed by a small int on
  the Node); backward is verified against the real reference's own `.backward()` gradients (not just its
  forward), matching to ~3e-7 relative — see `docs/GATED_DELTANET.md` §6 for the full numbers. **Stage 3
  (CUDA, FORWARD ONLY) is now also DONE** (branch `feature/gated-deltanet-stage3`) — the chunked-parallel
  form (`torch_chunk_gated_delta_rule`, not a re-run of the sequential CPU form), verified on real
  hardware against both Stage 1's CPU sequential reference and the real fixture; no CUDA backward, no CUDA
  decode (both refuse at runtime). Finding this bug along the way was the notable event: the CUDA port
  disagreeing with the CPU engine at a real mixed-layer model exposed a genuine, independent, pre-existing
  bug in `backend_cpu.cpp`'s `op_gdn` (dt_bias/a_log passed in swapped argument positions) that neither
  the fixture test nor the finite-difference gradient check could have caught from the CPU side alone —
  see `docs/GATED_DELTANET.md` §6 for the full account and numbers.
- **QSA** — **Stage 0 + Stage 1 DONE** (branch `feature/qsa-stage1`, `docs/QSA.md`): CPU forward only
  (`op_qsa`, `include/sub0/qsa_math.hpp`), correctness-gated against a real extracted layer-3 fixture, with
  a genuinely non-degenerate indexer (112 of 300 causally-visible entries dropped). No backward (loud
  `abort()`), no CUDA (`static_assert` refuses the build). Resolved design question: QSA needs NO new
  per-layer schedule axis — the real `layer_types` array IS `gdn_schedule_for(4)`, and every
  `full_attention` layer in the real model IS a QSA layer, so the three-way Attn/Gdn/Qsa classification is
  derived from the existing GDN stride plus a model-wide gate.
- **Gated Residual (hyper-connections)** — not previously tracked as a distinct backlog item; worth noting
  next to [[nanbeige-features-progress]]'s "remaining: ... mHC" line, since this is a second real,
  independent hyper-connection-family implementation (`Qwen4ExpTextGatedResidual`) to potentially draw on
  if/when that item is picked up. Not scoped or fixtured this round.
