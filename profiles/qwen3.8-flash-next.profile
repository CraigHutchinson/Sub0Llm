# Architecture profile for `sub0llm-configure --profile qwen3.8-flash-next`. Format: include/sub0/profile.hpp.
# Dims checked 2026-11-06 against Qwen/Qwen3.8-Flash-Next config.json (hidden 2560, 48 layers, 24 heads, 2 KV heads,
# head_dim 256, moe_intermediate_size = shared_expert_intermediate_size = 640, 512 experts / 10 per token, GDN 16/48x128).
# hc-count/hc-lowrank and qsa-indexer-* are engine-side choices with no config.json counterpart.
name = qwen3.8-flash-next
alias = qwen4-preview, qwen4

# Shared Qwen3.x geometry: GDN/QSA mixers, RoPE, external vocabulary.
head-dim = 256
rotary-dim = 64
rope-theta = 10000000
gdn-full-attn-stride = 4
gdn-key-heads = 16
gdn-key-head-dim = 128
gdn-value-head-dim = 128
qsa-indexer-n-heads = 4
qsa-indexer-kv-heads = 1
qsa-indexer-head-dim = 128
qsa-indexer-budget = 2048
qsa-indexer-compress-ratio = 4
tie-embeddings = 0
prec-param = 1
vocab-exact = 248320      # skipped when --vocab or --vocab-exact is given
seq = 128
pos-encoding = 1
gdn-value-heads = 48

# qwen3.8-flash-next-specific
dmodel = 2560
layers = 48
heads = 24
kv-heads = 2
hc-count = 4
hc-lowrank = 320
d-ff = 640
moe-quant-experts = 1
num-experts = 512
experts-per-tok = 10

# Weights (metadata only; see scripts/models.py)
model-card = Qwen/Qwen3.8-Flash-Next
model-source = unsloth/Qwen3.8-Flash-Next-GGUF
model-dir = Qwen3.8-Flash-Next-GGUF
model-include = UD-IQ1_S/*     # the tier docs/HANDOVER.md transplants from
