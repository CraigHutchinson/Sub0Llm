# Architecture profile for `sub0llm-configure --profile qwen3.6-35b`. Format: include/sub0/profile.hpp.
# Dims checked 2026-11-06 against Qwen/Qwen3.6-35B-A3B config.json (hidden 2048, 40 layers, 16 heads, 2 KV heads,
# moe_intermediate_size = shared_expert_intermediate_size = 512, 256 experts / 8 per token, GDN 16/32x128).
# d-ff was 1436 before this check, which matched nothing in config.json.
name = qwen3.6-35b
alias = qwen3.6-35b-a3b

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
gdn-value-heads = 32

# qwen3.6-35b-specific
dmodel = 2048
layers = 40
heads = 16
kv-heads = 2
d-ff = 512
num-experts = 256
experts-per-tok = 8
moe-quant-experts = 1

# Weights (metadata only; see scripts/models.py)
model-card = Qwen/Qwen3.6-35B-A3B
model-source = unsloth/Qwen3.6-35B-A3B-GGUF
model-dir = Qwen3.6-35B-A3B-GGUF
model-include = *UD-Q4_K_M.gguf     # choose the quant to suit your RAM
