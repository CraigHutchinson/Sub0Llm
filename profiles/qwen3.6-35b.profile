# Architecture profile for `sub0llm-configure --profile qwen3.6-35b`. Format: include/sub0/profile.hpp.
# TODO(source): dims are not yet tied to a fetched config.json -- see issue #7.
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
d-ff = 1436
num-experts = 256
experts-per-tok = 8
moe-quant-experts = 1
