# Architecture profile for `sub0llm-configure --profile qwen3.8-27b`. Format: include/sub0/profile.hpp.
# Dims checked 2026-11-06 against Qwen/Qwen3.8-27B config.json (hidden 5120, 64 layers, 24 heads, 4 KV heads,
# head_dim 256, intermediate 17408, vocab 248320, GDN 16/48x128, full-attention every 4th layer, rope_theta 1e7).
# No qsa-indexer-* axes: the real 27B GGUF has no indexer.* tensors, and no hc_* gated-residual tensors either (issue #25).
name = qwen3.8-27b
alias = qwen3.8-dense

# Shared Qwen3.x geometry: GDN/QSA mixers, RoPE, external vocabulary.
head-dim = 256
rotary-dim = 64
rope-theta = 10000000
gdn-full-attn-stride = 4
gdn-key-heads = 16
gdn-key-head-dim = 128
gdn-value-head-dim = 128
tie-embeddings = 0
prec-param = 1
vocab-exact = 248320      # skipped when --vocab or --vocab-exact is given
seq = 128
pos-encoding = 1
gdn-value-heads = 48

# qwen3.8-27b-specific
dmodel = 5120
layers = 64
heads = 24
kv-heads = 4
d-ff = 17408
gated-ffn = 1
num-experts = 0

# Weights (metadata only; see scripts/models.py)
model-card = Qwen/Qwen3.8-27B
model-source = unsloth/Qwen3.8-27B-GGUF
model-dir = Qwen3.8-27B-GGUF
model-include = *UD-Q4_K_M.gguf     # 15.3 GiB; your call -- see the repo's file sizes (5.8 GiB IQ1_S .. 46.6 GiB BF16)
