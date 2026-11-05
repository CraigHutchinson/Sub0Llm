---
name: rope-theta-tuning-reference
description: Reference paper for deriving a better RoPE theta (frequency base); SUB0_ROPE_THETA default 10000 — revisit with this
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

For tuning **RoPE theta** (`SUB0_ROPE_THETA`, currently the default base 10000 in CMakeLists / configurator `--rope-theta`): read this when revisiting the frequency base, especially for longer context.

Frontiers in Computer Science (2025): https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2025.1626899/pdf

Relevant because we moved to RoPE (pos-encoding default) + want longer effective context (the JOIN tokenizer already gives +38% text per window). Theta interacts with seq_len / extrapolation. Not yet acted on — a future tuning lever (could fold into the model-dimension `tune` sweep). See [[join-tokenizer-validated-result]].
