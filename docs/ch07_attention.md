# Chapter 07 — Attention Mechanisms

## Overview

The transformer's power comes from self-attention: every token can look at every
other token and weight its importance. This chapter implements the full
differentiable attention mechanism and connects it to the embedding pipeline
from Ch06.

## Scaled Dot-Product Attention

```
Q (queries)  (T, D_h)
K (keys)     (T, D_h)
V (values)   (T, D_h)

scores = Q @ K^T / sqrt(D_h)     shape: (T, T)
attn   = softmax(scores + mask)   shape: (T, T)
output = attn @ V                 shape: (T, D_h)
```

Dividing by `sqrt(D_h)` prevents dot products from growing large when
`D_h` is big — without it, softmax saturates and gradients vanish.

## Causal Masking

For language modelling, token `i` must not attend to future token `j > i`.
A mask of 0 (visible) or −∞ (future) is added before softmax:

```
mask[i, j] =   0   if j ≤ i
              -inf  if j > i
```

`exp(-inf) = 0`, so masked positions get exactly zero attention weight.

```
Causal attn[0, :] = [1.000, 0.000, 0.000, 0.000]  (only pos 0 visible)
Causal attn[2, :] = [0.312, 0.421, 0.267, 0.000]  (pos 0-2 visible)
```

## Multi-Head Attention

H parallel heads, each projecting to `D_h = D / H` dimensions:

```
For each head h:
  Q_h = x @ W_Q_h     K_h = x @ W_K_h     V_h = x @ W_V_h
  A_h = softmax(Q_h @ K_h^T / sqrt(D_h))
  out_h = A_h @ V_h

output = Σ_h out_h @ W_O_h    (concat + project)
```

Each head learns to attend to a different aspect of the sequence (syntax,
coreference, positional dependencies, etc.).

## API

```cpp
#include "sub0llm/nn/attention.hpp"
using namespace sub0llm::nn;

MultiHeadSelfAttention mha(
    /*embed_dim=*/256,
    /*num_heads=*/8,
    /*seed=*/42);

mha.embed_dim();   // 256
mha.num_heads();   // 8
mha.head_dim();    // 32

// Forward pass — causal mask applied when causal=true
Variable x(randn({6, 256}), true);
auto out = mha.forward(x, /*causal=*/true);   // (6, 256)

// Backward
sum(out).backward();

// Parameters: W_Q, W_K, W_V, W_O (one per head)
mha.parameters();   // vector<Variable*>
```

## Full Pipeline

```
token_ids (T,)
     │ Embedding (Ch06)
     ▼
x = tok_emb + pos_emb   (T, D)
     │ MultiHeadSelfAttention
     ▼
context   (T, D)   — gradient flows back through attention to embeddings
```

```cpp
// Token IDs → embeddings → attention (full pipeline)
auto tok_vec = tok_emb.forward(ids);          // (T, D)
auto pos_vec = pos_emb.forward(T);            // (T, D)
auto emb     = add(tok_vec, pos_vec);         // (T, D)
auto ctx     = attn_layer.forward(emb);       // (T, D)
sum(ctx).backward();                          // gradients flow to both embedding tables
```

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/attention.hpp` | `MultiHeadSelfAttention` |
| `src/nn/attention.cpp` | QKV projection, attention, output projection |
| `chapters/ch07_attention/main.cpp` | Demo (§1–§6) |
| `tests/test_attention.cpp` | Attention shape and gradient tests |
