# Chapter 06 — Embeddings

## Overview

Token IDs are integers with no geometric meaning — the model can't use
ID 42 being "close" to ID 43. An **embedding table** E[V × D] maps each token
to a D-dimensional continuous vector where semantically similar tokens end up
geometrically close (e.g. "king" − "man" + "woman" ≈ "queen").

This chapter builds the three layers between the tokenizer and the first
transformer block:

```
token_ids  →  Embedding (lookup)  →  tok_emb  (T, D)
                                           +
             PositionalEncoding (pos)  →  pos_emb  (T, D)
                                           ↓
                                      x = tok_emb + pos_emb  (T, D)
```

## Embedding Lookup

The forward pass is a **gather**: row `ids[t]` of the weight matrix is copied
to position `t` of the output.

The backward pass is **scatter-add**: upstream gradients are accumulated back to
the rows of the weight matrix. If token `t` appears at multiple positions, its
gradient row accumulates contributions from all of them.

```cpp
// Token 1 at positions 1 and 3 → grad[1] = 2× a single upstream row
```

## Positional Encodings

Three variants are implemented:

### Sinusoidal (Vaswani 2017)
```
PE[pos, 2i]   = sin(pos / 10000^(2i/D))
PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
```
Fixed — not learned. Row norms are constant (≈ √(D/2)).

### Learned
Standard trainable embedding table over positions 0…max_seq_len−1.
Initialised N(0, 1/√D).

### RoPE — Rotary Position Encoding (Su et al., 2021)
Rotates query/key pairs `(2i, 2i+1)` by angle `pos / base^(2i/D)`.

Key property: `dot(q_rot[pos_a], k_rot[pos_b])` depends only on
`(pos_a − pos_b)`, not on absolute positions → **relative attention** without
extra parameters.

Used in LLaMA, Mistral, Qwen, Gemma. Applied inside GQA in Ch10.

## API

```cpp
#include "sub0llm/nn/embedding.hpp"
#include "sub0llm/nn/positional_encoding.hpp"
using namespace sub0llm::nn;

// Token embedding table
Embedding tok_emb(/*vocab=*/50257, /*dim=*/768, /*seed=*/42);
auto emb_out = tok_emb.forward(token_ids);  // (T, D)
tok_emb.weight();                           // Variable — the D×V weight matrix

// Sinusoidal positional encoding (fixed, no params)
Tensor pe = sinusoidal_encoding(/*seq_len=*/T, /*dim=*/D);  // (T, D)

// Learned positional encoding (trainable)
LearnedPositionalEncoding pos_emb(/*max_seq=*/1024, /*dim=*/768, /*seed=*/1);
auto pos_out = pos_emb.forward(/*seq_len=*/T);   // (T, D)
pos_emb.weight();                                // Variable

// RoPE (applied inside GroupedQueryAttention, Ch10)
Tensor q_rot = apply_rope(q);   // (T, D_head) → rotated
Tensor k_rot = apply_rope(k);
```

## Vocabulary Sizes in Practice

| Model | V | D | Embedding params |
|-------|---|---|-----------------|
| GPT-2 small | 50,257 | 768 | 38.6 M |
| GPT-2 XL | 50,257 | 1,600 | 80.4 M |
| Llama 3.1-8B | 128,000 | 4,096 | 524 M (25% of all params) |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/embedding.hpp` | `Embedding`, `LearnedPositionalEncoding` |
| `src/nn/embedding.cpp` | Gather forward, scatter-add backward |
| `include/sub0llm/nn/positional_encoding.hpp` | `sinusoidal_encoding`, `apply_rope` |
| `chapters/ch06_embeddings/main.cpp` | Demo (§1–§7) |
| `tests/test_embeddings.cpp` | Embedding tests |
