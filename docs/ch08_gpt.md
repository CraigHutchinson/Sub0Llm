# Chapter 08 — GPT Architecture

## Overview

This chapter assembles the transformer building blocks from Ch05–Ch07 into a
complete GPT-2-style language model by adding the two missing pieces:

1. **Layer normalisation** — normalise each feature vector to zero mean / unit
   variance, then apply learnable scale γ and shift β.
2. **Feed-forward network (FFN)** — expand by 4×, apply GELU, project back.

## GPT Block (Pre-norm style)

```
x  ──►  LayerNorm  ──►  MultiHeadSelfAttention  ──►  + x   (residual #1)
  h  ──►  LayerNorm  ──►  FeedForward              ──►  + h   (residual #2)
```

## Full Model Architecture

```
token_ids (T,) Int32
     │
     ▼
Embedding (V, D) ──── weight-tied with LM head
     │ + LearnedPositionalEncoding
     ▼
TransformerBlock × N
  ├── pre-norm  LayerNorm(D)
  ├── MHA       MultiHeadSelfAttention(D, H) + residual
  ├── pre-norm  LayerNorm(D)
  └── FFN       Linear(D, 4D) → GELU → Linear(4D, D) + residual
     │
     ▼
LayerNorm(D)   (final)
     │
     ▼
logits = x @ tok_emb_weight^T   (T, V)   weight-tied
```

## API

```cpp
#include "sub0llm/nn/gpt.hpp"
using namespace sub0llm::nn;

// Construct model
GPT model(
    /*vocab_size=*/50257,
    /*embed_dim=*/768,
    /*num_heads=*/12,
    /*num_layers=*/12,
    /*max_seq_len=*/1024,
    /*seed=*/42);

// Forward pass
Tensor ids = ...;                 // (T,) Int32
Variable logits = model.forward(ids);   // (T, vocab_size) Variable

// Loss
Tensor targets = ...;             // (T,) Int32
auto loss = cross_entropy(logits, targets);  // scalar Variable

// Backward
loss.backward();

// Parameters
model.parameters();   // vector<Variable*> — all trainable params
```

## GELU Activation

```
GELU(x) = 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))

x:    -2     -1      0      1      2
gelu: -0.045 -0.159  0.000  0.841  1.955
grad:  0.019  0.084  0.500  0.916  1.085
```

Smoother than ReLU; non-zero gradient for x < 0 allows recovery from dying
units during training.

## Layer Normalisation

```
y = γ ⊙ (x − μ) / √(σ² + ε) + β

Normalises each token's D-dimensional feature vector independently.
γ (weight) and β (bias) are learnable — shape (D,) each.
```

## GPT-2 Parameter Counts

| Model | Layers | Heads | D | Params |
|-------|--------|-------|---|--------|
| GPT-2 small | 12 | 12 | 768 | 117 M |
| GPT-2 medium | 24 | 16 | 1024 | 345 M |
| GPT-2 XL | 48 | 25 | 1600 | 1.5 B |

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/gpt.hpp` | `GPT`, `TransformerBlock`, `LayerNorm`, `Linear`, `FeedForward` |
| `src/nn/gpt.cpp` | Full GPT implementation |
| `chapters/ch08_gpt/main.cpp` | Demo: GELU, LayerNorm, Linear, FFN, block, full model |
| `tests/test_gpt.cpp` | GPT forward/backward tests |
