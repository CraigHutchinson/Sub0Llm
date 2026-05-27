# Chapter 10 — Modern Architecture (LLaMA / Gemma / Qwen Style)

## Overview

GPT-2 (Ch08) uses LayerNorm, GELU, and full multi-head attention. State-of-the-art
models (LLaMA, Gemma, Mistral, Qwen) replace these with lighter, more expressive
components:

| Component | GPT-2 (Ch08) | Modern (Ch10) |
|-----------|-------------|---------------|
| Normalisation | LayerNorm (mean+var) | **RMSNorm** (var only, 2× cheaper) |
| Activation | GELU | **SwiGLU** (gated, more expressive) |
| Position | Learned absolute | **RoPE** (relative, baked into QK) |
| KV attention | Full MHA (H KV heads) | **GQA** (n_kv_heads ≤ H) |
| Training signal | Next-token only | **MTP heads** (K future tokens) |

## Architecture

```
token_ids (T,) Int32
     │
     ▼
Embedding (V, D)                         ← weight-tied with LM head
     │
     ▼  ┌──────────────────────────────────────────────┐
     │  │  ModernTransformerBlock                       │ × N layers
     │  │    RMSNorm → GQA (RoPE baked in) → residual  │
     │  │    RMSNorm → SwiGLU FFN → residual            │
     │  └──────────────────────────────────────────────┘
     │
     ▼
RMSNorm (final)
     │
     ▼
logits = x @ tok_emb_weight^T    (T, V)   weight-tied LM head
```

## Components

### RMSNorm
```
y = x / RMS(x) * γ     RMS(x) = sqrt(mean(x²) + ε)
```
No mean subtraction, no bias → 2× cheaper than LayerNorm. γ initialised to 1.

### SwiGLU Feed-Forward
```
d_ff = 8*D/3 rounded up to multiple of 64  (auto if d_ff=0)
y = (gate(x) * SiLU(up(x))) @ down^T

3 projections: gate: D→d_ff, up: D→d_ff, down: d_ff→D
```

### Grouped-Query Attention (GQA)
```
n_kv_heads ≤ n_heads
Each KV head serves (n_heads / n_kv_heads) query heads
```

Parameter savings vs full MHA (D=512, n_heads=8):

| n_kv_heads | Params | KV reduction |
|------------|--------|-------------|
| 8 (MHA) | 1,048,576 | 1× |
| 4 (GQA) | 786,432 | 2× |
| 2 (GQA) | 655,360 | 4× |
| 1 (MQA) | 589,824 | 8× |

## API

```cpp
#include "sub0llm/nn/modern_gpt.hpp"
using namespace sub0llm::nn;

ModernGPT model(
    /*vocab_size=*/32000,
    /*embed_dim=*/4096,
    /*n_heads=*/32,
    /*n_kv_heads=*/8,       // GQA: 4 query heads per KV head
    /*n_layers=*/32,
    /*d_ff=*/0,             // auto = 8*D/3 ↑ 64
    /*n_mtp_heads=*/0,      // set >0 for multi-token prediction (Ch19)
    /*seed=*/42);

// Standard forward
auto logits = model.forward(ids);   // (T, vocab_size)

// MTP forward (Ch10 preview, full treatment in Ch19)
auto heads = model.forward_mtp(ids);  // vector of K+1 Variables

model.vocab_size();
model.embed_dim();
model.parameters();   // all trainable parameters
```

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/modern_gpt.hpp` | `ModernGPT`, `RMSNorm`, `SwiGLUFeedForward`, `GroupedQueryAttention`, `ModernTransformerBlock` |
| `src/nn/modern_gpt.cpp` | Full modern architecture |
| `chapters/ch10_modern_arch/main.cpp` | Demo: §10.1 SiLU vs GELU, §10.2 RMSNorm, §10.3 SwiGLU, §10.4 RoPE, §10.5 GQA params, §10.6 forward+backward, §10.7 MTP, §10.8 training |
| `tests/test_modern_gpt.cpp` | ModernGPT tests |
