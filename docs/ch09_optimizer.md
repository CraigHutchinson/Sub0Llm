# Chapter 09 — Optimizers

## Overview

With a full GPT-style model and autograd, we need optimizers to update
parameters. This chapter implements gradient clipping and two fundamental
optimizers, then trains a tiny GPT on a synthetic circular sequence task.

## Gradient Clipping

Exploding gradients during early training are prevented by scaling all gradients
when their global L2 norm exceeds a threshold:

```
global_norm = sqrt(Σ_i ||g_i||²)
if global_norm > max_norm:
    g_i *= max_norm / global_norm
```

```cpp
float norm_after = clip_grad_norm(params, /*max_norm=*/1.0f);
```

## SGD with Momentum

```
v_{t+1} = momentum * v_t - lr * g_t
p_{t+1} = p_t + v_{t+1}
```

Momentum accumulates gradient history and dampens oscillations in narrow
valleys.

## Adam (Kingma & Ba, 2015)

```
m_{t+1} = β₁ * m_t + (1-β₁) * g        (first moment — mean)
v_{t+1} = β₂ * v_t + (1-β₂) * g²       (second moment — variance)
m̂ = m_{t+1} / (1 - β₁^t)               (bias correction)
v̂ = v_{t+1} / (1 - β₂^t)
p_{t+1} = p_t - lr * m̂ / (√v̂ + ε)
```

Adaptive per-parameter learning rates: parameters with large, consistent
gradients get a smaller effective step; sparse gradients get a larger step.

## API

```cpp
#include "sub0llm/nn/optimizer.hpp"
using namespace sub0llm::nn;

auto params = model.parameters();

// SGD
SGD sgd(params, /*lr=*/0.01f, /*momentum=*/0.9f);
sgd.zero_grad();
loss.backward();
sgd.step();

// Adam (recommended defaults: β₁=0.9, β₂=0.999, ε=1e-8)
Adam adam(params, /*lr=*/3e-4f);
adam.zero_grad();
loss.backward();
(void)clip_grad_norm(params, /*max_norm=*/1.0f);
adam.step();
```

## Training Demo — Circular Sequence

```
Task: predict next token in 0→1→2→…→7→0 (V=32, T=8)
Random-guess loss (log 32) = 3.4657

  step   1: loss = 3.4401
  step  10: loss = 3.1248
  step  20: loss = 2.7831
  step  30: loss = 2.2107
  step  40: loss = 1.6523
  step  50: loss = 1.1044
```

Adam converges noticeably faster than SGD on this task due to per-parameter
adaptive step sizes.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/nn/optimizer.hpp` | `SGD`, `Adam`, `clip_grad_norm` |
| `src/nn/optimizer.cpp` | Optimizer implementations |
| `chapters/ch09_optimizer/main.cpp` | Demo: clipping, SGD, Adam, training |
| `tests/test_optimizer.cpp` | Optimizer correctness tests |
