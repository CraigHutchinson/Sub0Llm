# Chapter 05 — Autograd Engine

## Overview

Training requires computing `dL/dθ` for every model parameter θ. Hand-deriving
gradients for each architecture is error-prone. This chapter builds a
**dynamic computation graph** (define-by-run, like PyTorch) and runs
**reverse-mode automatic differentiation** (backpropagation) through it.

The core insight: build the graph during the forward pass, then traverse it
in topological reverse order to accumulate gradients. Cost: ~2–3× a forward
pass. Accuracy: exact (not numerical approximation).

## Computation Graph

```
y = relu(x * w + b)

  x ─┐
      mul ─── add ─── relu ─── y
  w ─┘         │
               b

Each node stores:
  data : Tensor   (forward value)
  grad : Tensor   (backward accumulator)
  vjp  : closure  (maps upstream_grad → input_grad)
```

## API

```cpp
#include "sub0llm/autograd/variable.hpp"
#include "sub0llm/autograd/ops.hpp"
using namespace sub0llm::autograd;

// Leaf variables (requires_grad=true → gradient will be computed)
Variable x(Tensor({3}, DType::Float32), /*requires_grad=*/true);
Variable w(Tensor({3}, DType::Float32), true);
Variable b(Tensor({3}, DType::Float32), true);

// Build graph (forward pass)
auto h   = add(mul(x, w), b);   // h = x*w + b
auto act = relu(h);              // act = relu(h)
auto L   = sum(act);             // L = scalar loss

// Backward pass — computes all gradients
L.backward();

// Read gradients
w.grad();   // Tensor — dL/dw
b.grad();   // Tensor — dL/db

// Zero gradients before next step
w.zero_grad();
b.zero_grad();

// Full op set includes:
// add, sub, mul, scale, matmul, bias_add, transpose2d
// relu, gelu, tanh, sigmoid, softmax, log_softmax, log_sigmoid
// sum, narrow, cross_entropy, embedding_lookup
```

## Gradient Accumulation

When a variable is used multiple times, gradients from all paths are **summed**:

```cpp
auto y = sum(add(x, x));   // x used twice
y.backward();
x.grad()[0];               // = 2.0  (contribution from each path)
```

## Gradient Check (XOR Demo)

```
XOR task: 4 samples (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0

Network:  2 → 4 → 2  (hidden=4, Xavier init)
Epoch   0: loss = 0.8124
Epoch  50: loss = 0.4431
Epoch 100: loss = 0.2188
Epoch 150: loss = 0.0842
Epoch 200: loss = 0.0241
Accuracy: 4/4  (solves XOR)
```

Numerical gradient check: `|analytical - numerical| < 1e-4` on every parameter.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/autograd/variable.hpp` | `Variable` class, backward |
| `include/sub0llm/autograd/ops.hpp` | All differentiable ops |
| `src/autograd/variable.cpp` | Graph traversal, topological sort |
| `src/autograd/ops.cpp` | VJP closures for all ops |
| `src/autograd/embedding_ops.cpp` | Embedding gather + scatter-add backward |
| `chapters/ch05_autograd/main.cpp` | Demo (§1–§7) |
| `tests/test_autograd.cpp` | Gradient correctness tests |
