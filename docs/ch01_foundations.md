# Chapter 01 — Core Foundations

## Overview

Before any neural network can run, we need a tensor abstraction. Python projects
use NumPy or PyTorch transparently; building in C++23 exposes every memory
layout decision, every kernel dispatch, and every byte of a model weight.

This chapter establishes `Tensor` — the single data structure underpinning every
later chapter — along with the `DType` type system, the `Device` abstraction, and
the basic operations that the rest of the library builds on.

## Core Abstractions

```
DType enum ─── dtype_of<T> concept ─── dtype_size / dtype_name
     │
Device ────────── Device::cpu() / Device::cuda(n) / Device::openvino()
     │
Tensor ─────────── shape, strides, shared_storage (std::shared_ptr<std::byte[]>)
     │                   │
     │              reshape / transpose → view (no copy when contiguous)
     │              contiguous()        → materialises a copy when needed
     │
ops:: ──────────── add, sub, mul, div, matmul, sum, mean, max
                   relu, softmax, gelu, sigmoid
```

## DType System

| DType | Size | C++ type |
|-------|------|----------|
| Float32 | 4 B | `float` |
| Float16 | 2 B | `_Float16` |
| BFloat16 | 2 B | `bfloat16_t` |
| Float64 | 8 B | `double` |
| Int8 | 1 B | `int8_t` |
| Int32 | 4 B | `int32_t` |
| Int64 | 8 B | `int64_t` |

```cpp
dtype_of<float>      // → DType::Float32  (compile-time)
dtype_size(DType::Float16)  // → 2
dtype_name(DType::Int32)    // → "Int32"
```

## API

```cpp
#include "sub0llm/core/tensor.hpp"
#include "sub0llm/core/ops.hpp"
using namespace sub0llm;
using namespace sub0llm::ops;

// Factory functions
Tensor z  = zeros({3, 4});              // shape (3,4), Float32 by default
Tensor o  = ones({2, 3});
Tensor r  = randn({4, 4});              // N(0,1) random
Tensor a  = arange(6);                  // [0,1,2,3,4,5]

// Shape and strides
Tensor t3d = arange(24).reshape({2, 3, 4});  // view, no copy
Tensor mT  = arange(6).reshape({2,3}).transpose(0, 1);  // (3,2) view
bool c = mT.is_contiguous();               // false; call .contiguous() if needed

// Typed data access via std::span
Tensor x = zeros({5}, DType::Float32);
auto sp  = x.data_as<float>();             // std::span<float>
sp[0]    = 1.0f;

// Element-wise ops and reductions
Tensor c  = add(a, b);
Tensor d  = mul(a, 2.0f);
float  s  = sum(a);

// Activations
Tensor probs = softmax(logits);
Tensor h     = gelu(x);

// Matrix multiply
Tensor C = matmul(A, B);                // (M,K) × (K,N) → (M,N)
```

## Key Design Points

- **Strides**: `transpose` and `reshape` return views with adjusted strides — no
  data copy unless `contiguous()` is called.
- **Shared storage**: `std::shared_ptr<std::byte[]>` allows cheap copy semantics
  without reference counting per element.
- **DType dispatch**: ops select the right kernel branch at runtime based on
  `tensor.dtype()`.
- **`dtype_of<T>` concept**: maps C++ types to `DType` values at compile time,
  enabling type-safe `data_as<float>()` access.

## Files

| Path | Description |
|------|-------------|
| `include/sub0llm/core/dtype.hpp` | DType enum, traits, `dtype_of<T>` |
| `include/sub0llm/core/device.hpp` | Device abstraction |
| `include/sub0llm/core/tensor.hpp` | Tensor class |
| `include/sub0llm/core/ops.hpp` | Basic ops |
| `chapters/ch01_foundations/main.cpp` | Demo (§1–§10) |
| `tests/test_tensor.cpp` | Tensor unit tests |
