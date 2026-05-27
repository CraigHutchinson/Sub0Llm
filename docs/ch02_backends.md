# Chapter 02 — Compute Backends

## Overview

Chapter 01 gave us correct, readable tensor ops. Chapter 02 makes them fast by
adding a three-tier hardware dispatch that selects the best kernel without
changing any calling code:

```
ops::add(a, b)
  │
  ├─ a.device().is_cuda()      → backend::cuda::add()       (cuBLAS / custom kernels)
  ├─ a.device().is_openvino()  → backend::openvino::add()   (oneDNN, stub)
  └─ CPU                        → backend::cpu::add_f32()
                                     ├─ SUB0LLM_AVX512 → 16-wide _mm512_add_ps
                                     ├─ SUB0LLM_AVX2   →  8-wide _mm256_add_ps
                                     └─ else           → scalar loop
```

## SIMD Fundamentals

AVX2 processes 8 `float32` values per instruction cycle. A 1 M-element vector
add takes ~128 K iterations (1M / 8) instead of 1 M.

| Width | Instruction | Floats/cycle |
|-------|-------------|--------------|
| Scalar | `addss` | 1 |
| SSE4 | `_mm_add_ps` | 4 |
| AVX2 | `_mm256_add_ps` | 8 |
| AVX-512 | `_mm512_add_ps` | 16 |

The **tail pattern** handles the remaining `N % 8` elements with a scalar loop
so correctness is maintained regardless of tensor size.

## Cache-Blocked Matrix Multiply

Naïve matmul (O(M·N·K)) repeatedly evicts data from L1 cache. The blocked
version tiles the three loops so that each tile fits in L1/L2:

```
for ii in range(0, M, TILE_M):
  for kk in range(0, K, TILE_K):
    for jj in range(0, N, TILE_N):
      # inner micro-kernel — AVX2 FMA stays in registers
```

Typical throughput (benchmarked in §2.3):

| Matrix | Naive | Blocked + AVX2 |
|--------|-------|----------------|
| 256×256 | ~0.3 GFLOP/s | ~8–12 GFLOP/s |
| 512×512 | ~0.2 GFLOP/s | ~6–10 GFLOP/s |

## API

```cpp
// Same ops::add call regardless of device
Tensor a_cpu  = randn({1024*1024});
Tensor result = add(a_cpu, b_cpu);     // dispatches to CPU SIMD

// Device transfer
Tensor cpu_t = randn({4, 4});
Tensor gpu_t = cpu_t.to(Device::cuda(0));    // DMA copy host→device
Tensor back  = gpu_t.to(Device::cpu());      // DMA copy device→host

// Compile-time backend detection
#ifdef SUB0LLM_AVX2   // set by -DSUB0LLM_ENABLE_AVX2=ON
// AVX2 intrinsics available
#endif
```

## Build Flags

| CMake flag | Effect |
|------------|--------|
| `-DSUB0LLM_ENABLE_AVX2=ON` | Enables 8-wide float kernels |
| `-DSUB0LLM_ENABLE_CUDA=ON` | Builds CUDA kernels, enables `Tensor::to(cuda)` |
| `-DSUB0LLM_ENABLE_OPENVINO=ON` | Enables OpenVINO dispatch |

## Files

| Path | Description |
|------|-------------|
| `src/backends/cpu/kernels.cpp` | AVX2/AVX-512 element-wise kernels |
| `src/backends/cpu/matmul.cpp` | Cache-blocked matmul with FMA |
| `src/backends/cuda/` | CUDA element-wise and matmul kernels |
| `src/backends/openvino/` | OpenVINO dispatch stubs |
| `chapters/ch02_backends/main.cpp` | Benchmark demo (§1–§5) |
| `tests/test_backends.cpp` | Backend correctness tests |
