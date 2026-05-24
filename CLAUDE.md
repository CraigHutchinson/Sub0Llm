# sub0llm — Developer Notes for Claude Code

## Project goals

Ground-up LLM implementation in **C++23** — educational, from tensors to RLHF.
No Python in the core library; Python is only used in `tools/` for data prep and
plotting scripts.

## Build commands

```bash
# Configure (first time — downloads CPM deps, requires internet)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# Build everything
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure -V

# Run a chapter
./build/bin/ch01_foundations

# Release build with AVX2
cmake -B build-rel -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build-rel --parallel
```

## Code conventions

- **Namespace**: `sub0llm` for the library; `sub0llm::ops` for ops
- **Error handling**: `std::runtime_error` with `std::format` messages; no
  custom exception hierarchy yet (Ch05 may add one)
- **No raw new/delete**: use `std::shared_ptr<std::byte[]>` for owned storage
- **No comments on obvious code**: only add comments when the WHY is non-obvious
- **Concepts over SFINAE**: use `template<ComputeScalar T>` style
- **`[[nodiscard]]` everywhere** on pure functions that return a new value
- **`noexcept`** only when truly impossible to throw (metadata accessors, etc.)

## Current state (Ch01)

- `include/sub0llm/core/dtype.hpp` — DType enum, traits, `dtype_of<T>` concept mapping
- `include/sub0llm/core/device.hpp` — Device value type (CPU / CUDA / OpenVINO)
- `include/sub0llm/core/tensor.hpp` — Tensor: dynamic shape, strides, shared storage
- `include/sub0llm/core/ops.hpp` — Basic ops: add/sub/mul/div, reductions, matmul, activations
- `src/core/tensor.cpp`, `src/core/ops.cpp` — Naive CPU implementations
- `tests/test_tensor.cpp`, `tests/test_ops.cpp` — Catch2 unit tests
- `chapters/ch01_foundations/main.cpp` — Chapter narrative / demo

## What changes in Ch02

- Replace naive loops in `ops.cpp` with SIMD-dispatched kernels
- Add `src/backends/cpu_simd.cpp` (AVX2/AVX-512 kernels)
- Add `src/backends/cuda/` (CUDA kernels for matmul, element-wise)
- Add `src/backends/openvino.cpp` (OpenVINO dispatch)
- `Tensor::to(Device)` becomes fully implemented

## Git branch

All work goes to `claude/llm-cpp23-repo-init-1f1Il`.

## Dependencies (managed via CPM)

| Package | Use |
|---------|-----|
| spdlog | Logging throughout the library |
| nlohmann/json | Tokenizer vocab, model configs (Ch03+) |
| Catch2 v3 | Unit tests |
| OpenBLAS (system) | Optional BLAS-accelerated matmul (Ch02) |
| CUDAToolkit (system) | CUDA backend (Ch02) |
| OpenVINO (system) | Intel backend (Ch02) |
