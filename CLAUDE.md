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

## Current state (Ch01–Ch17, complete)

### Core
- `include/sub0llm/core/dtype.hpp` — DType enum, traits, `dtype_of<T>` concept mapping
- `include/sub0llm/core/device.hpp` — Device value type (CPU / CUDA / OpenVINO)
- `include/sub0llm/core/tensor.hpp` — Tensor: dynamic shape, strides, shared storage
- `include/sub0llm/core/ops.hpp` — Basic ops: add/sub/mul/div, reductions, matmul, activations

### Backends (Ch02)
- `src/backends/cpu/kernels.cpp`, `matmul.cpp` — SIMD-dispatched CPU kernels (AVX2/AVX-512)
- `src/backends/cuda/` — CUDA matmul and element-wise kernels
- `src/backends/openvino/` — OpenVINO dispatch

### Tokenizer (Ch03)
- `include/sub0llm/tokenizer/bpe.hpp`, `src/tokenizer/bpe.cpp` — Byte-Pair Encoding

### Data (Ch04)
- `include/sub0llm/data/dataset.hpp`, `dataloader.hpp` — Dataset and DataLoader

### Autograd (Ch05)
- `include/sub0llm/autograd/variable.hpp`, `ops.hpp` — Reverse-mode autograd, full op set including `log_sigmoid`
- `src/autograd/variable.cpp`, `ops.cpp`, `embedding_ops.cpp`

### Neural network modules (Ch06–Ch17)
- `include/sub0llm/nn/embedding.hpp` — Token and positional embeddings
- `include/sub0llm/nn/attention.hpp` — Multi-head attention (Ch07)
- `include/sub0llm/nn/gpt.hpp` — Vanilla GPT (Ch08)
- `include/sub0llm/nn/optimizer.hpp` — SGD, Adam, gradient clipping (Ch09)
- `include/sub0llm/nn/modern_gpt.hpp` — RMSNorm, SwiGLU, RoPE, GQA, ModernGPT+MTP (Ch10)
- `include/sub0llm/nn/scheduler.hpp`, `trainer.hpp` — LR schedulers, Trainer (Ch11)
- `include/sub0llm/nn/lora.hpp` — LoRA low-rank adaptation (Ch12)
- `include/sub0llm/nn/dpo.hpp` — Direct Preference Optimization loss (Ch13)
- `include/sub0llm/nn/sampler.hpp` — Greedy/temperature/top-k/top-p sampling, `generate` loop (Ch14)
- `include/sub0llm/nn/distillation.hpp` — Soft cross-entropy, knowledge distillation loss (Ch15)
- `include/sub0llm/nn/thinking.hpp` — `ThinkingConfig`, `ThinkingResult`, `generate_with_thinking`, `think_self_consistency` (Ch16)
- `include/sub0llm/nn/looped_gpt.hpp` — `LoopedGPT`: single block looped K times, `forward_k()` runtime budget (Ch17)

### Tests
329 Catch2 tests across 19 test files — all passing.

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
