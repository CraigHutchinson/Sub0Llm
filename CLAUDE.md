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

# Native release build — targets this machine's exact CPU (AVX-512, FMA, BMI, etc.)
# Enables: -march=native, -mtune=native, -ffp-contract=fast, -funroll-loops, LTO
# DO NOT distribute binaries from this build — they will SIGILL on other CPUs.
cmake -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUB0LLM_ENABLE_NATIVE=ON
cmake --build build-native --parallel
```

## Training policy

**Always use the native build for training runs** (`build-native/bin/`). It enables
`-march=native`, LTO, and fast-math, giving 3–4× throughput vs the debug build.
After any code change that affects a chapter binary, rebuild native before launching:

```bash
cmake --build build-native --parallel
# Then launch training, e.g.:
nohup ./build-native/bin/ch21_math_neurons \
  --phase train --ckpt-dir /tmp/ckpts --steps 3000 --token-mode real \
  > /tmp/train.log 2>&1 &
```

Tests use the debug build (`ctest --test-dir build`) — never run tests on native.

## Code conventions

- **Namespace**: `sub0llm` for the library; `sub0llm::ops` for ops
- **Error handling**: `std::runtime_error` with `std::format` messages; no
  custom exception hierarchy yet (Ch05 may add one)
- **No raw new/delete**: use `std::shared_ptr<std::byte[]>` for owned storage
- **No comments on obvious code**: only add comments when the WHY is non-obvious
- **Concepts over SFINAE**: use `template<ComputeScalar T>` style
- **`[[nodiscard]]` everywhere** on pure functions that return a new value
- **`noexcept`** only when truly impossible to throw (metadata accessors, etc.)

## Current state (Ch01–Ch22, complete)

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
- `include/sub0llm/nn/moe.hpp` — `MoEFeedForward`, `MoETransformerBlock`, `MoEGPT`: sparse top-k expert routing + load-balancing loss (Ch18)
- `include/sub0llm/nn/mtp.hpp` — `mtp_train_loss`, `mtp_generate`, `mtp_generate_stats`, `MtpGenStats`: Multi-Token Prediction — K+1 tokens per forward pass (Ch19)
- `include/sub0llm/nn/rlhf.hpp` — `RewardModel`, `reward_preference_loss`, `reinforce_loss`, `kl_penalty`: RLHF with Bradley-Terry preference training and KL-penalised REINFORCE (Ch20)
- `include/sub0llm/nn/math_nodes.hpp`, `src/nn/math_nodes.cpp` — `MathLayer`, `MathGPT`, `NumericRouter`: arithmetic-aware transformer with STE routing over 11 ops (FFN, Add, Sub, Mul, Div, IsLessThan, IsGreaterThan, IsEqual, Increment, Decrement, Sqrt); `apply_math_op`, `RouteType`, `RouteInfo`; configurable integer range via `NumericTokenizer(bpe, int_min, int_max)` (Ch21)
- `chapters/ch22_math_lm/` — General-purpose MathLM: parameter efficiency analysis, configurable int range (beyond int16), OOD generalization proof (exact compute vs memorization), mixed language+arithmetic training (Ch22)

### Autograd extensions
- `row_scale(x, v)` — scale each row i of (N,D) Variable x by scalar v[i,0]; used by MoE routing

### Tests
417 Catch2 tests across 22 test files — all passing.

## Git branch

All work goes to `claude/llm-cpp23-repo-init-1f1Il`.

## Dependencies (managed via CPM)

| Package | Use |
|---------|-----|
| spdlog | Logging throughout the library |
| nlohmann/json | Tokenizer vocab, model configs (Ch03+) |
| Catch2 v3 | Unit tests |
| OpenBLAS (system) | Optional BLAS-accelerated matmul (Ch02); dispatched for K≥64 |
| Eigen3 3.4.0 (CPM) | Header-only matmul fallback when system BLAS absent; dispatched for K≥64 |
| CUDAToolkit (system) | CUDA backend (Ch02) |
| OpenVINO (system) | Intel backend (Ch02) |
