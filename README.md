# sub0llm — LLMs from Scratch in C++23

A ground-up implementation of large language models in modern C++23, from raw
tensor operations through to pretraining, fine-tuning, RLHF, and beyond.
Inspired by Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
but entirely in C++ with explicit hardware backends (CPU SIMD, CUDA, OpenVINO).

---

## Why C++23?

Python implementations hide the mechanics behind NumPy/PyTorch. Building in
C++ exposes every memory layout decision, every kernel dispatch, and every
byte of a model weight — making it an ideal companion for deep understanding.
C++23 gives us `std::mdspan`, `std::expected`, `std::format`, and concepts to
write expressive code without sacrificing performance.

---

## Chapter Roadmap

### Completed

| # | Chapter | Key concepts | Doc |
|---|---------|-------------|-----|
| 01 | **Core Foundations** | Tensor, DType, Device, strides, factory ops, element-wise ops | [docs](docs/ch01_foundations.md) |
| 02 | **Compute Backends** | AVX2/AVX-512 SIMD, cache-blocked matmul, CUDA kernels, OpenVINO dispatch | [docs](docs/ch02_backends.md) |
| 03 | **Tokenization** | BPE algorithm, GPT-2 space marker, encode/decode, save/load | [docs](docs/ch03_tokenization.md) |
| 04 | **Dataset & DataLoader** | Sliding-window chunks, 1-token shift, shuffled mini-batches | [docs](docs/ch04_dataset.md) |
| 05 | **Autograd Engine** | Dynamic computation graph, reverse-mode AD, gradient check | [docs](docs/ch05_autograd.md) |
| 06 | **Embeddings** | Token embeddings (scatter-add backward), sinusoidal PE, learned PE, RoPE | [docs](docs/ch06_embeddings.md) |
| 07 | **Attention Mechanisms** | Scaled dot-product, causal masking, multi-head self-attention | [docs](docs/ch07_attention.md) |
| 08 | **GPT Architecture** | GELU, LayerNorm, FFN, TransformerBlock, full GPT-2 | [docs](docs/ch08_gpt.md) |
| 09 | **Optimizers** | SGD+momentum, Adam (bias correction), gradient clipping | [docs](docs/ch09_optimizer.md) |
| 10 | **Modern Architecture** | RMSNorm, SwiGLU, RoPE, GQA, ModernGPT (LLaMA/Gemma/Qwen style) | [docs](docs/ch10_modern_arch.md) |
| 11 | **Pretraining with MTP** | Cosine LR schedule, `narrow` op, MTP cross-entropy loss, full training loop | [docs](docs/ch11_pretraining.md) |
| 12 | **LoRA Fine-Tuning** | Low-rank adaptation, frozen base, gradient isolation, rank ablation | [docs](docs/ch12_finetuning.md) |
| 13 | **DPO Alignment** | Direct Preference Optimization, Bradley-Terry model, implicit reward | [docs](docs/ch13_alignment.md) |
| 14 | **Inference & Sampling** | Greedy, temperature, top-k, top-p, autoregressive generation loop | [docs](docs/ch14_inference.md) |
| 15 | **Knowledge Distillation** | Teacher-student training, soft targets, temperature scaling | [docs](docs/ch15_distillation.md) |
| 16 | **Thinking Tokens** | Chain-of-thought generation, special tokens, budget control, self-consistency | [docs](docs/ch16_thinking.md) |
| 17 | **LoopedGPT** | Universal Transformer, single shared block looped K times, inference budget | [docs](docs/ch17_looped_gpt.md) |
| 18 | **Mixture-of-Experts** | Sparse routing, top-k experts, load-balancing auxiliary loss | [docs](docs/ch18_moe.md) |
| 19 | **Multi-Token Prediction** | K+1 tokens per forward pass, aux weight, DeepSeek MTP comparison | [docs](docs/ch19_mtp.md) |
| 20 | **RLHF** | RewardModel, Bradley-Terry preference loss, REINFORCE, KL penalty | [docs](docs/ch20_rlhf.md) |

### Planned / Research

| # | Chapter | Key concepts | Doc |
|---|---------|-------------|-----|
| 21 | **Specialised Math Neurons** *(research concept)* | Numeric token range, exact arithmetic execution nodes, learned router, classical-computation integration | [research doc](docs/ch21_math_neurons_research.md) |

> **Test coverage**: 384 Catch2 tests across 22 test files — all passing.

---

## Build

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.25 |
| GCC | 13 or Clang 17 (C++23 support) |
| Ninja (optional) | any |

Optional backends:
- CUDA Toolkit ≥ 12 — pass `-DSUB0LLM_ENABLE_CUDA=ON`
- Intel OpenVINO ≥ 2024 — pass `-DSUB0LLM_ENABLE_OPENVINO=ON`

### Quick start (CPU only)

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build --parallel
```

### Run a chapter

```bash
./build/bin/ch01_foundations
./build/bin/ch10_modern_arch
./build/bin/ch20_rlhf
```

### Run tests

```bash
ctest --test-dir build --output-on-failure
```

### CUDA build

```bash
cmake -B build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
```

---

## Project structure

```
Sub0Llm/
├── CMakeLists.txt          # Root CMake
├── cmake/
│   ├── CPM.cmake           # Package manager (vendored)
│   ├── options.cmake       # Feature toggles
│   ├── compiler_flags.cmake
│   └── deps.cmake          # External dependencies
├── include/sub0llm/
│   ├── core/               # Tensor, DType, Device, ops
│   ├── autograd/           # Variable, differentiable ops
│   ├── tokenizer/          # BPE tokenizer
│   ├── data/               # Dataset, DataLoader
│   └── nn/                 # All neural network modules
├── src/                    # Library implementation
├── chapters/               # One executable per chapter (ch01–ch20)
├── docs/                   # Per-chapter documentation
├── tests/                  # Catch2 unit tests (384 tests)
└── tools/                  # Python scripts for data prep, plotting
```

---

## Dependencies (managed via CPM)

| Package | Use |
|---------|-----|
| spdlog | Logging throughout the library |
| nlohmann/json | Tokenizer vocab, model configs |
| Catch2 v3 | Unit tests |
| OpenBLAS (system) | Optional BLAS-accelerated matmul |
| CUDAToolkit (system) | CUDA backend |
| OpenVINO (system) | Intel backend |

---

## References

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — the Python counterpart
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for GGUF format, SIMD kernels, quantisation
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) — MTP modules
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [DPO paper](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Switch Transformer](https://arxiv.org/abs/2101.03961) — MoE load balancing
- [C++23 standard](https://en.cppreference.com/w/cpp/23) — `std::mdspan`, `std::expected`, `std::print`

---

## Licence

MIT
