# sub0llm — LLMs from Scratch in C++23

A ground-up implementation of large language models in modern C++23, from raw
tensor operations through to pretraining, fine-tuning, and inference.  Inspired
by Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
but entirely in C++ with explicit hardware backends (CPU SIMD, CUDA, OpenVINO).

---

## Why C++23?

Python implementations hide the mechanics behind NumPy/PyTorch.  Building in
C++ exposes every memory layout decision, every kernel dispatch, and every
byte of a model weight — making it an ideal companion for deep understanding.
C++23 gives us `std::mdspan`, `std::expected`, `std::format`, and concepts to
write expressive code without sacrificing performance.

---

## Chapter roadmap

| # | Chapter | Key concepts |
|---|---------|-------------|
| 01 | **Core Foundations** | Tensor, DType, Device, strides, factory ops |
| 02 | **Compute Backends** | AVX2/AVX-512 SIMD, CUDA kernels, OpenVINO dispatch |
| 03 | **Tokenization** | BPE, Unicode, vocabulary serialisation |
| 04 | **Dataset & DataLoader** | Streaming text, batching, shuffling |
| 05 | **Autograd Engine** | Computation graph, reverse-mode AD |
| 06 | **Embeddings** | Token embeddings, positional encodings, RoPE |
| 07 | **Attention Mechanisms** | Scaled dot-product, causal masking, MHA, GQA |
| 08 | **GPT Architecture** | LayerNorm, MLP (GELU), Transformer block, GPT-2 |
| 09 | **Optimizers** | SGD, Adam, AdamW, gradient clipping, LR schedules |
| 10 | **Pretraining** | Cross-entropy loss, training loop, checkpointing |
| 11 | **Model Serialisation** | Save/load, GGUF-compatible format, weight conversion |
| 12 | **Fine-Tuning** | Classification head, SFT, LoRA / QLoRA |
| 13 | **RLHF & DPO** | Reward modelling, PPO, Direct Preference Optimisation |
| 14 | **Inference & Serving** | KV cache, quantisation, sampling strategies |

> **Rationale for C++-native ordering vs the original:**
> Chapters 1–2 move hardware and tensor foundations to the front because C++ has
> no equivalent of `import torch`.  The autograd engine (Ch05) similarly must be
> built before any training can occur.  Model serialisation (Ch11) is promoted
> because GGUF compatibility is central to the C++ LLM ecosystem (llama.cpp).

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

### Run Chapter 01

```bash
./build/bin/ch01_foundations
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
│   ├── version.hpp
│   └── core/
│       ├── dtype.hpp       # DType enum + traits
│       ├── device.hpp      # Device abstraction
│       ├── tensor.hpp      # Tensor class
│       └── ops.hpp         # Basic ops (element-wise, matmul, activations)
├── src/                    # Library implementation
├── chapters/               # One executable per chapter
│   ├── ch01_foundations/
│   └── ...
├── tests/                  # Catch2 unit tests
└── tools/                  # Python scripts for data prep, plotting
```

---

## References

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — the Python counterpart
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for GGUF format, SIMD kernels, quantisation
- [C++23 standard](https://en.cppreference.com/w/cpp/23) — `std::mdspan`, `std::expected`, `std::print`

---

## Licence

MIT
