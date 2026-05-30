# sub0llm — LLMs from Scratch in C++23

A ground-up implementation of large language models in modern C++23, from raw
tensor operations through to pretraining, fine-tuning, RLHF, and a novel
exact-arithmetic reasoning head.  Inspired by Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) but entirely
in C++ with explicit hardware backends (CPU SIMD, CUDA, OpenVINO).

---

## Synopsis

sub0llm is a complete, self-contained LLM implementation split across 23
chapters.  Each chapter is a standalone executable that demonstrates a
coherent concept, builds on the previous chapters' infrastructure, and ships
with full unit-test coverage.

The project arc falls into four phases:

### Phase I — Infrastructure (Ch01–Ch05)
Build the bedrock: a stride-aware `Tensor` type, SIMD-dispatched compute
backends, a BPE tokenizer, a DataLoader, and a reverse-mode autograd engine.
By Ch05 we can write `loss.backward()` and gradients flow through arbitrary
expression trees.

### Phase II — Language Models (Ch06–Ch14)
Stack the standard LLM building blocks: token and positional embeddings, causal
multi-head attention, a vanilla GPT-2 architecture, Adam optimizer, then
upgrade to a modern LLaMA/Gemma/Qwen-style model (RMSNorm, SwiGLU, RoPE, GQA).
Ch11 adds a full pretraining loop with cosine LR schedule and Multi-Token
Prediction.  Ch12–Ch14 cover LoRA fine-tuning, DPO alignment, and
temperature/top-k/top-p sampling with an autoregressive generation loop.

### Phase III — Advanced Techniques (Ch15–Ch20)
Knowledge distillation (soft targets, temperature scaling), chain-of-thought
thinking tokens with budget control and self-consistency, the Universal
Transformer (LoopedGPT — one shared block looped K times), sparse
Mixture-of-Experts routing with load-balancing loss, K+1-token-per-step
Multi-Token Prediction, and RLHF with a Bradley-Terry reward model,
REINFORCE, and KL-penalised fine-tuning.

### Phase IV — Exact Arithmetic (Ch21–Ch23)
A novel hybrid architecture that replaces one transformer block with a
*math head*: a tiny learned router that selects one of 11 arithmetic
operations (Add, Sub, Mul, Div, comparisons, Sqrt, Inc, Dec) and executes it
exactly on the integer register.  The router adds only ~8% extra parameters
(a single `Linear(D, 11)` layer) yet delivers 100% OOD arithmetic accuracy
vs 0% for a pure-LLM baseline.

Ch22 shows the architecture is range-invariant — arbitrary integer ranges
beyond int16 — and that parameter efficiency scales favourably with model
size.  Ch23 extends this to **multi-step chain-of-thought reasoning**: because
each generated numeric token re-enters the register, a two-step chain
"A + B = C − D = E" remains exact at every step without any architecture
changes; errors do not compound.  Natural language word problems ("Alice has A,
Bob gives B, she now has A + B = C") work correctly because the last-two-
numerics rule picks the explicitly-written operands regardless of stray numbers
in prose.

---

## Why C++23?

Python implementations hide the mechanics behind NumPy/PyTorch.  Building in
C++ exposes every memory layout decision, every kernel dispatch, and every
byte of a model weight — making it an ideal companion for deep understanding.
C++23 gives us `std::format`, concepts, ranges, and `std::mdspan` to write
expressive code without sacrificing performance.

---

## Chapter Roadmap

| # | Chapter | Key concepts |
|---|---------|-------------|
| 01 | **Core Foundations** | `Tensor`, `DType`, `Device`, strides, factory ops, element-wise ops |
| 02 | **Compute Backends** | AVX2/AVX-512 SIMD, cache-blocked matmul, CUDA kernels, OpenVINO dispatch |
| 03 | **Tokenization** | BPE algorithm, GPT-2 space marker, encode/decode, save/load |
| 04 | **Dataset & DataLoader** | Sliding-window chunks, 1-token shift, shuffled mini-batches |
| 05 | **Autograd Engine** | Dynamic computation graph, reverse-mode AD, gradient check |
| 06 | **Embeddings** | Token embeddings (scatter-add backward), sinusoidal PE, learned PE, RoPE |
| 07 | **Attention Mechanisms** | Scaled dot-product, causal masking, multi-head self-attention |
| 08 | **GPT Architecture** | GELU, LayerNorm, FFN, TransformerBlock, full GPT-2 |
| 09 | **Optimizers** | SGD+momentum, Adam (bias correction), gradient clipping |
| 10 | **Modern Architecture** | RMSNorm, SwiGLU, RoPE, GQA, ModernGPT (LLaMA/Gemma/Qwen style) |
| 11 | **Pretraining** | Cosine LR schedule, `narrow` op, MTP cross-entropy loss, full training loop |
| 12 | **LoRA Fine-Tuning** | Low-rank adaptation, frozen base, gradient isolation, rank ablation |
| 13 | **DPO Alignment** | Direct Preference Optimization, Bradley-Terry model, implicit reward |
| 14 | **Inference & Sampling** | Greedy, temperature, top-k, top-p, autoregressive generation loop |
| 15 | **Knowledge Distillation** | Teacher-student training, soft targets, temperature scaling |
| 16 | **Thinking Tokens** | Chain-of-thought generation, special tokens, budget control, self-consistency |
| 17 | **LoopedGPT** | Universal Transformer, single shared block looped K times, inference budget |
| 18 | **Mixture-of-Experts** | Sparse top-k routing, expert load-balancing auxiliary loss |
| 19 | **Multi-Token Prediction** | K+1 tokens per forward pass, aux weight, DeepSeek MTP style |
| 20 | **RLHF** | RewardModel, Bradley-Terry preference loss, REINFORCE, KL penalty |
| 21 | **Specialised Math Neurons** | `NumericTokenizer` (configurable int range), `MathLayer`, STE router over 11 ops, exact arithmetic execution, OOD generalisation proof |
| 22 | **General-Purpose MathLM** | Parameter efficiency analysis (~8% overhead), int range scaling, mixed language + arithmetic training, OOD accuracy 100% vs 0% baseline |
| 23 | **Reasoned Arithmetic** | Multi-step chain-of-thought with exact math head, register walkthrough, natural language word problems, 3-step chains, OOD multi-step generalisation |

> **Test coverage**: 430 Catch2 tests across 23 test files — all passing.

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
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

For the full step-by-step guide including CUDA, native builds, Ollama synthetic
data, training commands, and troubleshooting, see
**[docs/local_machine_setup.md](docs/local_machine_setup.md)**.

### Run a chapter

```bash
./build/bin/ch01_foundations
./build/bin/ch10_modern_arch
./build/bin/ch20_rlhf
./build/bin/ch23_reasoned_math            # all sections
./build/bin/ch23_reasoned_math --phase register   # just the register walkthrough
```

### Run tests

```bash
ctest --test-dir build --output-on-failure
```

### Native release build (for training runs)

Enables `-march=native`, LTO, and fast-math — 3–4× throughput vs debug.
**Do not distribute binaries from this build.**

```bash
cmake -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUB0LLM_ENABLE_NATIVE=ON
cmake --build build-native --parallel
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
│   ├── tokenizer/          # BPE + NumericTokenizer
│   ├── data/               # Dataset, DataLoader
│   └── nn/                 # All neural network modules
├── src/                    # Library implementation
├── chapters/               # One executable per chapter (ch01–ch23)
├── tests/                  # Catch2 unit tests (430 tests, 23 files)
└── tools/                  # Python scripts for data prep, plotting
```

---

## Key library modules

| Header | What it provides |
|--------|-----------------|
| `core/tensor.hpp` | Dynamic-shape, stride-aware `Tensor`; shared storage |
| `core/ops.hpp` | Add, sub, mul, div, reductions, matmul, activations |
| `autograd/variable.hpp` | `Variable` — autograd node wrapping a `Tensor` |
| `autograd/ops.hpp` | Full differentiable op set; `cross_entropy`, `narrow`, `row_scale` |
| `tokenizer/bpe.hpp` | BPE tokenizer with save/load |
| `tokenizer/numeric_tokenizer.hpp` | Extends BPE with a configurable integer vocabulary (`int_min`…`int_max`), NaN and overflow tokens |
| `nn/modern_gpt.hpp` | `ModernGPT`: RMSNorm, SwiGLU, RoPE, GQA — production-style LLM |
| `nn/math_nodes.hpp` | `MathGPT`, `MathLayer`, `apply_math_op`, `RouteType` (11 ops), `RouteInfo` — exact arithmetic transformer |
| `nn/optimizer.hpp` | SGD, Adam, gradient clipping |
| `nn/lora.hpp` | LoRA low-rank adaptation |
| `nn/sampler.hpp` | Greedy / temperature / top-k / top-p + `generate` loop |
| `nn/moe.hpp` | Sparse MoE: `MoEFeedForward`, `MoEGPT`, load-balancing loss |
| `nn/rlhf.hpp` | `RewardModel`, preference loss, REINFORCE, KL penalty |

---

## Dependencies (managed via CPM)

| Package | Use |
|---------|-----|
| spdlog | Logging throughout the library |
| nlohmann/json | Tokenizer vocab, model configs |
| Catch2 v3 | Unit tests |
| cpp-httplib (CPM) | Cross-platform HTTP client for Ollama synthetic data (Ch24) |
| OpenBLAS (system) | Optional BLAS-accelerated matmul (dispatched for K≥64) |
| Eigen3 3.4.0 (CPM) | Header-only matmul fallback when system BLAS absent |
| CUDAToolkit (system) | CUDA backend |
| OpenVINO (system) | Intel backend |

---

## References

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — the Python counterpart
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for SIMD kernels, quantisation
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) — MTP modules
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [DPO paper](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [Switch Transformer](https://arxiv.org/abs/2101.03961) — MoE load balancing
- [C++23 standard](https://en.cppreference.com/w/cpp/23) — `std::format`, concepts, ranges

---

## Licence

MIT
