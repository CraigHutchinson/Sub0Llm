# sub0llm — LLMs from Scratch in C++23

A ground-up implementation of large language models in modern C++23, from raw
tensor operations through to pretraining, fine-tuning, RLHF, and a novel
exact-arithmetic reasoning head.  Inspired by Sebastian Raschka's
[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) but entirely
in C++ with explicit hardware backends (CPU SIMD, CUDA, OpenVINO).

---

## Synopsis

sub0llm is a complete, self-contained LLM implementation split across 26
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

### Phase V — Real-World Deployment (Ch24–Ch26)
Ch24 assembles every library component into a full pretraining loop with
BPE tokenization, streaming TextCorpus, Chinchilla-scaled batch sizing,
checkpoint save/resume, and a demonstration on Shakespeare.  A CLI tool and
OpenAI-compatible HTTP server (sub0llm-cli / sub0llm-server) expose any
trained or GGUF-loaded model for interactive use.

Ch25 adds production inference: a KV cache (~9× latency speedup),
sliding-window attention for O(n·W) memory, and RoPE NTK-aware scaling
for extending beyond training context length without fine-tuning.

Ch26 introduces **episodic memory**: a three-tier framework (working →
episodic → semantic) where fast-weight delta LoRA updates let the model
acquire novel facts (ones absent from pretraining) in seconds via a
targeted comprehension-pass → thinking-loop → gradient-write cycle.
A GGUF loader enables running the episodic pipeline on Qwen2/Qwen3
community weights.  The `sub0llm-episodic probe` command validates episodic
encoding on novel facts using a 4-condition test (high baseline NLL, NLL
reduction, query transfer, specificity).

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
| 24 | **Real-World Pretraining** | `TextCorpus` streaming BPE pipeline, Chinchilla scaling, checkpoint save/resume, Ollama synthetic data, Shakespeare demo |
| 25 | **Long-Context Inference** | KV cache (~9× speedup), sliding-window attention (O(n·W) memory), RoPE NTK-aware scaling, `generate_cached()` loop |
| 26 | **Episodic Memory** | Three-tier memory framework, fast-weight delta LoRA, comprehension pass → thinking loop → targeted write, GGUF loader (Qwen2/Qwen3), `sub0llm-episodic` CLI |

> **Test coverage**: 474 Catch2 tests across 26 test files — all passing.

---

## Build

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| CMake | 3.25 |
| GCC | 13 or Clang 17 (C++23 support) |
| Ninja (optional) | any |

Optional backends:
- CUDA Toolkit ≥ 12 — use the `cuda` or `cuda-native` preset
- Intel OpenVINO ≥ 2024 — use the `openvino` preset

### Quick start with presets

```bash
# List all available presets
cmake --list-presets=all

# Configure + build + test in one go (debug build, AVX2)
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

### Preset reference

| Preset | Build type | SIMD | Extras | Use for |
|--------|-----------|------|--------|---------|
| `debug` | Debug | AVX2 | — | Development, unit tests |
| `release` | Release | AVX2 | LTO | Distributable binaries |
| `avx512` | Release | AVX-512 | LTO | Skylake-X / Ice Lake+ servers |
| `native` | Release | host-native | LTO | **Training runs** (do not distribute) |
| `reldbg` | RelWithDebInfo | AVX2 | — | Profiling, crash investigation |
| `asan` | Debug | AVX2 | ASan + UBSan | Memory and undefined-behaviour checks |
| `cuda` | Release | AVX2 | CUDA, LTO | NVIDIA GPU inference |
| `cuda-native` | Release | host-native | CUDA, LTO | **GPU training** (do not distribute) |
| `openvino` | Release | AVX2 | OpenVINO, LTO | Intel hardware |
| `lib-only` | Release | AVX2 | LTO | Embedding sub0llm as a library |
| `ci` | Debug | AVX2 | — | CI pipelines |

Each preset writes its build tree to `build-<preset>` so multiple presets
can coexist without interfering.

> **Training policy**: always use `native` (CPU) or `cuda-native` (GPU). They
> enable `-march=native`, LTO, and FMA fusion for 3–4× throughput over `debug`.
> Never use these presets for binaries you intend to distribute.

### Manual configure (without presets)

```bash
# Debug (equivalent to --preset debug)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel

# Native release (equivalent to --preset native)
cmake -B build-native -G Ninja -DCMAKE_BUILD_TYPE=Release -DSUB0LLM_ENABLE_NATIVE=ON
cmake --build build-native --parallel
```

For the full step-by-step guide including CUDA, Ollama synthetic data,
training commands, and troubleshooting, see
**[docs/local_machine_setup.md](docs/local_machine_setup.md)**.

### Chapter docs

Each chapter has a companion design document in `docs/`:

| Doc | Contents |
|-----|----------|
| [ch01_foundations.md](docs/ch01_foundations.md) | Tensor internals, DType, Device |
| [ch02_backends.md](docs/ch02_backends.md) | SIMD dispatch, CUDA, OpenVINO |
| [ch03_tokenization.md](docs/ch03_tokenization.md) | BPE algorithm, GPT-2 encoding |
| [ch04_dataset.md](docs/ch04_dataset.md) | DataLoader, sliding-window chunks |
| [ch05_autograd.md](docs/ch05_autograd.md) | Reverse-mode AD, gradient check |
| [ch06_embeddings.md](docs/ch06_embeddings.md) | Token/positional embeddings, RoPE |
| [ch07_attention.md](docs/ch07_attention.md) | Scaled dot-product, causal masking |
| [ch08_gpt.md](docs/ch08_gpt.md) | GELU, LayerNorm, GPT-2 architecture |
| [ch09_optimizer.md](docs/ch09_optimizer.md) | SGD, Adam, gradient clipping |
| [ch10_modern_arch.md](docs/ch10_modern_arch.md) | RMSNorm, SwiGLU, GQA, ModernGPT |
| [ch11_pretraining.md](docs/ch11_pretraining.md) | Cosine LR, MTP loss, pretraining loop |
| [ch12_finetuning.md](docs/ch12_finetuning.md) | LoRA, frozen base, rank ablation |
| [ch13_alignment.md](docs/ch13_alignment.md) | DPO, Bradley-Terry, implicit reward |
| [ch14_inference.md](docs/ch14_inference.md) | Sampling strategies, generation loop |
| [ch15_distillation.md](docs/ch15_distillation.md) | Soft targets, temperature scaling |
| [ch16_thinking.md](docs/ch16_thinking.md) | Chain-of-thought, budget control |
| [ch17_looped_gpt.md](docs/ch17_looped_gpt.md) | Universal Transformer, runtime budget |
| [ch18_moe.md](docs/ch18_moe.md) | Sparse top-k routing, load-balancing loss |
| [ch19_mtp.md](docs/ch19_mtp.md) | Multi-Token Prediction, K+1 heads |
| [ch20_rlhf.md](docs/ch20_rlhf.md) | RewardModel, REINFORCE, KL penalty |
| [ch21_math_neurons.md](docs/ch21_math_neurons.md) | MathLayer, STE router, exact arithmetic |

### Run a chapter

```bash
./build-debug/bin/ch01_foundations
./build-debug/bin/ch10_modern_arch
./build-debug/bin/ch20_rlhf
./build-debug/bin/ch23_reasoned_math            # all sections
./build-debug/bin/ch23_reasoned_math --phase register   # just the register walkthrough
```

### Run tests

```bash
ctest --preset debug          # via preset (recommended)
ctest --test-dir build-debug --output-on-failure   # manual equivalent
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
│   ├── data/               # Dataset, DataLoader, TextCorpus
│   └── nn/                 # All neural network modules
├── src/                    # Library implementation
├── chapters/               # One executable per chapter (ch01–ch26)
├── tests/                  # Catch2 unit tests (474 tests, 26 files)
├── docs/                   # Per-chapter design documents
└── tools/
    ├── cli/                # sub0llm-cli — interactive inference CLI
    ├── server/             # sub0llm-server — OpenAI-compatible HTTP server
    ├── episodic/           # sub0llm-episodic — episodic memory CLI
    ├── gen_test_gguf.py    # Synthetic GGUF generator (llama/qwen2/qwen3 modes)
    └── download_model.py   # Download Qwen2/Qwen3 GGUF from HuggingFace Hub
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
| `tokenizer/numeric_tokenizer.hpp` | Extends BPE with configurable integer vocabulary (`int_min`…`int_max`), NaN and overflow tokens |
| `data/text_corpus.hpp` | `TextCorpus`: streaming BPE-tokenised corpus from files or JSONL, shuffled windows |
| `nn/modern_gpt.hpp` | `ModernGPT`: RMSNorm, SwiGLU, RoPE, GQA, explicit `head_dim` (Qwen3 support) |
| `nn/math_nodes.hpp` | `MathGPT`, `MathLayer`, `apply_math_op`, `RouteType` (11 ops) — exact arithmetic transformer |
| `nn/optimizer.hpp` | SGD, Adam, gradient clipping |
| `nn/lora.hpp` | LoRA low-rank adaptation |
| `nn/sampler.hpp` | Greedy / temperature / top-k / top-p + `generate` loop |
| `nn/moe.hpp` | Sparse MoE: `MoEFeedForward`, `MoEGPT`, load-balancing loss |
| `nn/rlhf.hpp` | `RewardModel`, preference loss, REINFORCE, KL penalty |
| `nn/checkpoint.hpp` | `save_checkpoint` / `load_checkpoint` with JSON header + binary weights |
| `nn/kv_cache.hpp` | `KVCache`: pre-allocated K/V buffers per layer for O(n) autoregressive inference |
| `nn/long_context.hpp` | `generate_cached()`: KV-cached generation with on-token callback and NTK RoPE scaling |
| `nn/gguf_loader.hpp` | `GGUFReader` + `load_gguf_model`: parse GGUF v2/v3, dequantise F32/F16/Q8_0, load Qwen2/Qwen3 weights |
| `nn/episodic_memory.hpp` | `EpisodicState`, `episodic_encode`, `merge`/`unmerge`, `save`/`load` — fast-weight episodic LoRA |

---

## CLI tools

Three binaries built alongside the chapters:

### sub0llm-cli — interactive inference
```bash
# Single-shot generation from a trained checkpoint
./build/bin/sub0llm-cli --model-dir /tmp/my_model --prompt "To be or not" --max-tokens 100

# Interactive REPL
./build/bin/sub0llm-cli --model-dir /tmp/my_model --interactive
```

### sub0llm-server — OpenAI-compatible HTTP server
```bash
./build/bin/sub0llm-server --model-dir /tmp/my_model --port 8080

curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"HAMLET:","max_tokens":80,"temperature":0.9}'
```

Both tools require a model directory produced by `ch24_real_training --phase train`
(contains `config.json`, `tokenizer/`, and a `*.ckpt` checkpoint file).

### sub0llm-episodic — episodic memory CLI
```bash
# Download a model (requires HuggingFace access)
python3 tools/download_model.py --preset qwen2-0.5b

# Show model config
./build/bin/sub0llm-episodic info --model models\Qwen3-0.6B-Q8_0.gguf

# Validity probe — tests 4 conditions with a novel fact
./build/bin/sub0llm-episodic probe `
    --model models\Qwen3-0.6B-Q8_0.gguf `
    --fact  "sub0llm is a C++23 educational LLM framework by CraigHutchinson" `
    --query "what is sub0llm used for"

# Write a fact to a persistent delta file
./build/bin/sub0llm-episodic write `
    --model models\Qwen3-0.6B-Q8_0.gguf `
    --fact  "sub0llm is a C++23 educational LLM framework by CraigHutchinson" `
    --delta /tmp/sub0llm.epis

# Recall — compare surprisal AND generated response, delta off vs on
./build/bin/sub0llm-episodic recall `
    --model models\Qwen3-0.6B-Q8_0.gguf `
    --query "what is sub0llm" `
    --delta /tmp/sub0llm.epis
```

The tool tokenises with the model's own GGUF BPE vocabulary (not raw bytes), so
the surprisal (NLL) signal is meaningful and generated text is readable.
`probe` and `recall` print the model's continuation of the query **with the
delta off vs on**, so the encoded fact's effect is directly visible — e.g. a
fact about "crystal lattices at 9 kelvin" steers the response toward
"superconducting material / magnetic storage". Each rehearsal step's loss and
gradient norm are streamed so a long write visibly converges.

| Flag | Default | Description |
|------|---------|-------------|
| `--lr F` | 5e-3 | rehearsal learning rate (gradients are clipped to ‖g‖≤1) |
| `--steps N` | 10 | elaborative-rehearsal gradient steps per surprising span |
| `--gen-tokens N` | 40 | length of the demonstrative query continuation (0 = skip) |
| `--train-layers N` | -1 | trailing transformer blocks to update: -1 = last half (freezes embedding + early blocks → faster + more specific), 0 = full model, N = last N blocks |
| `--angles N` | 0 | rehearse self-generated *question→fact* framings for retrieval linkage. The base model poses N questions about the fact; off-topic ones are auto-skipped, the rest are rehearsed so a *reworded* query transfers (fixes the single-phrasing failure). Cost ~×(1+kept). Improves transfer but increases specificity drift. |
| `--lora-rank R` | 0 | write into low-rank, base-frozen adapters on the late-layer FFNs instead of full weights (0=off). The base is provably untouched, so the memory is a tiny, cleanly-detachable delta. NB: composability/storage wins, but **not** better specificity (the adapter is a global FFN change). Higher `--lr` needed (2e-2–5e-2). |
| `--locality W` | 0 | penalise drift on a generic anchor (`W·MSE` of its logits vs the base). Keeps the write "light on existing memory" → specificity. The real PASS4 fix; `~1.0` works. |
| `--adaptive-steps` | off | scale rehearsal steps by novelty: fewer steps when the fact's baseline NLL is low (the model already half-knows it). |
| `--adaptive-lr` | off | likewise scale the learning rate by novelty (gentler step for familiar facts). Pairs with `--adaptive-steps`. |
| `--iterative` | off | grow the `--angles` mid-training — each new question is generated from the *partially-trained* model, so it links to what's already embedded rather than guessing from the base. |

Angles + locality must be trained **jointly** (one rehearsal session) — the tool
does this automatically. The default freezes the embedding and first half of the
blocks (~38% faster, more localised); `--train-layers 0` restores full-model updates.

### sub0llm-episodic suite

`sub0llm-episodic suite --model M [opts]` runs the probe over several built-in
facts with one model load and prints a comparison table — variance across facts is
real (some transfer, some don't), and the `baseNLL` column is the novelty signal
that drives `--adaptive-steps`. Example (bare, no angles/locality):

```
  baseNLL  fact-drop  queryNLL  P1 P2 P3 P4  core
    6.11      -44%      6.81    Y Y - -  fail   sub0llm ...
    5.28      -58%      4.19    Y Y Y -  PASS   Project Zephyr ...
    2.98      -58%      3.63    Y Y - -  fail   Mount Everest ...   (familiar → low baseNLL)
```

A config that passes **all four** checks on Qwen3-0.6B (PASS2 −55%, PASS3 transfers,
PASS4 −0.8% within the strict 2% bar) by writing into a frozen-base low-rank
partition and keeping it off existing knowledge:
`--lora-rank 8 --angles 3 --locality 1.0 --lr 5e-2 --steps 20`. The three guards —
frozen base, low rank, locality — together give learning + transfer + specificity
that none achieves alone.

**GGUF compatibility**: Qwen2 (0.5B–72B), Qwen3 (0.6B–235B, including models
with explicit `head_dim` != `embed_dim/n_heads`).  Quantisation: F32, F16, Q8_0.

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

- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) — the Python counterpart that inspired this project
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — reference for SIMD kernels, GGUF format, quantisation
- [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) — MTP modules (Ch19)
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al., 2021 (Ch12)
- [DPO paper](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023 (Ch13)
- [Switch Transformer](https://arxiv.org/abs/2101.03961) — MoE load balancing (Ch18)
- [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022 (Ch24)
- [NTK-aware RoPE scaling](https://arxiv.org/abs/2309.00071) — Chen et al., 2023 (Ch25)
- [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) — Behrouz et al., 2024 (Ch26 prior art)
- [ROME: Locating and Editing Factual Associations](https://arxiv.org/abs/2202.05262) — Meng et al., 2022 (Ch26 prior art)
- [Test-Time Training on Language Models](https://arxiv.org/abs/2407.04620) — Sun et al., 2024 (Ch26 prior art)
- [C++23 standard](https://en.cppreference.com/w/cpp/23) — `std::format`, concepts, ranges

---

## Licence

MIT
