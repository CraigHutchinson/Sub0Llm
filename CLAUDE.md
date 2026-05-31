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

## Current state (Ch01–Ch26, complete)

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
- `include/sub0llm/nn/modern_gpt.hpp` — RMSNorm, SwiGLU, RoPE, GQA, ModernGPT+MTP (Ch10); sliding-window attention, KV-cached `forward_one()`, RoPE NTK scaling, `make_kv_cache()` (Ch25)
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
- `chapters/ch23_reasoned_math/` — Reasoned Arithmetic: multi-step chain-of-thought with exact math head; register walkthrough, two-step and three-step chain training, natural language word problems, OOD multi-step generalization (Ch23)
- `include/sub0llm/data/text_corpus.hpp`, `src/data/text_corpus.cpp` — `TextCorpus`: streaming BPE-tokenised corpus from in-memory texts or JSONL/text files, shuffled non-overlapping windows (Ch24)
- `include/sub0llm/nn/checkpoint.hpp`, `src/nn/checkpoint.cpp` — `save_checkpoint`, `load_checkpoint`, `latest_checkpoint_path`, `latest_checkpoint_step`: binary checkpoint save/resume with JSON header (Ch24)
- `chapters/ch24_real_training/` — Real-World Pretraining: data landscape, Chinchilla scaling, TextCorpus pipeline demo, Ollama synthetic data API, training approach decision tree, full iterative training loop with checkpoint save/resume (Ch24)
- `include/sub0llm/nn/kv_cache.hpp` — `KVCache`: pre-allocated K/V buffers per layer/kv_head for O(n) autoregressive inference (Ch25)
- `include/sub0llm/nn/long_context.hpp`, `src/nn/long_context.cpp` — `LongContextConfig`, `generate_cached()`: KV-cached generation loop with temperature/top-k sampling and on_token callback (Ch25)
- `chapters/ch25_long_context/` — Long-Context Inference: KV cache (~9× speedup demo), sliding-window attention (banded causal mask, O(n·W) memory), RoPE NTK-aware scaling (extend beyond training length without fine-tuning), memory budget table for production models (Ch25)
- `chapters/ch26_episodic_memory/` — Episodic Memory design chapter: three-tier memory framework (working/episodic/semantic), CLS theory, prior art survey (fast weights, TTT layers, Titans Dec 2024, ROME/MEMIT, SHINE/Text-to-LoRA), schema acceleration model, proposed architecture (comprehension pass → thinking loop → targeted write), Online LoRA Path A plan, Titans NLM Path B roadmap (Ch26)

### Autograd extensions
- `row_scale(x, v)` — scale each row i of (N,D) Variable x by scalar v[i,0]; used by MoE routing

### Tools (CLI + server)
- `tools/cli/main.cpp` → `build/bin/sub0llm-cli` — Interactive inference CLI
- `tools/server/main.cpp` → `build/bin/sub0llm-server` — OpenAI-compatible HTTP server

Both tools require a **model directory** produced by `ch24_real_training --phase train`.
The model directory contains:
- `config.json` — architecture params (vocab_size, embed_dim, n_heads, n_kv_heads, n_layers)
- `tokenizer/` — `vocab.json` + `merges.txt` (GPT-2 format, loadable with `BPETokenizer::load()`)
- `step_XXXXXXXXX.ckpt` — latest model weights

#### End-to-end workflow

```bash
# 1. Train (native build — required for speed)
cmake --build build-native --parallel

nohup ./build-native/bin/ch24_real_training \
  --phase train \
  --ckpt-dir /tmp/my_model \
  --steps 10000 \
  --corpus data/shakespeare.txt \
  --vocab-size 1024 \
  > /tmp/train.log 2>&1 &

tail -f /tmp/train.log   # watch progress

# 2. Inference with the CLI
./build-native/bin/sub0llm-cli \
  --model-dir /tmp/my_model \
  --prompt "To be or not to be" \
  --max-tokens 100 \
  --temperature 0.8 \
  --top-k 20

# Interactive mode (read prompts from stdin):
./build-native/bin/sub0llm-cli --model-dir /tmp/my_model --interactive

# 3. Start the HTTP server
./build-native/bin/sub0llm-server --model-dir /tmp/my_model --port 8080

# OpenAI-compatible completion:
curl -s http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"HAMLET:","max_tokens":80,"temperature":0.9,"top_k":30}'

# Chat completion:
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"What is love?"}],"max_tokens":80}'

# Health check:
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
```

#### CLI flags
| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir DIR` | required | directory with config.json, tokenizer/, *.ckpt |
| `--prompt TEXT` | (empty) | initial prompt text |
| `--max-tokens N` | 200 | max new tokens to generate |
| `--temperature F` | 1.0 | sampling temperature >0 |
| `--top-k N` | 0 (off) | top-k filtering |
| `--top-p F` | 1.0 (off) | nucleus sampling threshold |
| `--greedy` | off | force deterministic argmax |
| `--interactive` | off | read prompts from stdin |
| `--seed N` | 42 | RNG seed |

#### Server flags
| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir DIR` | required | same layout as CLI |
| `--host HOST` | 0.0.0.0 | bind address |
| `--port PORT` | 8080 | listen port |
| `--threads N` | 4 | concurrent request handlers |
| `--seed N` | 42 | base RNG seed (incremented per request) |

#### Server endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | status, step, n_params |
| GET | `/v1/models` | model listing |
| POST | `/v1/completions` | text completion (OpenAI schema) |
| POST | `/v1/chat/completions` | chat completion (messages → prompt) |

#### Corpus notes
- `data/shakespeare.txt` (1.1 MB, 7222 paragraphs) is committed to the repo
- BPE training scales as O(corpus_lines × vocab_merges): 1000 lines + vocab 1024 ≈ 14 s
- Use `--vocab-size 512` for fast tokenizer, `--vocab-size 2048` for richer subwords
- The model dir is gitignored (`/data/*.ckpt` etc.); use Git LFS for checkpoints

#### Training dynamics observed (1000 Shakespeare paragraphs, 5M-param model)

| Step | Train loss | Eval loss | State |
|------|-----------|-----------|-------|
| 200 | 6.29 | **6.28** ← best generalisation | learning vocabulary |
| 1000 | 5.86 | 6.03 | structuring |
| 2000 | 3.82 | 6.33 | overfitting begins |
| 5000 | 0.37 | 7.88 | memorising |
| 10000 | 0.17 | 9.20 | fully memorised |

Key findings:
- **Chinchilla gap**: 5M-param model needs ~100M tokens to generalise; 577 samples × 64 tokens ≈ 37K tokens is 2700× under-resourced
- **Early stopping**: best eval checkpoint is around step 200–500 (minimum eval loss)
- **Memorisation is visible**: greedy decoding at step 9999 reproduces verbatim training passages at low temperature (e.g., "My nobler friends, I crave their pardons: For the mutable, rank-scented many" — Coriolanus Act III)
- **More data needed**: fix the BPE O(n²) bottleneck or use the GPT-2 tokenizer to use the full 7222-paragraph corpus

### Tests
442 Catch2 tests across 25 test files — all passing.

## Git branch

All work goes to `claude/llm-cpp23-repo-init-1f1Il`.

## Dependencies (managed via CPM)

| Package | Use |
|---------|-----|
| spdlog | Logging throughout the library |
| nlohmann/json | Tokenizer vocab, model configs (Ch03+) |
| Catch2 v3 | Unit tests |
| cpp-httplib 0.18.1 (CPM) | Cross-platform HTTP client for Ollama synthetic data (Ch24) |
| OpenBLAS (system) | Optional BLAS-accelerated matmul (Ch02); dispatched for K≥64 |
| Eigen3 3.4.0 (CPM) | Header-only matmul fallback when system BLAS absent; dispatched for K≥64 |
| CUDAToolkit (system) | CUDA backend (Ch02) |
| OpenVINO (system) | Intel backend (Ch02) |
