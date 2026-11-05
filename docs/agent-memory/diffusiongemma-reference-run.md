---
name: diffusiongemma-reference-run
description: How to run Google DiffusionGemma-26B locally (the real diffusion-LM oracle) + the finding that diffusion offload is GPU-favorable on the 8GB box (unlike AR decode)
metadata: 
  node_type: memory
  type: reference
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Set up 2026-06-18: a real open diffusion LLM running locally as a coherent-generation oracle (the reference our educational sub0diff model reaches toward). Confirms [[diffusion-contiguous-masking-lever]] / TRAINING_DESIGN §13.5 — the SAME masked-diffusion mechanism produces fully coherent text with enough data+capacity; our word-salad was data/capacity, not the mechanism.

**Model:** Google **DiffusionGemma-26B-A4B-it** (MoE 25.2B total / ~3.8B active, Gemma-4 backbone + diffusion head, denoises 256-tok canvas; released 2026-06-10, Apache-2.0). Q4_K_M GGUF (~16GB) at `D:/Craig/diffusiongemma/gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (from `unsloth/diffusiongemma-26B-A4B-it-GGUF`, un-gated).

**Runner:** llama.cpp **PR #24423** (`llama-diffusion-cli`; normal llama-cli can't sample it). Clone at `D:/Craig/diffusiongemma/llama.cpp` (branch `diffpr`). Two builds:
- CPU: `build/` (`cmake -B build -G Ninja -DGGML_CUDA=OFF` + clang).
- CUDA: `build-cuda/` (`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, built inside vcvars64 / VS18 — MSVC host is fine for llama.cpp, no clang gymnastics; CUDA 13.2, RTX 5070 sm_120). Run with CUDA bin on PATH.

**Run (GPU hybrid — best):** keep the bulky MoE experts on CPU, offload dense/attn to the 8GB VRAM:
`./build-cuda/bin/llama-diffusion-cli -m <gguf> -ngl 99 --n-cpu-moe 30 -p "..." -n 512`
(`--diffusion-steps`, entropy-bound decoder auto; `--diffusion-visual` to watch denoising.)

**KEY FINDING — diffusion offload is GPU-FAVORABLE on the 8GB box, unlike AR decode.** Same 256-tok gen: CPU pure **1.0 tok/s** (269s) vs GPU hybrid **13.2 tok/s** (19s) = **~14×**. This CONTRADICTS [[llama-cuda-shootout]] (AR Gemma-12B decode was PCIe-bound and LOST to CPU on this box): a diffusion step is a full-canvas forward (≈ batched prefill, not token-by-token decode), so it's compute-bound and the GPU wins big even with only dense layers in 8GB + experts on CPU. Generation is fully coherent (it even shows a chain-of-thought channel then drafts the answer). Useful as the correctness/quality oracle for our diffusion work ([[verify-correctness-against-reference-before-perf]]).
