---
name: llama-cuda-shootout
description: Shootout vs llama.cpp CUDA (RTX 5070 8GB) — Gemma-12B GPU decode is PCIe-bound and loses to our CPU
metadata: 
  node_type: memory
  type: project
  originSessionId: 26b08af2-2bc6-47fd-985b-b0a3d9b828eb
---

Shootout vs llama.cpp b9334 CUDA prebuilt (D:\tools\llamacpp\vendor\llama.cpp-prebuilt\b9334\cuda\, runs on Blackwell sm_120), Jun 2026, RTX 5070 Laptop 8GB.

**Qwen3-0.6B (0.64GB, fits VRAM — clean GPU case):** llama CUDA full forward = PP 16,163 tok/s, TG 323.5 tok/s. That PP ≈ ~19 TOPS effective. Our IMMA Q8 matmul standalone = ~3.5 TOPS (T=512) → llama's CUDA Q8 matmul (MMQ, multi-warp tiled) is ~5-6× faster than our single-warp IMMA. That 5-6× is the target for the deferred IMMA optimization passes (bigger multi-warp tiles, gather amortization). We have NO full GPU forward yet — needed for a true tok/s shootout.

**Gemma-12B (12GB Q8, OVERFLOWS 8GB VRAM → PCIe-paged) — the key finding:**
- llama CUDA -ngl 99: PP 92.6 tok/s, **TG 1.77 tok/s**.
- our CPU (this repo, GemmaModel): PP ~29 tok/s, **TG ~6.5 tok/s**.
- So GPU PP > CPU PP (92.6 vs 29: compute-bound, the 512-batch amortizes the PCIe weight stream), but **our CPU TG beats llama's GPU TG by 3.7× (6.5 vs 1.77)**. TG reads all 12GB/token; on 8GB VRAM that's PCIe streaming (~16-32 GB/s) << CPU system RAM (~100 GB/s). The GPU is bandwidth-starved on decode.

**-ngl PARTIAL-OFFLOAD SWEEP (Gemma-12B, the killer finding):** llama-bench -ngl 0,20,30,40,48 → PP/TG: ngl0(all-CPU) 318/5.92, **ngl20 483/8.67 ← OPTIMUM**, ngl30 97/4.89, ngl40 93/2.61, ngl48(all-GPU) 92/1.80. The optimum is "as many layers as FIT in 8GB VRAM without overflow" (~20/48 ≈ 5GB), and it beats BOTH extremes: 5.3×/4.8× over all-GPU, +52%/+46% over all-CPU, AND its TG (8.67) beats our pure-CPU TG (6.5). The 20→30 cliff is the VRAM-overflow boundary (spills to PCIe). KEY MECHANISM: the hybrid runs the resident layers on VRAM bandwidth ‖ the rest on system-RAM bandwidth → combined > either alone. So our hybrid engine should do PARTIAL LAYER OFFLOAD at the VRAM edge, not all-or-nothing. Optimal split shrinks as context grows (KV eats VRAM); llama -fitt auto-fits.

**REFINEMENT (locks the hybrid architecture):** at the ngl≈20 optimum, tested tensor-level placement vs flash-attn. Gemma-12B PP/TG: layer ngl20 fa-off 532/8.60, **layer ngl20 fa-ON 609/9.20 ← llama's BEST**, tensor FFN→CPU fa-off 564/7.62, tensor FFN→CPU fa-on 548/7.63. CONCLUSION: layer-level + flash-attn BEATS tensor-level (TG 9.20 vs 7.62). My hypothesis (FFN→CPU surgical split) was WRONG — fine-grained tensor placement crosses the GPU↔CPU boundary 2×/layer (~96 PCIe round-trips/token) vs a contiguous layer split's ONE crossing; boundary LATENCY dominates. So our hybrid = CONTIGUOUS layer split at the VRAM edge (minimize device-boundary crossings) + flash/fused attention on the GPU layers + our IMMA matmul (GPU) + llama-beating CPU path (CPU layers). The bar to beat for Gemma-12B is llama's 609 PP / 9.20 TG.

**q8_0 KV CACHE (key nuance):** at SAME ngl=20, q8 KV ≈ f16 (TG 9.03 vs 9.20) and at depth-4096 q8 KV is SLOWER (7.80 vs 8.33). Reason: for a 12GB model the per-token WEIGHT read dwarfs the KV read (KV ~0.7GB @4K ≈ 3% of total), so halving KV saves ~nothing on bandwidth and dequant overhead makes it net-negative. BUT q8 KV frees VRAM → push the layer split ngl 20→28 → **+20% PP&TG: 731 PP / 11.09 TG ← new Gemma-12B bar** (ngl28 + fa + q8 KV). So KV-compression's payoff for big-model-overflow is VRAM-FREEING (more GPU layers), NOT bandwidth. KV bandwidth only dominates for VRAM-resident models or very long context (32K+). TurboQuant/low-bit-KV (6×, lossy, rotation+QJL) would push ngl ~32-36 but breaks strict parity — a long-context/quality-mode lever, not the parity path. The reusable steal is the orthogonal-rotation primitive (invertible → rotated-q8 could give q8 quality at fewer bits, parity-safe-ish).

**llama tooling ideas worth stealing (-ot per-tensor placement = surgical hybrid; -ctk/-ctv KV-quant for TG bandwidth; -fa flash-attn; -np continuous batching; -t/-tb + -C/-Cb separate PP/TG threads+affinity = CONFIRMS our per-phase P/E policy; --poll spin/sleep blend; -dio/-mmap load tuning).**

**STRATEGY this dictates:**
- Models that FIT VRAM (Qwen-size): GPU is the play; build our forward + close the 5-6× matmul gap.
- Models that OVERFLOW VRAM (Gemma-12B @ 8GB): optimal engine is HYBRID — GPU prefill ‖ CPU decode — and we ALREADY own the decode half. This empirically justifies the overlapped-execution architecture for big-model-on-small-GPU. See [[cuda-build-setup]], [[optimization-target-is-hardware-ceiling]], [[ch27-optimization-shortlist]].
