---
name: ch27-gpu-forward-build-arc
description: Ch27 hybrid Gemma GPU engine build — staged plan, GPU primitives DONE+validated, next is on-device single-layer dataflow
metadata:
  node_type: memory
  type: project
  originSessionId: 26b08af2-2bc6-47fd-985b-b0a3d9b828eb
---

Building the hybrid Gemma-12B engine locked in [[llama-cuda-shootout]] (contiguous layer split
~28 GPU + 20 CPU, flash-attn on GPU layers, q8 KV frees VRAM; bar = beat llama **731 PP / 11.09 TG**).
User's plan: "imma first — optimize area by area, then go wide while optimizing each part, then a
final holistic optimize." IMMA (matmul area) done; now "go wide" = full GPU forward.

**Discipline (same as IMMA):** every GPU kernel mirrors the CPU oracle in `src/nn/gemma.cpp`
(`forward_one`) and is validated against an inline CPU reference via **`build-cuda/bin/sub0llm-qbench --layers`**
(relRMS, not bitwise — parity gate is greedy-token match). Build inside vcvars64 (see [[cuda-build-setup]]).

**STAGE 1 — GPU primitives: COMPLETE + validated (commits 19b1c69, 9383c03).** All in
`src/backends/cuda/kernels.cu` (+ `.cuh` launchers, `backend.cpp/.hpp` `*_dev` host wrappers with
non-CUDA throwing stubs):
- `quantize_q8` (f32→Q8, one warp/block) — **bit-exact 120/120 vs quantize_row_q8_0** (relRMS 0)
- `matmul_q8_0_mma` (IMMA, prior commit c03bf7e) — ~1e-7
- `rmsnorm` (weighted + no-weight) — **relRMS 0** (uses 1/sqrtf not rsqrtf; w=nullptr→no weight)
- `rope_neox` (NEOX half-split, ff=nullptr→no freq scaling) — ~1e-6 (cos/sin/pow rounding)
- `flash_attn_decode` (single-query online softmax, scale=1.0, one block/head, shared q/acc,
  tree-reduced per-pos dot) — ~1e-7, algebraically identical to CPU two-pass max/exp
- `geglu` (gelu_tanh(gate)⊙up; kK=1.5957691216 kC=0.044715 = cpu gelu_f32) — ~1e-8
Plus existing element-wise `add`/`mul` kernels for residuals/scales.

**STAGE 2 — on-device single-layer forward: DONE + validated (commit 50d5233).**
`backend::cuda::gemma_layer_decode_dev(GpuLayerDesc, x, pos, kcache, vcache, max_pos, out)` runs
one full Gemma layer on device in forward_one's exact order (rmsnorm→quantize→QKV matmul→per-head
q/k rmsnorm+rope & v rmsnorm & windowed KV store→flash_attn_decode→O matmul→post_attn norm+residual→
ffn norm→quantize→gate/up matmul→geglu→quantize→down matmul→post_ffw norm→(attn_out+ff)·out_scale).
Handles local (dh256, GQA-8kv, window, base1e4) and global (dh512, MQA, V=raw-K projection,
freq_factors, base1e6). Added `add_scale_kernel` ((a+b)·s) for the scaled residual. Validated via
`qbench --gpu-layer` vs a faithful scalar CPU ref (same Q8 weights/norms/KV/input): both layer
types **out relRMS ~5e-7**. CURRENT FORM is in-call upload (validation); Stage 3 makes weights+KV
device-resident and keeps activations on-device across layers (no per-token re-upload).

**PARITY NOTE (important for Stage 3):** Gemma attention scale = 1.0 (no 1/√dh). A ~1.0 q_norm
leaves scores huge (esp. dh512) → peaked softmax → amplifies the CPU(AVX hsum) vs GPU(tree-reduce)
score-dot summation-order diff to ~8e-4. NOT a bug — real Gemma folds the query pre-attn scaling
(~1/√head_dim) into q_norm at GGUF conversion, so real scores are O(1) and it lands ~5e-7. If a
Stage-3 GLOBAL-layer greedy mismatch shows up, match the flash score-reduction order to CPU.

**STAGE 3 step 1 — resident layers: DONE (commit 90229a7).** `backend::cuda::GemmaGpuLayers`
(pImpl) uploads a contiguous layer run's weights ONCE, holds device KV + reused scratch, ping-pongs
activations on-device across layers. Bit-identical (relRMS 0) to chaining the per-call form —
`qbench --gpu-resident`. Shared core `run_gpu_layer` used by both forms.

**STAGE 3 step 2 — hybrid GemmaModel forward: DONE, greedy parity PROVEN (commit 6057004).**
`GemmaModel::enable_gpu_layers(k, max_pos)` builds GpuLayerDesc from layers_[0,k) + constructs the
resident GemmaGpuLayers; `forward_one_hybrid` = embed → GPU layers [0,k) (device KV) → ONE D2H
hand-off → CPU `cpu_layer_` for [k,n) → head. forward_one refactored into shared embed_/cpu_layer_/
head_ (bit-identical to pre-refactor greedy on the 12B — reverified). GemmaModel is now move-only
(unique_ptr to incomplete GemmaGpuLayers, out-of-line specials). Tool: `--gpu-layers K` +
`--mode hybrid-check`. On real gemma-4-12b-it-Q8_0.gguf: **k=4 and k=16 both PASS — 22/22 greedy
tokens identical to pure CPU** (the llama-validated oracle), incl. global layers (5,11). The
[[ch27-gpu-forward-build-arc]] softmax-order note held: real weights → no greedy flip. GGUF lives
at models/gemma-4-12b-it-Q8_0.gguf (11.8GB); Qwen3-0.6B-Q8_0.gguf (0.6GB) also there.

**STAGE 3 step 3 — PERF: hybrid DECODE now BEATS CPU + VRAM edge mapped (commits 53ed169, 4a9d417).**
Two fixes: (1) batched per-head kernels (rmsnorm_heads/rope_heads/store_kv/flash_attn_decode_heads)
cut ~90→~25 launches/layer — transparent but barely moved wall-clock (decode wasn't launch-bound).
(2) THE win: `matmul_q8_0_gemv_kernel` — warp-per-row coalesced Q8 GEMV auto-dispatched for T==1
(ggml MMVQ pattern; the old one-thread-per-output kernel didn't saturate VRAM bw at T=1). Now the
GPU-resident layers are faster per-token than the CPU layers they replace.

K-SWEEP (cuda build, gemma-4-12b, -n24, parity PASS at every k): k16 4.72, **k24 5.00 ← optimum**,
k28 3.72, k32 2.66 tok/s; same-build CPU 3.74. tok/s peaks at k≈24 then a CLIFF — the VRAM-overflow
boundary (f32-KV weights >~6GB spill into WDDM shared sysmem, PCIe-bound), exactly llama's 20→30
cliff. Our edge ~24 layers vs llama ~28 because we use f32 KV (bigger); **device q8 KV is the lever
to push the edge higher** (backlog). CAVEAT: cuda preset is NOT -march=native/LTO, so absolute tok/s
< the native CPU's ~6.5; the +26% (k16) / +34% (k24) is the apples-to-apples same-build hybrid-vs-CPU
gain. For the true number vs llama 11.09 TG, need a native+CUDA build.

**NATIVE+CUDA NUMBERS (commit 4a583b8) — hybrid TG beats tuned CPU by +52%.** Built the
`cuda-native` preset (clang -march=native + LTO + CUDA). GOTCHA: a fresh configure inside vcvars64
auto-detected MSVC for C++ (the TTCMAKE_* preset typo → compiler not pinned), which fails to
compile the clang code; reconfigure with explicit `-DCMAKE_C_COMPILER=clang
-DCMAKE_CXX_COMPILER="C:/Program Files/LLVM/bin/clang++.exe"` (nvcc still uses cl.exe host by
default). Also fixed hybrid-check to use the per-phase core policy (compute cores prefill / E-cores
decode) and time ONLY the decode loop — the earlier all-cores pin understated the CPU baseline.
Real numbers on gemma-4-12b-it-Q8_0 (-n32, decode-only, parity PASS every k):
  tuned CPU decode (E-cores) 4.80 ; hybrid k16 6.13 (+26%) ; **hybrid k24 7.28 (+52%)** ← VRAM-edge
  optimum (f32 KV). Now in striking range of llama's best Gemma TG **11.09** (ngl28+fa+q8KV).

**DEVICE q8 KV — implemented (commit 8181456) + VALIDATED as a long-context win (--ctx sweep).**
store_kv_q8 (quantize on write) + flash_attn_decode_heads_q8 (dequant on read); GemmaGpuLayers(kv_q8)
/ enable_gpu_layers(...,kv_q8) / tools/gemma --q8-kv --ctx N. CORRECTION to my earlier WRONG claim:
q8 KV's payoff is VRAM-FREEING → FIT MORE LAYERS → faster (NOT bandwidth — KV per-token read is
negligible vs the 5.9GB weight read; matches [[llama-cuda-shootout]] "optimal split shrinks as
context grows, KV eats VRAM"). My short-context test (~40 pos) hid this — KV was negligible there.
Measured with --ctx (sizes KV capacity independent of token count):
  ctx 4096: f32 best k20=5.88, k24=4.20(spill) ; q8 k24=5.39 — ~break-even (f32 still fits k20).
  ctx 16384: f32 best **k12=5.14** (spills past 12) ; q8 best **k20=5.96** (fits 20) → **+16%**.
So q8 KV's advantage GROWS with context (f32 KV evicts layers from VRAM); at 32K it widens further.
q8 KV is LOSSY (~1%/value): --q8-kv FAILs strict greedy vs the f32 oracle (expected for lossy KV;
judge on quality/perplexity, not greedy-exactness). Default still f32 (greedy-exact, --gpu-resident
0.00, k24 PASS 22/22 @ 7.93); q8 is the recommended LONG-CONTEXT mode. NOTE the q8 flash re-decodes
the block f16 scale PER ELEMENT (32× redundant) — an optimization target (per-block decode).

**GPU PATHWAY OPTIMIZATION — decode GEMV PROFILED, alignment ruled out (commit d282f0f).** Tried
the obvious decode lever first: the T=1 Q8 GEMV runs ~120-204 GB/s (~27-45% peak). Hypothesized the
34-byte BlockQ8_0 (qs at +2, never 16B-aligned → scattered byte loads) was the cause; prototyped an
aligned-repack GEMV (qs[M*K] int8 16B-aligned + scales f16, coalesced int4 reads) as qbench
variant 2. FINDING: 2.4× ONLY for small weights (8-16MB, wk/wv/wq) that fit L2 — a rep-loop BENCH
ARTIFACT (L2 bandwidth); the large FFN matmuls (63MB, gate/up/down) exceed L2, stream from VRAM, and
show ~0 alignment gain. Real 12GB decode re-reads every 245MB layer from VRAM/token (L2 holds <1
layer) → large-shape VRAM-bound regime → alignment does nothing, FFN dominates. So weight-read
ALIGNMENT is NOT the decode bottleneck. The large-shape ceiling (~245 GFLOP/s) is VRAM-saturation /
X-reread / occupancy — NEEDS nsight to pin down (not available cleanly here). Prototype kept as infra.

**WHERE DECODE STANDS:** ~7.9 tok/s (k24, f32 KV) appears near this hardware's kernel ceiling
without profiler-guided VRAM work. Remaining decode levers (uncertain, profiling-dependent): stage X
to shared (cut the per-warp activation re-read on big GEMVs); q8-flash per-block scale; set up nsight.

**CONTROL BASELINE (interleaved, same-session — the only valid llama comparison).** User directive:
ALWAYS bench ours vs llama back-to-back in one session via `tools/scripts/bench_vs_llama.ps1`
(`-Mode hybrid` for GPU, `-Mode threads` for CPU; Latin-square thermal-balanced, greedy smoke-check,
CSV). NEVER cite a llama number from a prior session — the [[llama-cuda-shootout]] figures (731 PP /
11.09 TG) were a different session/config and proved NOT directly comparable. Fresh interleaved
result (commit ~5c.., gemma-4-12b, ours k24 f32 vs llama ngl28+fa, greedy-IDENTICAL smoke-check):
  ours TG 8.36 vs llama 10.77 = **78%** ; ours PP 10.2 vs llama 122 = 8% (token-by-token vs batched).
So real gap is ~1.28× on TG (much closer than the stale 11.09 implied) and PP is the big gap
(batched prefill). bench_vs_llama.ps1 unified both old scripts (bench_sweep_threads + the hybrid one).

**BATCHED PREFILL: DONE (commits 56e9bb6, d6aca20, 1f92d9b, dc45917) — PP 8% → 33% of llama.**
GemmaModel::forward_prefill_hybrid: embed → GPU layers [0,k) as ONE batched IMMA pass
(GemmaGpuLayers::prefill, fills device KV) → CPU tail [k,n) BATCHED via cpu_prefill_layer_
(factored from forward_prefill; row-reuse GEMM, not token-by-token) → head. New GPU kernels:
transpose, rope_prefill, store_kv_prefill, flash_attn_prefill_heads (all validated; orchestration
bit-exact vs sequential prefill in qbench --gpu-prefill). GREEDY PARITY HOLDS on real 12B (gen_ids
identical to CPU ref) AND forward_prefill refactor verified non-destructive (--mode greedy
unchanged). Interleaved control (bench_vs_llama -Mode hybrid, k24 vs ngl28+fa): **PP 40.2 vs llama
121 = 33%** (4× over the 8% token-by-token baseline), **TG 8.6 vs 11.2 = 77%**. Remaining PP gap:
CPU tail (24/48 layers on CPU) + IMMA ~5-6× behind MMQ. tools/gemma --mode hybrid uses it.

**NEXT levers to the hardware limit:** (1) IMMA→MMQ multi-warp tiles (the 5-6× matmul gap — helps
PP directly, the biggest remaining); (2) more GPU layers (q8 KV frees VRAM → fewer slow CPU-tail
layers); (3) overlap GPU‖CPU. TG (77%) is near the decode VRAM ceiling (needs nsight).

(historical) **BATCHED GPU PREFILL (the PP hardware-limit path).** PP is 8% of llama
(control baseline) ONLY because it's token-by-token GEMVs — IMMA tensor cores sit IDLE during
prefill. Measured headroom (qbench, gate/up M15360 K3840, prefill scale): T=512 IMMA **2927 GFLOP/s**
vs dp4a 739 (4×) vs CPU 72 — and IMMA still ~5-6× behind llama MMQ. So batched prefill (T prompt
tokens/layer in one pass = real GEMMs where IMMA is 4× the GEMV) is the compute-limit lever; current
IMMA is already good enough to start, then optimize IMMA→MMQ (5-6×) for more.
NEW KERNELS the batched GPU layer needs (most primitives exist at T>1): batched rmsnorm-ROWS (T rows
of width D; rmsnorm_heads is per-head/1-token), GPU transpose ((M,T)→(T,M) for per-token rope/norm),
batched CAUSAL attention (T queries each attend [0..pos] — the genuinely new kernel; flash_attn
prefill = grid over T·nH blocks, block(t,h) kvlen=t+1). matmul IMMA (T>1), quantize (T·nb blocks),
geglu (T·dff), residuals all already work. Validate vs CPU forward_prefill (relRMS/greedy). Then
populate the GPU layers' device KV during prefill so hybrid decode continues from it.

NOTE: bench exe hit a transient Windows "corrupted/unreadable" (0x80070570, AV lock) — delete + relink fixed it.

Backlog (lossy/quality, later): q8 KV on device; TurboQuant rotated low-bit KV (long context);
close IMMA→MMQ ~5-6× gap (multi-warp tiles); full Qwen3-0.6B GPU forward; DeltaNet distillation.
See [[ch27-optimization-shortlist]], [[verify-correctness-against-reference-before-perf]].
