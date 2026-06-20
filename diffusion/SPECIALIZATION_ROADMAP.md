# Specialization & Backend Roadmap — follow-up work plan

Persisted so a fresh session can pick this up cleanly. Branch: `claude/llm-cpp23-repo-init-1f1Il`.
This captures the **layered specialize-frontend → specialized-engine → backends** architecture
and **all pending work** toward it. Companion design notes: `diffusion/TRAINING_DESIGN.md`,
memory index `MEMORY.md`.

---

## 0. Baseline — what already landed (this session)

The spine for everything below is in place and committed:

- **Config module** (`include/sub0llm/config/{schema,json}.hpp`, `src/config/json.cpp`;
  `diffusion/include/sub0diff/config/run_config.hpp`, `diffusion/src/run_config.cpp`) —
  heap-free field-reflection; **BuildTime vs Runtime `Scope` tag on every field**; layered
  resolve (defaults → `run_config.json` → CLI); **simdjson** parse; `code_sha` + `config_sha`
  stamped into `run_config.json`. `--ckpt-dir X` resumes with exact settings. (commit a441df7)
- **Self-scaling pool caps** — TensorPool/BlockPool high-water-mark cap (no fixed threshold),
  `pool_stats.hpp` instrumentation behind `-DSUB0LLM_POOL_STATS`. (commit 8987949)
- **Full-core scaling finding** — heap fix unlocks W>8 in bench; sustained gain modest
  (E-core straggler-capped); run resumed at W=16/B=16. (commit 3311a4f)
- **Cleanup 3a** — `trainstate`/`tokcache`/`inspect` extracted to reusable `sub0diff` units;
  main.cpp 1257→1070. (commit 8326392)
- **Tests 3b** — `test_pool.cpp`, `test_support.cpp`, +config edges; 578/578. (commit 2e63857)

`Tensor::to(Device)` device transfer (CPU↔CUDA memcpy) ALREADY EXISTS (`src/core/tensor.cpp:133`).

---

## 1. Target architecture — the three layers

The user's vision (verbatim intent): a build pipeline that specializes the engine to a fixed
model shape, with a clean split between what is baked at compile time and what stays runtime.

1. **Specialize FRONTEND (compile driver)** — turns a CLI/config invocation into a *definition*:
   the `RunConfig` BuildTime fields (model shape) → generated `constexpr` spec source + CMake
   build of a specialized engine+backend. One invocation = one tagged build.
2. **Specialized (constexpr) ENGINE** — model dims (vocab/seq/embed/layers/heads/d_ff) become
   `constexpr` (not CLI args), so the whole graph is statically sized. This is the endpoint that
   lets **AOT stack/preallocated buffers replace the pools** (sizes known at compile time → no
   runtime allocator needed; the self-scaling pool was the runtime-dims stopgap).
3. **Runtime BACKEND layers** — `constexpr`-aware dispatch over **cpu / cuda** (and OpenVINO).
   Same engine, swappable backend.

**Tagging:** each specialized build is keyed by **`code_sha` + `config_sha`** (the config module
already computes both; `config_sha` folds only the BuildTime fields). Same shape ⇒ same sha ⇒
reusable build.

**Build-time vs runtime split (already encoded):** model dimensions = `Scope::BuildTime`
(compiled into the constexpr engine); operational knobs (paths, threads, verbosity, eval cadence,
optimizer, curriculum) = `Scope::Runtime` (stay CLI/`run_config.json`). The split is already
machine-checkable via the `Scope` tag — the specialization frontend consumes exactly the
BuildTime set.

**Existing scaffolding to build on:** `diffusion/tools/specialize/` (`sub0diff-specialize`,
host-axis today), `sub0diff/spec/host_spec.hpp` (HostSpec concept + `verify_host`),
`sub0diff/spec/diffusion_spec.hpp` (`DiffusionStaticSpec` concept). The "model axis" was always
the planned next axis (informally "Ch32"). This roadmap is that work, now fed by the config module.

---

## 2. Pending work (prioritized)

### A. Stage 4 — CUDA backend for diffusion training  (~22h, phased; see §3)
The larger throughput lever. Gap audit done: device transfer exists; **Phase 0 (device plumbing)
is now DONE** — backward grads land on the input's device and `Variable::to`/`Denoiser::to` exist.
**Phases 1–6 DONE** (softmax, rms_norm, matmul_tb, rope, silu fwd+bwd / embed-scatter bwd,
narrow + scale) — CUDA kernels + dispatch + parity tests + CPU-vs-GPU bench rows (594 CPU / 551
core CUDA cases green). **Only Phase 7 remains:** the end-to-end step — device d2d `copy()` +
`Node::accumulate_grad` (today both host memcpy; this is the ONE dependency every wired backward
dispatch has deferred to), then `weighted_cross_entropy` on CUDA, then the Denoiser fwd+bwd
gradient-checked vs CPU + a mini on-device training loop. Needs the CUDA build.

### B. Constexpr (specialized) engine — the "model axis"
Turn BuildTime config into a `constexpr` model spec and monomorphize the Denoiser on it; then
replace runtime-sized pool buffers with **AOT/stack buffers sized from the constexpr dims**
(the user's "embrace embedded methodologies, no heap pressure" endpoint). Steps:
- Generate a `StaticSpec`-satisfying struct from `RunConfig` BuildTime fields (extend
  `sub0diff-specialize` model axis; `diffusion_spec.hpp` already defines the concept).
- A `constexpr`-parameterized Denoiser/engine variant (template on the spec) alongside the
  runtime one (keep the runtime path for experiments).
- AOT buffer arena sized at compile time → drop the pool dependency on the specialized path
  (pools remain for the runtime/experiment path). Verify parity vs the runtime engine.
- Wire `code_sha`+`config_sha` into the specialized build artifact name/dir.

### C. Heterogeneous worker-split  (memory: `heterogeneous-worker-split-followup`)
Give P-cores more windows than E-cores so the data-parallel barrier finish-times match —
the real lever for "full core count" (sustained W=16 only ~single-digit% over W=8 because an
E-core@1-window ≈ a P-core@2-windows). Change: weight per-worker window count by core class in
`ParallelTrainer` (`diffusion/include/sub0diff/train/parallel.hpp`). Needs a maintenance window
(paused run) to re-measure sustained (not best-of-3) throughput.

### D. Deeper main.cpp modularization (Stage 3, beyond 3a)
The eval/early-stop logic is inline lambdas (`eval_nelbo`, `eval_nelbo_at`, recall) tightly
coupled to `run()`'s locals. The reusable *cores* already live in `sub0diff/eval/recovery.hpp`
and `sub0diff/train/curriculum.hpp`, so this is low-value / higher-risk on the live training
path — do it only if a future curriculum (the organic per-level one) needs a cleaner eval hook.

### E. nlohmann → simdjson migration  (memory: `simdjson-migration-backlog`)
Migrate remaining nlohmann reads to the simdjson `JsonDoc` facade; drop nlohmann from CPM once
nothing needs it. simdjson is parse-only → pair each migrated reader with a small writer.

### F. Organic per-level curriculum  (memory: `ch29-organic-curriculum-followup`)
Gated on the founded run proving USEFUL. Generalize `FrontierCurriculum` to per-level plateau
tracking + adaptive importance sampling. Add a cheap mid-k forward-transfer probe at next restart.

### G. Diffusion loss-weighting & per-level cadence  (training-science track, not architecture)
`TRAINING_DESIGN.md` §13.3 / memory `diffusion-loss-weighting-and-level-cadence`. A/B the per-window
`(1/t)` NELBO weight vs per-token-equal vs SNR-style (may underlie noise-level gradient conflict +
early global-NELBO stop), and whether the curriculum advances levels while still learning. Run in the
[[ch31]] sandbox / a fresh run — NOT on the live founded run. Listed here so it isn't lost; it's
distinct from the specialize/backend line above.

---

## 3. Stage 4 CUDA — phased plan (from the gap audit)

Diffusion engine ops (fwd+bwd) and their CUDA status, then the build order. Device transfer
exists; the gaps are VJP device-args + kernels + autograd integration.

| Phase | Work | Kernels | CUDA build? |
|------|------|---------|-------------|
| **0 ✅ DONE** | **Device plumbing** — backward VJP closures now allocate grads on the input's device (`src/autograd/ops.cpp`: log_softmax, layer_norm ×3, rms_norm ×2, rope, narrow, row_scale ×2, log_sigmoid); added `Variable::to(Device)` (in-place, keeps Node identity → optimizer `Variable*`s stay valid) and `Denoiser::to(Device)` (loops `parameters()`). Tests: `Variable::to`/`Denoiser::to` identity+parity, **a `{CPU,index=1}` host-backed sentinel** that exercises the closure device-threading end-to-end on CPU (the allocator dispatches on `is_cpu()`, ignoring index, so a real fwd+bwd lands grads on the input's device value — would fail against the old `{CPU,0}` default), plus CUDA round-trips. **Validated on both builds:** debug 585/585 green; cuda build run on the RTX 5070 — 542 core + 41 diffusion Catch cases green, device round-trips exercise live H2D/D2H. **Deferred to Phase 1:** forward-output `zeros()` allocs in those same ops still default to CPU — the constexpr/CUDA forward phases thread that once the kernels make CUDA forward runnable. | 0 | No |
| **1 ✅ DONE** | softmax fwd — `softmax_rows_f32_kernel` (block-per-row, shared-mem max+Σexp reduction) in `kernels.cu`; `backend::cuda::softmax` + `ops::softmax` routes to it when `device().is_cuda()`. Parity test (`[backends][cuda]`) relRMS <1e-4 + rows sum to 1 on the RTX 5070 (actual ~1.6e-7). **Benchmarked** (`bench_kernels --iters` → "Device softmax" rows): kernel-only GPU time vs CPU, 3.6×/12×/53× as rows×cols grows (parity printed per row). GPU ~289 GB/s at the largest shape = well under the 5070 ceiling → headroom (block-per-row underfills at small row counts). | 1 | Yes |
| **2 ✅ DONE** | rms_norm fwd+bwd — 3 kernels (`rms_norm_fwd_kernel` exposing x_norm/inv_rms/out that the inference `rmsnorm_kernel` doesn't; `rms_norm_bwd_x_kernel`; `rms_norm_bwd_w_kernel` column-parallel over rows) + `backend::cuda::rms_norm_{fwd,bwd_x,bwd_w}` device-ptr wrappers; `autograd::rms_norm` dispatches fwd + both bwd closures by device. Parity: kernel-level fwd/bwd_x/bwd_w vs CPU (relRMS <1e-4, actual ~5–9e-8) + autograd fwd end-to-end on the RTX 5070. Benched (`rms_norm fwd` row): 0.63×→17× as T×D grows (launch-bound at small sizes, ~86 GB/s plateau = block-per-row underfill, a later lever). **NOTE:** autograd BACKWARD end-to-end on CUDA waits on Phase 7 — `Node::accumulate_grad`/`copy()` use host `memcpy` (no device d2d); the bwd kernels are validated directly here. | 3 | Yes |
| **3 ✅ DONE** | matmul_tb (Aᵀ·B — weight grads) — `matmul_tb_f32_kernel` (16×16 tiled, contraction over M, left tile reads A transposed) + `backend::cuda::matmul_tb`; `ops::matmul_tb` now dispatches to it when `device().is_cuda()` (replaced the broken transpose-then-matmul fallback that did a host memcpy on device ptrs). Parity test via the public `ops::matmul_tb` (non-tile-multiple dims, relRMS <1e-4, actual ~2–6e-6 from f32 reduction-order). Benched (`matmul_tb` row, GFLOP/s): 8–48× vs CPU at ~1.25–1.6 TFLOP/s f32 — large headroom vs the 5070 f32 peak (naive tile; cuBLAS/wmma later). | 1 | Yes |
| **4 ✅ DONE** | rope fwd+bwd — `rope_fwd_kernel`/`rope_bwd_kernel` (half-split, one thread per (t,i) pair over x(T,Dh) with precomputed cos/sin (T,Dh/2); distinct from the single-vector inference `rope_neox_kernel`) + `backend::cuda::rope_{fwd,bwd}`; `autograd::rope` dispatches fwd (moves host cos/sin to device) + bwd closure. Parity: fwd end-to-end + bwd kernel vs the autograd CPU grad (relRMS 0 — pure mul/add, no reduction). Benched (`rope fwd` row): 0.22×→2.28× (launch/bandwidth-bound; only wins at larger T·Dh). | 2 | Yes |
| **5 ✅ DONE** | silu fwd+bwd + embedding-scatter bwd — `silu_f32_kernel`/`silu_bwd_f32_kernel` (elementwise, expf) + `embed_bwd_f32_kernel` (scatter-add via `atomicAdd` into a zeroed grad_w). `ops::silu` now dispatches fwd to CUDA; `autograd::silu` bwd closure + `autograd::embedding_lookup` bwd closure dispatch by device (indices snapshotted on the weight's device). Parity: silu fwd+bwd vs CPU (relRMS ~7.6e-8 — fast-sigmoid is accurate, 2e-3 tol is slack), embed-scatter vs CPU with repeated indices N>V (relRMS <1e-4, atomics exercised). Benched: silu fwd 5.75×→61× (395 GB/s at 1M). | 3 | Yes |
| **6 ✅ DONE** | narrow + scale on CUDA — `mul_scalar_f32_kernel` (autograd::scale primitive, both directions via `ops::mul(a,scalar)` dispatch); `ops::narrow` dispatches a contiguous d2d memcpy (dim-0 narrow is a contiguous block) and `autograd::narrow` bwd scatters into a zeroed buffer (`zeros()` cudaMemsets + `memcpy_d2d` at offset). Parity: scale + narrow fwd vs CPU (relRMS 0/1e-6), narrow bwd scatter vs autograd CPU grad. Benched: scale 15.4× (342 GB/s). **weighted_cross_entropy moved to Phase 7** (it's the loss entry point; needs a log + NLL-reduction kernel that belongs with end-to-end). | 1 | Yes |
| **7 — step 1 ✅ DONE** | device d2d `copy()` (tensor.cpp now dispatches `memcpy_d2d` for cuda) + `ones()` fixed for cuda (host-build→transfer; the scalar backward seed). `Node::accumulate_grad` needs no change (uses copy + the already-cuda `ops::add`). **End-to-end gradcheck PASSES on GPU**: `silu(rms_norm(x·0.5, w))` full backward graph matches CPU grads (≤1e-3) — every Phase 2-6 bwd dispatch composes on-device. **Step 2 ✅ DONE:** `weighted_cross_entropy` (the diffusion loss, confirmed via diffusion_loss.hpp) on CUDA — `wce_fwd_kernel` (single-block reduction → scalar loss + Σwᵢ) + `wce_bwd_kernel` (grad[i,j]=wᵢ·(g/wsum)·(probs−onehot), no atomics). Self-contained cuda branch in `autograd::weighted_cross_entropy` (moves host targets/weights to device, D2H's wsum for the closure). Parity on GPU: loss value + backward grad vs CPU (≤1e-3/1e-4), incl. the scalar `backward()` seed. **Step 3a ✅ DONE:** embedding FORWARD gather kernel (`embed_fwd_f32_kernel`, host bounds-check then device gather) + `matmul_bt` kernel (`matmul_bt_f32_kernel`, A·Bᵀ, replaced the broken transpose fallback in `ops::matmul_bt`). → **matmul and embedding are now FULLY fwd+bwd on CUDA**, validated by end-to-end gradchecks (matmul bwd uses matmul_bt for dL/dA + matmul_tb for dL/dB; embedding fwd gather + bwd scatter with repeated indices). **Step 3b ✅ DONE:** softmax BACKWARD (`softmax_bwd_f32_kernel`, gx=y·(g−rowsum(g·y)); dispatched in `autograd::softmax`), transpose-contiguous (`copy_strided_f32_kernel` strided gather → non-contiguous cuda `copy()` materialises transpose/permute for the multi-head reshape; dispatched in `tensor.cpp copy()`), and the fused **Adam(W) optimizer step** (`adam_step_f32_kernel`, in-place p/m/v; AdamW wd_keep; dispatched in `Adam::step`). Each with a `[backends][cuda]` parity test (relRMS <1e-4/1e-5 on the RTX 5070). → **every op in the Denoiser fwd+bwd+update path now runs on CUDA. CAPSTONE ✅ DONE:** a mini end-to-end Denoiser training step (embed→conditioning→GQA attn→SwiGLU FFN→LM head→diffusion loss→backward→Adam) runs on the GPU and reduces the loss (test "Denoiser end-to-end training step on CUDA reduces loss"). The integration surfaced 4 gaps the op-parity tests missed, all fixed: (a) the **conditioning constant** (time_embedding) is host-built → `.to(x.device())`; (b) **matmul forward** must `.contiguous()` its operands (attention passes `transpose2d(K)`) like matmul_bt/tb already do; (c) **bias_add** fwd + bias-grad were host-only loops (the FFN's Linear layers; attention is bias-free) → added `bias_add_fwd`/`bias_add_bwd_b` kernels + dispatch; (d) **`Tensor::item<T>()`** host-memcpy'd a device scalar (the loss value) → bring to host first. Localised via stderr stage-markers + compute-sanitizer (0 GPU errors ⇒ host-side deref). Validated: CUDA core 565 + diffusion 42 green; CPU core 567 + diffusion 42 green. **The diffusion engine can now train on the GPU.** Remaining (perf, not correctness): a full-step bench vs CPU + wiring `ch29 --device cuda`. | done | Yes |

Files: kernels in `src/backends/cuda/kernels.cu`; dispatch wrappers `src/backends/cuda/backend.{hpp,cpp}`;
routing `src/core/ops.cpp` + backward routing `src/autograd/ops.cpp`. `GemmaGpuLayers` stays the
inference engine; training needs the autograd path above, not that class.
Correctness oracle: CPU parity / finite-difference gradcheck (memory `verify-correctness-against-reference-before-perf`).
Build: `cuda-native` with explicit clang (memory `cuda-build-setup`, `ch27-gpu-forward-build-arc`).

**Bench harness (standing tool for every kernel phase):** `benchmarks/bench_kernels.cpp` →
`bench_device_ops()` is the per-op CPU-vs-GPU harness. Each kernel adds a `backend::cuda::*_bench`
(preallocated device buffers + GPU events, kernel-only timing — mirrors `matmul_q8_0_bench`,
NOT the per-iter `Tensor` alloc path) and a row that prints CPU ms, GPU ms, speedup, and parity
relRMS. Build+run: `cmake --build --preset cuda --target bench_kernels` then
`./build-cuda/bin/bench_kernels.exe --iters N`. A perf number is only trustworthy once relRMS is
at f32-epsilon — gate on the `[backends][cuda]` parity test first.

**Recommended start:** ~~Phase 0–6~~ DONE. Next: **Phase 7 (end-to-end) — start with device d2d
`copy()` + `Node::accumulate_grad`.** Today `copy()` (tensor.cpp) does a host `std::memcpy` and
`Node::accumulate_grad` (variable.cpp) does `copy()` / `ops::add` — so a backward graph on CUDA dies
at the first grad accumulation. Fix `copy()` to dispatch `memcpy_d2d` for cuda tensors (and confirm
`ops::add` cuda — it already dispatches); that single change unblocks running the full autograd
backward on-device, which validates EVERY bwd dispatch wired in Phases 2–6 end-to-end (gradcheck a
small graph vs CPU). Then add `weighted_cross_entropy` on CUDA (softmax done; needs a log + weighted
-NLL forward-reduction kernel + the backward grad kernel), then the Denoiser fwd+bwd gradcheck + a
mini on-device training loop. Bench a full training step (fwd+bwd) CPU vs GPU.

---

## 4. Notes / guardrails for the new session

- Verify against current code before acting — file:line citations here are 2026-06-17.
- Don't introduce fixed thresholds where the scale is known (memory
  `derive-bounds-from-scale-not-fixed-thresholds`) — the constexpr engine makes those `constexpr`.
- Keep the runtime engine path working alongside the specialized one (experiments need it).
- The live founded+curriculum run (W=16, ckpt `D:/tmp/ch29_curric_founded`, ~k=17→63) is climbing;
  honest-resume + `run_config.json` mean any rebuild can resume with bare `--ckpt-dir`.
- Tests are the contract: pool/config/support now covered (578). Add gradcheck tests per CUDA phase.
