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
**Phase 1 (softmax fwd) also DONE** — CUDA kernel + dispatch + parity test + CPU-vs-GPU bench
(586 CPU / 543 core + 41 diffusion CUDA cases green). Remaining: ~6 CUDA kernels (rms_norm bwd,
matmul_tb, rope, silu/embedding bwd, …) and `GemmaGpuLayers` is inference-only. Phases 2+ need the
CUDA build (`cuda`/`cuda-native` preset — see memory `cuda-build-setup`).

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
| **2** | rms_norm fwd+bwd | 2 | Yes |
| **3** | matmul_tb (A^T·B — weight grads; currently `require_cpu`) | 1 | Yes |
| **4** | rope fwd+bwd | 1 | Yes |
| **5** | silu backward; embedding-scatter backward | 2 | Yes |
| **6** | narrow/scale dispatch + weighted_cross_entropy on CUDA | 0 | Yes |
| **7** | end-to-end: Denoiser fwd+bwd on CUDA, gradient-checked vs CPU; mini training loop | 0 | Yes |

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

**Recommended start:** ~~Phase 0~~ ~~Phase 1~~ DONE. Next: **Phase 2 (rms_norm fwd+bwd)** — fwd
kernel already exists (`launch_rmsnorm` / `rmsnorm_kernel`, used by the Gemma path), so wire
`ops`/autograd `rms_norm` dispatch + add the bwd kernels (`rms_norm_bwd_x` / `rms_norm_bwd_w`),
parity-test each, and add a bench row. Add a finite-difference gradcheck-on-CUDA per backward
phase — that is where the Phase-0 backward closures' device arg finally gets exercised end-to-end.

---

## 4. Notes / guardrails for the new session

- Verify against current code before acting — file:line citations here are 2026-06-17.
- Don't introduce fixed thresholds where the scale is known (memory
  `derive-bounds-from-scale-not-fixed-thresholds`) — the constexpr engine makes those `constexpr`.
- Keep the runtime engine path working alongside the specialized one (experiments need it).
- The live founded+curriculum run (W=16, ckpt `D:/tmp/ch29_curric_founded`, ~k=17→63) is climbing;
  honest-resume + `run_config.json` mean any rebuild can resume with bare `--ckpt-dir`.
- Tests are the contract: pool/config/support now covered (578). Add gradcheck tests per CUDA phase.
