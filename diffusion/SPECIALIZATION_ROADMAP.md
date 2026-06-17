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
The larger throughput lever. Gap audit done: device transfer exists, but autograd VJP closures
allocate grads device-agnostically, 7 CUDA kernels are missing, and `GemmaGpuLayers` is
inference-only. Phase 0 is CPU-testable; the rest need the CUDA build (`cuda`/`cuda-native`
preset — see memory `cuda-build-setup`).

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

---

## 3. Stage 4 CUDA — phased plan (from the gap audit)

Diffusion engine ops (fwd+bwd) and their CUDA status, then the build order. Device transfer
exists; the gaps are VJP device-args + kernels + autograd integration.

| Phase | Work | Kernels | CUDA build? |
|------|------|---------|-------------|
| **0** | **Device plumbing** — make autograd VJP closures allocate grads on the input's device (audit: `src/autograd/ops.cpp` `zeros/ones`/`copy` in backward closures lack a device arg → grads land on CPU); add `Denoiser::to(Device)` (per-param via `Tensor::to`, but params are `Variable`s — may need `Variable::to`). **CPU-testable** (device round-trips; `to(cpu)` identity). | 0 | No |
| **1** | softmax fwd (+ wire `ops::softmax` CUDA branch; currently throws on CUDA) | 1 | Yes |
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

**Recommended start:** Phase 0 alone in the first clean session (no GPU needed, but touches the
shared autograd core — every model depends on it, so gate on the full 578-test suite). Then the
kernel phases in a CUDA-build session.

---

## 4. Notes / guardrails for the new session

- Verify against current code before acting — file:line citations here are 2026-06-17.
- Don't introduce fixed thresholds where the scale is known (memory
  `derive-bounds-from-scale-not-fixed-thresholds`) — the constexpr engine makes those `constexpr`.
- Keep the runtime engine path working alongside the specialized one (experiments need it).
- The live founded+curriculum run (W=16, ckpt `D:/tmp/ch29_curric_founded`, ~k=17→63) is climbing;
  honest-resume + `run_config.json` mean any rebuild can resume with bare `--ckpt-dir`.
- Tests are the contract: pool/config/support now covered (578). Add gradcheck tests per CUDA phase.
