# Device backends — the abstraction seam (CUDA today; ROCm / OpenVINO / others by design)

Status (2026-07-17): **design + bridge layer shipped; consumer migration incremental.** The neutral seam
(`include/sub0/device_backend.hpp`) is real and callable today, implemented as a zero-cost bridge over
the existing CUDA backend. New code targets the neutral seam; existing `sub0_cuda_*` call sites migrate
mechanically as they are next touched (each migration shrinks the scattered legacy declarations).

## What exists, and what was wrong with it for multi-backend

The CPU engine (`src/backend_cpu.cpp`) is **always the engine**: the whole `sub0::` API (graph,
forward/backward, sampling, AdamW/Muon, the binding mechanisms) compiles for every build. A device
backend is an **add-on accelerator** reached through an `extern "C"` seam across a DLL boundary
(`sub0_backend_cuda.dll`), selected at **build time** (`SUB0_COMPUTE=AUTO/CPU/GPU/HYBRID`,
`cmake/Backends.cmake`) per the project's compile-time-decision philosophy — no runtime backend
registry, no vtables on hot paths.

Three things made that seam CUDA-locked rather than device-neutral:

1. **The seam is named `sub0_cuda_*`** (~37 exported functions). A ROCm backend would have to either
   impersonate CUDA (wrong, and lies to logs/telemetry) or duplicate every call site.
2. **No canonical header.** Consumers each declare the externs they use (`decode.hpp` declares 6,
   `train_stage.cpp` ~14, `memplan.hpp`, tools, tests more) — signature drift between sites is a real,
   observed risk class.
3. **Capabilities are hardcoded as assumptions in consumers**: "the device trains, the CPU generates",
   "interception is CPU-only", "binding/content-embed windows can't run on the device" (the hybrid
   CPU/GPU router's `needs_cpu` test). Those are *facts about the current CUDA backend*, not laws — an
   OpenVINO backend would invert the first (inference-only), and a future CUDA `encode_slot` kernel
   flips the last.

## The design

### 1. One canonical, backend-NEUTRAL C seam: `sub0_dev_*`

`include/sub0/device_backend.hpp` is the single home of the production seam declarations. The names say
*device*, not *cuda*. The seam stays `extern "C"` + POD types only (it crosses a DLL boundary; the same
reasoning `registry.hpp` documents for keeping simdjson types inside `sub0_core`).

Production surface (the ~20 functions consumers actually use), grouped:

| group | functions |
|---|---|
| lifecycle | `sub0_dev_init`, `sub0_dev_shutdown`, `sub0_dev_caps` |
| params/opt transfer | `sub0_dev_upload_params`, `sub0_dev_download_params`, `sub0_dev_upload_opt`, `sub0_dev_download_opt` |
| training | `sub0_dev_train_reserve`, `sub0_dev_train_step`, `sub0_dev_backward`, `sub0_dev_time_train_step` |
| memory/planning | `sub0_dev_train_footprint`, `sub0_dev_train_predicted_mb`, `sub0_dev_free_mem_mb` |
| decode | `sub0_dev_kv_reset`, `sub0_dev_forward_one` |
| precision | `sub0_dev_set_tf32` (a *hint*; a backend without the concept ignores it) |

**Deliberately NOT in the neutral seam**: the diagnostic/parity-check functions
(`sub0_cuda_attn_check`, `sub0_cuda_muon_ns_check`, `sub0_cuda_*_check` …). Those exist to validate a
*specific backend's kernels against the CPU reference* — they are inherently per-backend, live in
per-backend test files (`tests/cuda_tests.cpp` is, by nature, a CUDA-backend test suite), and each new
backend brings its own parity suite. Forcing them through a neutral seam would be abstraction theater.

### 2. Capability struct instead of hardcoded assumptions

```c
struct Sub0DeviceCaps {
    const char* name;                  // "cuda", "rocm", "openvino", "none"
    int supports_train;                // full train_step/backward/opt-state residency
    int supports_decode;               // kv_reset/forward_one
    int supports_interception;         // decode-time expand/combine/compute injection
    int supports_binding_compose;      // encode_slot-class composed embeddings (scratch/persistent/sentinel)
    int supports_opt_state;            // upload/download_opt round-trip
    int supports_tf32;                 // the tf32 hint is meaningful
};
```

Consumers branch on **caps**, not on `#if defined(SUB0_BUILD_CUDA)` + folklore. Concretely:

- `decode.hpp` currently falls back to CPU whenever interception callbacks are supplied — that rule
  becomes `caps.supports_interception == 0`, and a future backend that CAN intercept gets used
  automatically with zero consumer changes.
- The hybrid CPU/GPU router's `needs_cpu = !src.doc_bindings.empty()` becomes
  `needs_cpu = !src.doc_bindings.empty() && !caps.supports_binding_compose` — the day a backend ships
  `encode_slot` kernels, binding windows start routing to the device by flipping one caps bit in that
  backend, not by editing the training loop. **This is the designed landing zone for GPU-accelerating
  the scratch/sentinel/persistent mechanisms** (today's spikes run them CPU-only).
- An **inference-only backend (OpenVINO's natural shape)** advertises
  `supports_train=0, supports_decode=1`: generation/eval offload works, and the train stage keeps the
  CPU (or another device) without any consumer special-casing per backend name.

The compile-time gate `SUB0_BUILD_DEVICE` (aliasing today's `SUB0_BUILD_CUDA`) still answers the one
question that must be compile-time: "is any device backend linked into this build at all?" Everything
finer-grained is a caps query (cheap, called at setup, never in a hot loop).

### 3. Link-time backend selection (unchanged philosophy, generalized axis)

Exactly **one** device backend library per build. `cmake/Backends.cmake` keeps `SUB0_COMPUTE` as the
user-facing axis and gains (when a second backend lands) `SUB0_DEVICE=CUDA|ROCM|OPENVINO` selecting
*which* implementation provides the `sub0_dev_*` symbols. No runtime registry: adding a backend is a new
implementation TU + a CMake branch, not a plugin system.

### 4. Migration path (each step independently shippable)

1. **DONE — bridge**: `device_backend.hpp` declares the neutral seam; under `SUB0_BUILD_CUDA` the
   neutral functions are zero-cost inline forwards to `sub0_cuda_*`; without any device backend they
   are stubs (fail-fast returns + a `"none"` caps struct), so consumers can drop their own `#if`
   scaffolding as they migrate. `sub0_dev_caps()` for CUDA reports today's truthful capability set
   (train+decode+opt+tf32, no interception, no binding-compose).
2. **Consumer migration** (mechanical, per-file): replace scattered `extern "C" sub0_cuda_*`
   declarations + calls with the header + `sub0_dev_*`; replace assumption-branches with caps queries.
   Order: `memplan.hpp` → `decode.hpp` → `train_stage.cpp`/`tune` → tools.
3. **Symbol flip**: once no production consumer names `sub0_cuda_*`, rename the exports in
   `backend_cuda.cu` to `sub0_dev_*` natively and delete the bridge forwards (the per-backend test
   suite keeps whatever backend-private symbols it needs).
4. **Second backend**, when wanted: implement `sub0_dev_*` in `src/backend_rocm.cpp` (HIP's API shape
   is deliberately CUDA-like; the port is mostly mechanical) or an OpenVINO inference-only TU; add the
   `SUB0_DEVICE` CMake branch; ship its parity-test suite alongside.

### How to add a backend (checklist for the future)

1. New TU implementing every `sub0_dev_*` symbol (stub + honest caps for anything unsupported).
2. `sub0_dev_caps()` tells the truth — consumers are DESIGNED to degrade around zeros.
3. A parity test suite vs the CPU reference (the `cuda_tests.cpp` precedent: differentials, not smoke).
4. A `cmake/Backends.cmake` branch wiring sources + toolchain, keeping single-backend-per-build.
5. `docs/BACKENDS.md` (this file) gains the backend's row in the caps table.
