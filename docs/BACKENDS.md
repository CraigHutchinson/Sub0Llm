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
- The hybrid CPU/GPU router's `needs_cpu = !src.doc_bindings.empty()` became
  `needs_cpu = !src.doc_bindings.empty() && !caps.supports_binding_compose` (2026-07-18, DONE) — the
  day a backend ships `encode_slot` kernels, binding windows start routing to the device by flipping
  one caps bit in that backend, not by editing the training loop. **This was the designed landing zone
  for GPU-accelerating the scratch/sentinel/persistent mechanisms** (today's spikes still run
  sentinel/persistent CPU-only; production content-embed now routes through the device).
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
2. **Consumer migration — DONE (2026-07-18)**: replaced scattered `extern "C" sub0_cuda_*` declarations
   + calls with the header + `sub0_dev_*` in `decode.hpp` and `train_stage.cpp`/`tune`.
   `memplan.hpp` and `tools/` needed no changes (no direct `sub0_cuda_*` usage outside the deliberately-
   exempt per-backend diagnostic files, `cuda_selftest.cpp`/`cuda_tests.cpp`). Found and fixed a real
   pre-existing bug along the way: `train_stage.cpp`'s `hybrid_train` branch called the raw
   `sub0_cuda_upload_params` unconditionally, but that name was only ever declared under
   `SUB0_BUILD_CUDA` -- a genuine `-DSUB0_COMPUTE=CPU` build of `sub0_train` failed to compile
   ("undeclared identifier"), reproduced via `git stash` on the original code before fixing. Verified on
   real hardware in both configurations after: CPU-only default suite green (117 cases, 9,192,142
   assertions), CUDA `[cuda]`-tagged suite green (41 cases, 2,029,105 assertions). What's left in
   `sub0_cuda_*` now is only `backend_cuda.cu` itself plus the two permanently-exempt diagnostic files --
   assumption-branches -> caps queries (the hybrid router's `needs_cpu` flip, see below -- DONE
   2026-07-18) was separate follow-up work, not blocked on this.
3. **Symbol flip**: once no production consumer names `sub0_cuda_*`, rename the exports in
   `backend_cuda.cu` to `sub0_dev_*` natively and delete the bridge forwards (the per-backend test
   suite keeps whatever backend-private symbols it needs).
4. **Second backend**, when wanted: implement `sub0_dev_*` in `src/backend_rocm.cpp` (HIP's API shape
   is deliberately CUDA-like; the port is mostly mechanical) or an OpenVINO inference-only TU; add the
   `SUB0_DEVICE` CMake branch; ship its parity-test suite alongside.

### Binding-compose on CUDA (the first caps flip -- DONE + merged 2026-07-17)

`supports_binding_compose=1` for the CUDA backend: the device now composes overridden embed rows
(param-free MeanPool/Hash/HRR arms) itself, so the hybrid router's follow-up (below) can stop
forcing binding windows (content-embed / sentinel-pair / persistent) onto the CPU. Implemented by a
delegated Fable agent, reviewed line-by-line against the CPU `encode_slot`/`encode_slot_bwd`
reference (adjoint math, bounds validation, CUDA-graph-capture lifetime), and independently
re-verified on hardware: the 3 new parity test cases (46 assertions) plus the full `[cuda]`-tagged
suite (41 cases, 2,029,105 assertions) all green from a from-scratch rebuild. Two minor
non-blocking findings on file (a test-coverage gap on some validation-reject branches; a wire-
constant duplication between `device_backend.hpp` and `backend_cuda.cu` that only the CUDA TU's
`static_assert` currently guards) -- tracked as follow-up, not merge blockers. Design retained below
for reference; the **consuming flip** (`train_stage.cpp`'s hybrid `needs_cpu`) is also DONE now, see
its own bullet below.

Goal (met): `supports_binding_compose=1` for the CUDA backend, so the hybrid router stops forcing
binding windows (content-embed / sentinel-pair / persistent) onto the CPU. Param-free encoders only
(MeanPool/Hash/HRR -- the learned-`enc_w` encoders keep their documented CPU/single-thread limit).

- **Host-computed override table, device-composed rows.** The host already walks every window's ids and
  owns the binding tables, so it computes, per step: `override_idx[b*T+t]` (-1 = plain lookup, else an
  index into a flat entry array) + `entries[] = {frag_offset, frag_len, encoding}` + `frags[]`
  (concatenated fragment token ids). One upload alongside ids/targets; a new
  `sub0_dev_set_window_bindings(...)` seam call before `train_step`/`backward`, cleared per step.
  Device never parses binding SEMANTICS (which token follows which sigil, which table) -- that stays
  host-side where it already exists; the device only composes rows it is told to compose.
- **Kernels**: `embed_kernel`/`embed_act_kernel` gain an override branch (compose fragment rows via the
  encoder arm: mean / RoPE-rotate-sum / HRR circular-conv against a device-resident copy of
  `hrr_role_table`); `embed_backward_token_kernel` gains the adjoint (scatter-add into fragment rows --
  atomicAdd, same as the existing id scatter). `embed_one_body` (decode) reads the same table when
  installed. Naive O(C^2) HRR first (parity before performance, per project rule).
- **Parity tests** (`cuda_tests.cpp` precedent): forward differential vs CPU `encode_slot` per encoder
  arm; backward gradient-scatter differential vs CPU `encode_slot_bwd`; inertness (no table installed =
  bit-identical to today); then a full train-step differential on a binding-heavy synthetic batch.
- **Flip (DONE 2026-07-18)**: `sub0_dev_caps().supports_binding_compose = 1`, and `train_stage.cpp`'s
  hybrid `needs_cpu` became `!src.doc_bindings.empty() && !dev_binding_compose` (`dev_binding_compose`
  cached once from `sub0_dev_caps()`, not re-queried per window). The routing boolean alone isn't
  enough -- `GpuTrainer::backward_only`'s materialization was already internal to the class, so the
  caller (`sub0_train_stage`'s `hybrid_train` branch) independently re-derives each GPU sub-batch
  window's binding state the same way `materialize_cpu_window` already did for the CPU sub-batch: for
  every row `i` in the compacted GPU sub-batch, resolve its owning document via `doc_of(src.docs,
  starts[b])`, build a `ScratchBindings` view over that document's `doc_bindings`, and walk the
  window's tokens (`sb.bound`/`sb.fragments`) to flatten bound positions into the
  `override_idx`/`entries`/`frags` wire format -- installed via `sub0_dev_set_window_bindings` right
  before `gpu.backward_only`, cleared right after (its own documented per-step contract). No
  per-encoding gate is needed in that loop: `ContentEmbedKind`'s persisted enum only ever selects
  MeanPool/Hash/HRR, so every bound position a production run can produce already has a device kernel.
  Because production `content_embed` never selects a learned encoder, `needs_cpu` is unconditionally
  false whenever `dev_binding_compose` is true -- the CPU sub-batch side of `hybrid_train` goes idle in
  that case (kept, not deleted: the capability check stays per-window so a future *partial*-capability
  backend degrades correctly instead of needing a second code path). Verified: a new differential test
  (`cuda_tests.cpp`, "doc_of-resolved bindings across multiple windows per document match CPU") proves
  the doc_of-based multi-window-per-document resolution this adds (the existing end-to-end binding-
  compose test only ever used one distinct table per window, never a shared per-document one) against
  the CPU `train_batch` content-embed path; full `[cuda]` suite green (42 cases, 2,029,113 assertions)
  and CPU-only default suite green (117 cases, 9,192,142 assertions) from a from-scratch rebuild of
  both configs. Real hardware A/B on a live `--blend-config` (base + scratchspike, `content_embed:
  hrr`) training run, same corpus/schedule/seed, code toggled via `git stash`: steady-state throughput
  ~40-57% higher with the flip (e.g. 517K vs 365K tok/s, 441K vs 309K tok/s) with both runs' loss/
  val_nelbo curves equally clean (no divergence) -- direct confirmation the GPU sub-batch is actually
  absorbing content-embed windows now, not a no-op flip.

### How to add a backend (checklist for the future)

1. New TU implementing every `sub0_dev_*` symbol (stub + honest caps for anything unsupported).
2. `sub0_dev_caps()` tells the truth — consumers are DESIGNED to degrade around zeros.
3. A parity test suite vs the CPU reference (the `cuda_tests.cpp` precedent: differentials, not smoke).
4. A `cmake/Backends.cmake` branch wiring sources + toolchain, keeping single-backend-per-build.
5. `docs/BACKENDS.md` (this file) gains the backend's row in the caps table.
