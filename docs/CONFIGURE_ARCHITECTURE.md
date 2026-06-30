# Configuration architecture — `sub0llm-configure` owns the config

Goal: make `sub0llm-configure` the **user-runnable source of truth** for configuration. It auto-derives
sensible defaults (model size from the corpus, system config by probing the machine, tokenizer/vocab
from the corpus) and bakes them into generated headers. Changing a default is `sub0llm-configure --x`
+ build — **not** a CMake reconfigure. CMake stops carrying settings the configurator can decide.

## The three build states

1. **Fresh checkout (no generated headers).** `sub0llm-configure` builds with *no* dependency on the
   generated headers (it includes only `casing/tokenizer/unigram/memplan`, never `sub0_config.hpp`), so
   it can always be built and run first. The user *may* run it explicitly to (re)generate the headers;
   as a convenience the CMake build also auto-generates them (idempotent: same corpus + sidecar → same
   headers), so a plain `cmake --build` of a fresh checkout still works without a manual step.
2. **Configured (model baked, tune runtime-tweakable).** The corpus header bakes the model dims +
   vocab (compile-time, for the folded hot loops). Tune params (threads / GPU batch / TF32) have
   configurator-provided defaults but stay **runtime-adjustable** (the tune cache / CLI), so a model
   can be retuned without a recompile.
3. **Fully tuned (everything baked).** The tuned params are also baked compile-time → the maximally
   optimized build.

## Split generated headers — DONE

`generated/sub0_config.hpp` is now a thin **umbrella** (`#include "sub0_corpus.hpp"` +
`"sub0_system.hpp"`) so corpus and machine vary independently. The engine still includes only
`sub0_config.hpp`, so nothing downstream changed.
- **`sub0_corpus.hpp`** — the *model identity*: `D_MODEL/N_LAYERS/N_HEADS/SEQ_LEN/D_FF/D_HEAD`, `VOCAB`,
  `USE_TERNARY`, `JOIN_TOKENIZER`, `POS_ENCODING/ROPE_THETA`, and the `DEFAULT_CORPUS/CORPUS_TOK/TOKENIZER`
  paths. Regenerated when the corpus or a pinned dim changes. **State 2 freezes this.**
- **`sub0_system.hpp`** — the *machine + tuning*: precision (`Dtype`, `GEMM/ACT/...`), `HW_CONCURRENCY`,
  `DEFAULT_THREADS/WINDOWS_PER_THREAD/GPU_BATCH`, and the compute backend (`HAS_CUDA/COMPUTE_MODE/
  CUDA_ARCH/GPU_VRAM_MB/GPU_SHARED_MEM_MB/CUDA_TF32/ATTN_BWD_PER_QUERY`). Re-tuned/re-probed without
  touching the model identity; the tune knobs default here and are baked in **state 3**.

## `<corpus>.model` sidecar — DONE

So a chosen model size **persists across re-runs** (a build-time auto-regen keeps a pin instead of
re-auto-sizing). The configurator seeds `<corpus>.model` (key=value: `d_model/n_layers/n_heads/seq_len/
vocab`) on first run; later runs read it. Resolution precedence is **CLI (`--dmodel`…, nonzero) >
sidecar > auto-size**, chained via `sub0::config::fill_defaults`. Edit the sidecar to pin a size, then
rebuild — no CMake reconfigure. This is what makes a re-config a *tool run, not a CMake reconfigure*.

## What the configurator decides (removed from CMake)

| Setting | Source | Status |
|---|---|---|
| `d_model / n_layers / n_heads / seq_len / vocab` | **auto-size from corpus bytes** (`autosize`), persisted in `<corpus>.model`; `--dmodel N` pins | **DONE** — removed from CMake |
| `threads / windows_per_thread / gpu_batch / tf32 / attn_bwd` | tune cache (`parse_tune_cache`); state-2 runtime, state-3 baked | **DONE** — configurator folds the cache into `sub0_system.hpp` |
| precision (`GEMM/ACT` Dtype) | `--prec-*` + detected f16 capability (`resolve_precision`) | **DONE** — configurator owns it (errors on an unsupported choice) |
| `has_cuda / cuda_arch` | **CMake** — these gate + target the `nvcc` build (`CUDA_ARCHITECTURES sm_${ARCH}`) at *configure* time, before the configurator runs | **stays in CMake by design** (see below) |
| `gpu_vram_mb / gpu_shared_mb` | CMake device probe (header + the GPU-batch-fit guard) | stays with CMake's device detection |
| corpus, ternary, pos-encoding | explicit user intent | a CLI knob |

### Why CUDA stays a CMake responsibility (resolves old stage 3)

The original plan had the configurator self-probe the device (`nvidia-smi`) and drop
`--has-cuda/--cuda-arch` from CMake. That is **not** correct: `SUB0_CUDA_ARCH` drives the `nvcc` build
itself (`CUDA_ARCHITECTURES`, `sm_${ARCH}`) at CMake *configure* time — **before** the configurator is
built or run. Having the configurator own it would (a) need a chicken-and-egg bootstrap and (b) let the
emitted `CUDA_ARCH` constexpr desync from the arch the backend was actually compiled for. So the CUDA
*build* config legitimately belongs to CMake; the configurator owns the *corpus + runtime* config.
(`nvidia-smi` self-probe is verified to return e.g. `12.0, 8151` here, so the data is available if a
future fully-decoupled bootstrap ever wants it — but it is not a win today.)

## Auto-sizing ladder (current)

`autosize(corpus_bytes)` — coarse, overridable: `<64 MB → d192 L6 H6 seq256` (tinystories);
`<512 MB → d320 L8 H8`; `<4 GB → d448 L11 H7 seq256` (fineweb-smoke); `<32 GB → d640 L14 H8 seq512`;
else `d768 L16 H8 seq512`. The `--dump-vocab` vocab-curve already reports the ideal vocab; the
sizing ladder is the coarse default, to be refined against measured val-loss per corpus.

## Migration stages

1. **Dims + vocab auto-sized, removed from CMake.** ✅ (`autosize`; CMake no longer passes `--dmodel`…).
2. **Split headers** (`sub0_corpus.hpp` + `sub0_system.hpp` behind the `sub0_config.hpp` umbrella). ✅
3. **CUDA detection** — resolved: stays in CMake (it drives `nvcc` at configure time; see above). ✅
4. **Decouple re-config from a CMake reconfigure** — ✅ via auto-size + the `<corpus>.model` sidecar:
   changing a size is `sub0llm-configure …` (or edit the sidecar) then `cmake --build`. The
   `add_custom_command` remains the convenience that auto-generates the headers for a fresh checkout.

All four stages are complete; the pure decisions live in `sub0::config` (`config_util.hpp`) and are
unit-tested in `tests/config_tests.cpp`.
