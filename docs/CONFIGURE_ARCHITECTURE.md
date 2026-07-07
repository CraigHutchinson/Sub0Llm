# Configuration architecture — `sub0llm-configure` owns the config

Goal: make `sub0llm-configure` the **user-runnable source of truth** for configuration. It auto-derives
sensible defaults (model size from the corpus, system config by probing the machine, tokenizer/vocab
from the corpus) and bakes them into generated headers. Changing a default is `sub0llm-configure --x`
+ build — **not** a CMake reconfigure. CMake stops carrying settings the configurator can decide.

## The three build states

1. **Fresh checkout (no generated headers).** `sub0llm-configure` builds with *no* dependency on the
   generated headers (it includes only `casing/tokenizer/unigram/memplan`, never `sub0_config.hpp`), so
   it can always be built and run first: `cmake --build <dir> --target sub0llm-configure`, then
   `sub0llm-configure --corpus <c>`. There is no CMake-orchestrated auto-generation step — the engine and
   stage-tool targets are unconditionally defined but simply fail to compile with a plain "file not
   found" on the generated header until that run has happened (see
   [WORKFLOW_ARCHITECTURE.md](WORKFLOW_ARCHITECTURE.md)).
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

## Auto-sizing (current)

`autosize(corpus_bytes, vram_mb, size_scale)` — formula-based, overridable: every dimension is a
smooth, monotonic function of corpus scale, not a handful of hand-picked per-bucket values. Token
count is estimated from raw bytes (~4 bytes/token, refined for real by tokenization); the target
parameter budget follows a tokens/param ratio well above pure Chinchilla-optimal (100:1, informed by
real small-model practice — TinyStories' own reference configs, SmolLM2, the FineWeb-Edu paper's own
ablation model); that budget decomposes into `d_model`/`n_layers` via a width/depth aspect ratio that
itself scales with model size (deeper/narrower at small scale, matching TinyStories/SmolLM2's own real
configs; wider at large scale, matching SmolLM2-1.7B), snapped to `head_dim=64`-multiples
(head-divisibility falls out automatically); vocab follows Heaps'-law-style sublinear growth in token
count, as a separate axis from the capacity budget; `seq_len` scales mildly with model width. A
hardware-aware clamp (`vram_mb`, 0 = unknown/CPU-only = no-op) shrinks the shape until it fits the
detected GPU's VRAM at batch=1, so pointing the configurator at a big corpus never outright fails on a
modest card by default (an explicit `--dmodel` override that intentionally exceeds VRAM still hits the
existing hard-error path below, correctly). `--size-scale` (default 1.0) is a caller-chosen multiplier
on the target-parameter budget — a minimal/fast/safe vs. more generous starting point, same formula.
See the function's own doc comment in `include/sub0/config_util.hpp` for the full reasoning and
citations. The
`--dump-vocab` vocab-curve remains the source of truth for the IDEAL vocab once a corpus is actually
scanned; this is the informed starting point before that analysis exists.

## Migration stages

1. **Dims + vocab auto-sized, removed from CMake.** ✅ (`autosize`; CMake no longer passes `--dmodel`…).
2. **Split headers** (`sub0_corpus.hpp` + `sub0_system.hpp` behind the `sub0_config.hpp` umbrella). ✅
3. **CUDA detection** — resolved: stays in CMake (it drives `nvcc` at configure time; see above). ✅
4. **Decouple re-config from a CMake reconfigure** — ✅ via auto-size + the `<corpus>.model` sidecar:
   changing a size is `sub0llm-configure …` (or edit the sidecar) then `cmake --build`. There is no
   CMake-side auto-generation left to keep in sync with this — `sub0llm-configure` is run directly (or
   via `scripts/workflow.ps1`); see [WORKFLOW_ARCHITECTURE.md](WORKFLOW_ARCHITECTURE.md) for why that is
   the deliberate end state, not a stopgap.

All four stages are complete; the pure decisions live in `sub0::config` (`config_util.hpp`) and are
unit-tested in `tests/config_tests.cpp`.
