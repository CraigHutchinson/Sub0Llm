# Configuration architecture — `sub0-configure` owns the config

Goal: make `sub0-configure` the **user-runnable source of truth** for configuration. It auto-derives
sensible defaults (model size from the corpus, system config by probing the machine, tokenizer/vocab
from the corpus) and bakes them into generated headers. Changing a default is `sub0-configure --x`
+ build — **not** a CMake reconfigure. CMake stops carrying settings the configurator can decide.

## The three build states

1. **Fresh checkout (no generated headers).** `sub0-configure` builds with *no* dependency on the
   generated headers (it includes only `casing/tokenizer/unigram/memplan`, never `sub0_config.hpp`).
   The user builds it, runs it once to generate the headers, then builds the engine. The CMake build
   does **not** implicitly run the configurator.
2. **Configured (model baked, tune runtime-tweakable).** The corpus header bakes the model dims +
   vocab (compile-time, for the folded hot loops). Tune params (threads / GPU batch / TF32) have
   configurator-provided defaults but stay **runtime-adjustable** (the tune cache / CLI), so a model
   can be retuned without a recompile.
3. **Fully tuned (everything baked).** The tuned params are also baked compile-time → the maximally
   optimized build.

## Split generated headers

`generated/sub0_config.hpp` → two files so corpus and machine vary independently:
- **`sub0_corpus.hpp`** — `D_MODEL`, `N_LAYERS`, `N_HEADS`, `SEQ_LEN`, `VOCAB`, RoPE/ternary/precision,
  and the `DEFAULT_CORPUS/CORPUS_TOK/TOKENIZER` paths. Regenerated when the corpus or a pinned dim
  changes. **State 2 freezes this.**
- **`sub0_system.hpp`** — `HAS_CUDA`, `CUDA_ARCH`, `GPU_VRAM_MB`, `GPU_SHARED_MB`, `COMPUTE`,
  `CUDA_TF32`, `DEFAULT_THREADS/WINDOWS_PER_THREAD/GPU_BATCH`. Probed from the machine; the tune knobs
  default here and are baked in **state 3**.

## What the configurator decides (removed from CMake)

| Setting | Source | Status |
|---|---|---|
| `d_model / n_layers / n_heads / seq_len` | **auto-size from corpus bytes** (`autosize_dims`); `--dmodel N` pins | **DONE** — removed from CMake; configurator owns it |
| `vocab` | corpus scale / the `--dump-vocab` curve knee | next: auto from the curve |
| `has_cuda / cuda_arch / gpu_vram_mb / gpu_shared_mb` | configurator probes the device (`nvidia-smi`/driver API) at run time | planned (CMake still finds the CUDA *toolkit* to decide whether to compile the nvcc backend) |
| `threads / gpu_batch / tf32` | tune defaults (state 2 runtime, state 3 baked) | planned |
| corpus, ternary, pos-encoding, precision | explicit user intent | stays a CLI/CMake knob |

## Auto-sizing ladder (current)

`autosize_dims(corpus_bytes)` — coarse, overridable: `<64 MB → d192 L6 H6 seq256` (tinystories);
`<512 MB → d320 L8 H8`; `<4 GB → d448 L11 H7 seq256` (fineweb-smoke); `<32 GB → d640 L14 H8 seq512`;
else `d768 L16 H8 seq512`. The `--dump-vocab` vocab-curve already reports the ideal vocab; the
sizing ladder is the coarse default, to be refined against measured val-loss per corpus.

## Migration stages

1. **Dims auto-sized, removed from CMake.** ✅ (`autosize_dims`; CMake no longer passes `--dmodel`...).
2. **Split headers** (`sub0_corpus.hpp` + `sub0_system.hpp`); engine includes both.
3. **Configurator self-probes the device** (`nvidia-smi`), drop `--has-cuda/--cuda-arch/--gpu-vram`
   from the CMake command (CMake keeps only the toolkit-find that gates building the nvcc backend).
4. **Decouple the run from the build** — the CMake `add_custom_command` becomes a *fallback* (only if
   the headers are missing) or is dropped; the documented flow is `sub0-configure …` then `cmake
   --build`. A re-config is a tool run, not a CMake reconfigure.
