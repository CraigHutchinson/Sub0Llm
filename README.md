# Sub0Llm

A single-engine **CPU transformer language model in C++23**. The model's dimensions and its
BPE vocabulary are *baked in at build time*: a configurator tokenizes the corpus, derives the
vocabulary, and emits a `constexpr` config header the engine is compiled against. Training is
data-parallel across cores; storage is static (BSS), not heap. There is no Python runtime.

## Prerequisites

- **Clang** (C++23: `std::print`, `<format>`), **CMake ≥ 3.20**, **Ninja**.
- An **OpenMP** runtime (`libomp`) for the multi-threaded training path. The build *fails loudly*
  if OpenMP is missing (the data-parallel path needs it); to build single-threaded on purpose,
  pass `-DSUB0_REQUIRE_OPENMP=OFF`.
- *(Optional, GPU training)* the **CUDA Toolkit** (`nvcc`) and an NVIDIA device. The CUDA backend
  is auto-detected; see [Building with the CUDA backend](#building-with-the-cuda-backend-windows)
  for the one Windows-specific setup step.

## Build

```sh
cmake --preset native          # configure (tokenizes the corpus, bakes the config header)
cmake --build --preset native  # build -> out/build/native/sub0llm(.exe)
ctest --preset native          # run the unit tests
```

The `native` preset is an `-march=native` Release build. `debug` and `release` presets also exist.

### Building with the CUDA backend (Windows)

When a CUDA toolkit and device are present, the GPU training backend is built automatically. On
Windows, `nvcc` compiles `.cu` files using **MSVC `cl.exe`** as its host compiler, so `cl.exe`
and the MSVC/Windows-SDK headers and libraries must be on your environment. A plain PowerShell
session does not have these, and configure fails with:

```
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

Fix: run the commands from a **Visual Studio developer environment** so the MSVC toolchain is on
`PATH`. Either:

- open the **“x64 Native Tools Command Prompt for VS”** from the Start menu, **or**
- initialize an existing shell once by running `vcvars64.bat`, e.g. from PowerShell:

  ```pwsh
  & "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
  ```

  (adjust the path to your VS edition/year), **or** launch from the *Developer PowerShell for VS*.

Then run the normal build commands above from that environment. The CPU-only build has no such
requirement — only the CUDA backend needs `cl.exe`.

## Quick start

```sh
exe=out/build/native/sub0llm        # (.exe on Windows)

$exe train                          # train on the baked-in corpus into an auto-named model dir
$exe models                         # list trained models (which load into this build)
$exe gen <model.bin> "One day,"     # generate a continuation
$exe autotemp <model.bin>           # pick a sampling temperature by matching held-out text
```

`train` with **no path** creates and registers a structured model directory (see *Models* below);
pass an explicit path to control it yourself. `--steps 0` (default) auto-sizes the run to the
corpus and stops on a validation plateau.

## Using a larger corpus (out-of-core)

The default corpus is small (`data/tinystories.txt`). To train on something big like
**FineWeb-Edu**, fetch it with the helper (needs `pip install huggingface_hub duckdb`):

```sh
python scripts/get_fineweb.py --out data/fineweb_edu.txt        # ~46GB; --shards N for a slice
cmake --preset native -DSUB0_CORPUS="$(pwd)/data/fineweb_edu.txt"
cmake --build --preset native
$exe train
```

`SUB0_CORPUS_TOK=AUTO` (default) decides by scale whether to pre-tokenize the corpus to
`corpus.tok` (fast random-access training) or **tokenize on demand** from the raw text when the
token copy would not fit comfortably in RAM — so a corpus larger than memory just works. The
configurator caches its scan (`<corpus>.words`) so re-configuring with a different vocab skips
the multi-pass scan.

## Models

Each trained model is its own directory under `models/`, named for its identity:

```
models/sub0llm_<corpus>_d<D>l<L>h<H>sq<SEQ>v<VOCAB>[t]_<gitSHA>/
  model.bin  model.bin.ckpt  meta.txt
```

The "registry" is the set of `meta.txt` files (no separate index to drift out of sync).

```sh
$exe models            # list models; '*' = loadable by this build, 'x' = incompatible architecture
$exe models --prune    # delete models whose architecture this build can no longer load
```

The auto-derived path is deterministic, so re-running `train` resumes the same model from its
checkpoint.

## Configuration

Model dimensions and the corpus are CMake cache variables (re-tokenizes/recompiles on change):

```sh
cmake --preset native -DSUB0_D_MODEL=128 -DSUB0_N_LAYERS=6 -DSUB0_N_HEADS=4 -DSUB0_SEQ_LEN=64
```

`SUB0_D_MODEL`, `SUB0_N_LAYERS`, `SUB0_N_HEADS`, `SUB0_SEQ_LEN`, `SUB0_TERNARY`, `SUB0_CORPUS`,
`SUB0_CORPUS_TOK` (ON/OFF/AUTO), `SUB0_EXACT_MATH` (exact vs fast transcendentals).

### Positional encoding

`SUB0_POS_ENCODING` selects how position is injected (compile-time):

- **`ROPE`** (default) — rotary embeddings applied to Q/K inside attention. Encodes *relative*
  position in the attention dot-product, uses no learned position table, and extends to longer
  contexts far better than learned absolute embeddings. `SUB0_ROPE_THETA` (default `10000`) is its
  frequency base.
- **`ABSOLUTE`** — a learned `pos_emb[SEQ_LEN, D_MODEL]` table added to the token embedding.

```sh
cmake --preset native -DSUB0_POS_ENCODING=ROPE -DSUB0_ROPE_THETA=10000
```

The two schemes share the same weight layout, so a model is only comparable to others built with
the **same** `SUB0_POS_ENCODING`.

## Tooling

| Command | Purpose |
|---|---|
| `train` | train (auto-sized, plateau-stopped, crash-safe checkpoints) |
| `gen` | sample a continuation (`--temp`, `--topk`, `--n`) |
| `models [--prune]` | list / prune trained models |
| `report` | diagnose model sizing vs its corpus; per-knob retrain guidance |
| `memplan` | predicted train/gen memory footprints (breakdown + batch sweep vs VRAM) |
| `autotemp` | pick a coherence temperature by matching held-out perplexity |
| `vocab` | print the BPE vocabulary table |
| `bench` | cycle-accurate hot-path benchmark (the optimization control) |
| `tune` | auto-tune threads / batch granularity for throughput |

## Layout

```
src/        engine (engine.cpp) + stages (train_stage, gen_stage)
include/    public headers (core, casing, tokmap, registry, tune, coherence)
tools/      sub0-configure (build-time tokenizer)
scripts/    data-acquisition helpers (get_fineweb.py)
tests/      Catch2 unit tests
docs/       design notes (TOKENIZER_DESIGN.md)
```
