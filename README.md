# Sub0Llm

A single-engine **CPU/GPU transformer language model in C++23**. The model's dimensions and its
vocabulary are *baked in at build time*: a configurator tokenizes the corpus, **auto-sizes** the model
to the corpus scale, derives the vocabulary, and emits `constexpr` config headers the engine is compiled
against. Training is data-parallel across cores (or on the CUDA device when one is present); storage is
static (BSS), not heap. There is no Python runtime.

The build is **self-configuring**: point it at a corpus and it picks a sensible model size, vocabulary,
precision, thread count and (if a GPU is present) the device backend — every choice overridable.

## Prerequisites

- **Clang** (C++23: `std::print`, `<format>`), **CMake ≥ 3.20**, **Ninja**.
- An **OpenMP** runtime (`libomp`) for the multi-threaded training path. The build *fails loudly*
  if OpenMP is missing; to build single-threaded on purpose, pass `-DSUB0_REQUIRE_OPENMP=OFF`.
- *(Optional, GPU training)* the **CUDA Toolkit** (`nvcc`) and an NVIDIA device. The CUDA backend
  is auto-detected; see [Building with the CUDA backend](#building-with-the-cuda-backend-windows)
  for the one Windows-specific setup step.

## Build

```sh
cmake --preset native                                        # configure the build (CUDA/backend detection)
cmake --build --preset native --target sub0llm-configure      # build just the configurator
out/build/native/sub0llm-configure --corpus data/tinystories.txt  # tokenize + auto-size + bake headers
cmake --build --preset native                                 # build the engine + stage tools
ctest --preset native                                          # run the unit tests
```

The `native` preset is an `-march=native` Release build. `debug` and `release` presets also exist.

`sub0llm-configure` is a plain executable with no generated inputs, so it always builds first. The engine
and stage-tool targets are unconditionally defined too, but simply fail to compile with a plain "file not
found" on the generated header until the configure step above has actually run once — there is no
CMake-orchestrated config step, and the build never regenerates config behind your back. Re-running
`sub0llm-configure` (a different corpus, an override flag) and rebuilding is how you reconfigure.

On configure it emits, into the build's `generated/` dir, **split** headers the engine compiles against
(see [Configuration](#configuration)):

- `sub0_corpus.hpp` — the model *identity* (dims, vocab, tokenizer scheme), frozen per corpus
- `sub0_system.hpp` — the *machine* (precision, thread/batch defaults, CUDA backend), per host
- `sub0_config.hpp` — a thin umbrella over both (what the engine includes)

The stage tools (`sub0llm-train/-gen/-tune`) are separate executables, built alongside `sub0llm` in the
step above. `scripts/workflow.ps1` wraps this whole sequence into one command; see
[docs/WORKFLOW_ARCHITECTURE.md](docs/WORKFLOW_ARCHITECTURE.md) for the full staged-workflow design.

### Building with the CUDA backend (Windows)

When a CUDA toolkit and device are present, the GPU training backend is built automatically. On
Windows, `nvcc` compiles `.cu` files using **MSVC `cl.exe`** as its host compiler, so `cl.exe` and the
MSVC/Windows-SDK headers/libraries must be on your environment. A plain PowerShell session does not have
these, and configure fails with `nvcc fatal : Cannot find compiler 'cl.exe' in PATH`.

Fix: run the build from a **Visual Studio developer environment** — open the *“x64 Native Tools Command
Prompt for VS”*, or initialize a shell once with `vcvars64.bat`:

```pwsh
& "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
```

The CPU-only build has no such requirement — only the CUDA backend needs `cl.exe`.

## Quick start

```sh
exe=out/build/native/sub0llm        # (.exe on Windows)

$exe train                          # train on the baked-in corpus into an auto-named model dir
$exe models                         # list trained models (which load into this build)
$exe gen <model.bin> "One day,"     # generate a continuation
$exe autotemp <model.bin>           # pick a sampling temperature by matching held-out text
```

`train` with **no path** creates and registers a structured model directory (see *Models* below);
`--steps 0` (default) auto-sizes the run to the corpus and stops on a validation plateau. The GPU
backend is used automatically when it was built (see *Compute backend* below).

## Configuration

**You do not set the model dimensions** — the configurator auto-sizes `d_model / n_layers / n_heads /
seq_len / vocab` from the corpus byte-scale. To re-configure, run the configurator directly (or edit the
sidecar) and rebuild — **no CMake reconfigure** is needed, because the dimensions live in the generated
headers the tool owns, not in CMake:

```sh
configure=out/build/native/sub0llm-configure
$configure --corpus data/tinystories.txt              # auto-size + emit the headers
$configure --corpus data/tinystories.txt --dmodel 256 # pin a dimension (the rest still auto-size)
cmake --build --preset native                         # recompiles against the new headers
```

A chosen size **persists** in a `<corpus>.model` sidecar (key=value), so it survives later re-runs.
The resolution precedence is **CLI flag > sidecar > auto-size**.

| What | Owner | Notes |
|---|---|---|
| `d_model / n_layers / n_heads / seq_len / vocab` | configurator | auto-sized from corpus; `--dmodel`… pins; `<corpus>.model` persists |
| tokenizer (Unigram LM, JOIN scheme), precision | configurator | derived; `--join`, `--prec-gemm/--prec-act` override |
| compute backend (CPU vs GPU) | configurator | **GPU when the CUDA backend was built**, else CPU; `--compute` pins |
| whether CUDA *exists* / is built | CMake | detects the toolkit + device; `-DSUB0_COMPUTE=CPU` forces a CPU-only build |
| corpus, positional encoding | configurator | `--corpus` (required), `--pos-encoding`/`--rope-theta` |
| ternary block weights | CMake cache (`SUB0_TERNARY`) **and** configurator (`--ternary`) | CMake picks the *source files* compiled in (ternary is CPU-only for now); the configurator bakes the *flag* the engine reads — keep both in sync |

See [docs/CONFIGURE_ARCHITECTURE.md](docs/CONFIGURE_ARCHITECTURE.md) for the build-state model and
[docs/WORKFLOW_ARCHITECTURE.md](docs/WORKFLOW_ARCHITECTURE.md) for the staged user-driven workflow roadmap.

### Compute backend

CMake checks whether CUDA *exists* (toolkit + device → it builds the `nvcc` device backend); the
configurator decides whether to *use* it (`COMPUTE_MODE`). With a device present, `cmake --preset native`
builds the GPU training backend and the configurator defaults to it. Force CPU-only with
`-DSUB0_COMPUTE=CPU`, or keep the backend but run on the CPU with `sub0llm-configure --compute 0`.

### Positional encoding

`sub0llm-configure --pos-encoding` selects how position is injected (baked as `constexpr` into the
generated header): **`1` = RoPE** (default) — rotary embeddings on Q/K, relative position, no learned
table, extends to longer contexts; **`0` = ABSOLUTE** — a learned `pos_emb[SEQ_LEN, D_MODEL]` table.
`--rope-theta` (default `10000`) is RoPE's frequency base. A model is only comparable to others built
with the **same** scheme.

## Using a larger corpus (out-of-core)

The default corpus is small (`data/tinystories.txt`). For something big like **FineWeb-Edu**:

```sh
python scripts/get_fineweb.py --out data/fineweb_edu.txt          # ~46GB; --shards N for a slice
out/build/native/sub0llm-configure --corpus data/fineweb_edu.txt  # auto-sizes UP to a larger model
cmake --build --preset native
$exe train
```

`--corpus-pretok` (`2` = AUTO, the default) decides by scale whether to pre-tokenize the corpus to
`corpus.tok` (fast random-access training) or **tokenize on demand** from the raw text when the token
copy would not fit comfortably in RAM (estimated `corpus.tok` size vs half of physical RAM) — so a
corpus larger than memory just works; `1`/`0` force it on/off. The configurator caches its scan
(`<corpus>.words`) so re-configuring with a different vocab skips the multi-pass scan.

## Models

Each trained model is its own directory under `models/`, named for its identity:

```
models/sub0llm_<corpus>_d<D>l<L>h<H>sq<SEQ>v<VOCAB>[t][r]_<gitSHA>/
  model.bin  model.bin.ckpt  meta.txt
```

The "registry" is the set of `meta.txt` files (no separate index to drift out of sync).

```sh
$exe models            # list models; '*' = loadable by this build, 'x' = incompatible architecture
$exe models --prune    # delete models whose architecture this build can no longer load
```

The auto-derived path is deterministic, so re-running `train` resumes the same model from its checkpoint.

## Tooling

| Command | Purpose |
|---|---|
| `train` | train (auto-sized, plateau-stopped, crash-safe checkpoints) |
| `gen` | sample a continuation (`--temp`, `--topk`, `--n`) |
| `models [--prune]` | list / prune trained models |
| `report` | diagnose model sizing vs its corpus; per-knob retrain guidance |
| `memplan` | predicted train/gen memory footprints (breakdown + batch sweep vs VRAM) |
| `autotemp` | pick a coherence temperature by matching held-out perplexity |
| `vocab` | print the (Unigram) vocabulary table |
| `bench` | cycle-accurate hot-path benchmark (the optimization control) |
| `tune` | auto-tune threads / batch granularity for throughput |

The pipeline stages are **also standalone executables** — `sub0llm-configure`, `sub0llm-train`,
`sub0llm-gen`, `sub0llm-tune` — so each stage can be built/run on its own (`sub0llm-train --steps 0` ≡
`sub0llm train --steps 0`). The stage tools share one runner definition with the umbrella
(`include/sub0/cli_stages.hpp`); each compiles against whatever generated header already exists, so a
missing config header is a plain compile error, not a silent auto-regenerate. `sub0llm-configure` is the
config-independent front (`--dump-vocab` also writes readable corpus/token/vocab-curve analysis files).

## Layout

```
src/        engine (engine_core + backend_cpu/backend_cuda) + stages (train_stage, gen_stage) + driver
include/    public headers (core, casing, tokenizer, unigram, memplan, registry, config_util, cli_stages)
tools/      sub0llm-configure + sub0llm-{train,gen,tune} (thin stage mains) + cuda_selftest
cmake/      backend detection + the sub0_build_facts.hpp.in template
scripts/    data-acquisition helpers (get_fineweb.py)
tests/      Catch2 unit tests (engine + the engine-free tokenizer/config suite)
docs/       design notes (tokenizer, configure + workflow architecture, CUDA review)
```
