# Local Machine Setup — Building and Running sub0llm

This guide covers everything needed to clone, build, and run sub0llm on your
own hardware: Windows, Linux, and macOS, with or without a CUDA GPU.

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Windows build](#2-windows-build)
3. [Clone and configure (Linux / macOS)](#3-clone-and-configure-linux--macos)
4. [CPU-only build](#4-cpu-only-build)
5. [Native build for training runs](#5-native-build-for-training-runs)
6. [CUDA build](#6-cuda-build)
7. [Verify CUDA works](#7-verify-cuda-works)
8. [Run the chapters](#8-run-the-chapters)
9. [Run a training loop](#9-run-a-training-loop)
10. [Synthetic data with Ollama](#10-synthetic-data-with-ollama)
11. [Performance guide](#11-performance-guide)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

### Which platform are you on?

| Platform | Recommended path |
|----------|-----------------|
| **Windows 10/11** | Section 2 below — Visual Studio 2022 + CMake |
| **Ubuntu / Debian** | Sections 3–6 |
| **Fedora / RHEL** | Sections 3–6 (dnf commands shown) |
| **macOS** | Sections 3–5 (Clang via Xcode; no CUDA) |

---

## 2. Windows Build

Windows is a first-class supported platform. The recommended toolchain is
**Visual Studio 2022** (which bundles MSVC, CMake, and Ninja). All library
code and chapter binaries build and run on Windows without modification.

### Step 1 — Install Visual Studio 2022

Download the **Community** edition (free) from
[visualstudio.microsoft.com](https://visualstudio.microsoft.com/).

During installation, select these workloads:

- ☑ **Desktop development with C++**
- ☑ **C++ CMake tools for Windows** (inside the C++ workload, Individual Components)

That installs MSVC 19.x, CMake 3.29+, Ninja, and the Windows SDK.

> **Minimum version**: MSVC 19.38 (VS 2022 17.8) for full `std::format` and
> C++23 support. Use `winver` + Help → About in VS to check.

### Step 2 — Install Git

Download from [git-scm.com](https://git-scm.com/download/win) and install
with default options. Tick **"Git from the command line and also from 3rd-party
software"** so Git is on PATH.

### Step 3 — Open a Developer terminal

Open **"Developer Command Prompt for VS 2022"** (search Start menu).
This sets up the MSVC compiler, CMake, and Ninja on PATH automatically.

All commands in this section should be run in that prompt (or in
**Developer PowerShell for VS 2022** if you prefer PowerShell syntax).

### Step 4 — Clone the repository

```bat
git clone https://github.com/CraigHutchinson/Sub0Llm.git
cd Sub0Llm
git checkout claude/llm-cpp23-repo-init-1f1Il
```

### Step 5 — Configure (first time — downloads dependencies)

```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

CPM downloads spdlog, nlohmann/json, Catch2, and Eigen3 automatically.
This requires internet access and takes 30–60 seconds on first run.

Expected output (last lines):
```
-- sub0llm: BLAS not found — checking Eigen fallback
-- sub0llm: Eigen3 3.4.0 fetched via CPM — Eigen matmul dispatch enabled (K>=64)
-- Configuring done
-- Build files have been written to: ...\Sub0Llm\build
```

> **Note**: OpenBLAS is not in the default Windows package managers. Eigen3
> (fetched automatically via CPM) is used instead — it provides the same
> functionality with no extra installation.

### Step 6 — Build

```bat
cmake --build build --parallel
```

Expected: all targets link cleanly. The binaries land in `build\bin\`.

### Step 7 — Run the tests

```bat
ctest --test-dir build --output-on-failure
```

Expected:
```
100% tests passed, 0 tests failed out of 442
```

> **Checkpoint ✓** — all 442 tests green confirms the library, autograd,
> transformer, math head, checkpointing, and corpus streaming are all correct.

### Step 8 — Run a chapter

```bat
.\build\bin\ch01_foundations.exe
.\build\bin\ch24_real_training.exe --phase estimate
.\build\bin\ch23_reasoned_math.exe --phase register
```

### Step 9 — Release build with AVX2 (faster)

```bat
cmake -B build-rel -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build-rel --parallel
.\build-rel\bin\ch24_real_training.exe --phase train --steps 500
```

> The `^` character is the line-continuation in `cmd.exe`. In PowerShell use
> a backtick `` ` `` instead.

### Step 10 — CUDA build on Windows (optional, requires NVIDIA GPU)

#### Install CUDA Toolkit for Windows

1. Download **CUDA Toolkit ≥ 12** from
   [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).
   Choose: **Windows → x86_64 → 11/10 → exe (network)**.

2. Run the installer. Select **"Express"** — this installs the toolkit,
   the NVIDIA driver, and Visual Studio integration together.

3. Reboot when prompted.

4. Verify:
   ```bat
   nvcc --version
   nvidia-smi
   ```

   Expected:
   ```
   nvcc: NVIDIA (R) Cuda compiler driver, release 12.x
   +-------------------------------+
   | GPU 0: NVIDIA GeForce RTX ... |
   +-------------------------------+
   ```

#### Configure and build with CUDA

```bat
cmake -B build-cuda -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSUB0LLM_ENABLE_CUDA=ON ^
    -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build-cuda --parallel
```

CMake automatically finds `nvcc` from the CUDA Toolkit installation.
If it cannot find it, set:

```bat
set CUDAToolkit_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
cmake -B build-cuda -G Ninja -DSUB0LLM_ENABLE_CUDA=ON ^
    -DCUDAToolkit_ROOT="%CUDAToolkit_ROOT%"
```

#### Verify CUDA

```bat
ctest --test-dir build-cuda --output-on-failure
.\build-cuda\bin\ch02_backends.exe
```

### Windows-specific notes

| Topic | Detail |
|-------|--------|
| Path separator | Windows uses `\`; CMake and Ninja handle this automatically |
| Binary extension | Executables have `.exe`; scripts don't need to specify it in CMake |
| Line ending | Git is configured to check out with `\r\n` by default — the source handles both |
| `build-native` | `-march=native` is GCC/Clang only; MSVC uses `/arch:AVX2` or `/arch:AVX512` instead, which `build-rel` uses automatically when the flag is set |
| Ollama on Windows | Install from [ollama.com](https://ollama.com) — has a native Windows installer; runs as a system tray app on port 11434 same as Linux |
| Terminal | **Windows Terminal** (from the Microsoft Store, free) gives a much better experience than the legacy `cmd.exe` window |

---

## 3. Clone and configure (Linux / macOS)

### Compiler and tools

| Tool | Minimum | How to check |
|------|---------|--------------|
| GCC  | 13.x    | `gcc --version` |
| Clang | 17.x  | `clang --version` |
| CMake | 3.25   | `cmake --version` |
| Ninja | any    | `ninja --version` |
| Git  | any     | `git --version` |

Install on Ubuntu/Debian:
```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git
# GCC 13 on Ubuntu 22.04:
sudo apt install -y gcc-13 g++-13
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130
```

Install on Fedora/RHEL:
```bash
sudo dnf install -y gcc-c++ cmake ninja-build git
```

Install on macOS (Clang 17+ ships with Xcode 15+):
```bash
xcode-select --install
brew install cmake ninja
```

### Optional: OpenBLAS (speeds up large-matrix ops)

```bash
# Ubuntu/Debian
sudo apt install -y libopenblas-dev

# Fedora
sudo dnf install -y openblas-devel
```

CMake detects OpenBLAS automatically — no flag needed.

### Optional: CUDA Toolkit (GPU backend)

Required only if you have an NVIDIA GPU. See [Section 6](#6-cuda-build).

---

## 3. Clone and configure (Linux / macOS)

```bash
git clone https://github.com/CraigHutchinson/Sub0Llm.git
cd Sub0Llm
git checkout claude/llm-cpp23-repo-init-1f1Il
```

First-time CMake configuration downloads all CPM dependencies (spdlog,
nlohmann/json, Catch2, Eigen3). This requires internet access and takes
about 30–60 seconds.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Expected output (last few lines):
```
-- sub0llm: BLAS found — cblas_sgemm dispatch enabled (K>=64)
-- sub0llm: Eigen3 3.4.0 fetched via CPM — Eigen matmul dispatch enabled (K>=64)
-- Configuring done
-- Build files have been written to: .../Sub0Llm/build
```

---

## 4. CPU-only build

Build everything and run the test suite to confirm a clean baseline:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Expected output:
```
100% tests passed, 0 tests failed out of 442
Total Test time (real) = N sec
```

> **Checkpoint ✓** — all 442 tests green means every module from tensor ops
> through to the math reasoning head is working correctly on your machine.

---

## 5. Native build for training runs

The native build enables `-march=native`, LTO, and fast-math. On modern
hardware this gives **3–5× throughput** versus the debug build and is the
correct choice for any training run.

```bash
cmake -B build-native -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_NATIVE=ON
cmake --build build-native --parallel
```

> **Do not distribute binaries from this build.** `-march=native` generates
> instructions for your exact CPU; the binary will `SIGILL` on other machines.

Verify the native build works:

```bash
./build-native/bin/ch24_real_training --phase estimate
```

You should see a Chinchilla scaling table and, at the end, throughput close
to the numbers below:

| CPU class | Demo model (~5M params) | Expected tok/s |
|-----------|------------------------|----------------|
| Intel Xeon / Core with AVX-512 | ~300–600 |
| AMD Zen 4 with AVX-512 | ~250–500 |
| Apple M-series (Clang) | ~400–800 |
| Intel Core with AVX2 only | ~150–300 |

---

## 6. CUDA build (Linux)

### 6a. Install the CUDA Toolkit

The project requires **CUDA Toolkit ≥ 12**.

**Ubuntu / Debian** (recommended: use NVIDIA's official repo):
```bash
# Add NVIDIA package repo (example for Ubuntu 22.04 + CUDA 12.4)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4

# Add to PATH and LD_LIBRARY_PATH (add these to ~/.bashrc)
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
nvcc --version
nvidia-smi
```

Expected:
```
nvcc: NVIDIA (R) Cuda compiler driver
Cuda compilation tools, release 12.4, ...

+-------------------------+
| NVIDIA-SMI  ...  Driver |
| GPU  0: GeForce RTX ...  |
+-------------------------+
```

**Fedora / RHEL**:
```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install -y cuda-toolkit-12-4
```

**If you already have the toolkit** but CMake cannot find it, set:
```bash
export CUDAToolkit_ROOT=/usr/local/cuda
```

### 6b. Configure with CUDA

```bash
cmake -B build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_CUDA=ON \
    -DSUB0LLM_ENABLE_NATIVE=ON
```

Expected output includes:
```
-- sub0llm: CUDA toolkit 12.4 found
-- Build files have been written to: .../Sub0Llm/build-cuda
```

### 6c. Build

```bash
cmake --build build-cuda --parallel
```

The build compiles `.cu` files with `nvcc` and links `libcublas`, `libcurand`,
and `libcudart`. A successful build outputs:
```
[N/N] Linking CXX executable bin/tests
```

with no errors.

---

## 7. Verify CUDA works

Run the test suite against the CUDA build:

```bash
ctest --test-dir build-cuda --output-on-failure
```

All 442 tests should pass. The test suite exercises the tensor operations,
autograd, and all neural network modules. The CUDA backend is exercised by
`test_backends.cpp`.

Run the backends chapter to see GPU dispatch in action:

```bash
./build-cuda/bin/ch02_backends
```

Expected output includes lines like:
```
  CUDA matmul (M=256, N=256, K=256):  result verified ✓
  CUDA add element-wise (N=1M):       result verified ✓
```

> **What currently uses CUDA vs CPU**
>
> | Component | CUDA build | CPU build |
> |-----------|-----------|-----------|
> | Element-wise ops (Tensor add/mul/relu) | GPU kernels (Ch02) | AVX2/AVX-512 |
> | Matrix multiply (Tensor matmul) | GPU tiled kernel (Ch02) | OpenBLAS / AVX2 |
> | Autograd engine | CPU | CPU |
> | Transformer forward pass | CPU | CPU |
> | Training loop | CPU | CPU |
>
> The CUDA backend dispatches low-level tensor operations when a Tensor lives on
> a CUDA device. The training pipeline (autograd + transformer) currently runs
> on CPU. Wiring the transformer to allocate weights on the GPU device and route
> all ops through the CUDA dispatch is the natural next step (future chapter).

---

## 8. Run the chapters

Every chapter binary lives in `build-native/bin/` (or `build/bin/` for debug).

```bash
# Chapter overview / quick demo for each chapter:
./build-native/bin/ch01_foundations
./build-native/bin/ch05_autograd
./build-native/bin/ch08_gpt
./build-native/bin/ch10_modern_arch
./build-native/bin/ch16_thinking
./build-native/bin/ch20_rlhf
./build-native/bin/ch21_math_neurons --phase compare
./build-native/bin/ch22_math_lm
./build-native/bin/ch23_reasoned_math
./build-native/bin/ch24_real_training
```

Chapters with `--phase` options:

| Chapter | Phases | Notes |
|---------|--------|-------|
| `ch21_math_neurons` | `train`, `eval`, `compare`, `train_real`, `all` | `train` runs for 3000 steps |
| `ch23_reasoned_math` | `register`, `two_step`, `word`, `three`, `ood`, `all` | |
| `ch24_real_training` | `landscape`, `estimate`, `pipeline`, `synth`, `approaches`, `train`, `all` | `train` runs 2000 steps |

**On Windows**, replace `/` path separators and use the `.exe` suffix:
```bat
.\build-rel\bin\ch24_real_training.exe --phase estimate
.\build-rel\bin\ch23_reasoned_math.exe --phase all
.\build-rel\bin\ch21_math_neurons.exe  --phase compare
```

---

## 9. Run a training loop

### Demo training (runs in ~30 seconds, no data needed)

**Linux / macOS:**
```bash
./build-native/bin/ch24_real_training \
    --phase all \
    --ckpt-dir /tmp/sub0llm_ckpts
```

**Windows (Developer Command Prompt):**
```bat
.\build-rel\bin\ch24_real_training.exe ^
    --phase all ^
    --ckpt-dir %TEMP%\sub0llm_ckpts
```

### Extended training (2000 steps, ~3 minutes)

**Linux / macOS:**
```bash
./build-native/bin/ch24_real_training \
    --phase train \
    --steps 2000 \
    --ckpt-dir /tmp/sub0llm_ckpts
```

**Windows:**
```bat
.\build-rel\bin\ch24_real_training.exe ^
    --phase train ^
    --steps 2000 ^
    --ckpt-dir %TEMP%\sub0llm_ckpts
```

### Resume from a checkpoint

Training auto-resumes from the latest checkpoint in `--ckpt-dir`. Run the
command a second time to pick up where it left off. The output shows:
```
Resuming from step N
```

### Math specialisation training (Ch21)

**Linux / macOS** (background process):
```bash
nohup ./build-native/bin/ch21_math_neurons \
    --phase train \
    --ckpt-dir /tmp/ch21_ckpts \
    --steps 3000 \
    --token-mode real \
    > /tmp/ch21_train.log 2>&1 &

tail -f /tmp/ch21_train.log
```

**Windows** (background via `start`):
```bat
start /B .\build-rel\bin\ch21_math_neurons.exe ^
    --phase train ^
    --ckpt-dir %TEMP%\ch21_ckpts ^
    --steps 3000 ^
    --token-mode real ^
    > %TEMP%\ch21_train.log 2>&1

type %TEMP%\ch21_train.log
```

### Multi-step reasoning training (Ch23)

```bash
# Linux / macOS
./build-native/bin/ch23_reasoned_math --phase two_step
./build-native/bin/ch23_reasoned_math --phase ood
./build-native/bin/ch23_reasoned_math --phase all
```

```bat
:: Windows
.\build-rel\bin\ch23_reasoned_math.exe --phase two_step
.\build-rel\bin\ch23_reasoned_math.exe --phase ood
.\build-rel\bin\ch23_reasoned_math.exe --phase all
```

---

## 10. Synthetic data with Ollama

Chapter 24's `--phase synth` and `--phase train` can generate training data
from a local Ollama instance instead of using the hardcoded fallback corpus.

### Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**

Download the installer from [ollama.com/download](https://ollama.com/download/windows).
Run `OllamaSetup.exe` — it installs Ollama as a background service and adds
`ollama` to PATH. It starts automatically and runs on `localhost:11434`.

### Pull a model and start the server

**Linux / macOS:**
```bash
ollama pull llama3       # 4.7 GB
ollama serve &           # starts in background (Linux); macOS uses the menu bar app
```

**Windows** (Ollama starts automatically after install; just pull the model):
```bat
ollama pull llama3
:: Server is already running as a Windows service — no "serve" needed
```

Verify it's running:
```
curl http://localhost:11434
:: or in PowerShell:
Invoke-WebRequest http://localhost:11434
```

### Run with Ollama data generation

**Linux / macOS:**
```bash
./build-native/bin/ch24_real_training --phase synth
./build-native/bin/ch24_real_training \
    --phase train --steps 2000 \
    --ckpt-dir /tmp/sub0llm_ckpts
```

**Windows:**
```bat
.\build-rel\bin\ch24_real_training.exe --phase synth
.\build-rel\bin\ch24_real_training.exe ^
    --phase train --steps 2000 ^
    --ckpt-dir %TEMP%\sub0llm_ckpts
```

Expected output on both platforms:
```
Attempting Ollama at localhost:11434... [OK]
Generated N samples (avg M tokens each)
```

The training loop queries Ollama for 5 prompt templates (factual, Q&A,
reasoning, definitions, examples). Generated text is tokenised with BPE and
fed directly into the training loop — no data files needed.

> **Model choice**: smaller Ollama models (`llama3`, `phi3`, `gemma3:4b`)
> generate faster; larger models (`llama3:70b`) produce higher-quality data.
> Data quality matters more than quantity for small-model training.

---

## 11. Performance guide

### CPU throughput scaling

Training throughput (tokens/second) scales roughly as:

```
throughput ∝ 1 / (n_layers × D²)
```

Benchmarks on native build with 6-layer ModernGPT:

| Model dim D | ~Params | Approx tok/s (AVX-512) | Approx tok/s (AVX2 only) |
|-------------|---------|------------------------|--------------------------|
| 64  | ~1M   | ~3,000 | ~1,500 |
| 128 | ~4M   | ~700   | ~350 |
| 256 | ~14M  | ~200   | ~100 |
| 512 | ~55M  | ~55    | ~28 |
| 768 | ~125M | ~24    | ~12 |

### Time to Chinchilla-optimal (CPU, native build, 1 thread)

| Model | Params | Optimal tokens | Time (AVX-512) |
|-------|--------|---------------|----------------|
| demo  | 14M | 280M tokens | ~16 days |
| small | 55M | 1.1B tokens | ~230 days |

**Conclusion**: CPU training is viable for research experiments at D≤128
(~4M params) with custom datasets of a few million tokens. For anything
larger, cloud GPU is the right tool (see the Chinchilla table in
`ch24_real_training --phase estimate`).

### GPU throughput (when CUDA is active)

On an NVIDIA A100 80GB (once the training pipeline is wired to CUDA devices):

| Model dim D | ~Params | Expected tok/s |
|-------------|---------|----------------|
| 256 | ~14M | ~15,000–25,000 |
| 512 | ~55M | ~5,000–8,000 |
| 768 | ~125M | ~2,000–3,500 |

This is roughly 50–100× faster than CPU, reducing the D=256 Chinchilla run
from 16 days to ~4 hours.

---

## 12. Troubleshooting

### CMake cannot find GCC 13

```
CMake Error: ... does not support C++23
```

Force the compiler explicitly:
```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=gcc-13 \
    -DCMAKE_CXX_COMPILER=g++-13
```

### CUDA not found despite toolkit being installed

```
CMake Error: Failed to find nvcc
```

Set the root explicitly:
```bash
cmake -B build-cuda -G Ninja \
    -DSUB0LLM_ENABLE_CUDA=ON \
    -DCUDAToolkit_ROOT=/usr/local/cuda-12.4
```

Or set the environment variable before running cmake:
```bash
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

### `libcuda.so` not found at runtime

```
error while loading shared libraries: libcuda.so.1: cannot open shared object file
```

The CUDA runtime library is in the driver, not the toolkit. Ensure the
NVIDIA driver is installed:
```bash
sudo apt install -y nvidia-driver-550   # or latest available
sudo reboot
```

### CPM download fails (no internet in air-gapped environment)

```
[CPM] Downloading spdlog ...
error: failed to connect
```

Download the CPM cache on a connected machine first:
```bash
# On connected machine:
cmake -B build -G Ninja -DCPM_SOURCE_CACHE=~/.cpm_cache
# Copy ~/.cpm_cache to the air-gapped machine, then:
cmake -B build -G Ninja -DCPM_SOURCE_CACHE=/path/to/cpm_cache
```

### Tests fail: `Catch2: no tests ran`

The test binary must be built before running ctest:
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Training produces NaN loss immediately

This usually means the learning rate is too high for the model size. The
default 3e-3 works well for D≤256. For larger models, try:
```bash
# Ch24 training — the learning rate is hard-coded; edit main.cpp
# Adam adam(params, 1e-3f);  ← use 1e-3 for D=512, 5e-4 for D=768
```

### Segfault or SIGILL on startup

Binary built with `-march=native` on one machine and run on another. Always
use the `build/` (debug) or `build-rel/` (AVX2-safe release) binaries for
distribution. Only use `build-native/` on the machine it was compiled on.

### Windows: Winsock or networking errors building ch24

Chapter 24 uses **cpp-httplib** (fetched automatically via CPM) which
initialises Winsock2 internally. No manual `ws2_32` link is needed.
If you see any socket-related linker error, ensure the
**Windows SDK** is installed by reinstalling the
"Desktop development with C++" workload in Visual Studio Installer and
ticking "Windows 10/11 SDK".

### Windows: `'std::format' is not a member of 'std'`

Your MSVC version is too old. `std::format` requires MSVC 19.29 (VS 2022
17.0). Update Visual Studio via **Help → Check for Updates**.

### Windows: CMake generates `.sln` instead of `build.ninja`

You forgot `-G Ninja`. If Visual Studio opens instead of a terminal build,
rerun with:
```bat
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```
Alternatively, open the "Developer Command Prompt for VS 2022" (not a regular
Command Prompt) — it puts Ninja on PATH.

### Windows: Ninja not found

Ninja is installed with Visual Studio's C++ workload. If it's not on PATH,
either:
- Use the **Developer Command Prompt for VS 2022** (it sets up PATH)
- Or install Ninja separately: `winget install Ninja-build.Ninja`

### Windows: `ollama` not found after installing Ollama

Close and reopen the terminal after installing — the installer adds Ollama
to PATH but the current session won't pick it up until restarted.

---

## Quick-reference: all build configurations

### Linux / macOS

```bash
# Debug build (tests, development)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# AVX2 release (portable on x86 machines made after ~2013)
cmake -B build-rel -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build-rel --parallel

# Native release (maximum performance, this machine only)
cmake -B build-native -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_NATIVE=ON
cmake --build build-native --parallel

# CUDA build — Linux (requires NVIDIA GPU + Toolkit ≥ 12)
cmake -B build-cuda -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSUB0LLM_ENABLE_CUDA=ON \
    -DSUB0LLM_ENABLE_NATIVE=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

### Windows (Developer Command Prompt for VS 2022)

```bat
:: Debug build (tests, development)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure

:: AVX2 release
cmake -B build-rel -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build-rel --parallel

:: CUDA build (requires NVIDIA GPU + CUDA Toolkit ≥ 12)
cmake -B build-cuda -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSUB0LLM_ENABLE_CUDA=ON ^
    -DSUB0LLM_ENABLE_AVX2=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

> **Note**: `-DSUB0LLM_ENABLE_NATIVE=ON` is a GCC/Clang-only flag and is
> silently ignored on MSVC. Use `-DSUB0LLM_ENABLE_AVX2=ON` (or AVX512) on
> Windows for the equivalent optimised release build.
