---
name: cuda-build-setup
description: "How to build/run the CUDA preset on this Windows+clang machine, and the GPU's hard constraints"
metadata: 
  node_type: memory
  type: project
  originSessionId: 26b08af2-2bc6-47fd-985b-b0a3d9b828eb
---

The CUDA backend (src/backends/cuda/) was Ch02-era pedagogical f32 kernels (add/mul/relu/tiled-matmul), NOT wired to the Gemma engine. As of Jun 2026 the `cuda` preset BUILDS and is GPU-validated (commit 05eeb5c).

**Machine:** NVIDIA RTX 5070 Laptop, **8 GB VRAM**, compute cap **12.0 (sm_120 / Blackwell)**, CUDA Toolkit **13.2** at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`. cl.exe (MSVC host for nvcc) at `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231\...`.

**vcvars is primarily a CONFIGURE-time requirement (user clarified 2026-06-23), with a build-time caveat
that depends on the host compiler.** CMake needs vcvars64 active during `cmake --preset <cuda*>` to LOCATE
the compiler/nvcc-host and STORE paths in the cache. At BUILD time:
- **clang-configured tree** → clang is self-contained (finds its own + MSVC headers via -fms-compatibility),
  so `cmake --build` needs NO vcvars. This is the user's "build doesn't need vcvars" case and the INTENDED
  native config (clang gives real `-march=native`; see [[ch27-gpu-forward-build-arc]]).
- **cl.exe-configured tree** → the stored path is the compiler BINARY, but cl.exe still needs the MSVC
  INCLUDE/LIB ENV that vcvars sets, so building in a raw shell fails with `C1083: Cannot open ... 'cstdint'`.
  VS Code's CMake Tools auto-activates the dev env, so it "just works" there; a raw bash shell does not, so
  build under vcvars. (Verified 2026-06-23: build-cuda and build-cuda-native both came up as MSVC /arch:AVX2,
  NOT clang -march=native — the `_base` TTCMAKE typo means the compiler is auto-detected, and under vcvars
  auto-detect picks cl.exe. For device=cuda TRAINING this barely matters: the GPU does the matmuls, so a
  cl.exe-AVX2 host ≈ a clang-native host; the real native win is for CPU runs.)
The git SHA stamped into the model dir (`g<sha>`) is captured at CONFIGURE time, so reconfigure after new
commits or the provenance dir lags (build-cuda-native left at the Boost commit tagged dirs `g5c33952`).
To RUN a CUDA binary, put `...\CUDA\v13.2\bin` on PATH (for cudart64_13.dll).

**Four toolchain fixes that made it build (all committed):** spdlog /Zc:__cplusplus stripped under clang-GNU-frontend (deps.cmake); our -W/-m/-f flags guarded to COMPILE_LANGUAGE:CXX so nvcc doesn't forward them to cl.exe (compiler_flags.cmake); kernels.cu CUDA_SEPARABLE_COMPILATION OFF (src/CMakeLists.txt); CMAKE_CUDA_RUNTIME_LIBRARY None + link CUDA::cudart explicitly, because clang++ link driver can't resolve the BARE cudart.lib/cudadevrt.lib CMake injects (options.cmake).

**VRAM:** 8 GB dedicated, but the GPU exposes **55.4 GB total via shared system memory** (Windows WDDM) — so Gemma-12B CAN be loaded/exercised on the GPU via unified/managed memory (cudaMallocManaged), it just pages the VRAM-overflow over PCIe (bandwidth-gated for that portion). So Qwen3-0.6B (0.64 GB, fully VRAM-resident) is the baseline for OPTIMIZING the GPU kernel (no PCIe noise); Gemma-12B is exercisable for correctness/functional runs.

**CUDA Q8 matmul kernel (commit b3ef4d0):** matmul_q8_0_kernel in kernels.cu — Q8_0×Q8_0, one thread per output, int8 dot via __dp4a, per-block f16 scales. Validated vs CPU matmul_q8_0_q8_0 (relRMS ~1e-7 = f32 epsilon). Benched in qbench `--batch N` (prints a "CUDA Q8 dp4a" row). Perf on Qwen3-0.6B shapes (T=64, kernel-only): ~450-540 GFLOP/s = **8-9.6× over the CPU** (which already beats llama). NEGATIVE RESULT: column register-blocking (CPU NC trick) REGRESSED on GPU (484→282) — the naive M×T-thread kernel already gets weight reuse from L2 cache, and manual blocking just cuts occupancy + adds register pressure. More threads + L2 > hand-blocking here. ~0.5 TOPS is still far below Blackwell int8 tensor-core peak (tens of TOPS) — the real next lever is **int8 tensor cores (IMMA/wmma) or cuBLASLt int8 GEMM**, complicated by Q8_0's per-32-block scales (cuBLASLt wants per-tensor/row scale). See [[ch27-optimization-shortlist]], [[optimization-target-is-hardware-ceiling]].

KNOWN latent bug (not yet fixed, builds work via auto-detect): CMakePresets.json `_base` sets `TTCMAKE_C/CXX_COMPILER` (typo for `CMAKE_*`), so the compiler is auto-detected rather than pinned.
