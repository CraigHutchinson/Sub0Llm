# cmake/Backends.cmake — compute-backend selection (the CPU / GPU / HYBRID axis).
#
# sub0llm configures for ONE optimal compute target for the host and compiles only
# that path -- the project's compile-time-decision philosophy: no runtime backend
# switch for the pure cases, so every dead path is eliminated and every optimization
# pathway stays open. SUB0_COMPUTE picks the target:
#
#   CPU    - scalar + OpenMP backend (src/backend_cpu.cpp)         [always the engine]
#   GPU    - CPU engine + CUDA device TRAINING (src/backend_cuda.cu, nvcc)
#   HYBRID - CPU + CUDA, autotuned split                           [Phase 3 -- not yet]
#   AUTO   - resolve the best available at configure time (-> GPU if a CUDA device is
#            present, else CPU)                                    [default]
#
# The CPU backend is ALWAYS the engine (the sub0:: API: forward / generation / sampling
# / eval); it is the baseline and the GPU's numerical-parity reference. GPU mode keeps
# that CPU engine and ADDITIONALLY builds the CUDA backend so the training loop runs on
# the device (the train-stage fast-path) -- generation/eval stay on the CPU engine until
# a full GPU engine lands (Phase 3 / HYBRID). So SUB0_COMPUTE drives whether the CUDA
# backend is built (SUB0_BUILD_CUDA, derived below), not which engine sub0_core compiles.
#
# Outputs (consumed by the top-level build):
#   SUB0_COMPUTE_RESOLVED - the concrete mode after AUTO resolution
#   SUB0_BACKEND_SOURCES  - the engine translation unit(s) for sub0_core (always CPU)
#   SUB0_BUILD_CUDA       - whether to build the CUDA device-training backend (derived)

set(SUB0_COMPUTE "AUTO" CACHE STRING "Compute backend: AUTO / CPU / GPU / HYBRID")
set_property(CACHE SUB0_COMPUTE PROPERTY STRINGS AUTO CPU GPU HYBRID)

# Performance-knob expression (see include/sub0/knob.hpp). SUB0_TUNING ON builds knobs as
# runtime-mutable so the autotuner can sweep them; OFF (default) bakes them as compile-time
# constants so hot paths fold the branch away. Individual knobs (e.g. SUB0_CUDA_TF32) carry
# the baked value -- normally written by the autotuner into the tune cache and emitted by the
# configurator, like DEFAULT_THREADS.
option(SUB0_TUNING "Build performance knobs as runtime-tunable (autotuner) instead of baked constants" OFF)
option(SUB0_CUDA_TF32 "Bake TF32 tensor-core GEMM math ON (measured: no win at small K, default OFF)" OFF)
if(SUB0_CUDA_TF32)
  set(SUB0_CUDA_TF32_FLAG 1)
else()
  set(SUB0_CUDA_TF32_FLAG 0)
endif()

# CUDA detection (facts only -- the GPU backend compiles in Phase 2). find_package
# locates the toolkit WITHOUT enabling the CUDA language (that happens in Phase 2 when
# .cu files exist); nvidia-smi gives the first device's compute capability + VRAM. These
# are baked into the generated config (HAS_CUDA / CUDA_ARCH / GPU_VRAM_MB) so the engine
# knows the host's GPU at compile time.
set(SUB0_HAS_CUDA 0)
set(SUB0_CUDA_ARCH 0)
set(SUB0_GPU_VRAM_MB 0)
set(SUB0_GPU_SHARED_MB 0)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  find_program(SUB0_NVIDIA_SMI nvidia-smi)
  if(SUB0_NVIDIA_SMI)
    execute_process(
      COMMAND "${SUB0_NVIDIA_SMI}" --query-gpu=compute_cap,memory.total --format=csv,noheader,nounits
      OUTPUT_VARIABLE _sub0_smi OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE _sub0_smi_rc)
    if(_sub0_smi_rc EQUAL 0 AND _sub0_smi)
      string(REGEX MATCH "^[^\r\n]+" _sub0_smi_line "${_sub0_smi}")   # first GPU only
      string(REPLACE "," ";" _sub0_smi_fields "${_sub0_smi_line}")
      list(GET _sub0_smi_fields 0 _sub0_cc)      # e.g. "12.0"
      list(GET _sub0_smi_fields 1 _sub0_vram)    # e.g. " 8151"
      string(STRIP "${_sub0_vram}" SUB0_GPU_VRAM_MB)
      string(REPLACE "." "" _sub0_arch "${_sub0_cc}")   # "12.0" -> "120"
      string(STRIP "${_sub0_arch}" SUB0_CUDA_ARCH)
      set(SUB0_HAS_CUDA 1)
    endif()
  endif()
  # Shared GPU memory: on Windows (WDDM) the GPU can address a slice of system RAM as
  # overflow -- exceeding dedicated VRAM thrashes over PCIe instead of OOMing, and the
  # same region can back zero-copy/mapped host transfers. The WDDM default is ~half of
  # system RAM (matches Task Manager's "Shared GPU memory" / DXGI SharedSystemMemory).
  # 0 on platforms without it (e.g. Linux discrete CUDA, where overflow hard-OOMs).
  if(SUB0_HAS_CUDA AND WIN32)
    cmake_host_system_information(RESULT _sub0_ram_mb QUERY TOTAL_PHYSICAL_MEMORY)
    math(EXPR SUB0_GPU_SHARED_MB "${_sub0_ram_mb} / 2")
  endif()
  message(STATUS "CUDA: toolkit ${CUDAToolkit_VERSION} found; device sm_${SUB0_CUDA_ARCH}, "
                 "${SUB0_GPU_VRAM_MB} MB VRAM + ${SUB0_GPU_SHARED_MB} MB shared (overflow)")
else()
  message(STATUS "CUDA: no toolkit found -- CPU-only host")
endif()

# AUTO resolves to the best AVAILABLE backend: GPU when a CUDA device was detected at
# configure time, else CPU.
set(SUB0_COMPUTE_RESOLVED "${SUB0_COMPUTE}")
if(SUB0_COMPUTE STREQUAL "AUTO")
  if(SUB0_HAS_CUDA)
    set(SUB0_COMPUTE_RESOLVED "GPU")
  else()
    set(SUB0_COMPUTE_RESOLVED "CPU")
  endif()
endif()

# The engine sub0_core compiles is ALWAYS the CPU backend (the sub0:: API + parity reference).
# GPU mode additionally builds the CUDA device-training backend; SUB0_BUILD_CUDA is DERIVED from
# the resolved mode (it is not an independent toggle), so SUB0_COMPUTE is the single switch.
set(SUB0_BACKEND_SOURCES src/backend_cpu.cpp)

if(SUB0_COMPUTE_RESOLVED STREQUAL "CPU")
  set(SUB0_BUILD_CUDA OFF CACHE BOOL "Build the CUDA device-training backend (derived from SUB0_COMPUTE)" FORCE)
elseif(SUB0_COMPUTE_RESOLVED STREQUAL "GPU")
  if(NOT SUB0_HAS_CUDA)
    message(FATAL_ERROR
      "SUB0_COMPUTE=GPU but no CUDA toolkit/device was detected at configure time. Install the "
      "CUDA toolkit and ensure a device is visible (nvidia-smi), or build with -DSUB0_COMPUTE=CPU "
      "(or AUTO, which falls back to CPU when no device is present).")
  endif()
  set(SUB0_BUILD_CUDA ON CACHE BOOL "Build the CUDA device-training backend (derived from SUB0_COMPUTE)" FORCE)
elseif(SUB0_COMPUTE_RESOLVED STREQUAL "HYBRID")
  message(FATAL_ERROR
    "SUB0_COMPUTE=HYBRID (an autotuned CPU/GPU split) is not implemented yet (Phase 3). Use "
    "-DSUB0_COMPUTE=GPU for device training, or -DSUB0_COMPUTE=CPU.")
else()
  message(FATAL_ERROR "SUB0_COMPUTE='${SUB0_COMPUTE}' is invalid (choose AUTO / CPU / GPU / HYBRID).")
endif()

# BitNet/ternary is CPU-only for now: the GPU backend will implement the dense FP path
# first, so a GPU build must not silently ignore ternary weights. Fail loud at configure
# time. TODO(ternary-gpu): implement absmean re-quant + straight-through estimator on the
# device (mirror ternarize_into() in src/backend_cpu.cpp), then lift this guard.
if(SUB0_TERNARY AND NOT SUB0_COMPUTE_RESOLVED STREQUAL "CPU")
  message(FATAL_ERROR
    "SUB0_TERNARY=ON is CPU-only for now: the GPU backend implements the dense FP path "
    "only (TODO(ternary-gpu)). Reconfigure with -DSUB0_COMPUTE=CPU, or build a GPU "
    "target with -DSUB0_TERNARY=OFF.")
endif()

message(STATUS "compute backend: SUB0_COMPUTE=${SUB0_COMPUTE} -> ${SUB0_COMPUTE_RESOLVED} "
               "(engine: ${SUB0_BACKEND_SOURCES})")
if(SUB0_COMPUTE_RESOLVED STREQUAL "GPU")
  message(STATUS "  GPU mode: training runs on the device (CUDA); generation/eval on the CPU engine")
endif()

# CMake decides whether the CUDA backend EXISTS / is built; the configurator decides whether to USE it
# (COMPUTE_MODE). Pass the build fact as a 0/1 int for --has-cuda; the configurator defaults the mode to
# GPU when the backend is built, else CPU. (SUB0_COMPUTE stays the build-side switch via SUB0_BUILD_CUDA.)
if(SUB0_BUILD_CUDA)
  set(SUB0_BUILD_CUDA_INT 1)
else()
  set(SUB0_BUILD_CUDA_INT 0)
endif()
