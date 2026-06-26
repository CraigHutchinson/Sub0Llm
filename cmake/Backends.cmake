# cmake/Backends.cmake — compute-backend selection (the CPU / GPU / HYBRID axis).
#
# sub0llm configures for ONE optimal compute target for the host and compiles only
# that path -- the project's compile-time-decision philosophy: no runtime backend
# switch for the pure cases, so every dead path is eliminated and every optimization
# pathway stays open. SUB0_COMPUTE picks the target:
#
#   CPU    - scalar + OpenMP backend (src/backend_cpu.cpp)   [default, always built]
#   GPU    - CUDA backend (src/backend_cuda.cu, nvcc)        [Phase 2 -- not yet]
#   HYBRID - CPU + CUDA, autotuned split                     [Phase 3 -- not yet]
#   AUTO   - resolve the best available at configure time    [Phase 1: -> CPU/GPU]
#
# Outputs (consumed by the top-level build):
#   SUB0_COMPUTE_RESOLVED - the concrete mode after AUTO resolution
#   SUB0_BACKEND_SOURCES  - the compute-backend translation unit(s) for sub0_core
#
# The CPU backend is ALWAYS compiled: it is the baseline and the numerical parity
# reference the GPU backend is validated against.

set(SUB0_COMPUTE "CPU" CACHE STRING "Compute backend: AUTO / CPU / GPU / HYBRID")
set_property(CACHE SUB0_COMPUTE PROPERTY STRINGS AUTO CPU GPU HYBRID)

# CUDA detection (facts only -- the GPU backend compiles in Phase 2). find_package
# locates the toolkit WITHOUT enabling the CUDA language (that happens in Phase 2 when
# .cu files exist); nvidia-smi gives the first device's compute capability + VRAM. These
# are baked into the generated config (HAS_CUDA / CUDA_ARCH / GPU_VRAM_MB) so the engine
# knows the host's GPU at compile time.
set(SUB0_HAS_CUDA 0)
set(SUB0_CUDA_ARCH 0)
set(SUB0_GPU_VRAM_MB 0)
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
  message(STATUS "CUDA: toolkit ${CUDAToolkit_VERSION} found; device sm_${SUB0_CUDA_ARCH}, ${SUB0_GPU_VRAM_MB} MB VRAM")
else()
  message(STATUS "CUDA: no toolkit found -- CPU-only host")
endif()

# AUTO resolves to the best AVAILABLE backend. Until the GPU backend lands (Phase 2)
# there is nothing to resolve to but CPU, even when a CUDA device is present; HAS_CUDA
# is still baked so a Phase-2 build can flip AUTO -> GPU.
set(SUB0_COMPUTE_RESOLVED "${SUB0_COMPUTE}")
if(SUB0_COMPUTE STREQUAL "AUTO")
  set(SUB0_COMPUTE_RESOLVED "CPU")
endif()

set(SUB0_BACKEND_SOURCES src/backend_cpu.cpp)

if(SUB0_COMPUTE_RESOLVED STREQUAL "CPU")
  # baseline only
elseif(SUB0_COMPUTE_RESOLVED STREQUAL "GPU" OR SUB0_COMPUTE_RESOLVED STREQUAL "HYBRID")
  message(FATAL_ERROR
    "SUB0_COMPUTE=${SUB0_COMPUTE} selects a GPU backend, which is not implemented yet "
    "(Phase 2: src/backend_cuda.cu via nvcc). Build with -DSUB0_COMPUTE=CPU for now.")
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
               "(sources: ${SUB0_BACKEND_SOURCES})")

# Integer encoding of the resolved backend for the configurator (--compute): 0/1/2.
if(SUB0_COMPUTE_RESOLVED STREQUAL "GPU")
  set(SUB0_COMPUTE_MODE_INT 1)
elseif(SUB0_COMPUTE_RESOLVED STREQUAL "HYBRID")
  set(SUB0_COMPUTE_MODE_INT 2)
else()
  set(SUB0_COMPUTE_MODE_INT 0)
endif()
