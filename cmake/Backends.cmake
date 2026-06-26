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

# Phase 1 will probe here (check_language(CUDA) + find_package(CUDAToolkit) + a device
# query) and resolve AUTO -> GPU when a usable CUDA 13 toolkit and an sm_120 device are
# present, else CPU, baking HAS_CUDA / COMPUTE_MODE / CUDA_ARCH into the generated
# config. Until the GPU backend exists, AUTO resolves to CPU.
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
