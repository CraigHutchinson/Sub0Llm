# cmake/SuperBuild.cmake — optional multi-compiler super-build (recursive self-call).
#
# WHY: the C++ engine builds with clang, but a CUDA backend must be built with nvcc --
# clang only partially supports CUDA <= 12.9 and this project targets CUDA 13 (sm_120).
# nvcc and clang cannot share one global flag set: -march=native / -fopenmp=libomp /
# -fno-math-errno are clang-only and would break nvcc's host compile. The super-build
# isolates each compiler in its own CMake sub-build and then collects every artifact
# into ONE bin dir, so the driver discovers all backend DLLs side by side. It also lets
# each future backend (iGPU / NPU / ...) bring its own optimal toolchain without
# contaminating the others.
#
# DEFAULT OFF: a CPU-only build needs only clang, so the plain single-tree build is used
# and artifacts land in ${CMAKE_BINARY_DIR} exactly as before (the driver finds them
# next to itself). Turn ON only when a component needs a different compiler than the host.
#
# MECHANISM: when SUB0_SUPERBUILD=ON the top-level project builds NO targets itself; it
# ExternalProject_Add's THIS SAME source tree once per component with -DSUB0_SUPERBUILD=OFF
# and pins each child's RUNTIME/LIBRARY_OUTPUT_DIRECTORY to a shared bin dir. The
# top-level CMakeLists returns immediately after including this module when it is ON.

option(SUB0_SUPERBUILD "Multi-compiler super-build: isolate each backend's toolchain in its own sub-build" OFF)

if(NOT SUB0_SUPERBUILD)
  return()  # plain single-tree build -- the top-level continues to define the real targets
endif()

include(ExternalProject)

# Shared output dir every child build emits into, so the driver and every backend DLL
# end up co-located (Windows loader / $ORIGIN rpath discovery both search this dir).
set(SUB0_SUPERBUILD_BIN "${CMAKE_BINARY_DIR}/bin")
file(MAKE_DIRECTORY "${SUB0_SUPERBUILD_BIN}")

# Cache vars forwarded into every child (the user-facing config surface + the shared
# output dir). Backend selection is forwarded so a child knows which path to build.
set(_sub0_forward
  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  -DSUB0_NATIVE=${SUB0_NATIVE}
  -DSUB0_EXACT_MATH=${SUB0_EXACT_MATH}
  -DSUB0_CORPUS=${SUB0_CORPUS}
  -DSUB0_D_MODEL=${SUB0_D_MODEL}
  -DSUB0_N_LAYERS=${SUB0_N_LAYERS}
  -DSUB0_N_HEADS=${SUB0_N_HEADS}
  -DSUB0_SEQ_LEN=${SUB0_SEQ_LEN}
  -DSUB0_TERNARY=${SUB0_TERNARY}
  -DSUB0_CORPUS_TOK=${SUB0_CORPUS_TOK}
  -DSUB0_COMPUTE=${SUB0_COMPUTE}
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=${SUB0_SUPERBUILD_BIN}
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${SUB0_SUPERBUILD_BIN})

# Component: host (clang) -- the engine core, the CPU backend, the stage libraries, the
# driver and the tests. This is the entire product today. The CUDA component slots in
# next to it later as a second ExternalProject_Add that drives nvcc (a -DCMAKE_CUDA_*
# toolchain) and emits sub0_backend_cuda.dll into the same shared bin dir; it is built
# BEFORE the host so the host links/copies it.
ExternalProject_Add(sub0_host
  SOURCE_DIR   "${CMAKE_SOURCE_DIR}"
  CMAKE_GENERATOR "${CMAKE_GENERATOR}"
  CMAKE_ARGS   ${_sub0_forward}
               -DSUB0_SUPERBUILD=OFF
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  BUILD_ALWAYS ON
  INSTALL_COMMAND "")

# TODO(phase2): add the CUDA component, built before sub0_host:
#   ExternalProject_Add(sub0_cuda SOURCE_DIR "${CMAKE_SOURCE_DIR}/cuda"
#     CMAKE_ARGS ${_sub0_forward} -DSUB0_SUPERBUILD=OFF
#                -DCMAKE_CUDA_COMPILER=<nvcc> -DCMAKE_CUDA_ARCHITECTURES=120 ...)
#   ExternalProject_Add_StepDependencies(sub0_host configure sub0_cuda)

message(STATUS "super-build: orchestrating component sub-builds into ${SUB0_SUPERBUILD_BIN}")
