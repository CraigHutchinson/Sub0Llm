include(CMakeDependentOption)

# ── Feature toggles ──────────────────────────────────────────────────────────
option(SUB0LLM_BUILD_CHAPTERS "Build chapter example programs" ON)
option(SUB0LLM_BUILD_TESTS    "Build test suite"               ON)
option(SUB0LLM_BUILD_TOOLS    "Build CLI and server tools"     ON)

# ── Backend options ──────────────────────────────────────────────────────────
option(SUB0LLM_ENABLE_CUDA      "Enable CUDA backend (requires NVIDIA GPU + toolkit)" OFF)
option(SUB0LLM_ENABLE_OPENVINO  "Enable OpenVINO backend (requires Intel OpenVINO)"   OFF)

# ── Accelerator / matmul back-ends ───────────────────────────────────────────
# Priority: BLAS (system) > Eigen (CPM) > AVX2 intrinsics > scalar blocked.
# BLAS is tried first via find_package; Eigen is fetched via CPM as a portable
# C++-only alternative with no Fortran dependency.  Both are gated on minimum
# matrix size (K >= 64) so small-D experiments still use the fast AVX2 path.
option(SUB0LLM_ENABLE_EIGEN     "Fetch Eigen3 via CPM for large-matrix matmul" ON)

# ── SIMD / architecture options ──────────────────────────────────────────────
# SUB0LLM_ENABLE_NATIVE: use -march=native -mtune=native; subsumes AVX2/AVX-512
# options and auto-detects all ISA extensions for compile-time dispatch macros.
# Recommended for release builds on the build host; not for distributable bins.
option(SUB0LLM_ENABLE_NATIVE  "Use -march=native/-mtune=native (host-optimised release)" OFF)
option(SUB0LLM_ENABLE_AVX2    "Enable AVX2 SIMD optimisations"                          ON)
option(SUB0LLM_ENABLE_AVX512  "Enable AVX-512 SIMD optimisations"                       OFF)

# ── Dev options ───────────────────────────────────────────────────────────────
# LTO: default ON for Release/RelWithDebInfo, OFF otherwise
if(CMAKE_BUILD_TYPE MATCHES "^(Release|RelWithDebInfo)$")
    option(SUB0LLM_ENABLE_LTO "Enable Link-Time Optimisation" ON)
else()
    option(SUB0LLM_ENABLE_LTO "Enable Link-Time Optimisation" OFF)
endif()
option(SUB0LLM_ENABLE_SANITIZERS         "Enable address/UB sanitizers (Debug)"    OFF)
option(SUB0LLM_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"        OFF)

# Wire up BUILD_TESTING
if(SUB0LLM_BUILD_TESTS)
    set(BUILD_TESTING ON CACHE BOOL "" FORCE)
else()
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
endif()

# ── Vectorisation report ─────────────────────────────────────────────────────
# Emits GCC/Clang vectorisation diagnostics during compilation.
# Useful for verifying that SIMD kernels and hot loops are auto-vectorised.
# Do NOT enable in CI — the diagnostic output is noisy.
option(SUB0LLM_ENABLE_VEC_REPORT "Emit GCC/Clang vectorisation report (-fopt-info-vec)" OFF)

# Validate CUDA + SIMD aren't both off in a way that breaks things
if(SUB0LLM_ENABLE_CUDA)
    enable_language(CUDA)
endif()
