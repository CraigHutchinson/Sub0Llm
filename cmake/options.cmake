include(CMakeDependentOption)

# ── Feature toggles ──────────────────────────────────────────────────────────
option(SUB0LLM_BUILD_CHAPTERS "Build chapter example programs" ON)
option(SUB0LLM_BUILD_TESTS    "Build test suite"               ON)

# ── Backend options ──────────────────────────────────────────────────────────
option(SUB0LLM_ENABLE_CUDA      "Enable CUDA backend (requires NVIDIA GPU + toolkit)" OFF)
option(SUB0LLM_ENABLE_OPENVINO  "Enable OpenVINO backend (requires Intel OpenVINO)"   OFF)

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

# Validate CUDA + SIMD aren't both off in a way that breaks things
if(SUB0LLM_ENABLE_CUDA)
    enable_language(CUDA)
endif()
