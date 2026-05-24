include(CMakeDependentOption)

# ── Feature toggles ──────────────────────────────────────────────────────────
option(SUB0LLM_BUILD_CHAPTERS "Build chapter example programs" ON)
option(SUB0LLM_BUILD_TESTS    "Build test suite"               ON)

# ── Backend options ──────────────────────────────────────────────────────────
option(SUB0LLM_ENABLE_CUDA      "Enable CUDA backend (requires NVIDIA GPU + toolkit)" OFF)
option(SUB0LLM_ENABLE_OPENVINO  "Enable OpenVINO backend (requires Intel OpenVINO)"   OFF)

# ── SIMD options ─────────────────────────────────────────────────────────────
option(SUB0LLM_ENABLE_AVX2    "Enable AVX2 SIMD optimisations"   ON)
option(SUB0LLM_ENABLE_AVX512  "Enable AVX-512 SIMD optimisations" OFF)

# ── Dev options ───────────────────────────────────────────────────────────────
option(SUB0LLM_ENABLE_LTO       "Enable Link-Time Optimisation"          OFF)
option(SUB0LLM_ENABLE_SANITIZERS "Enable address/UB sanitizers (Debug)"  OFF)
option(SUB0LLM_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

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
