# External dependencies via CPM
# https://github.com/cpm-cmake/CPM.cmake

# ── Logging ───────────────────────────────────────────────────────────────────
CPMAddPackage("gh:gabime/spdlog@1.15.1")

# spdlog's CMakeLists adds MSVC-only options (e.g. /Zc:__cplusplus) under if(MSVC).
# clang++ targeting the MSVC ABI makes CMake set MSVC=1, but its GNU-style command-line
# frontend treats a /-prefixed token as an input filename and errors ("no such file").
# This stays latent on cached trees but breaks a fresh configure (e.g. the CUDA preset).
# Strip MSVC /-flags from spdlog whenever we drive clang through its GNU frontend.
if(TARGET spdlog AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
        AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    foreach(_prop COMPILE_OPTIONS INTERFACE_COMPILE_OPTIONS)
        get_target_property(_opts spdlog ${_prop})
        if(_opts)
            list(FILTER _opts EXCLUDE REGEX "^/")
            set_target_properties(spdlog PROPERTIES ${_prop} "${_opts}")
        endif()
    endforeach()
endif()

# ── JSON (tokenizer configs, model configs) ───────────────────────────────────
CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.11.3
    OPTIONS "JSON_BuildTests OFF" "JSON_Install OFF"
)

# ── JSON parsing, fast path (config module) ───────────────────────────────────
# simdjson is the optimal *parser* (SIMD, on-demand, minimal allocation) — used by the
# config module to read run_config.json / model config.json. nlohmann stays for the
# checkpoint/tokenizer DOM writing it already does; simdjson does not emit JSON, so the
# config module's writer is a tiny dependency-free streamer (see config/schema.hpp).
# SIMDJSON_DEVELOPER_MODE OFF (consumer mode) builds only the static lib — no CLI tools,
# fuzzers, or benchmarks. (The older SIMDJSON_JUST_LIBRARY knob is deprecated in 3.10.)
CPMAddPackage(
    NAME simdjson
    GITHUB_REPOSITORY simdjson/simdjson
    VERSION 3.10.1
    OPTIONS "SIMDJSON_DEVELOPER_MODE OFF"
)

# ── Testing ───────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    CPMAddPackage("gh:catchorg/Catch2@3.7.1")
    if(Catch2_ADDED)
        list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
    endif()
    include(Catch)
endif()

# ── Optional: BLAS (system) + Eigen3 (CPM) for CPU matrix ops ────────────────
# Priority: BLAS > Eigen > AVX2 > scalar.
# BLAS (system): cblas_sgemm for any BLAS provider (OpenBLAS, MKL, Accelerate).
# Eigen (CPM):   header-only C++, no Fortran; used when system BLAS is absent.
# Both kick in only for K >= 64 so small-D training uses the faster AVX2 path.
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "sub0llm: BLAS found — cblas_sgemm dispatch enabled (K>=64)")
else()
    message(STATUS "sub0llm: BLAS not found — checking Eigen fallback")
endif()

if(SUB0LLM_ENABLE_EIGEN AND NOT BLAS_FOUND)
    CPMAddPackage(
        NAME Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        DOWNLOAD_ONLY  YES   # header-only; we add the include path manually
    )
    if(Eigen3_ADDED)
        add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
        target_include_directories(Eigen3::Eigen INTERFACE "${Eigen3_SOURCE_DIR}")
        set(SUB0LLM_EIGEN_AVAILABLE TRUE CACHE INTERNAL "")
        message(STATUS "sub0llm: Eigen3 3.4.0 fetched via CPM — Eigen matmul dispatch enabled (K>=64)")
    endif()
elseif(SUB0LLM_ENABLE_EIGEN AND BLAS_FOUND)
    message(STATUS "sub0llm: BLAS takes priority over Eigen; Eigen will not be fetched")
endif()

# ── Boost (Beast + ASIO for the Ch32 viz server) ──────────────────────────────
# Using the cmake-first tarball (fastest CPM download — no recursive git clone).
# Boost.Beast and Boost.ASIO are header-only; Boost.System is effectively
# header-only since 1.77.  BOOST_CONTEXT_IMPLEMENTATION winfib avoids the MASM
# assembler (ml64.exe) which is not in PATH in the CUDA preset environment.
CPMAddPackage(
    NAME Boost
    VERSION 1.91.0
    URL https://github.com/boostorg/boost/releases/download/boost-1.91.0-1/boost-1.91.0-1-cmake.tar.xz
    URL_HASH SHA256=cc5dc5006ecbdf0051f90979be31b4eee5987d9ae14ae9fb9c03cfa43fa3cdad
    OPTIONS
        "BOOST_ENABLE_CMAKE ON"
        "BOOST_INCLUDE_LIBRARIES beast"
        "BUILD_SHARED_LIBS OFF"
        "BOOST_CONTEXT_IMPLEMENTATION winfib"
)

# ── Suppress warnings in all CPM / third-party headers ───────────────────────
# Moving their include paths into INTERFACE_SYSTEM_INCLUDE_DIRECTORIES causes
# the compiler to treat them like system headers — our -Wsign-conversion,
# -Wconversion, etc. flags never fire on code we do not own.
foreach(_3p_target
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        simdjson::simdjson
        Boost::headers
        Boost::beast
        Boost::asio
        Boost::system)
    if(TARGET ${_3p_target})
        # ALIAS targets do not support set_target_properties — resolve to the
        # real target before setting INTERFACE_SYSTEM_INCLUDE_DIRECTORIES.
        get_target_property(_real ${_3p_target} ALIASED_TARGET)
        if(NOT _real)
            set(_real ${_3p_target})
        endif()
        get_target_property(_incs ${_real} INTERFACE_INCLUDE_DIRECTORIES)
        if(_incs)
            set_target_properties(${_real} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_incs}")
        endif()
    endif()
endforeach()
unset(_3p_target)
unset(_real)
unset(_incs)

# ── Optional: CUDA ────────────────────────────────────────────────────────────
if(SUB0LLM_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "sub0llm: CUDA toolkit ${CUDAToolkit_VERSION} found")
endif()

# ── Optional: OpenVINO ────────────────────────────────────────────────────────
if(SUB0LLM_ENABLE_OPENVINO)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
    message(STATUS "sub0llm: OpenVINO ${OpenVINO_VERSION} found")
endif()
