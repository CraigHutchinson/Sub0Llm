# External dependencies via CPM
# https://github.com/cpm-cmake/CPM.cmake

# ── Logging ───────────────────────────────────────────────────────────────────
CPMAddPackage("gh:gabime/spdlog@1.15.1")

# ── JSON (tokenizer configs, model configs) ───────────────────────────────────
CPMAddPackage(
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.11.3
    OPTIONS "JSON_BuildTests OFF" "JSON_Install OFF"
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
