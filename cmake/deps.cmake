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

# ── Optional: BLAS for CPU matrix ops ─────────────────────────────────────────
find_package(BLAS QUIET)
if(BLAS_FOUND)
    message(STATUS "sub0llm: BLAS found — using accelerated BLAS for CPU matmul")
else()
    message(STATUS "sub0llm: BLAS not found — using naive CPU matmul")
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
