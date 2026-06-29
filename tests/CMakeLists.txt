# tests/CMakeLists.txt — unit test suite driven by Catch2 v3 (fetched via CPM).

CPMAddPackage("gh:catchorg/Catch2@3.7.1")

# Catch2's CMake helpers (catch_discover_tests) live in its extras/ dir.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

add_executable(sub0_tests
  tokenizer_tests.cpp
  engine_tests.cpp
  layout_tests.cpp
  tune_tests.cpp
  coherence_tests.cpp
)

# Tests compile against the engine API and the generated config header.
add_dependencies(sub0_tests sub0_generate_config)
target_include_directories(sub0_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${GEN_DIR}")
target_link_libraries(sub0_tests PRIVATE sub0_core sub0_tok Catch2::Catch2WithMain)

# Pure tokenizer-library tests: link ONLY sub0_tok + Catch2, NOT the engine. The tokenizer
# is config-independent, so this target builds and runs without sub0_core / the generated
# header -- fast to iterate, and unaffected when a training run locks the engine DLLs.
add_executable(sub0_tok_tests tok_lib_tests.cpp)
target_include_directories(sub0_tok_tests PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0_tok_tests PRIVATE sub0_tok Catch2::Catch2WithMain)
catch_discover_tests(sub0_tok_tests)

# GPU backend smoke test: compiled + linked only when the CUDA backend is built. The
# nvcc-built sub0_backend_cuda.dll sits next to sub0_core.dll; its CUDA runtime deps are
# covered by the toolkit bin added to DL_PATHS below.
if(SUB0_BUILD_CUDA)
  target_sources(sub0_tests PRIVATE cuda_tests.cpp)
  target_link_libraries(sub0_tests PRIVATE sub0_backend_cuda)
endif()

# Find the sub0_core shared lib next to the test binary at runtime.
set_target_properties(sub0_tests PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

# On Windows the loader has no rpath; DL_PATHS adds the needed DLL directories to the search
# path used during test discovery and execution: sub0_core.dll (and sub0_backend_cuda.dll) sit
# next to the test binary, and when the CUDA backend is built sub0_backend_cuda.dll additionally
# needs the CUDA runtime DLLs (cudart/cublas) from the toolkit bin.
#
# DISCOVERY_MODE PRE_TEST: enumerate the test cases at `ctest` time, NOT during the build. Running
# the CUDA-linked test exe at build time made every build depend on the CUDA runtime DLLs being
# loadable in the build shell's environment (a missing/unfindable cublas surfaced as a cryptic
# 0xc000007b and FAILED the build). Deferring to test time -- where DL_PATHS is applied -- keeps
# the build robust on any host.
set(_sub0_test_dl_paths "$<TARGET_FILE_DIR:sub0_core>")
if(SUB0_BUILD_CUDA AND CUDAToolkit_BIN_DIR)
  list(APPEND _sub0_test_dl_paths "${CUDAToolkit_BIN_DIR}")
endif()
catch_discover_tests(sub0_tests
  DL_PATHS ${_sub0_test_dl_paths}
  DISCOVERY_MODE PRE_TEST)
