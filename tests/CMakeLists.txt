# tests/CMakeLists.txt — unit test suite driven by Catch2 v3 (fetched via CPM).

CPMAddPackage("gh:catchorg/Catch2@3.7.1")

# Catch2's CMake helpers (catch_discover_tests) live in its extras/ dir.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

add_executable(sub0_tests
  tokenizer_tests.cpp
  engine_tests.cpp
  tune_tests.cpp
  coherence_tests.cpp
)

# Tests compile against the engine API and the generated config header.
add_dependencies(sub0_tests sub0_generate_config)
target_include_directories(sub0_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${GEN_DIR}")
target_link_libraries(sub0_tests PRIVATE sub0_core Catch2::Catch2WithMain)

# Find the sub0_core shared lib next to the test binary at runtime.
set_target_properties(sub0_tests PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

# On Windows the loader has no rpath; DL_PATHS adds sub0_core's directory to the
# search path used during both test discovery and execution.
catch_discover_tests(sub0_tests
  DL_PATHS "$<TARGET_FILE_DIR:sub0_core>")
