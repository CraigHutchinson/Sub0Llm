include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/Sub0MathTooling.cmake")
sub0math_add_benchmark(sub0llm-bench-denseq "${PROJECT_SOURCE_DIR}/benchmarks/dense_quant_perf.cpp")
target_include_directories(sub0llm-bench-denseq PRIVATE "${PROJECT_SOURCE_DIR}/include")
# Vendor warnings are isolated by upstream SYSTEM include paths.
target_compile_options(sub0llm-bench-denseq PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>)
add_test(NAME sub0llm-perf-fixture COMMAND sub0llm-bench-denseq --list)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME sub0llm-perf-tooling COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tools/perf)
set_tests_properties(sub0llm-perf-tooling PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
add_test(NAME sub0llm-perf-provenance COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/check_sub0perf_vendor.py")

add_test(NAME sub0llm-perf-workload COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tools/perf_extensions/tests)
set_tests_properties(sub0llm-perf-workload PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_test(NAME sub0llm-reference-gates COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s tools/reference/tests)
set_tests_properties(sub0llm-reference-gates PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_test(NAME sub0llm-chat-state COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s scripts/tests)
set_tests_properties(sub0llm-chat-state PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_executable(sub0llm-phase-fixture "${PROJECT_SOURCE_DIR}/tools/reference/tests/phase_fixture.cpp")
add_test(NAME sub0llm-native-phase-contract COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tools/reference/tests/check_phase_fixture.py" $<TARGET_FILE:sub0llm-phase-fixture>)
add_executable(sub0llm-quality-fixture "${PROJECT_SOURCE_DIR}/tools/reference/tests/quality_fixture.cpp")
add_test(NAME sub0llm-native-quality-contract COMMAND "${Python3_EXECUTABLE}"
  "${PROJECT_SOURCE_DIR}/tools/reference/tests/check_quality_fixture.py" $<TARGET_FILE:sub0llm-quality-fixture>)

add_executable(sub0llm-batch-projection-fixture "${PROJECT_SOURCE_DIR}/tools/reference/tests/batch_projection_fixture.cpp")
target_include_directories(sub0llm-batch-projection-fixture PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME sub0llm-batch-projection-contract COMMAND sub0llm-batch-projection-fixture)

add_executable(sub0llm-q8k-projection-fixture "${PROJECT_SOURCE_DIR}/tools/reference/tests/q8k_projection_fixture.cpp")
target_include_directories(sub0llm-q8k-projection-fixture PRIVATE "${PROJECT_SOURCE_DIR}/include")
add_test(NAME sub0llm-q8k-projection-contract COMMAND sub0llm-q8k-projection-fixture)
find_package(Threads REQUIRED)
target_link_libraries(sub0llm-q8k-projection-fixture PRIVATE Threads::Threads)

add_executable(sub0llm-q16-projection-fixture "${PROJECT_SOURCE_DIR}/tools/reference/tests/q16_projection_fixture.cpp")
target_include_directories(sub0llm-q16-projection-fixture PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(sub0llm-q16-projection-fixture PRIVATE Threads::Threads)
add_test(NAME sub0llm-q16-projection-contract COMMAND sub0llm-q16-projection-fixture)

# Separate experiment suite keeps established default-kernel baseline names stable.
sub0math_add_benchmark(sub0llm-bench-q16-tile "${PROJECT_SOURCE_DIR}/benchmarks/q16_tile_perf.cpp")
target_include_directories(sub0llm-bench-q16-tile PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_options(sub0llm-bench-q16-tile PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>)
add_executable(sub0llm-q16-tile-fixture "${PROJECT_SOURCE_DIR}/tools/reference/tests/q16_tile_fixture.cpp")
target_include_directories(sub0llm-q16-tile-fixture PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_options(sub0llm-q16-tile-fixture PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Werror>)
target_link_libraries(sub0llm-q16-tile-fixture PRIVATE Threads::Threads)
add_test(NAME sub0llm-q16-tile-contract COMMAND sub0llm-q16-tile-fixture)
