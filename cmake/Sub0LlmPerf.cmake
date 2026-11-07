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
