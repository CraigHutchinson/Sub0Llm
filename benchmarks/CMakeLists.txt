# benchmarks/CMakeLists.txt — microbenchmarks via Catch2's benchmark support (opt-in, SUB0_BUILD_BENCHMARKS).
#
# Catch2 is already the project's test framework and builds cleanly on this clang-on-Windows toolchain
# (Google Benchmark's cmake injects MSVC /W4 /MP into clang++ and mis-probes std::regex). Scope: HOST /
# CPU hot paths -- device (CUDA) throughput uses cudaEvent timing in sub0-cuda-selftest.
#
#   frontend_bench.cpp — the engine-free frontend the configurator runs over the WHOLE corpus at ingest.

CPMAddPackage("gh:catchorg/Catch2@3.7.1")   # no-op if the tests already fetched it (CPM dedups)

add_executable(sub0_frontend_bench frontend_bench.cpp)
target_include_directories(sub0_frontend_bench PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0_frontend_bench PRIVATE sub0_frontend Catch2::Catch2WithMain)
