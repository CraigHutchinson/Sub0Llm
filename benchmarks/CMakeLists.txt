# benchmarks/CMakeLists.txt — microbenchmarks via Catch2's benchmark support (opt-in, SUB0_BUILD_BENCHMARKS).
#
# Catch2 is already the project's test framework and builds cleanly on this clang-on-Windows toolchain
# (Google Benchmark's cmake injects MSVC /W4 /MP into clang++ and mis-probes std::regex). Scope: HOST /
# CPU hot paths -- device (CUDA) throughput uses cudaEvent timing in sub0-cuda-selftest.
#
#   frontend_bench.cpp — the engine-free frontend the configurator runs over the WHOLE corpus at ingest.
#   moe_expert_bench.cpp — one routed MoE expert's decode cost, split into disk / dequant / transpose /
#                          FFN. Plain main(), not Catch2: its first arm measures a page's FIRST touch,
#                          which a benchmark harness' warm-up destroys. See that file's own header.

CPMAddPackage("gh:catchorg/Catch2@3.7.1")   # no-op if the tests already fetched it (CPM dedups)

add_executable(sub0_frontend_bench frontend_bench.cpp)
target_include_directories(sub0_frontend_bench PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0_frontend_bench PRIVATE sub0_frontend Catch2::Catch2WithMain)

# Header-only too (gguf.hpp + transplant.hpp + moe_quant.hpp + moe_math.hpp), and deliberately linked
# against NOTHING: no sub0_core, no sub0_frontend, no generated sub0_config.hpp. Every dimension comes
# from the real sidecar's own header at runtime, which is what lets this measure a real 48-layer artifact
# from a build configured for anything at all. Release already supplies -O3 -g (see the top-level
# CMAKE_CXX_FLAGS_RELEASE comment on why -g stays in Release), so a profiler resolves its symbols; add
# -march=native only through the existing project-wide SUB0_NATIVE option, never a per-target one, so its
# numbers always describe the same ISA baseline the engine in the same build tree was compiled for.
add_executable(sub0_moe_expert_bench moe_expert_bench.cpp)
target_include_directories(sub0_moe_expert_bench PRIVATE "${CMAKE_SOURCE_DIR}/include")
if(SUB0_HAS_OPENMP_SIMD_FLAG)
  target_compile_options(sub0_moe_expert_bench PRIVATE -fopenmp-simd)
endif()

# Header-only numerical core; recognize its SIMD pragmas just as the CPU engine does.
add_executable(sub0_muon_bench muon_bench.cpp)
target_include_directories(sub0_muon_bench PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0_muon_bench PRIVATE Catch2::Catch2WithMain)
if(SUB0_HAS_OPENMP_SIMD_FLAG)
  target_compile_options(sub0_muon_bench PRIVATE -fopenmp-simd)
endif()
