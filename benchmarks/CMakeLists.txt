add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE sub0llm)
sub0llm_apply_compile_options(bench_kernels)
set_target_properties(bench_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Memory-bandwidth ceiling probe (STREAM-style) — self-contained, no lib dependency.
# Establishes the "exercised maximum" DRAM bandwidth this machine can hit, bracketing
# the theoretical JEDEC peak above and the LLM real decode throughput below.
add_executable(bench_membw bench_membw.cpp)
sub0llm_apply_compile_options(bench_membw)
set_target_properties(bench_membw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
