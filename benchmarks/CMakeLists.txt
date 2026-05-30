add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE sub0llm)
sub0llm_apply_compile_options(bench_kernels)
set_target_properties(bench_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
