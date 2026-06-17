add_executable(bench_kernels bench_kernels.cpp)
target_link_libraries(bench_kernels PRIVATE sub0llm)
# Reach the private CUDA backend header (backends/cuda/backend.hpp) for the device-vs-CPU op
# bench — same as sub0llm-qbench. No-op on CPU builds (the include is SUB0LLM_CUDA-gated).
target_include_directories(bench_kernels PRIVATE ${CMAKE_SOURCE_DIR}/src)
sub0llm_apply_compile_options(bench_kernels)
set_target_properties(bench_kernels PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Memory-bandwidth ceiling probe + Q8 decode-ladder diagnostic. STREAM kernels plus the
# real backend::cpu int8 GEMV, to localize which rung (pure read / Q8 byte stream / int8
# kernel) falls off the exercised DRAM bandwidth ceiling. Links the lib for the real Q8 dot.
add_executable(bench_membw bench_membw.cpp)
target_link_libraries(bench_membw PRIVATE sub0llm)
sub0llm_apply_compile_options(bench_membw)
set_target_properties(bench_membw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
