add_executable(ch29_diffusion_training main.cpp)

target_link_libraries(ch29_diffusion_training PRIVATE sub0diff)

sub0llm_apply_compile_options(ch29_diffusion_training)

set_target_properties(ch29_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ch29b — isolated data-parallel throughput benchmark (separate exe so the main
# training binary can stay running/locked while we profile threading scaling).
add_executable(ch29b_diffusion_training bench_parallel.cpp)
target_link_libraries(ch29b_diffusion_training PRIVATE sub0diff)
sub0llm_apply_compile_options(ch29b_diffusion_training)
set_target_properties(ch29b_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
