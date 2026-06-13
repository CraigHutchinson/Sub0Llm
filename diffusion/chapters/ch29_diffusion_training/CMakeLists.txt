add_executable(ch29_diffusion_training main.cpp)

target_link_libraries(ch29_diffusion_training PRIVATE sub0diff)

sub0llm_apply_compile_options(ch29_diffusion_training)

set_target_properties(ch29_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
