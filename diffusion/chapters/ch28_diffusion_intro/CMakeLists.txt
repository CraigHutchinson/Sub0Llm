add_executable(ch28_diffusion_intro main.cpp)

target_link_libraries(ch28_diffusion_intro PRIVATE sub0diff)

sub0llm_apply_compile_options(ch28_diffusion_intro)

set_target_properties(ch28_diffusion_intro PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
