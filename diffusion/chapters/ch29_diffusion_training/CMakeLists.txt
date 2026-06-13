add_executable(ch29_diffusion_training main.cpp)

target_link_libraries(ch29_diffusion_training PRIVATE sub0diff)

sub0llm_apply_compile_options(ch29_diffusion_training)

set_target_properties(ch29_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Same source, second target name: lets a rebuilt binary be produced (and tested)
# while a long training run holds a file lock on ch29_diffusion_training.exe.
add_executable(ch29b_diffusion_training main.cpp)
target_link_libraries(ch29b_diffusion_training PRIVATE sub0diff)
sub0llm_apply_compile_options(ch29b_diffusion_training)
set_target_properties(ch29b_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
