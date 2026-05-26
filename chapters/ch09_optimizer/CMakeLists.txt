add_executable(ch09_optimizer main.cpp)

target_link_libraries(ch09_optimizer PRIVATE sub0llm)
sub0llm_apply_compile_options(ch09_optimizer)

set_target_properties(ch09_optimizer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
