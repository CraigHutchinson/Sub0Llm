add_executable(ch16_thinking main.cpp)
target_link_libraries(ch16_thinking PRIVATE sub0llm)
sub0llm_apply_compile_options(ch16_thinking)
set_target_properties(ch16_thinking PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
