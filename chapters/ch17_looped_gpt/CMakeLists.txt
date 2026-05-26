add_executable(ch17_looped_gpt main.cpp)
target_link_libraries(ch17_looped_gpt PRIVATE sub0llm)
sub0llm_apply_compile_options(ch17_looped_gpt)
set_target_properties(ch17_looped_gpt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
