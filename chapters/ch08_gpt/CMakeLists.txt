add_executable(ch08_gpt main.cpp)

target_link_libraries(ch08_gpt PRIVATE sub0llm)
sub0llm_apply_compile_options(ch08_gpt)

set_target_properties(ch08_gpt PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
