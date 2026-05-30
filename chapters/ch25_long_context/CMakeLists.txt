add_executable(ch25_long_context main.cpp)
target_link_libraries(ch25_long_context PRIVATE sub0llm)
sub0llm_apply_compile_options(ch25_long_context)
set_target_properties(ch25_long_context PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
