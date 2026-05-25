add_executable(ch07_attention main.cpp)

target_link_libraries(ch07_attention PRIVATE sub0llm)
sub0llm_apply_compile_options(ch07_attention)

set_target_properties(ch07_attention PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
