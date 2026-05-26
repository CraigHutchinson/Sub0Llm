add_executable(ch14_inference main.cpp)

target_link_libraries(ch14_inference PRIVATE sub0llm)

sub0llm_apply_compile_options(ch14_inference)

set_target_properties(ch14_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
