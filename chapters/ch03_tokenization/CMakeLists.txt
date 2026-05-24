add_executable(ch03_tokenization main.cpp)
target_link_libraries(ch03_tokenization PRIVATE sub0llm)
sub0llm_apply_compile_options(ch03_tokenization)
set_target_properties(ch03_tokenization PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
