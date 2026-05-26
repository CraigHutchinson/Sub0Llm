add_executable(ch13_alignment main.cpp)
target_link_libraries(ch13_alignment PRIVATE sub0llm)
sub0llm_apply_compile_options(ch13_alignment)
set_target_properties(ch13_alignment PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
