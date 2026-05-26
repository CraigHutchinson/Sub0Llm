add_executable(ch18_moe main.cpp)
target_link_libraries(ch18_moe PRIVATE sub0llm)
sub0llm_apply_compile_options(ch18_moe)
set_target_properties(ch18_moe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
