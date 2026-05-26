add_executable(ch20_rlhf main.cpp)
target_link_libraries(ch20_rlhf PRIVATE sub0llm)
sub0llm_apply_compile_options(ch20_rlhf)
set_target_properties(ch20_rlhf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
