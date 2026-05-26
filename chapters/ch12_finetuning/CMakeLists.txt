add_executable(ch12_finetuning main.cpp)
target_link_libraries(ch12_finetuning PRIVATE sub0llm)
sub0llm_apply_compile_options(ch12_finetuning)
set_target_properties(ch12_finetuning PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
