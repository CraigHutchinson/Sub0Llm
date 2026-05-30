add_executable(ch24_real_training main.cpp)
target_link_libraries(ch24_real_training PRIVATE sub0llm
    $<$<PLATFORM_ID:Windows>:ws2_32>)
sub0llm_apply_compile_options(ch24_real_training)
set_target_properties(ch24_real_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
