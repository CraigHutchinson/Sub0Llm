add_executable(ch22_math_lm main.cpp)
target_link_libraries(ch22_math_lm PRIVATE sub0llm)
sub0llm_apply_compile_options(ch22_math_lm)
set_target_properties(ch22_math_lm PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
