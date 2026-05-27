add_executable(ch21_math_neurons main.cpp)
target_link_libraries(ch21_math_neurons PRIVATE sub0llm)
sub0llm_apply_compile_options(ch21_math_neurons)
set_target_properties(ch21_math_neurons PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
