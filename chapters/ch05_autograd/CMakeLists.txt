add_executable(ch05_autograd main.cpp)

target_link_libraries(ch05_autograd PRIVATE sub0llm)
sub0llm_apply_compile_options(ch05_autograd)

set_target_properties(ch05_autograd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
