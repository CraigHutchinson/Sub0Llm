add_executable(ch02_backends main.cpp)

target_link_libraries(ch02_backends PRIVATE sub0llm)

sub0llm_apply_compile_options(ch02_backends)

set_target_properties(ch02_backends PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
