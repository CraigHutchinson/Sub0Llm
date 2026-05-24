add_executable(ch01_foundations main.cpp)

target_link_libraries(ch01_foundations PRIVATE sub0llm)

sub0llm_apply_compile_options(ch01_foundations)

set_target_properties(ch01_foundations PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
