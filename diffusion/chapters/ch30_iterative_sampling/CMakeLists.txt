add_executable(ch30_iterative_sampling main.cpp)

target_link_libraries(ch30_iterative_sampling PRIVATE sub0diff)

sub0llm_apply_compile_options(ch30_iterative_sampling)

set_target_properties(ch30_iterative_sampling PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
