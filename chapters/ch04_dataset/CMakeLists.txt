add_executable(ch04_dataset main.cpp)

target_link_libraries(ch04_dataset PRIVATE sub0llm)
sub0llm_apply_compile_options(ch04_dataset)

set_target_properties(ch04_dataset PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
