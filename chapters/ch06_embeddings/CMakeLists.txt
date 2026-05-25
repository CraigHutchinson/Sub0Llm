add_executable(ch06_embeddings main.cpp)

target_link_libraries(ch06_embeddings PRIVATE sub0llm)
sub0llm_apply_compile_options(ch06_embeddings)

set_target_properties(ch06_embeddings PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
