add_executable(ch19_mtp main.cpp)
target_link_libraries(ch19_mtp PRIVATE sub0llm)
sub0llm_apply_compile_options(ch19_mtp)
set_target_properties(ch19_mtp PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
