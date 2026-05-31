add_executable(ch26_episodic_memory main.cpp)
target_link_libraries(ch26_episodic_memory PRIVATE sub0llm)
sub0llm_apply_compile_options(ch26_episodic_memory)
set_target_properties(ch26_episodic_memory PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
