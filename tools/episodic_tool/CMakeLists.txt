add_executable(sub0llm-episodic main.cpp)
target_link_libraries(sub0llm-episodic PRIVATE sub0llm)
sub0llm_apply_compile_options(sub0llm-episodic)
set_target_properties(sub0llm-episodic PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
