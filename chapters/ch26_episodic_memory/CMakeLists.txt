add_executable(ch26_episodic_memory main.cpp)
target_link_libraries(ch26_episodic_memory PRIVATE sub0llm)
# Eigen3 is used in §5 for truncated-SVD LoRA compression demo
if(TARGET Eigen3::Eigen)
    target_link_libraries(ch26_episodic_memory PRIVATE Eigen3::Eigen)
    target_compile_definitions(ch26_episodic_memory PRIVATE SUB0LLM_HAVE_EIGEN)
endif()
sub0llm_apply_compile_options(ch26_episodic_memory)
set_target_properties(ch26_episodic_memory PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
