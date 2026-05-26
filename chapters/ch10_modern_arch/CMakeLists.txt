add_executable(ch10_modern_arch main.cpp)

target_link_libraries(ch10_modern_arch PRIVATE sub0llm)
sub0llm_apply_compile_options(ch10_modern_arch)

set_target_properties(ch10_modern_arch PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
