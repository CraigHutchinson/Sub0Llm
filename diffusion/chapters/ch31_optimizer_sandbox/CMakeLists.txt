add_executable(ch31_optimizer_sandbox main.cpp)

target_link_libraries(ch31_optimizer_sandbox PRIVATE sub0diff)

sub0llm_apply_compile_options(ch31_optimizer_sandbox)

set_target_properties(ch31_optimizer_sandbox PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
