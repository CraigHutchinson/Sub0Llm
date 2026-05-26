add_executable(ch15_distillation main.cpp)
target_link_libraries(ch15_distillation PRIVATE sub0llm)
sub0llm_apply_compile_options(ch15_distillation)
set_target_properties(ch15_distillation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
