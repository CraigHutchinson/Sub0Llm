add_executable(ch11_pretraining main.cpp)

target_link_libraries(ch11_pretraining PRIVATE sub0llm)
sub0llm_apply_compile_options(ch11_pretraining)

set_target_properties(ch11_pretraining PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
