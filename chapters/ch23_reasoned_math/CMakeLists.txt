add_executable(ch23_reasoned_math main.cpp)
target_link_libraries(ch23_reasoned_math PRIVATE sub0llm)
sub0llm_apply_compile_options(ch23_reasoned_math)
set_target_properties(ch23_reasoned_math PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
