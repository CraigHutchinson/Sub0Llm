add_executable(ch24_real_training main.cpp)
target_link_libraries(ch24_real_training PRIVATE sub0llm Boost::beast Boost::system)
sub0llm_apply_compile_options(ch24_real_training)
set_target_properties(ch24_real_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
