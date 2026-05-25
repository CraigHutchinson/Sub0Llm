add_executable(tests
    test_tensor.cpp
    test_ops.cpp
    test_backends.cpp
    test_tokenizer.cpp
    test_dataset.cpp
    test_autograd.cpp
    test_embeddings.cpp
)

target_link_libraries(tests
    PRIVATE
        sub0llm
        Catch2::Catch2WithMain
)

sub0llm_apply_compile_options(tests)

set_target_properties(tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

include(CTest)
catch_discover_tests(tests)
