add_executable(tests
    test_tensor.cpp
    test_ops.cpp
    test_backends.cpp
    test_quant.cpp
    test_sentencepiece.cpp
    test_tokenizer.cpp
    test_dataset.cpp
    test_autograd.cpp
    test_embeddings.cpp
    test_attention.cpp
    test_gpt.cpp
    test_optimizer.cpp
    test_modern_gpt.cpp
    test_pretraining.cpp
    test_lora.cpp
    test_dpo.cpp
    test_sampler.cpp
    test_distillation.cpp
    test_thinking.cpp
    test_looped_gpt.cpp
    test_moe.cpp
    test_mtp.cpp
    test_rlhf.cpp
    test_math_nodes.cpp
    test_numeric_router.cpp
    test_reasoning.cpp
    test_checkpoint.cpp
    test_text_corpus.cpp
    test_ch25_long_context.cpp
    test_ch26_episodic_memory.cpp
    test_workspace.cpp
    test_cpu_topology.cpp
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
