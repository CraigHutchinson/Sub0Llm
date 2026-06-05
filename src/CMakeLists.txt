set(SUB0LLM_SOURCES
    core/tensor.cpp
    core/ops.cpp
    backends/cpu/kernels.cpp
    backends/cpu/matmul.cpp
    backends/cpu/quant.cpp
    backends/cuda/backend.cpp
    backends/openvino/backend.cpp
    tokenizer/bpe.cpp
    data/dataset.cpp
    data/dataloader.cpp
    autograd/variable.cpp
    autograd/ops.cpp
    autograd/embedding_ops.cpp
    nn/embedding.cpp
    nn/positional_encoding.cpp
    nn/attention.cpp
    nn/gpt.cpp
    nn/optimizer.cpp
    nn/modern_gpt.cpp
    nn/scheduler.cpp
    nn/trainer.cpp
    nn/lora.cpp
    nn/dpo.cpp
    nn/sampler.cpp
    nn/distillation.cpp
    nn/thinking.cpp
    nn/looped_gpt.cpp
    nn/moe.cpp
    nn/mtp.cpp
    nn/rlhf.cpp
    tokenizer/numeric_tokenizer.cpp
    nn/numeric_router.cpp
    nn/math_nodes.cpp
    data/text_corpus.cpp
    nn/checkpoint.cpp
    nn/long_context.cpp
    nn/episodic_memory.cpp
    nn/gguf_loader.cpp
)


# CUDA .cu files must be listed separately so CMake uses nvcc for them.
if(SUB0LLM_ENABLE_CUDA)
    list(APPEND SUB0LLM_SOURCES backends/cuda/kernels.cu)
    set_source_files_properties(backends/cuda/kernels.cu
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

add_library(sub0llm STATIC ${SUB0LLM_SOURCES})

target_include_directories(sub0llm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}   # so backends/… can include each other
)

target_link_libraries(sub0llm
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
    PRIVATE
        $<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>
        $<$<BOOL:${SUB0LLM_EIGEN_AVAILABLE}>:Eigen3::Eigen>
        $<$<BOOL:${SUB0LLM_ENABLE_CUDA}>:CUDA::cublas CUDA::curand CUDA::cudart>
        $<$<BOOL:${SUB0LLM_ENABLE_OPENVINO}>:openvino::runtime>
)

# Propagate BLAS/Eigen availability as compile definitions for dispatch in matmul.cpp
target_compile_definitions(sub0llm PRIVATE
    $<$<BOOL:${BLAS_FOUND}>:SUB0LLM_BLAS>
    $<$<BOOL:${SUB0LLM_EIGEN_AVAILABLE}>:SUB0LLM_EIGEN>
)

target_compile_features(sub0llm PUBLIC cxx_std_23)

sub0llm_apply_compile_options(sub0llm)

set_target_properties(sub0llm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"
)

# nvcc has no "CUDA23" dialect; the C++23 requirement on this mixed-language
# target would otherwise propagate to the .cu sources and fail configuration.
# Device code only needs C++20, so pin the CUDA dialect explicitly.
if(SUB0LLM_ENABLE_CUDA)
    set_target_properties(sub0llm PROPERTIES
        CUDA_STANDARD          20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES     native
    )
endif()
