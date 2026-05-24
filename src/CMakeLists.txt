add_library(sub0llm STATIC
    core/tensor.cpp
    core/ops.cpp
)

target_include_directories(sub0llm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(sub0llm
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
    PRIVATE
        $<$<BOOL:${BLAS_FOUND}>:${BLAS_LIBRARIES}>
        $<$<BOOL:${SUB0LLM_ENABLE_CUDA}>:CUDA::cublas CUDA::curand>
        $<$<BOOL:${SUB0LLM_ENABLE_OPENVINO}>:openvino::runtime>
)

target_compile_features(sub0llm PUBLIC cxx_std_23)

sub0llm_apply_compile_options(sub0llm)

set_target_properties(sub0llm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY  "${CMAKE_BINARY_DIR}/lib"
)
