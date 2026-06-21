# ── diffusion/ — the text-diffusion (DiffusionGemma) volume (Ch28+) ─────────────
#
# A separate specialized track that links libsub0llm and reuses its transformer
# blocks, autograd, tokenizer, and the Ch27 monomorphization backbone. The diffusion-
# specific contracts (compile-time model + host specs) and the small amount of
# platform code (host detection) live here in libsub0diff.

# ── libsub0diff ──────────────────────────────────────────────────────────────────
add_library(sub0diff STATIC
    src/host_detect.cpp
    src/denoiser.cpp
    src/recovery.cpp
    src/oov_cliff.cpp
    src/sampler.cpp
    src/model_io.cpp
    src/run_config.cpp
)
target_include_directories(sub0diff PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(sub0diff PUBLIC sub0llm)
sub0llm_apply_compile_options(sub0diff)
set_target_properties(sub0diff PROPERTIES LINKER_LANGUAGE CXX)

# ── sub0diff-specialize (host axis today; model axis in Ch32) ────────────────────
add_executable(sub0diff-specialize tools/specialize/main.cpp)
target_link_libraries(sub0diff-specialize PRIVATE sub0diff)
sub0llm_apply_compile_options(sub0diff-specialize)
set_target_properties(sub0diff-specialize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── recall_probe — structured single-forward recall capability probe ─────────────
add_executable(sub0diff-recall-probe tools/recall_probe/main.cpp)
target_link_libraries(sub0diff-recall-probe PRIVATE sub0diff)
sub0llm_apply_compile_options(sub0diff-recall-probe)
set_target_properties(sub0diff-recall-probe PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── chapters ─────────────────────────────────────────────────────────────────────
if(SUB0LLM_BUILD_CHAPTERS)
    add_subdirectory(chapters)
endif()

# ── tests ────────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
