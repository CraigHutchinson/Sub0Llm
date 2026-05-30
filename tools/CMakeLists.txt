# ── sub0llm-cli ───────────────────────────────────────────────────────────────
# Interactive inference CLI: load a trained ModernGPT checkpoint and generate
# text from a prompt (single-shot or interactive readline loop).
add_executable(sub0llm-cli cli/main.cpp)
target_link_libraries(sub0llm-cli PRIVATE sub0llm)
sub0llm_apply_compile_options(sub0llm-cli)
set_target_properties(sub0llm-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── sub0llm-server ────────────────────────────────────────────────────────────
# OpenAI-compatible HTTP inference server (POST /v1/completions,
# POST /v1/chat/completions) backed by a trained ModernGPT checkpoint.
add_executable(sub0llm-server server/main.cpp)
target_link_libraries(sub0llm-server PRIVATE sub0llm httplib)
sub0llm_apply_compile_options(sub0llm-server)
set_target_properties(sub0llm-server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
