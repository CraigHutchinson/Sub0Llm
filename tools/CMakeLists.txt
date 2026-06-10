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

# ── sub0llm-episodic ──────────────────────────────────────────────────────────
# Episodic memory CLI: write/recall/info subcommands for GGUF models.
add_subdirectory(episodic)

# ── sub0llm-specialize (Ch27) ─────────────────────────────────────────────────
# Compile-time specialization front-end: read a GGUF's metadata and emit a
# constexpr <model>_spec.hpp + <model>_arch.json so a dedicated binary can be
# monomorphized to one model's exact shapes.
add_executable(sub0llm-specialize specialize/main.cpp)
target_link_libraries(sub0llm-specialize PRIVATE sub0llm)
sub0llm_apply_compile_options(sub0llm-specialize)
set_target_properties(sub0llm-specialize PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── sub0llm-verify (Ch27) ─────────────────────────────────────────────────────
# Correctness harness: tokenize / greedy / logits / nll for a GGUF model, to diff
# against llama.cpp (the correctness oracle) before trusting any optimization.
add_executable(sub0llm-verify verify/main.cpp)
target_link_libraries(sub0llm-verify PRIVATE sub0llm)
sub0llm_apply_compile_options(sub0llm-verify)
set_target_properties(sub0llm-verify PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── sub0llm-gemma (Ch27) ──────────────────────────────────────────────────────
# Faithful Gemma 4 text inference: the dedicated per-layer-heterogeneous GemmaModel
# (local/global attention, dual RoPE, GeGLU, logit soft-cap) loaded Q8-on-load,
# with tokenize / logits / greedy modes to gate parity against llama.cpp.
add_executable(sub0llm-gemma gemma/main.cpp)
target_link_libraries(sub0llm-gemma PRIVATE sub0llm)
sub0llm_apply_compile_options(sub0llm-gemma)
set_target_properties(sub0llm-gemma PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── sub0llm-qbench (Ch27) ─────────────────────────────────────────────────────
# Profile Q8 weight-consumption strategies (dequant-on-the-fly vs quantized int8
# matmul vs f32) across real model GEMV shapes — measure, don't assume.
add_executable(sub0llm-qbench qbench/main.cpp)
target_link_libraries(sub0llm-qbench PRIVATE sub0llm)
# qbench reaches the private CUDA backend header (backends/cuda/backend.hpp) to bench the
# device Q8 matmul against the CPU path; needs the library's src/ on its include path.
target_include_directories(sub0llm-qbench PRIVATE ${CMAKE_SOURCE_DIR}/src)
sub0llm_apply_compile_options(sub0llm-qbench)
set_target_properties(sub0llm-qbench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
