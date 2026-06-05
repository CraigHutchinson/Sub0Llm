# ── ch27_specialized_build ────────────────────────────────────────────────────
# Specialized Model Compilation: monomorphize the model to one GGUF's exact shapes.
#
# The chapter binary compiles against a *generated* spec header. The default
# target spec is committed (generated/qwen3_0_6b_q8_0_spec.hpp) so the build works
# without any model file present. Override with -DCH27_SPEC=<ident> to point at a
# different generated header (e.g. a Gemma 4 spec once you have the GGUF).

set(CH27_SPEC "qwen3_0_6b_q8_0" CACHE STRING
    "Generated spec stem in generated/ to monomorphize ch27 against")

# Map the stem to the struct identifier (PascalCase). For the committed default we
# know it; for user-supplied stems, allow an explicit -DCH27_SPEC_TYPE override.
set(CH27_SPEC_TYPE "" CACHE STRING
    "Fully-qualified spec struct type (default inferred for the committed Qwen3 spec)")
if(CH27_SPEC_TYPE STREQUAL "")
    if(CH27_SPEC STREQUAL "qwen3_0_6b_q8_0")
        set(CH27_SPEC_TYPE "sub0llm::spec::Qwen3_0_6B_Q8_0")
    else()
        message(FATAL_ERROR
            "ch27: set -DCH27_SPEC_TYPE=<fully::qualified::Struct> for spec '${CH27_SPEC}'")
    endif()
endif()

set(CH27_SPEC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/generated/${CH27_SPEC}_spec.hpp")
if(NOT EXISTS "${CH27_SPEC_HEADER}")
    message(FATAL_ERROR
        "ch27: generated spec header not found: ${CH27_SPEC_HEADER}\n"
        "  Generate it with: sub0llm-specialize --model <model.gguf> "
        "--out-dir ${CMAKE_CURRENT_SOURCE_DIR}/generated")
endif()

add_executable(ch27_specialized_build main.cpp microbench_dynamic.cpp)
target_link_libraries(ch27_specialized_build PRIVATE sub0llm)
sub0llm_apply_compile_options(ch27_specialized_build)
target_include_directories(ch27_specialized_build PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/generated")
target_compile_definitions(ch27_specialized_build PRIVATE
    "SUB0LLM_CH27_SPEC_HEADER=\"${CH27_SPEC}_spec.hpp\""
    "SUB0LLM_CH27_SPEC_TYPE=${CH27_SPEC_TYPE}"
    "SUB0LLM_CH27_SPEC_NAME=\"${CH27_SPEC}\"")
set_target_properties(ch27_specialized_build PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Optional convenience target: regenerate the default spec from the local GGUF.
# Only useful when models/Qwen3-0.6B-Q8_0.gguf is present.
add_custom_target(ch27-regen-spec
    COMMAND sub0llm-specialize
            --model "${CMAKE_SOURCE_DIR}/models/Qwen3-0.6B-Q8_0.gguf"
            --out-dir "${CMAKE_CURRENT_SOURCE_DIR}/generated"
    DEPENDS sub0llm-specialize
    COMMENT "Regenerating ch27 spec from Qwen3-0.6B GGUF"
    VERBATIM)
