# Ch32 tree-predictor / hierarchy track. P1 1c OOV A/B experiment.
add_executable(ch32_oov_ab oov_ab.cpp)
target_link_libraries(ch32_oov_ab PRIVATE sub0diff)

# Phase-0 M2 topic-drift baseline (corpus vs flat-model generations).
add_executable(ch32_m2_drift m2_drift.cpp)
target_link_libraries(ch32_m2_drift PRIVATE sub0diff)

# P2 2b gist-conditioning A/B (flat Denoiser vs GistDenoiser).
add_executable(ch32_gist_ab gist_ab.cpp)
target_link_libraries(ch32_gist_ab PRIVATE sub0diff)

# P2 2e coarse-to-fine vs flat (efficiency/context benchmark).
add_executable(ch32_hier_ab hier_ab.cpp)
target_link_libraries(ch32_hier_ab PRIVATE sub0diff)

# P2 HierDenoiser generation + M2-on-generations (flat vs coarse-to-fine).
add_executable(ch32_hier_gen hier_gen.cpp)
target_link_libraries(ch32_hier_gen PRIVATE sub0diff)

# P2 context-length ceiling: where flat OOMs, hier keeps running (one model/N per process).
add_executable(ch32_hier_ceiling hier_ceiling.cpp)
target_link_libraries(ch32_hier_ceiling PRIVATE sub0diff)

# P3 mixed-N training: variable-N MERA robustness across lengths.
add_executable(ch32_mera_mixedn mera_mixedn.cpp)
target_link_libraries(ch32_mera_mixedn PRIVATE sub0diff)

# Viz Phase A: generate a traced passage -> tools/viz/trace.json for the web scrubber.
add_executable(ch32_viz_gen viz_gen.cpp)
target_link_libraries(ch32_viz_gen PRIVATE sub0diff)

# Viz Phase C: interactive server — trains a MERA once, serves POST /v1/generate_trace + the static
# viewer so the web UI can drive generation live (prompt box + param knobs + A/B trace diff).
add_executable(ch32_viz_server viz_server.cpp)
target_link_libraries(ch32_viz_server PRIVATE sub0diff httplib)

# Viz Phase D: training-time trajectory — snapshot a fixed-prompt traced generation every K steps so the
# viewer can scrub across training and watch the same prompt's denoising sharpen as the model learns.
add_executable(ch32_viz_train viz_train.cpp)
target_link_libraries(ch32_viz_train PRIVATE sub0diff)

# Resumable CUDA trainer for the hierarchical MERA (and flat) denoiser: reuses the Ch29 consistent
# config layer + binary checkpoint format. `--ckpt-dir X` alone rebuilds the exact arch from
# run_config.json and continues from the latest step with Adam state intact. Stamp code_sha (resume tag).
add_executable(ch32_mera_train mera_train.cpp)
target_link_libraries(ch32_mera_train PRIVATE sub0diff)
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
    OUTPUT_VARIABLE SUB0DIFF_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _git_rc)
if(NOT _git_rc EQUAL 0 OR SUB0DIFF_GIT_SHA STREQUAL "")
    set(SUB0DIFF_GIT_SHA "dev")
endif()
target_compile_definitions(ch32_mera_train PRIVATE SUB0DIFF_CODE_SHA="${SUB0DIFF_GIT_SHA}")

# Apply the standard warning/SIMD/release flags + feature defines (SUB0LLM_AVX2 / SUB0LLM_CUDA / …) to
# every ch32 target — these were missing chapter-wide, so the executables' own code (training-loop
# driver, device guards) ran without AVX2 and saw SUB0LLM_CUDA undefined. The heavy compute already
# lives in sub0diff (built with the flags); this gets the chapter binaries the same treatment.
foreach(_t
    ch32_oov_ab ch32_m2_drift ch32_gist_ab ch32_hier_ab ch32_hier_gen ch32_hier_ceiling
    ch32_mera_mixedn ch32_viz_gen ch32_viz_server ch32_viz_train ch32_mera_train)
    sub0llm_apply_compile_options(${_t})
endforeach()
