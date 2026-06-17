add_executable(ch29_diffusion_training main.cpp)

target_link_libraries(ch29_diffusion_training PRIVATE sub0diff)

# Stamp the build with the short git SHA → run_config.json's _code_sha (the code half of the
# specialized-build tag). Resolved at configure time; "dev" if git is unavailable.
execute_process(
    COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
    OUTPUT_VARIABLE SUB0DIFF_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET RESULT_VARIABLE _git_rc)
if(NOT _git_rc EQUAL 0 OR SUB0DIFF_GIT_SHA STREQUAL "")
    set(SUB0DIFF_GIT_SHA "dev")
endif()
target_compile_definitions(ch29_diffusion_training PRIVATE
    SUB0DIFF_CODE_SHA="${SUB0DIFF_GIT_SHA}")

sub0llm_apply_compile_options(ch29_diffusion_training)

set_target_properties(ch29_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ch29b — isolated data-parallel throughput benchmark (separate exe so the main
# training binary can stay running/locked while we profile threading scaling).
add_executable(ch29b_diffusion_training bench_parallel.cpp)
target_link_libraries(ch29b_diffusion_training PRIVATE sub0diff)
sub0llm_apply_compile_options(ch29b_diffusion_training)
set_target_properties(ch29b_diffusion_training PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
