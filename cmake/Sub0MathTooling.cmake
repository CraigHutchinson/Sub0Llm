# Performance tooling shared by the Sub0 family (Sub0Math and every library above it).
#
#   sub0math_target_perf_options(<target>)
#       Applies the profiling, vectorization-report and native-architecture options to a target.
#   sub0math_target_deterministic_fp(<target> [INTERFACE|PUBLIC|PRIVATE])
#       Forbids fused multiply-add contraction so results are bit-identical across -march levels and
#       compilers (-ffp-contract=off, /fp:precise). Default scope PRIVATE; see docs/PERFORMANCE.md.
#   sub0math_add_benchmark(<target> <sources>...)
#       Builds a nanobench executable using tools/perf/bench_runner.hpp, plus these targets:
#         <target>_json             run it once and write build/perf/<target>.json (nanobench format)
#         <target>_runs             SUB0MATH_BENCH_REPEATS interleaved, calibrated rounds -> <target>.samples.json
#         <target>_check            judge those rounds against perf/baselines/<host>.json; fails on a regression
#         <target>_baseline         record perf/baselines/<host>.json from the same rounds (needs a reason once
#                                   a baseline exists: bench_check.py baseline --reason ...)
#       The pre-push gate and the after-build check run at the quick level (a time budget, SUB0MATH_PERF_QUICK_BUDGET;
#       pairs and controls always, the other benchmarks rotating) unless the push carries a tag matching
#       SUB0MATH_PERF_FULL_REFS or SUB0MATH_PERF_LEVEL=full is set: docs/PERFORMANCE.md, "Levels".
#       and installs the pre-push gate and post-checkout hooks (SUB0MATH_PERF_GIT_HOOKS, off on CI),
#       and, unless SUB0MATH_PERF_ON_BUILD is OFF, a quick check after every rebuild of <target>
#       (its code or a header it uses changed), reporting regressions and progressions implicitly.
#         <target>_vtune_hotspots   VTune hotspots      (SUB0MATH_PROFILE_FILTER, _SECONDS)
#         <target>_vtune_uarch      VTune microarchitecture exploration
#         <target>_advisor_roofline Intel Advisor vectorization survey and roofline
#         <target>_perf_record      Linux perf sampling with call graphs
#
#   sub0math_add_test(<name> [FULL] COMMAND <command> [args...])
#       add_test() that can label a long suite FULL (ctest label "full"). `ctest -LE full` is then the quick
#       run for everyday work; plain `ctest` runs everything (milestones, CI). Tests made by a discovery macro
#       are labelled with its own argument: gtest_discover_tests(<t> PROPERTIES LABELS full) or
#       doctest_discover_tests(<t> ADD_LABELS 1 TEST_SUFFIX ...) / catch_discover_tests(<t> PROPERTIES LABELS full).
#   sub0math_label_full_tests(<test>...)
#       Adds the "full" label to tests that already exist (same directory scope).
#
# Consumers include this file from the Sub0Math source tree (CPM sets Sub0Math_SOURCE_DIR).

include_guard(GLOBAL)

# Paths are resolved inside each function from CMAKE_CURRENT_FUNCTION_LIST_DIR: include_guard skips
# re-inclusion, so variables set here would not reach a consumer's directory scope.

option(SUB0MATH_PROFILE_FLAGS "Debug info and frame pointers in optimised builds, for profilers" OFF)
option(SUB0MATH_VECTORIZATION_REPORT "Emit compiler vectorization remarks" OFF)
option(SUB0MATH_NATIVE_ARCH "Tune for the build machine's CPU (local experiments only)" OFF)
set(SUB0MATH_PROFILE_FILTER "" CACHE STRING "Benchmark name substring the profiling targets run")
set(SUB0MATH_PROFILE_SECONDS "10" CACHE STRING "Soak seconds per benchmark under a profiler")
set(SUB0MATH_BENCH_REPEATS "5" CACHE STRING "Interleaved rounds per budget check (tools/perf/bench_check.py)")
set(SUB0MATH_PERF_ON_BUILD "REPORT" CACHE STRING
    "After a benchmark executable is rebuilt: OFF, REPORT (print regressions and progressions) or ENFORCE (fail the build on a regression)")
set_property(CACHE SUB0MATH_PERF_ON_BUILD PROPERTY STRINGS OFF REPORT ENFORCE)
set(SUB0MATH_PERF_QUICK_ROUNDS "3" CACHE STRING "Rounds for the after-build check (noisy benchmarks get up to twice as many)")
set(SUB0MATH_PERF_QUICK_BUDGET "90" CACHE STRING
    "Target seconds of measuring for the quick level (the pre-push gate and the after-build check); 0 = no budget. Override per run with SUB0MATH_PERF_BUDGET or --budget")
set(SUB0MATH_PERF_FULL_REFS "v*;milestone/*" CACHE STRING
    "Tag-name patterns (a list): pushing a matching tag runs the pre-push gate at the full level. Override a run with SUB0MATH_PERF_LEVEL=full|quick")
set(SUB0MATH_PERF_QUICK_EPOCHS "0" CACHE STRING
    "nanobench epochs in the after-build check; 0 = default, the same method as baselines and records")
if(DEFINED ENV{CI})
    set(_sub0math_hooks_default OFF)
else()
    set(_sub0math_hooks_default ON)
endif()
option(SUB0MATH_PERF_GIT_HOOKS "Install the pre-push performance gate and post-checkout note in this repository's git hooks"
       ${_sub0math_hooks_default})

# Where the scripts are, for shells and CI: `python "$(cat build/sub0math_tooling_dir.txt)/profile.py" ...`.
# CPM may place Sub0Math anywhere, so consumers should not hard-code the path.
get_filename_component(_sub0math_tooling_dir "${CMAKE_CURRENT_LIST_DIR}/../tools/perf" ABSOLUTE)
file(WRITE "${CMAKE_BINARY_DIR}/sub0math_tooling_dir.txt" "${_sub0math_tooling_dir}")

cmake_host_system_information(RESULT _sub0math_host QUERY HOSTNAME)
string(TOLOWER "${_sub0math_host}" _sub0math_host)
set(SUB0MATH_PERF_HOST "${_sub0math_host}" CACHE STRING "Baseline name in perf/baselines/")

function(sub0math_target_perf_options target)
    if(SUB0MATH_PROFILE_FLAGS)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/Zi /Oy->
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-g -fno-omit-frame-pointer>)
        target_link_options(${target} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/DEBUG>)
    endif()
    if(SUB0MATH_VECTORIZATION_REPORT)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/Qvec-report:2>
            $<$<CXX_COMPILER_ID:GNU>:-fopt-info-vec-optimized -fopt-info-vec-missed>
            $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Rpass=loop-vectorize|slp-vectorizer
                -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize>)
    endif()
    if(SUB0MATH_NATIVE_ARCH)
        target_compile_options(${target} PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-march=native>)
    endif()
endfunction()

# Floating-point determinism: no contraction of a * b + c into a fused multiply-add. GCC defaults to
# -ffp-contract=fast in gnu modes and Clang/MSVC contract under some flags, so a build with FMA hardware
# (-march=native, SUB0MATH_NATIVE_ARCH, -march=x86-64-v3) would otherwise round differently from one
# without. Scope is INTERFACE for header-only libraries so consumers' translation units inherit it.
function(sub0math_target_deterministic_fp target)
    set(scope PRIVATE)
    if(ARGC GREATER 1)
        set(scope "${ARGV1}")
    endif()
    if(NOT scope MATCHES "^(INTERFACE|PUBLIC|PRIVATE)$")
        message(FATAL_ERROR "sub0math_target_deterministic_fp: scope must be INTERFACE, PUBLIC or PRIVATE, got '${scope}'")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} ${scope} $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>)
    elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC") # clang-cl
        target_compile_options(${target} ${scope} $<$<COMPILE_LANGUAGE:CXX>:/fp:precise>
                               $<$<COMPILE_LANGUAGE:CXX>:/clang:-ffp-contract=off>)
    else() # GCC, Clang, AppleClang
        target_compile_options(${target} ${scope} $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
    endif()
    # MinGW-w64 GCC guarantees only 16-byte stack alignment, but AVX code (-march=x86-64-v3 or native) spills
    # 32-byte vectors with aligned moves, which fault. Assembling vector moves as unaligned ones fixes it at no
    # measurable cost; without it, a private consumer's tests segfaulted under -march=x86-64-v3 (2026-10-20).
    if(MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-Wa,-muse-unaligned-vector-move" SUB0MATH_HAVE_UNALIGNED_VECTOR_MOVE)
        if(SUB0MATH_HAVE_UNALIGNED_VECTOR_MOVE)
            target_compile_options(${target} ${scope} $<$<COMPILE_LANGUAGE:CXX>:-Wa,-muse-unaligned-vector-move>)
        endif()
    endif()
endfunction()

# Quality gates come in the same two levels as the performance gate. A suite that takes long is labelled
# "full"; everyday runs exclude it (ctest -LE full, or the "quick" test preset), milestones and CI run everything.
set(SUB0MATH_FULL_TEST_LABEL "full")

function(sub0math_label_full_tests)
    foreach(test IN LISTS ARGN)
        if(NOT TEST ${test})
            message(FATAL_ERROR "sub0math_label_full_tests: no test named '${test}' in this directory")
        endif()
        set_property(TEST ${test} APPEND PROPERTY LABELS "${SUB0MATH_FULL_TEST_LABEL}")
    endforeach()
endfunction()

function(sub0math_add_test name)
    cmake_parse_arguments(PARSE_ARGV 1 T "FULL" "" "COMMAND")
    if(NOT T_COMMAND)
        message(FATAL_ERROR "sub0math_add_test(<name> [FULL] COMMAND <command> [args...]): COMMAND is required")
    endif()
    add_test(NAME ${name} COMMAND ${T_COMMAND})
    if(T_FULL)
        sub0math_label_full_tests(${name})
    endif()
endfunction()

function(sub0math_add_benchmark target)
    get_filename_component(tooling "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tools/perf" ABSOLUTE)
    get_filename_component(vendor "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../tests/vendor" ABSOLUTE)
    add_executable(${target} ${ARGN})
    target_include_directories(${target} SYSTEM PRIVATE "${tooling}" "${vendor}")
    target_compile_features(${target} PRIVATE cxx_std_23)
    sub0math_target_perf_options(${target})

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(NOT Python3_Interpreter_FOUND)
        return()
    endif()
    set(results "${CMAKE_BINARY_DIR}/perf/${target}.json")
    set(samples "${CMAKE_BINARY_DIR}/perf/${target}.samples.json")
    set(baseline "${PROJECT_SOURCE_DIR}/perf/baselines/${SUB0MATH_PERF_HOST}.json")
    set(check "${tooling}/bench_check.py")
    set(profile "${tooling}/profile.py")

    add_custom_target(${target}_json
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/perf"
        COMMAND $<TARGET_FILE:${target}> --json "${results}"
        DEPENDS ${target} USES_TERMINAL VERBATIM)
    # Interleaved, calibrated, load-aware rounds: see tools/perf/bench_check.py.
    add_custom_target(${target}_runs
        COMMAND ${Python3_EXECUTABLE} "${check}" run --exe $<TARGET_FILE:${target}> --out "${samples}"
                --rounds ${SUB0MATH_BENCH_REPEATS}
        DEPENDS ${target} USES_TERMINAL VERBATIM)
    add_custom_target(${target}_check
        COMMAND ${Python3_EXECUTABLE} "${check}" check --results "${samples}" --baseline "${baseline}"
        DEPENDS ${target}_runs USES_TERMINAL VERBATIM)
    add_custom_target(${target}_baseline
        COMMAND ${Python3_EXECUTABLE} "${check}" baseline --results "${samples}" --baseline "${baseline}"
        DEPENDS ${target}_runs USES_TERMINAL VERBATIM)

    # Implicit report: the executable relinks only when its code, or a header it uses, changed.
    if(NOT SUB0MATH_PERF_ON_BUILD STREQUAL "OFF")
        set(mode_args --report-only)
        if(SUB0MATH_PERF_ON_BUILD STREQUAL "ENFORCE")
            set(mode_args "")
        endif()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "perf: quick interleaved check of ${target} (SUB0MATH_PERF_ON_BUILD=${SUB0MATH_PERF_ON_BUILD})"
            COMMAND ${Python3_EXECUTABLE} "${check}" auto --exe $<TARGET_FILE:${target}> --baseline "${baseline}"
                    --rounds ${SUB0MATH_PERF_QUICK_ROUNDS} --epochs ${SUB0MATH_PERF_QUICK_EPOCHS}
                    --budget ${SUB0MATH_PERF_QUICK_BUDGET}
                    --out "${CMAKE_BINARY_DIR}/perf/${target}.quick.json" ${mode_args}
                    --record --target ${target} --repo "${PROJECT_SOURCE_DIR}"
            VERBATIM)
    endif()

    # Pre-push gate: blocks a push whose benchmarks regressed against the committed baseline or the
    # local record of the commit it started from, unless the regression was accepted (bench_check.py accept).
    if(SUB0MATH_PERF_GIT_HOOKS AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
        if(CMAKE_CONFIGURATION_TYPES)
            set(hook_config Release)
            set(hook_exe "${CMAKE_CURRENT_BINARY_DIR}/Release/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        else()
            set(hook_config "")
            set(hook_exe "${CMAKE_CURRENT_BINARY_DIR}/${target}${CMAKE_EXECUTABLE_SUFFIX}")
        endif()
        # SUB0MATH_PERF_PINS: dependency pins the pre-push hook checks are on the dependency's remote, each
        # "file=cmake/Dependencies.cmake,var=X_REVISION,repo=owner/name[,checkout=../X][,name=X]" (docs/PERFORMANCE.md).
        set(pin_args "")
        foreach(pin IN LISTS SUB0MATH_PERF_PINS)
            list(APPEND pin_args --pin "${pin}")
        endforeach()
        execute_process(
            COMMAND ${Python3_EXECUTABLE} "${check}" install-hooks --repo "${PROJECT_SOURCE_DIR}" --build-dir "${CMAKE_BINARY_DIR}"
                    --target ${target} --exe "${hook_exe}" --baseline "${baseline}" --config "${hook_config}" ${pin_args}
                    --full-refs "${SUB0MATH_PERF_FULL_REFS}" --quick-budget ${SUB0MATH_PERF_QUICK_BUDGET}
            RESULT_VARIABLE hook_result)
        if(NOT hook_result EQUAL 0)
            message(WARNING "sub0perf: installing the git hooks failed (${hook_result})")
        endif()
    endif()

    foreach(analysis vtune-hotspots vtune-uarch advisor-roofline perf-record)
        string(REPLACE "-" "_" suffix "${analysis}")
        add_custom_target(${target}_${suffix}
            COMMAND ${Python3_EXECUTABLE} "${profile}" ${analysis} --exe $<TARGET_FILE:${target}>
                    --filter "${SUB0MATH_PROFILE_FILTER}" --seconds ${SUB0MATH_PROFILE_SECONDS}
                    --out "${CMAKE_BINARY_DIR}/profiles"
            DEPENDS ${target} USES_TERMINAL VERBATIM)
    endforeach()
endfunction()
