include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# ── Warning flags ─────────────────────────────────────────────────────────────
set(SUB0LLM_WARNING_FLAGS)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND SUB0LLM_WARNING_FLAGS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wno-unused-parameter
        -Wconversion
        -Wsign-conversion
    )
    if(SUB0LLM_ENABLE_WARNINGS_AS_ERRORS)
        list(APPEND SUB0LLM_WARNING_FLAGS -Werror)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    list(APPEND SUB0LLM_WARNING_FLAGS /W4)
    if(SUB0LLM_ENABLE_WARNINGS_AS_ERRORS)
        list(APPEND SUB0LLM_WARNING_FLAGS /WX)
    endif()
endif()

# ── SIMD / architecture flags ─────────────────────────────────────────────────
set(SUB0LLM_SIMD_FLAGS)

if(SUB0LLM_ENABLE_NATIVE)
    # -march=native subsumes all individual SIMD flags.
    # We probe which ISA extensions the host CPU actually provides so the
    # SUB0LLM_AVX2 / SUB0LLM_AVX512 compile-time dispatch macros are correct.
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(CMAKE_REQUIRED_FLAGS "-march=native")
        check_cxx_source_compiles(
            "#ifndef __AVX2__\n#error no AVX2\n#endif\nint main(){}"
            _native_has_avx2)
        check_cxx_source_compiles(
            "#ifndef __AVX512F__\n#error no AVX512\n#endif\nint main(){}"
            _native_has_avx512)
        unset(CMAKE_REQUIRED_FLAGS)

        if(_native_has_avx2)
            set(SUB0LLM_ENABLE_AVX2  ON CACHE BOOL "AVX2 detected via -march=native"   FORCE)
        endif()
        if(_native_has_avx512)
            set(SUB0LLM_ENABLE_AVX512 ON CACHE BOOL "AVX-512 detected via -march=native" FORCE)
        endif()

        list(APPEND SUB0LLM_SIMD_FLAGS -march=native -mtune=native)
        message(STATUS
            "sub0llm: -march=native — avx2=${_native_has_avx2} avx512=${_native_has_avx512}")
    else()
        message(WARNING "sub0llm: SUB0LLM_ENABLE_NATIVE requires GCC/Clang; ignoring")
    endif()

elseif(SUB0LLM_ENABLE_AVX512)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        # Core AVX-512 subsets (widely available on Skylake-X / Ice Lake+)
        list(APPEND SUB0LLM_SIMD_FLAGS
            -mavx512f      # foundation
            -mavx512bw     # byte/word
            -mavx512dq     # double/quadword
            -mavx512vl     # 128/256-bit vector lengths (enables VEX forms)
            -mavx512cd     # conflict detection
        )
        # VNNI (int8 dot-product, Ice Lake+) — optional, checked at configure time
        check_cxx_compiler_flag(-mavx512vnni _has_avx512vnni)
        if(_has_avx512vnni)
            list(APPEND SUB0LLM_SIMD_FLAGS -mavx512vnni)
            message(STATUS "sub0llm: AVX-512 VNNI enabled (int8 dot-product)")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND SUB0LLM_SIMD_FLAGS /arch:AVX512)
    endif()
    message(STATUS "sub0llm: AVX-512 SIMD enabled")

elseif(SUB0LLM_ENABLE_AVX2)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND SUB0LLM_SIMD_FLAGS -mavx2 -mfma)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND SUB0LLM_SIMD_FLAGS /arch:AVX2)
    endif()
    message(STATUS "sub0llm: AVX2 SIMD enabled")
endif()

# ── Release-mode performance flags ────────────────────────────────────────────
# -ffp-contract=fast : FMA fusion only — safe; does not alter NaN/inf semantics
# -funroll-loops     : loop unrolling (helps SIMD-vectorised matmul inner loops)
# NOTE: -ffast-math is intentionally omitted; apply_math_op relies on std::isnan.
set(SUB0LLM_RELEASE_FLAGS)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    list(APPEND SUB0LLM_RELEASE_FLAGS
        $<$<CONFIG:Release>:-ffp-contract=fast;-funroll-loops>
        $<$<CONFIG:RelWithDebInfo>:-ffp-contract=fast>
    )
endif()

# ── LTO ───────────────────────────────────────────────────────────────────────
if(SUB0LLM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
        message(STATUS "sub0llm: LTO enabled")
    else()
        message(WARNING "sub0llm: LTO requested but not supported: ${ipo_output}")
    endif()
endif()

# ── Sanitizers ────────────────────────────────────────────────────────────────
if(SUB0LLM_ENABLE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(SUB0LLM_SANITIZER_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)
        message(STATUS "sub0llm: Address + UB sanitizers enabled")
    endif()
endif()

# ── Helper to apply flags to a target ─────────────────────────────────────────
function(sub0llm_apply_compile_options target)
    # Warning/SIMD/release flags are GNU/clang command-line syntax (-Wall, -mavx2, …).
    # Guard them to C++ sources: on a CUDA target, nvcc forwards unknown compile options
    # to the host compiler (cl.exe on Windows), which rejects -W/-m flags (D8021). The .cu
    # device code needs none of them anyway — its arch comes from CUDA_ARCHITECTURES.
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${SUB0LLM_WARNING_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${SUB0LLM_SIMD_FLAGS}>
        $<$<COMPILE_LANGUAGE:CXX>:${SUB0LLM_RELEASE_FLAGS}>
        $<$<AND:$<BOOL:${SUB0LLM_ENABLE_SANITIZERS}>,$<COMPILE_LANGUAGE:CXX>>:${SUB0LLM_SANITIZER_FLAGS}>
    )
    target_link_options(${target} PRIVATE
        $<$<BOOL:${SUB0LLM_ENABLE_SANITIZERS}>:${SUB0LLM_SANITIZER_FLAGS}>
    )
    target_compile_definitions(${target} PRIVATE
        $<$<BOOL:${SUB0LLM_ENABLE_AVX2}>:SUB0LLM_AVX2>
        $<$<BOOL:${SUB0LLM_ENABLE_AVX512}>:SUB0LLM_AVX512>
        $<$<BOOL:${SUB0LLM_ENABLE_CUDA}>:SUB0LLM_CUDA>
        $<$<BOOL:${SUB0LLM_ENABLE_OPENVINO}>:SUB0LLM_OPENVINO>
    )
    # Vectorisation report: -fopt-info-vec prints which loops were vectorised.
    if(SUB0LLM_ENABLE_VEC_REPORT AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE -fopt-info-vec)
    endif()
endfunction()
