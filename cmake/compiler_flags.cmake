include(CheckCXXCompilerFlag)

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

# ── SIMD flags ────────────────────────────────────────────────────────────────
set(SUB0LLM_SIMD_FLAGS)

if(SUB0LLM_ENABLE_AVX512)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        list(APPEND SUB0LLM_SIMD_FLAGS -mavx512f -mavx512bw -mavx512dq)
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
    target_compile_options(${target} PRIVATE
        ${SUB0LLM_WARNING_FLAGS}
        ${SUB0LLM_SIMD_FLAGS}
        $<$<BOOL:${SUB0LLM_ENABLE_SANITIZERS}>:${SUB0LLM_SANITIZER_FLAGS}>
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
endfunction()
