cmake_minimum_required(VERSION 3.28...4.2)
project(sub0llm VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# sub0llm is a staged build. The model's dimensions and vocabulary are compiled
# in as constexpr constants, so the configuration is generated *during the build*
# and the engine is compiled against it:
#
#   sub0-configure (stage 1, config-independent)
#        | reads the corpus + the SUB0_* args below
#        v
#   generated/sub0_config.hpp   (constexpr dims + corpus-derived vocab)
#        |
#        v
#   libsub0_core (shared)   --->  libsub0_train (shared)  \
#                           --->  libsub0_gen   (shared)   >--> sub0llm (driver)
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Emit compile_commands.json for clangd / editor tooling (harmless for Ninja/Makefiles).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Native tuning. This project is always built on the machine the model will run
# on, so an exact-CPU build is the intended default for real runs (the `native`
# preset). It is opt-in and probed for support so portable/cross builds and
# non-GCC/Clang toolchains still configure cleanly.
option(SUB0_NATIVE "Optimize for the building machine's exact CPU (-march=native)" OFF)
if(SUB0_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=native" SUB0_HAS_MARCH_NATIVE)
  if(SUB0_HAS_MARCH_NATIVE)
    # Release-only: a -march=native Debug build would also be CPU-specific but is
    # not what anyone debugging wants, and the speed only matters for real runs.
    # Scoped to the CXX language so a future CUDA backend's nvcc host compile never
    # sees this clang-only flag (the super-build isolates toolchains too; this is the
    # belt-and-suspenders for the single-tree case).
    add_compile_options($<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>)
    message(STATUS "SUB0_NATIVE: building for this machine's CPU (-march=native)")
  else()
    message(WARNING "SUB0_NATIVE requested but the compiler rejects -march=native; using a generic build")
  endif()
endif()

# OpenMP detection lives in cmake/OpenMP.cmake and is included on the normal build path
# below (after the super-build short-circuit), so it sets SUB0_OPENMP_TARGET before
# sub0_core links it -- without blocking the super-build orchestrator on an OpenMP probe.

# Transcendental math path: fast vectorizable approximations (default) vs the exact
# std::exp/std::erf path. Compile-time only -- the unused branch is eliminated.
option(SUB0_EXACT_MATH "Use exact std::exp/std::erf instead of the fast approximations" OFF)

# --- Dependency management via CPM -----------------------------------------
include(cmake/get_cpm.cmake)

# TODO(cli): preferred future CLI library is nathan-baggs/clap (BSL-1.0, header-only
# + CMake, reflection-based `clap::parse<Args>()` API). It needs C++26 reflection
# (P2996) + annotations (P3394), which today only build on gcc trunk -- Clang 22.1.6
# has no reflection support yet. Switch when Clang ships P2996. (clap supersedes the
# earlier amedoeyes/cli idea, which was AGPL + xmake-only.) Until then CLI11 stays.
CPMAddPackage("gh:CLIUtils/CLI11@2.4.2")

# TODO(json): when JSON support is needed (e.g. config/metadata/tokenizer files),
# use simdjson (Apache-2.0): CPMAddPackage("gh:simdjson/simdjson@3.10.1"). Not added
# yet -- the project has no JSON usage, so we avoid an unused dependency for now.

# TODO(tune): the `tune` autotuner uses an in-tree search (include/sub0/tune.hpp).
# Prior art surveyed -- the mature auto-tuning stacks (Optuna, NNI, scikit-optimize,
# Vizier, FLAML) are Python; the C++ options are biteopt (header-only derivative-free
# global optimiser), CMA-ES, NLopt (DIRECT) and dlib's find_max_global. None were
# pulled: our space is tiny, discrete and low-cardinality, where a deterministic coarse
# grid + basin refinement is both near-optimal and -- unlike the stochastic optimisers
# -- unit testable, and none of them supply the Knob/Objective framework we actually
# want. Escalate to biteopt (gh:avaneev/biteopt, header-only) only if the search space
# becomes continuous / high-dimensional.

# --- Stage-1 "runtime arguments": passed through as cache variables ----------
set(SUB0_CORPUS   "${CMAKE_SOURCE_DIR}/data/fineweb_smoke.txt" CACHE FILEPATH "training corpus (drives vocab)")
set(SUB0_D_MODEL  192 CACHE STRING "embedding / residual width")
set(SUB0_N_LAYERS 8  CACHE STRING "transformer blocks")
set(SUB0_N_HEADS  6  CACHE STRING "attention heads")
set(SUB0_SEQ_LEN  64 CACHE STRING "context window")
set(SUB0_TERNARY  0  CACHE STRING "1 = BitNet-style ternary block weights")

# Pre-tokenize the whole corpus to corpus.tok at build time (fast random-access training)?
# OFF skips that ~2x-on-disk token copy: training then tokenizes windows on demand from the
# raw corpus -- the out-of-core path for a corpus too large to duplicate. The header and
# tokenizer.bin are emitted either way, so the engine still builds.
#
# AUTO (the default) decides by SCALE rather than a fixed threshold: pre-tokenizing is a
# clear win while the token copy stays page-cacheable, so emit it when the estimated
# corpus.tok (~2x the text) fits in half of physical RAM, and switch to on-demand for a
# larger corpus (where the mmap can't stay resident and the disk cost is large). Override
# with -DSUB0_CORPUS_TOK=ON or =OFF.
set(SUB0_CORPUS_TOK "AUTO" CACHE STRING "Pre-tokenize corpus.tok: ON / OFF / AUTO (by corpus size vs RAM)")
set_property(CACHE SUB0_CORPUS_TOK PROPERTY STRINGS AUTO ON OFF)

cmake_host_system_information(RESULT _ram_mb QUERY TOTAL_PHYSICAL_MEMORY)
set(_corpus_bytes 0)
if(EXISTS "${SUB0_CORPUS}")
  file(SIZE "${SUB0_CORPUS}" _corpus_bytes)
endif()
math(EXPR _est_tok_mb  "(${_corpus_bytes} * 2) / 1048576")   # corpus.tok ~ 2x the text
math(EXPR _half_ram_mb "${_ram_mb} / 2")
if(_est_tok_mb GREATER _half_ram_mb)
  set(_auto_tok 0)   # token copy would not fit comfortably in RAM -> on demand
else()
  set(_auto_tok 1)   # fits -> pre-tokenize for fast random-access training
endif()

if(SUB0_CORPUS_TOK STREQUAL "AUTO")
  set(SUB0_CORPUS_TOK_FLAG ${_auto_tok})
elseif(SUB0_CORPUS_TOK)            # ON / 1 / TRUE
  set(SUB0_CORPUS_TOK_FLAG 1)
else()                             # OFF / 0 / FALSE
  set(SUB0_CORPUS_TOK_FLAG 0)
endif()
message(STATUS "corpus.tok: SUB0_CORPUS_TOK=${SUB0_CORPUS_TOK} -> emit=${SUB0_CORPUS_TOK_FLAG} "
               "(corpus ${_est_tok_mb}MB est-tok vs ${_half_ram_mb}MB half-RAM)")

# --- Compute-backend selection (CPU / GPU / HYBRID axis) ---------------------
# Picks SUB0_BACKEND_SOURCES for sub0_core and enforces the backend invariants
# (e.g. ternary is CPU-only for now). Needs SUB0_TERNARY (set above).
include(cmake/Backends.cmake)

# --- Multi-compiler super-build orchestrator (opt-in) -----------------------
# Default OFF -> plain single-tree clang build, artifacts in ${CMAKE_BINARY_DIR} next
# to the driver. ON -> isolates each backend's toolchain (clang vs nvcc) in its own
# sub-build; the orchestrator defines the child builds and the top-level returns here.
include(cmake/SuperBuild.cmake)
if(SUB0_SUPERBUILD)
  return()
endif()

# --- OpenMP (data-parallel training) ----------------------------------------
include(cmake/OpenMP.cmake)

# --- Optional CUDA device-training backend ----------------------------------
# SUB0_BUILD_CUDA is DERIVED from SUB0_COMPUTE in cmake/Backends.cmake (GPU/AUTO-with-device
# -> ON), so SUB0_COMPUTE is the single user-facing switch. When ON, builds src/backend_cuda.cu
# with nvcc (sm_${SUB0_CUDA_ARCH}) into a shared lib + self-test, and links it into the train
# stage for the device training fast-path. enable_language(CUDA) is called here so nvcc is set
# up before the targets below; the actual targets are defined next to sub0_core.
if(SUB0_BUILD_CUDA)
  if(NOT SUB0_HAS_CUDA)
    message(FATAL_ERROR "SUB0_BUILD_CUDA=ON but no CUDA toolkit/device was detected at configure time.")
  endif()
  enable_language(CUDA)
  message(STATUS "CUDA backend: nvcc enabled, target sm_${SUB0_CUDA_ARCH}")
endif()

# --- Stage 1: the configurator (host tool, depends on nothing generated) -----
add_executable(sub0-configure tools/configurator.cpp)
target_include_directories(sub0-configure PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0-configure PRIVATE CLI11::CLI11)

# --- Stage 2: code generation — run the configurator to emit the config ------
set(GEN_DIR    "${CMAKE_BINARY_DIR}/generated")
set(GEN_HEADER "${GEN_DIR}/sub0_config.hpp")
file(MAKE_DIRECTORY "${GEN_DIR}")

# Persisted tuned-defaults cache. `sub0llm tune` writes the best threads /
# windows-per-thread here; the configurator reads it back to bake DEFAULT_THREADS /
# DEFAULT_WINDOWS_PER_THREAD into the header. Touched once so it always exists as a
# build dependency — a tune-updated (newer) cache then retriggers config regen.
set(GEN_TUNE_CACHE "${GEN_DIR}/sub0_tune.cache")
if(NOT EXISTS "${GEN_TUNE_CACHE}")
  file(WRITE "${GEN_TUNE_CACHE}" "")
endif()

add_custom_command(
  OUTPUT  "${GEN_HEADER}"
  BYPRODUCTS "${GEN_DIR}/corpus.tok" "${GEN_DIR}/tokenizer.bin"
  COMMAND sub0-configure
            --corpus  "${SUB0_CORPUS}"
            --dmodel  ${SUB0_D_MODEL}
            --layers  ${SUB0_N_LAYERS}
            --heads   ${SUB0_N_HEADS}
            --seq     ${SUB0_SEQ_LEN}
            --ternary ${SUB0_TERNARY}
            --corpus-tok ${SUB0_CORPUS_TOK_FLAG}
            --tune-cache "${GEN_TUNE_CACHE}"
            --has-cuda ${SUB0_HAS_CUDA}
            --cuda-arch ${SUB0_CUDA_ARCH}
            --gpu-vram-mb ${SUB0_GPU_VRAM_MB}
            --gpu-shared-mb ${SUB0_GPU_SHARED_MB}
            --compute ${SUB0_COMPUTE_MODE_INT}
            --cuda-tf32 ${SUB0_CUDA_TF32_FLAG}
            -o "${GEN_HEADER}"
  DEPENDS sub0-configure "${SUB0_CORPUS}" "${GEN_TUNE_CACHE}"
  COMMENT "Configuring: deriving vocab from ${SUB0_CORPUS} and baking constexpr config"
  VERBATIM
)
add_custom_target(sub0_generate_config DEPENDS "${GEN_HEADER}")

# --- Stage 3: engine core (shared), compiled against the generated header ----
# Two translation units, one shared library: engine_core.cpp is the backend-agnostic
# part (tokenizer, serialization, sampler, config paths) and backend_cpu.cpp is the
# CPU compute backend (ops, autograd, train_batch, AdamW). A future GPU backend
# replaces backend_cpu.cpp behind the same include/sub0/core.hpp API.
add_library(sub0_core SHARED src/engine_core.cpp ${SUB0_BACKEND_SOURCES})
add_dependencies(sub0_core sub0_generate_config)
target_include_directories(sub0_core PUBLIC "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
# The AdamW hot loop only ever sqrt()s non-negative values and never reads errno, so
# the errno side-effect of the math libcalls is dead weight that stops the optimizer
# from emitting vectorized vsqrtps. Dropping it changes no computed value (it is not a
# fast-math reassociation flag) and lets the update loop vectorize.
target_compile_options(sub0_core PRIVATE
  $<$<CXX_COMPILER_ID:Clang,GNU>:-fno-math-errno>)
if(SUB0_EXACT_MATH)
  target_compile_definitions(sub0_core PRIVATE SUB0_EXACT_MATH)
endif()
if(SUB0_OPENMP_TARGET)
  target_link_libraries(sub0_core PUBLIC ${SUB0_OPENMP_TARGET})
endif()
# Propagated to every TU compiled against the core so a build that lost OpenMP fails
# loudly at compile time (see the #error in engine.cpp) instead of running serially.
if(SUB0_REQUIRE_OPENMP)
  target_compile_definitions(sub0_core PUBLIC SUB0_REQUIRE_OPENMP)
endif()

# Performance-knob tuning mode (include/sub0/knob.hpp): PUBLIC so any TU using a Knob sees it.
# OFF (default) bakes knobs as compile-time constants; ON makes them runtime-tunable.
if(SUB0_TUNING)
  target_compile_definitions(sub0_core PUBLIC SUB0_TUNING)
endif()

# --- CUDA backend skeleton (Phase 2; built only when SUB0_BUILD_CUDA) --------
# Compiled by nvcc; meets the clang world only at the extern "C" self-test boundary.
# Grows into the real GPU backend (replacing backend_cpu.cpp under SUB0_COMPUTE=GPU).
if(SUB0_BUILD_CUDA)
  add_library(sub0_backend_cuda SHARED src/backend_cuda.cu)
  add_dependencies(sub0_backend_cuda sub0_generate_config)
  target_include_directories(sub0_backend_cuda PRIVATE "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
  set_target_properties(sub0_backend_cuda PROPERTIES
    CUDA_ARCHITECTURES "${SUB0_CUDA_ARCH}"
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  # Flush single-precision denormals to zero on the device, matching the CPU backend's
  # FTZ/DAZ (set_flush_denormals in backend_cpu.cpp): denormals are underflow noise in the
  # hot loops and the slow-path handling is wasted work. NOT --use_fast_math (that would
  # also relax div/sqrt rounding); just the denormal flush.
  target_compile_options(sub0_backend_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>)
  target_link_libraries(sub0_backend_cuda PRIVATE CUDA::cudart CUDA::cublas)
  if(SUB0_TUNING)
    target_compile_definitions(sub0_backend_cuda PRIVATE SUB0_TUNING)
  endif()

  add_executable(sub0-cuda-selftest tools/cuda_selftest.cpp)
  target_link_libraries(sub0-cuda-selftest PRIVATE sub0_backend_cuda)
  set_target_properties(sub0-cuda-selftest PROPERTIES
    BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endif()

# --- Stage 4: backend stage libraries ---------------------------------------
add_library(sub0_train SHARED src/train_stage.cpp)
target_link_libraries(sub0_train PRIVATE sub0_core)

# Phase 2e: when the CUDA backend is built, link it into the train stage and flag the GPU
# training fast-path on. The train stage reaches the device through the extern "C" seam
# (sub0_cuda_train_step etc.); without this it transparently runs the CPU loop.
if(SUB0_BUILD_CUDA)
  target_link_libraries(sub0_train PRIVATE sub0_backend_cuda)
  target_compile_definitions(sub0_train PRIVATE SUB0_BUILD_CUDA)
endif()

# The train/models stages tag each model with the code version that produced it and lay
# them out under a models root. Pass both as compile definitions (configure-time) ONLY to
# this stage, so a new commit does not force an engine-core rebuild. Reconfigure to refresh.
execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" rev-parse --short HEAD
                OUTPUT_VARIABLE SUB0_GIT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET RESULT_VARIABLE _git_rc)
if(NOT _git_rc EQUAL 0 OR SUB0_GIT_SHA STREQUAL "")
  set(SUB0_GIT_SHA "nogit")
endif()
target_compile_definitions(sub0_train PRIVATE
  SUB0_GIT_SHA="${SUB0_GIT_SHA}"
  SUB0_MODELS_ROOT="${CMAKE_SOURCE_DIR}/models")

add_library(sub0_gen SHARED src/gen_stage.cpp)
target_link_libraries(sub0_gen PRIVATE sub0_core)

# --- Stage 5: the thin driver executable ------------------------------------
add_executable(sub0llm src/driver.cpp)
add_dependencies(sub0llm sub0_generate_config)          # for DEFAULT_CORPUS in the header
target_include_directories(sub0llm PRIVATE "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
target_link_libraries(sub0llm PRIVATE sub0_train sub0_gen CLI11::CLI11)

# Let the driver find the stage libs next to it without install/LD_LIBRARY_PATH.
set_target_properties(sub0llm PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

# --- Tests ------------------------------------------------------------------
option(SUB0_BUILD_TESTS "Build the unit test suite" ON)
if(SUB0_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
