cmake_minimum_required(VERSION 3.20)
project(sub0llm CXX)

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
    add_compile_options($<$<CONFIG:Release>:-march=native>)
    message(STATUS "SUB0_NATIVE: building for this machine's CPU (-march=native)")
  else()
    message(WARNING "SUB0_NATIVE requested but the compiler rejects -march=native; using a generic build")
  endif()
endif()

find_package(OpenMP)

# OpenMP drives the data-parallel training path. Without it the engine silently falls
# back to single-thread stubs and `tune`/`bench` show no scaling at all -- an easily
# missed regression -- so a missing OpenMP is a hard, explained error by default.
option(SUB0_REQUIRE_OPENMP "Fail configuration if OpenMP is unavailable (the data-parallel path needs it)" ON)

if(OpenMP_CXX_FOUND)
  set(SUB0_OPENMP_TARGET OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP: enabled via find_package (${OpenMP_CXX_FLAGS})")
else()
  # find_package's library probe is fragile with clang on Windows: it can fail to
  # locate libomp even though the compiler fully supports -fopenmp=libomp (which
  # auto-links the runtime). Probe the flag directly and drive it ourselves if it
  # works, so a flaky find_package no longer silently disables all parallelism.
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS "-fopenmp=libomp")
  check_cxx_source_compiles(
    "#include <omp.h>\nint main() { return omp_get_max_threads() > 0 ? 0 : 1; }"
    SUB0_OPENMP_FLAG_WORKS)
  unset(CMAKE_REQUIRED_FLAGS)
  if(SUB0_OPENMP_FLAG_WORKS)
    add_library(sub0_openmp INTERFACE)
    target_compile_options(sub0_openmp INTERFACE -fopenmp=libomp)
    target_link_options(sub0_openmp INTERFACE -fopenmp=libomp)
    set(SUB0_OPENMP_TARGET sub0_openmp)
    message(STATUS "OpenMP: find_package failed; using -fopenmp=libomp directly")
  elseif(SUB0_REQUIRE_OPENMP)
    message(FATAL_ERROR
      "OpenMP not found and '-fopenmp=libomp' does not compile.\n"
      "  The data-parallel training path REQUIRES OpenMP; without it the engine runs\n"
      "  single-threaded and `tune`/`bench` show no thread scaling (a silent perf\n"
      "  regression). Install an OpenMP runtime (libomp) for this toolchain and\n"
      "  reconfigure. To build single-threaded ON PURPOSE, pass -DSUB0_REQUIRE_OPENMP=OFF.")
  else()
    message(WARNING "OpenMP unavailable -- building SINGLE-THREADED (SUB0_REQUIRE_OPENMP=OFF).")
  endif()
endif()

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
set(SUB0_CORPUS   "${CMAKE_SOURCE_DIR}/data/tinystories.txt" CACHE FILEPATH "training corpus (drives vocab)")
set(SUB0_D_MODEL  96 CACHE STRING "embedding / residual width")
set(SUB0_N_LAYERS 5  CACHE STRING "transformer blocks")
set(SUB0_N_HEADS  4  CACHE STRING "attention heads")
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
            -o "${GEN_HEADER}"
  DEPENDS sub0-configure "${SUB0_CORPUS}" "${GEN_TUNE_CACHE}"
  COMMENT "Configuring: deriving vocab from ${SUB0_CORPUS} and baking constexpr config"
  VERBATIM
)
add_custom_target(sub0_generate_config DEPENDS "${GEN_HEADER}")

# --- Stage 3: engine core (shared), compiled against the generated header ----
add_library(sub0_core SHARED src/engine.cpp)
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

# --- Stage 4: backend stage libraries ---------------------------------------
add_library(sub0_train SHARED src/train_stage.cpp)
target_link_libraries(sub0_train PRIVATE sub0_core)

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
