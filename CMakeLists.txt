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

# --- Stage-1 "runtime arguments": passed through as cache variables ----------
set(SUB0_CORPUS   "${CMAKE_SOURCE_DIR}/data/tinystories.txt" CACHE FILEPATH "training corpus (drives vocab)")
set(SUB0_D_MODEL  96 CACHE STRING "embedding / residual width")
set(SUB0_N_LAYERS 3  CACHE STRING "transformer blocks")
set(SUB0_N_HEADS  4  CACHE STRING "attention heads")
set(SUB0_SEQ_LEN  64 CACHE STRING "context window")
set(SUB0_TERNARY  0  CACHE STRING "1 = BitNet-style ternary block weights")

# --- Stage 1: the configurator (host tool, depends on nothing generated) -----
add_executable(sub0-configure tools/configurator.cpp)
target_include_directories(sub0-configure PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0-configure PRIVATE CLI11::CLI11)

# --- Stage 2: code generation — run the configurator to emit the config ------
set(GEN_DIR    "${CMAKE_BINARY_DIR}/generated")
set(GEN_HEADER "${GEN_DIR}/sub0_config.hpp")
file(MAKE_DIRECTORY "${GEN_DIR}")
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
            -o "${GEN_HEADER}"
  DEPENDS sub0-configure "${SUB0_CORPUS}"
  COMMENT "Configuring: deriving vocab from ${SUB0_CORPUS} and baking constexpr config"
  VERBATIM
)
add_custom_target(sub0_generate_config DEPENDS "${GEN_HEADER}")

# --- Stage 3: engine core (shared), compiled against the generated header ----
add_library(sub0_core SHARED src/engine.cpp)
add_dependencies(sub0_core sub0_generate_config)
target_include_directories(sub0_core PUBLIC "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
if(OpenMP_CXX_FOUND)
  target_link_libraries(sub0_core PUBLIC OpenMP::OpenMP_CXX)
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
target_link_libraries(sub0llm PRIVATE sub0_train sub0_gen)

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
