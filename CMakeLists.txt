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

find_package(OpenMP)

# --- Stage-1 "runtime arguments": passed through as cache variables ----------
set(SUB0_CORPUS   "${CMAKE_SOURCE_DIR}/data/english.txt" CACHE FILEPATH "training corpus (drives vocab)")
set(SUB0_D_MODEL  96 CACHE STRING "embedding / residual width")
set(SUB0_N_LAYERS 3  CACHE STRING "transformer blocks")
set(SUB0_N_HEADS  4  CACHE STRING "attention heads")
set(SUB0_SEQ_LEN  64 CACHE STRING "context window")
set(SUB0_TERNARY  0  CACHE STRING "1 = BitNet-style ternary block weights")

# --- Stage 1: the configurator (host tool, depends on nothing generated) -----
add_executable(sub0-configure tools/configurator.cpp)

# --- Stage 2: code generation — run the configurator to emit the config ------
set(GEN_DIR    "${CMAKE_BINARY_DIR}/generated")
set(GEN_HEADER "${GEN_DIR}/sub0_config.hpp")
file(MAKE_DIRECTORY "${GEN_DIR}")
add_custom_command(
  OUTPUT  "${GEN_HEADER}"
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
