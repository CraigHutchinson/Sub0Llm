cmake_minimum_required(VERSION 3.25)

project(Sub0LLM
    VERSION 0.1.0
    DESCRIPTION "LLMs from Scratch in C++23 — tensor foundations to RLHF"
    HOMEPAGE_URL "https://github.com/craighutchinson/sub0llm"
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── C++23 ─────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Default build type ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
message(STATUS "sub0llm: Build type — ${CMAKE_BUILD_TYPE}")

# ── Compile commands for tooling (clangd, etc.) ───────────────────────────────
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Project modules ───────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(options)
include(compiler_flags)
include(CPM)
include(deps)

# ── Library & chapters ────────────────────────────────────────────────────────
add_subdirectory(src)

# enable_testing() must precede every add_subdirectory() that registers tests
# (diffusion/tests would otherwise emit no CTest entries).
if(BUILD_TESTING)
    enable_testing()
endif()

if(SUB0LLM_BUILD_CHAPTERS)
    add_subdirectory(chapters)
endif()

# ── Diffusion-LM track (Ch28+) ────────────────────────────────────────────────
# A separate specialized volume that links libsub0llm and builds the text-diffusion
# (DiffusionGemma-style) model around compile-time monomorphization.
if(SUB0LLM_BUILD_DIFFUSION)
    add_subdirectory(diffusion)
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# ── Tools (CLI + server) ──────────────────────────────────────────────────────
if(SUB0LLM_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
add_subdirectory(benchmarks)

# ── Install ───────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
