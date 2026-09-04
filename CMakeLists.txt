cmake_minimum_required(VERSION 3.28...4.2)
project(sub0llm VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# sub0llm is a staged build. The model's dimensions and vocabulary are compiled
# in as constexpr constants, so the configuration is generated *during the build*
# and the engine is compiled against it:
#
#   sub0llm-configure (stage 1, config-independent)
#        | reads the corpus + the SUB0_* args below
#        v
#   generated/sub0_config.hpp   (constexpr dims + corpus-derived vocab)
#        |
#        v
#   libsub0_core (shared)   --->  libsub0_train (shared)  \
#                           --->  libsub0_gen   (shared)   >--> sub0llm (driver)
# ---------------------------------------------------------------------------

# C++26, adopted deliberately ahead of full compiler support: this project tracks the newer standard
# and refactors toward it, rather than settling on C++23 and migrating later under pressure. The
# whole tree builds and both suites pass at -std=c++26 on clang today (verified at the bump: engine
# 1774719/125, frontend 113132/155). Reflection (P2996) is the eventual prize -- see
# spikes/reflect/ -- but upstream clang has not shipped it yet, so nothing here depends on it.
#
# IMPORTANT, and the one real constraint this creates: the CUDA translation unit cannot follow.
# nvcc forces its own `--std` onto the HOST compiler for the unit it drives, and on THIS platform
# that dialect tops out at c++20 -- measured, not assumed:
#
#     nvcc 13.2.78 -std=c++23  ->  nvcc fatal: Value 'c++23' is not defined for option 'std'
#
# and identically with `-ccbin clang++`, so it is nvcc's own baked-in list, not the host compiler's
# capability. NVIDIA's C++ language-support appendix does document `--std=c++23`, but its own
# requirements table qualifies it: GCC >= 14 / Clang >= 18 / nvc++ >= 24.3, and "Microsoft Visual
# Studio is not supported for C++23". That is a Linux-host statement; Windows nvcc does not offer it.
# Re-test the two lines above after a CUDA upgrade rather than trusting either this comment or the
# docs -- when it starts compiling, raise CUDA_STANDARD and delete this paragraph.
#
# So every header reachable from src/backend_cuda.cu -- layout.hpp, core.hpp, memplan.hpp and what
# they pull in -- must stay C++20-compatible. Put C++23/26-only constructs in .cpp files, or in
# headers the .cu never sees. A violation surfaces as a confusing nvcc host-compilation error deep
# in a header, not a clean "unsupported feature" message, so it is worth knowing the rule up front.
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
# Emit compile_commands.json for clangd / editor tooling (harmless for Ninja/Makefiles).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -g alongside -O3: RelWithDebInfo-style Release -- full optimization, but every Release build also
# emits CodeView debug info (a .pdb next to each DLL/EXE via lld-link), so a profiler (VTune, WPA) can
# resolve real function names instead of `func@0xADDRESS` without a separate Debug build or reconfigure.
# Debug info doesn't change the generated machine code, just link time and on-disk .pdb size.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

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

# JSON: simdjson (Apache-2.0), for config.json (include/sub0/registry.hpp's RunConfig --
# read_config_json in src/run_config.cpp). Single-pass ondemand reads only (this project's JSON
# convention: register a handler per key, one forward pass -- no DOM tree), matching how sparse
# these config files are. Writing is hand-rolled inline in registry.hpp (no library needed to emit
# a flat object of scalars) -- simdjson is a parser, not a serializer.
CPMAddPackage("gh:simdjson/simdjson@3.10.1")

# TODO(tune): the `tune` autotuner uses an in-tree search (include/sub0/tune.hpp).
# Prior art surveyed -- the mature auto-tuning stacks (Optuna, NNI, scikit-optimize,
# Vizier, FLAML) are Python; the C++ options are biteopt (header-only derivative-free
# global optimiser), CMA-ES, NLopt (DIRECT) and dlib's find_max_global. None were
# pulled: our space is tiny, discrete and low-cardinality, where a deterministic coarse
# grid + basin refinement is both near-optimal and -- unlike the stochastic optimisers
# -- unit testable, and none of them supply the Knob/Objective framework we actually
# want. Escalate to biteopt (gh:avaneev/biteopt, header-only) only if the search space
# becomes continuous / high-dimensional.

# --- Stage-1 "runtime arguments" -------------------------------------------
# The corpus, model dims, positional encoding, precision, and pretokenize-vs-on-demand choice are
# ALL sub0llm-configure's own CLI flags now (--corpus/--dmodel/--pos-encoding/--bf16/--prec-gemm/
# --prec-act/--corpus-pretok, each with its own sensible default) -- run it directly
# (`sub0llm-configure --corpus <c> ...`, or scripts/workflow.ps1), not through CMake. The staged
# workflow (see docs/WORKFLOW_ARCHITECTURE.md) never orchestrates that step, so there is nothing
# here to mirror it into; a CMake cache variable would only invite two sources of truth to drift.
#
# SUB0_TERNARY is the one exception: cmake/Backends.cmake ALSO reads it directly (ternary is
# CPU-only for now, a real build-time backend-selection constraint independent of the configurator).
set(SUB0_TERNARY  0  CACHE STRING "1 = BitNet-style ternary block weights")

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
# --- Frontend library: the engine-free, pre-model layer (config-free) -------
# The shared "frontend": the truecasing + Unigram/BPE tokenizer (corpus scan, vocab learning,
# encode/detokenize, (de)serialize) compiled here, plus the header-only pre-model logic exposed via the
# PUBLIC include dir -- config_util (auto-size/precision/tune-cache), memplan (device-memory planning),
# registry (model identity/compat), casing. Factored out so BOTH the configurator and the runtime engine
# share ONE definition, and so the engine-free tests cover it without building the engine. Depends only
# on the standard library (no generated config), so the configurator links it without the engine.
add_library(sub0_frontend STATIC src/tokenizer.cpp src/unigram.cpp src/tok_interchange.cpp src/run_config.cpp)
target_include_directories(sub0_frontend PUBLIC "${CMAKE_SOURCE_DIR}/include")
# run_config.cpp is registry.hpp's out-of-line half (read_config_json / read_state, both simdjson).
# It belongs HERE, not in sub0_core: it includes only registry.hpp + simdjson -- no engine, no
# generated config -- and registry is named above as frontend surface precisely so "the engine-free
# tests cover it without building the engine". While it sat in sub0_core, sub0_frontend_tests could
# only reach registry's *header-only* half, which is why the readers had to stay inline in the header.
target_link_libraries(sub0_frontend PRIVATE simdjson::simdjson)
set_target_properties(sub0_frontend PROPERTIES POSITION_INDEPENDENT_CODE ON)
# WS6: casing.hpp's normalize_text/truecase_tokenize use `#pragma omp simd` (this project's
# existing auto-vectorization convention, see src/backend_cpu.cpp) over their hot per-byte classify
# loops. `sub0_frontend` never links the full OpenMP runtime (only sub0_core does, for the
# data-parallel training path -- see cmake/OpenMP.cmake) -- without at least `-fopenmp-simd`, those
# pragmas are silently inert (ignored, not an error) and the loops stay plain scalar. This flag
# enables ONLY simd-pragma recognition, no threading runtime, so it's free to add here regardless
# of whether the full OpenMP probe above succeeded.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp-simd" SUB0_HAS_OPENMP_SIMD_FLAG)
if(SUB0_HAS_OPENMP_SIMD_FLAG)
  target_compile_options(sub0_frontend PRIVATE -fopenmp-simd)
endif()

add_executable(sub0llm-configure tools/configurator.cpp)
target_include_directories(sub0llm-configure PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0llm-configure PRIVATE CLI11::CLI11 sub0_frontend)

# WS7: engine-free tokenizer inspection/interchange (docs/WORKFLOW_ARCHITECTURE.md's "Tokenizer /
# vocab as engine-free frontend tools" section) -- links sub0_frontend only, same as
# sub0llm-configure above, never the engine. Config-independent (no GEN_DIR dependency), unlike
# sub0llm-configure below, since it only reads an already-built tokenizer.tok, not a corpus.
add_executable(sub0llm-tokenizer tools/sub0llm-tokenizer.cpp)
target_include_directories(sub0llm-tokenizer PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(sub0llm-tokenizer PRIVATE CLI11::CLI11 sub0_frontend)

# WP4c: the offline weight transplant (docs/WP4_SCOPE.md S4). Unusual in exactly one way, and the
# reason is worth stating at the target rather than only in the source: it is compiled against
# tests/qwen4_real_axes/sub0_config.hpp -- Qwen3.8-Flash-Next's REAL RunConfig axes, with N_LAYERS
# forced to 4 for the real repeating sub-stack (S7 Q3) -- INSTEAD of ${GEN_DIR}'s generated config. So
# it links no engine library: sub0_core is compiled against the build's own config, and two definitions
# of sub0::PARAM_LAYOUT in one binary is an ODR violation. Same reasoning, and the same shape, as the
# sub0_qwen4_shape_tests target in tests/CMakeLists.txt.
#
# The axes header is SHARED with that test rather than copied here: the real model's RunConfig has one
# source of truth or it has none. sub4_prefix.hpp (same directory) carries the two literals both
# targets assert against, which is what makes "this 4-layer layout IS the real model's first four
# layers" a compile-time claim instead of an argument.
add_executable(sub0llm-transplant tools/sub0llm-transplant.cpp)
target_include_directories(sub0llm-transplant PRIVATE
  "${CMAKE_SOURCE_DIR}/tests/qwen4_real_axes"
  "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(sub0llm-transplant PRIVATE SUB0_QWEN4_LAYERS=4)
target_link_libraries(sub0llm-transplant PRIVATE CLI11::CLI11)
# 512 experts x 4 layers makes PARAM_LAYOUT a ~6,200-tensor consteval walk, done several times over
# (make_param_layout, DECAY_RANGES, MUON_RANGES, MUON_MAX_MN/MM, total_param_floats) -- past clang's
# default 1,048,576-step budget. Same flag, same reason, as sub0_qwen4_shape_tests.
if(NOT MSVC)
  target_compile_options(sub0llm-transplant PRIVATE -fconstexpr-steps=1000000000)
endif()

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

# Bake the build-known facts (detected device caps + the generated-output paths) INTO sub0llm-configure,
# so the command line below stops mirroring them and the tool is self-sufficient when run directly.
# These become the CLI defaults; each stays overridable. See docs/WORKFLOW_ARCHITECTURE.md (Stage 1).
configure_file("${CMAKE_SOURCE_DIR}/cmake/sub0_build_facts.hpp.in" "${GEN_DIR}/sub0_build_facts.hpp" @ONLY)
target_include_directories(sub0llm-configure PRIVATE "${GEN_DIR}")

# --- Stage 3: engine core (shared), compiled against the generated header ----
# Two translation units, one shared library: engine_core.cpp is the backend-agnostic
# part (tokenizer, serialization, sampler, config paths) and backend_cpu.cpp is the
# CPU compute backend (ops, autograd, train_batch, AdamW). A future GPU backend
# replaces backend_cpu.cpp behind the same include/sub0/core.hpp API.
add_library(sub0_core SHARED src/engine_core.cpp src/blend_schedule.cpp ${SUB0_BACKEND_SOURCES})
target_include_directories(sub0_core PUBLIC "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
# The runtime tokenizer (encode/detokenize/load) is the same sub0_frontend used by the
# configurator, linked PRIVATE so its symbols stay internal to the engine DLL.
target_link_libraries(sub0_core PRIVATE sub0_frontend)
# blend_schedule.cpp's parse_blend_schedule_json (blend_schedule.hpp's ScheduleSpec) uses simdjson to
# parse blend_schedule.json. PRIVATE: consumers (train_stage.cpp etc.) only ever call the plain
# std::-typed functions declared in their headers -- simdjson's own types never leak across that boundary.
# (registry.hpp's own readers moved to sub0_frontend with run_config.cpp -- see above.)
target_link_libraries(sub0_core PRIVATE simdjson::simdjson)
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
  # Suppress nvcc diag #20280-D on embed_add_act_kernel<__nv_bfloat16>: it warns that a __global__
  # template instantiation "must have a definition in the current translation unit" under whole-
  # program compilation (-rdc=false, our mode -- this project is a single .cu file with no cross-TU
  # device linking). Verified benign: the sole instantiation point (backend_cuda.cu, forward_train)
  # is in the SAME translation unit as the definition, so the condition the warning describes is
  # already satisfied; the kernel runs correctly (parity-tested by the forward/backward CUDA tests).
  # nvcc's own suggested fix for a genuinely benign case is -static-global-template-stub=false or
  # this diag suppression -- picked the latter since it is scoped to exactly this advisory.
  target_compile_options(sub0_backend_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe;--diag_suppress=20280>)
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
# sub0_frontend (the tokenizer) links in DIRECTLY, not just transitively via sub0_core (which links
# it PRIVATE, so its symbols aren't re-exported): --spell-mix deserializes tokenizer.tok into a raw
# sub0::tok::Tokenizer and generates the uncombine curriculum (sub0/spellspike.hpp) as a blend source.
# It's a static PIC lib, so this embeds its own copy in sub0_train.dll -- the stateless tokenizer
# functions have no cross-DLL shared state, so a second copy is harmless.
target_link_libraries(sub0_train PRIVATE sub0_core sub0_frontend)
# MSVC's CRT headers flag std::getenv deprecated (favoring _dupenv_s); the two SUB0_TRAIN_CPU /
# SUB0_FIXED_SEQ toggles are simple read-only env lookups with no buffer to overflow, so the
# "insecure" warning has nothing to protect against here -- suppress it rather than take on a
# Windows-only API for zero safety benefit.
target_compile_definitions(sub0_train PRIVATE _CRT_SECURE_NO_WARNINGS)

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
# sub0_frontend links in directly (as with sub0_train above): a model trained with --spell-mix drives
# gen's uncombine/combine interceptor, which deserializes tokenizer.tok into a raw sub0::tok::Tokenizer
# to build the expand/combine callbacks the decode loop calls.
target_link_libraries(sub0_gen PRIVATE sub0_core sub0_frontend)

# Phase 2e follow-up: GPU KV-cache decode (sub0_cuda_forward_one), parity-validated against a trained
# model (cuda_tests.cpp) -- link it in the same way sub0_train links the training fast-path above.
if(SUB0_BUILD_CUDA)
  target_link_libraries(sub0_gen PRIVATE sub0_backend_cuda)
  target_compile_definitions(sub0_gen PRIVATE SUB0_BUILD_CUDA)
endif()

# --- Stage 5: the thin driver executable ------------------------------------
add_executable(sub0llm src/driver.cpp)
target_include_directories(sub0llm PRIVATE "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
target_link_libraries(sub0llm PRIVATE sub0_train sub0_gen CLI11::CLI11)

# Let the driver find the stage libs next to it without install/LD_LIBRARY_PATH.
set_target_properties(sub0llm PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

# --- Stage 5b: the per-stage executables (thin mains over the shared runners) -
# Each is an OPTIONAL, separately-buildable stage tool that reuses the same sub0cli::run_* runner the
# umbrella `sub0llm` dispatches to (cli_stages.hpp) -- so arg parsing + stage dispatch are defined once.
# Each compiles against whatever config header already exists in ${GEN_DIR}: run
# `sub0llm-configure --corpus <c>` first (see docs/WORKFLOW_ARCHITECTURE.md) -- a missing header is a
# clear compile error, not a silent auto-regenerate.
foreach(_stage train gen tune)
  add_executable(sub0llm-${_stage} tools/sub0llm-${_stage}.cpp)
  target_include_directories(sub0llm-${_stage} PRIVATE "${CMAKE_SOURCE_DIR}/include" "${GEN_DIR}")
  target_link_libraries(sub0llm-${_stage} PRIVATE CLI11::CLI11)
  set_target_properties(sub0llm-${_stage} PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
endforeach()
# train + tune live in the train stage lib; gen in the gen stage lib.
target_link_libraries(sub0llm-train PRIVATE sub0_train)
target_link_libraries(sub0llm-tune  PRIVATE sub0_train)
target_link_libraries(sub0llm-gen   PRIVATE sub0_gen)

# --- Tests ------------------------------------------------------------------
option(SUB0_BUILD_TESTS "Build the unit test suite" ON)
if(SUB0_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# --- Benchmarks (opt-in) ----------------------------------------------------
# First-class microbenchmarks for the optimization work: statistical timing (warmup + repetitions +
# mean/stddev) instead of ad-hoc printf. Uses Catch2's benchmark support (already a project dep and it
# builds cleanly on this clang-on-Windows toolchain -- Google Benchmark's cmake injects MSVC /W4 /MP
# flags into clang++ and mis-probes std::regex, so it is not viable here). Host/CPU hot paths only;
# device (CUDA) throughput stays on the cudaEvent-based sub0-cuda-selftest. Opt-in. See benchmarks/.
option(SUB0_BUILD_BENCHMARKS "Build the microbenchmark suite (Catch2 benchmarks)" OFF)
if(SUB0_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
