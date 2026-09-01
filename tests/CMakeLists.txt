# tests/CMakeLists.txt — unit test suite driven by Catch2 v3 (fetched via CPM).

CPMAddPackage("gh:catchorg/Catch2@3.7.1")

# Catch2's CMake helpers (catch_discover_tests) live in its extras/ dir.
list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
include(Catch)

# Spikes are exploratory and NOT meant to persist: a spike that succeeds has its finding merged into the
# mainline, after which the spike's own scaffolding is dead weight that still costs build time on every
# configuration and still has to be kept compiling through unrelated refactors. A spike that is parked
# does not need to keep being verified at all.
#
# OFF by default. Turn on to revive one:  cmake -DSUB0_BUILD_SPIKE_TESTS=ON ...
#
# Membership rule: a spike belongs here ONLY once nothing in src/ or tools/ includes its header. That is
# the objective test for "the finding was merged and the scaffolding is what is left". scratchspike and
# corpus_collapse deliberately do NOT qualify -- train_stage.cpp consumes both as blend sources, so their
# tests cover a production path and stay in the default build.
option(SUB0_BUILD_SPIKE_TESTS "Build tests for PARKED spikes whose findings are already merged" OFF)

# factspike: KV-trace memoization / piece-embedding transfer. Findings merged -- the marker-inclusive
# span replay it validated lives in corpus_collapse.hpp, which cites it. Nothing in src/ or tools/
# includes factspike.hpp. Parked 2026-07-26; see memory scratch-spikes-parked-2026-07-26 and
# docs/FACTSPIKE.md for what it established and how to resume it.
set(SUB0_SPIKE_TEST_SOURCES factspike_engine_tests.cpp)
set(SUB0_SPIKE_FRONTEND_TEST_SOURCES factspike_dataset_tests.cpp)
if(NOT SUB0_BUILD_SPIKE_TESTS)
  set(SUB0_SPIKE_TEST_SOURCES "")
  set(SUB0_SPIKE_FRONTEND_TEST_SOURCES "")
endif()

add_executable(sub0_tests
  tokenizer_tests.cpp
  engine_tests.cpp
  arch_identity_tests.cpp
  layout_tests.cpp
  tune_tests.cpp
  coherence_tests.cpp
  run_config_tests.cpp
  loss_mask_tests.cpp
  spellspike_engine_tests.cpp
  scratchspike_engine_tests.cpp
  repeatspike_engine_tests.cpp
  wordspike_engine_tests.cpp
  corpus_collapse_engine_tests.cpp
  ${SUB0_SPIKE_TEST_SOURCES}
  persistent_slots_engine_tests.cpp
  persistent_scratchspike_engine_tests.cpp
  pairspike_engine_tests.cpp
  sentinel_pair_engine_tests.cpp
  scratch_embed_tests.cpp
  arithspike_engine_tests.cpp
  nodespike_engine_tests.cpp
  chainspike_engine_tests.cpp
  node_frame_engine_tests.cpp
  scalar_reentry_engine_tests.cpp
  numeric_bind_engine_tests.cpp
  gsm8k_engine_tests.cpp
  blend_schedule_tests.cpp
  blended_capstone_engine_tests.cpp
)

# Tests compile against the engine API and the generated config header.
target_include_directories(sub0_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${GEN_DIR}")
target_link_libraries(sub0_tests PRIVATE sub0_core sub0_frontend Catch2::Catch2WithMain)

# Engine-free FRONTEND tests: link ONLY sub0_frontend + Catch2, NOT the engine. The tokenizer + the
# pre-model logic are config-independent, so this target builds and runs without sub0_core / the
# generated header -- fast to iterate, and unaffected when a training run locks the engine DLLs.
#   tok_lib_tests.cpp        — pinned tokenizer worked examples (one construct per assertion)
#   tok_roundtrip_fuzz.cpp   — generative net: dogfood the project's own source + random
#                              mutation/blob fuzz + a token-pattern report. SUB0_SOURCE_DIR
#                              lets it read the real tree for the dogfood/pattern cases.
#   tok_interchange_tests.cpp — WS7: to_interchange()'s struct-producing export shape
#   ws6_simd_differential_tests.cpp — WS6: scalar-vs-SIMD-rewrite byte/token-identical differential
#   spellspike_tests.cpp     — spelling-curriculum generator structure (engine-free; the model harness is in sub0_tests)
#   blend_tests.cpp          — weighted per-window corpus blender (engine-free; rng-neutrality + window bounds)
#   scratchspike_tests.cpp   — scratch-token (context-translation) curriculum structure (engine-free; harness in sub0_tests)
#   config_tests.cpp         — pure config decisions (sub0::config): autosize/tune-cache/precision/vocab-curve
#   frontend_tests.cpp       — memplan VRAM math (clamp + tuner cap) + registry identity/compat (prune)
#   gguf_tests.cpp           — GGUF header/KV/tensor-table parsing + Llama-MHA import-compat report
#   muon_tests.cpp           — Newton-Schulz orthogonalization math (the Muon optimizer's core primitive)
#   lr_schedule_tests.cpp    — the WSD learning-rate schedule's boundary conditions (sub0/lr_schedule.hpp)
#   casing_policy_tests.cpp  — HIDDEN [.casing]: measures the truecasing NAME policy against a real
#                              corpus slice from data/. Not run by default (slow, needs the corpus).
#   ngram_qwen4_fixture_tests.cpp — n-gram embeddings' "concat" convention cross-checked against REAL
#                              Qwen/Qwen3.8-Flash-Next weight values (tests/fixtures/qwen4_preview/,
#                              per docs/NGRAM_EMBEDDING.md sec 8's fixture-check note). Skips gracefully
#                              (WARN, not fail) if the fixtures are absent.
#   gdn_qwen4_fixture_tests.cpp — Gated DeltaNet Stage 1's primary correctness gate: sub0::gdn::forward
#                              (include/sub0/gdn_math.hpp, engine-free) checked against REAL extracted
#                              Qwen/Qwen3.8-Flash-Next GDN layer-0 weights + activations
#                              (tests/fixtures/qwen4_preview/gdn_layer0_small_*), plus mutation-style
#                              numerical property checks (docs/GATED_DELTANET.md sec 6). Skips gracefully
#                              if the fixtures are absent.
add_executable(sub0_frontend_tests tok_lib_tests.cpp tok_roundtrip_fuzz.cpp tok_interchange_tests.cpp ws6_simd_differential_tests.cpp spellspike_tests.cpp blend_tests.cpp scratchspike_tests.cpp config_tests.cpp frontend_tests.cpp gguf_tests.cpp muon_tests.cpp lr_schedule_tests.cpp nodes_tests.cpp gsm8k_tests.cpp op_curriculum_tests.cpp modality_tests.cpp casing_policy_tests.cpp ngram_qwen4_fixture_tests.cpp gdn_qwen4_fixture_tests.cpp ${SUB0_SPIKE_FRONTEND_TEST_SOURCES})
target_include_directories(sub0_frontend_tests PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_compile_definitions(sub0_frontend_tests PRIVATE SUB0_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(sub0_frontend_tests PRIVATE sub0_frontend Catch2::Catch2WithMain)
catch_discover_tests(sub0_frontend_tests)

# Eval-seam integration tests: the held-out NELBO evaluation driven end-to-end through a MOCK device
# backend, so the device dispatch (window batching, ids/targets pairing, group recombination) is
# covered on EVERY machine rather than only where a GPU happens to be present.
#
# Its own executable, and deliberately NOT part of sub0_tests: SUB0_BUILD_MOCK_DEVICE reroutes the
# inline sub0_dev_* seam to the mock, and sub0_tests both links sub0_backend_cuda and contains
# cuda_tests.cpp, which needs the real one. Two definitions of the same inline functions in one binary
# is an ODR violation, so they must stay separate binaries. For the same reason this target links
# sub0_core only -- never sub0_train, which is compiled against the real seam.
add_executable(sub0_eval_seam_tests eval_seam_tests.cpp mock_device_backend.cpp)
target_include_directories(sub0_eval_seam_tests PRIVATE
  "${CMAKE_SOURCE_DIR}/include"
  "${GEN_DIR}")
target_compile_definitions(sub0_eval_seam_tests PRIVATE SUB0_BUILD_MOCK_DEVICE=1)
target_link_libraries(sub0_eval_seam_tests PRIVATE sub0_core Catch2::Catch2WithMain)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sub0_eval_seam_tests PRIVATE OpenMP::OpenMP_CXX)
endif()
set_target_properties(sub0_eval_seam_tests PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")
catch_discover_tests(sub0_eval_seam_tests
  DL_PATHS "$<TARGET_FILE_DIR:sub0_core>"
  DISCOVERY_MODE PRE_TEST)

# GPU backend smoke test: compiled + linked only when the CUDA backend is built. The
# nvcc-built sub0_backend_cuda.dll sits next to sub0_core.dll; its CUDA runtime deps are
# covered by the toolkit bin added to DL_PATHS below.
if(SUB0_BUILD_CUDA)
  target_sources(sub0_tests PRIVATE cuda_tests.cpp)
  target_link_libraries(sub0_tests PRIVATE sub0_backend_cuda)
  # ...and compile the NEUTRAL seam (include/sub0/device_backend.hpp) in its device configuration.
  # Without this the target linked the CUDA backend while every sub0_dev_* in the header still
  # resolved to the no-device fail-fast stub, so anything driving a device through the seam rather
  # than through a hand-written extern "C" silently took the CPU fallback and "passed". Every other
  # test here already passes use_gpu=false explicitly, so nothing else changes behaviour.
  target_compile_definitions(sub0_tests PRIVATE SUB0_BUILD_CUDA)
endif()

# Find the sub0_core shared lib next to the test binary at runtime.
set_target_properties(sub0_tests PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")

# On Windows the loader has no rpath; DL_PATHS adds the needed DLL directories to the search
# path used during test discovery and execution: sub0_core.dll (and sub0_backend_cuda.dll) sit
# next to the test binary, and when the CUDA backend is built sub0_backend_cuda.dll additionally
# needs the CUDA runtime DLLs (cudart/cublas) from the toolkit bin.
#
# DISCOVERY_MODE PRE_TEST: enumerate the test cases at `ctest` time, NOT during the build. Running
# the CUDA-linked test exe at build time made every build depend on the CUDA runtime DLLs being
# loadable in the build shell's environment (a missing/unfindable cublas surfaced as a cryptic
# 0xc000007b and FAILED the build). Deferring to test time -- where DL_PATHS is applied -- keeps
# the build robust on any host.
set(_sub0_test_dl_paths "$<TARGET_FILE_DIR:sub0_core>")
if(SUB0_BUILD_CUDA AND CUDAToolkit_BIN_DIR)
  list(APPEND _sub0_test_dl_paths "${CUDAToolkit_BIN_DIR}")
endif()
catch_discover_tests(sub0_tests
  DL_PATHS ${_sub0_test_dl_paths}
  DISCOVERY_MODE PRE_TEST)
