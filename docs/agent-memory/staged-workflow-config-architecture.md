---
name: staged-workflow-config-architecture
description: "Staged user-driven workflow (configure→tune→train→gen): configurator owns config, CMake checks CUDA existence; Stage 1 (bake facts) DONE, gating + exe-split PENDING; doc WORKFLOW_ARCHITECTURE.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Operationalizing the [[architecture-solidification-plan]] "thin tools" goal for the config/build side.
Design + decisions in **docs/WORKFLOW_ARCHITECTURE.md** (+ docs/CONFIGURE_ARCHITECTURE.md). Target: each
stage is a user-callable tool that writes config the *next build* bakes in as `constexpr`, so re-config
is a tool run, **not** a CMake reconfigure.

**Done (this session):** Unigram = default tokenizer; configurator AUTO-SIZES dims+vocab from corpus
(removed from CMake); split headers (sub0_corpus.hpp + sub0_system.hpp behind sub0_config.hpp umbrella);
`<corpus>.model` sidecar persists pins (precedence CLI > sidecar > auto-size); pure decisions in
`sub0::config` (config_util.hpp) unit-tested (200 engine-free assertions); **configurator DECIDES compute
(GPU when the CUDA backend was built, else CPU; --compute pins) while CMake only CHECKS CUDA existence**
(SUB0_CUDA_ARCH must stay CMake-side — it drives nvcc CUDA_ARCHITECTURES at configure time); **Stage 1**
= build facts baked into sub0-configure via `configure_file`→sub0_build_facts.hpp (device caps + output
paths), CMake command shrunk (dropped --has-cuda/--cuda-arch/--gpu-*/-o/--tune-cache).

**Also DONE since:** driver SPLIT into thin `sub0llm-{configure,train,gen,tune}` exes over one shared
runner set (include/sub0/cli_stages.hpp), diagnostics stay on the `sub0llm` umbrella; renamed
sub0-configure→sub0llm-configure; **SUB0_AUTO_CONFIGURE** option (default ON) to disable build-time
auto-config (OFF swaps the OUTPUT add_custom_command for a COMMAND-only target — gating just the
dependency was insufficient because the header is a custom-command OUTPUT); gpu_batch CLAMP (stale
over-budget batch clamps to memplan::max_batch_for_vram instead of hard-erroring). See
[[frontend-layer-testability-and-tooling]] for the next cleanup (frontend lib + tokenizer tool).

**Pending stages (WORKFLOW_ARCHITECTURE.md):** (3) extract a `sub0_frontend` lib
(config_util/memplan/casing/tokenizer/unigram/registry) — one named home + a `sub0_frontend_tests`
target; (stretch) strict EXISTS(GEN_HEADER)+CMAKE_CONFIGURE_DEPENDS gate so a fresh checkout builds only
the configurator first; state-1 frontend_cuda self-probe.

**Resolved forks:** configure_file ONLY for CMake-known facts (device/paths); keep C++ `sub0::config`
emit for the smart derivation (auto-size/precision/vocab-knee — configure_file can't compute it). One
configure binary (decision vs emission already layered). NO implicit stage-chaining (Windows exe-lock).

**Aside:** scripts/get_fineweb.py is COMMITTED, which conflicts with [[python-not-committed-cpp-library]]
(no .py committed) — it's a corpus-download helper; flag for the user to confirm keep-or-port.
