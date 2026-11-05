---
name: simdjson-migration-backlog
description: Backlog — migrate all nlohmann/json reads to simdjson; drop nlohmann once nothing needs it
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Backlog (user, 2026-06-17): migrate every nlohmann/json use that simdjson can replace over to simdjson, and once nothing in the tree still needs nlohmann, DROP the nlohmann CPM dependency entirely. simdjson was added in the Ch29 config module (commit a441df7) and is already the parser there; nlohmann remains for the legacy JSON *writers* and DOM reads elsewhere.

**Why:** one JSON library, the fast SIMD one; less build surface and dependency weight. **How to apply:** simdjson is parse-only (no emit), so each migration must pair a simdjson reader with either the config module's dependency-free streaming writer or a small hand writer. Audit `nlohmann` / `json.hpp` uses (checkpoint headers `src/nn/checkpoint.cpp`, tokenizer vocab `src/tokenizer/bpe.cpp`, model `config.json` in `diffusion/src/model_io.cpp` + `tools/cli`/`tools/server`, `text_corpus`, diffusion recovery). Reuse the `sub0llm::config::JsonDoc` facade pattern (`include/sub0llm/config/json.hpp`) for reads. Remove the `CPMAddPackage(nlohmann_json ...)` in `cmake/deps.cmake` + the link in `src/CMakeLists.txt` only after the last include is gone. Relates to the config-module spine [[diffusion-nelbo-ceiling-reframe]].

**Ch30 config-module rework (user, 2026-06-17, concrete instance).** `diffusion/src/model_io.cpp::load_model_dir` reads ONLY `config.json` (model dims, via nlohmann) — it predates the Ch29 config module and ignores the `run_config.json` the trainer now writes. Verified Ch30 (`ch30_iterative_sampling`) still LOADS the current checkpoint fine via this legacy path (V=514/T=64, step 318860), so no breakage — but to "fully update" Ch30: make `load_model_dir` consume `run_config.json` via the RunConfig module (`sub0diff::config`, the BuildTime fields are the authoritative arch) with a `config.json` fallback for old dirs, dropping the duplicated nlohmann arch parse. Small, file-local to model_io.cpp; sequence it so the native/debug rebuild doesn't collide with the active Stage-4 session's working tree. Part of the config-as-single-source-of-truth goal ([[specialization-backend-roadmap]]).
