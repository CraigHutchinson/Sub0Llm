---
name: json-write-simdjson-and-cpp26-reflection
description: "Backlog - extend simdjson to config.json WRITES too (not just reads), and investigate a C++26 reflection toolchain to replace registry.hpp's hand-rolled enum_name/enum_parse boilerplate"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

`include/sub0/registry.hpp`'s `RunConfig` JSON path is currently split: `src/run_config.cpp`'s READ side uses simdjson (`ondemand`, single forward pass, per [[json-reads-forward-ondemand-not-dom-facade]]), but the WRITE side (`detail::json_write`/`json_write_field` in registry.hpp) is hand-rolled `ostream <<` code. User asked for a follow-up task (raised while working on the content-embed RunConfig field, 2026-07-15) to look at using simdjson more fully for JSON read/write across the board, since it's more performant — do the write side too, not just reads.

Separately, user raised a second, related idea: `detail::enum_name`/`enum_parse` (the `{"off","on"}`/`{"adamw","muon"}` mapping tables in registry.hpp) are hand-rolled boilerplate that C++26 reflection (`std::meta`, once compilers ship it) could likely generate automatically from the `RunConfig` struct/enum definitions instead. This may need a prerequisite task: get set up with a C++26 toolchain (trunk/nightly build of a compiler with reflection support, since it's not in a stable release as of this session) specifically to evaluate this.

**How to apply**: don't start either of these speculatively — they're both real but not urgent. Good timing: after the scratch-token content-embed work finishes (`registry.hpp`'s `SUB0_RUN_CONFIG_FIELDS` X-macro is actively being extended right now, so mid-flight is the wrong time to also refactor its plumbing). When picked up: (1) simdjson-write is the safer, smaller first step (a `simdjson`-side serializer or a `simdjson_builder`-style API if that project has one by then; verify current simdjson version's write-support maturity, since older simdjson focused purely on parsing) — check before assuming; (2) the C++26/reflection idea is exploratory and toolchain-dependent — worth a quick feasibility check (does any mainstream compiler ship usable reflection support yet) before committing to a nightly-toolchain migration, since that's a much bigger, riskier lift than the write-side change and could destabilize the whole build for an uncertain payoff.
