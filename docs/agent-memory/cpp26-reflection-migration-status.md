---
name: cpp26-reflection-migration-status
description: "MEASURED 2026-07-28 — codebase already compiles+passes at -std=c++26 (only gguf regresses); reflection itself is blocked on toolchain, not on us"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-07-28T14:23:04.064Z
---

User priority (2026-07-28): move to a C++26-capable compiler to get **P2996 static reflection** and
delete the serialization boilerplate ([[json-write-simdjson-and-cpp26-reflection]],
[[compile-time-decisions-preferred]]).

## Measured facts, not recalled ones

**RESOLVED 2026-07-28: the repo is fully green at `-std=c++26`** (`8a01fff`). The 68 gguf failures
were NOT a C++26 regression -- they were a real dangling-span bug present since the file was written.
`Reader` is non-owning, but 14 test sites constructed it from a temporary (`Reader r(b.finish());`);
the freed pages happened to still hold the bytes under C++23. Fixed at the class:
`Reader(std::vector<std::uint8_t>&&) = delete`. **Lesson: a "new standard broke X" symptom is more
often latent UB the new codegen exposed than a language change.** The tell was that `tensor_bytes`
returned the right SIZE from the wrong memory, so size assertions passed and only values were wrong.

**The language version is NOT the blocker.** A full clang build at `-DCMAKE_CXX_STANDARD=26`
(clang 22.1.6, Ninja) is green on both suites: engine 1,774,719 / 125 and frontend 113,132 / 155 at
the d96/english.txt config. Nothing else in the codebase needed changing. C++23 remains the
committed default -- the standard bump itself is still to do (see plan below).

**Reflection is unavailable on this toolchain.** Verified directly, not assumed:
- `clang++ -freflection` → `unknown argument`.
- `#include <meta>` → `'meta' file not found`; no `<meta>` anywhere on the system.
- `^^S` parses as **Objective-C blocks** syntax, not the reflection operator.

**Ecosystem, mid-2026**: P2996 was adopted into C++26 at the Sofia 2025 WG21 meeting. GCC trunk
implements most of it. Upstream Clang has NOT integrated it yet. MSVC has no public support.
Bloomberg's `clang-p2996` fork is the most complete implementation and its own README says
**"DO NOT use this project to build any artifacts destined for production"** — highly experimental,
crashes expected, memory use unoptimised.

## The platform constraint that decides the path

This project is Windows + CUDA, clang targeting `x86_64-pc-windows-msvc`. **The GCC-trunk path is
not open**: nvcc's host compiler on Windows must be MSVC or clang-cl, so following GCC would mean
moving CUDA builds to Linux/WSL — a far larger change than the boilerplate it would save.

## Staged plan

1. **Fix the gguf C++26 float-decode regression** (1 file, 68 assertions). Do this first — it is
   real work with value regardless of reflection, and it is the only thing standing between the repo
   and a clean C++26 build.
2. **Bump `CMAKE_CXX_STANDARD` to 26** once (1) is green on both backends. Cheap, removes the
   language-version question from the critical path, and stops C++23-era idioms accreting.
3. **Prototype reflection against `clang-p2996` in a throwaway build dir only** — EVALUATION, never
   production, per its own README. Target exactly one thing: replacing `SUB0_RUN_CONFIG_FIELDS` +
   `write_config_json`/`read_config_json` with `nonstatic_data_members_of(^^RunConfig)`. That proves
   the design and sizes the win before anything is committed to it.
4. **Adopt when upstream clang ships P2996.** Migration stays localized because serialization already
   funnels through the one X-macro.

## The actionable finding, available TODAY with no toolchain change

The spike's real conclusion is not "reflection replaces the X-macro". It is that **the X-macro does
not cover `write_state`/`read_state`** -- 9 fields hand-listed in TWO places with no compiler check
that they agree. That is precisely the shape of the meta.txt drift bug already eaten once
([[roadmap-2026-07-21]] §7c). **Give `RunState` its own X-macro exactly as `RunConfig` has** -- a
same-day change on the current compiler, and it captures the entire correctness win.

Reflection then reduces to readability + queryability (`field_count()` as a `static_assert`, P3394
annotations replacing the name-keyed `enum_name`/`enum_parse` tables). Worth taking when upstream
Clang ships P2996; **not** worth building an experimental LLVM fork for.

**Do not block other work on this.**
