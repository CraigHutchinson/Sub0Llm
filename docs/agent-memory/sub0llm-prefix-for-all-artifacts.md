---
name: sub0llm-prefix-for-all-artifacts
description: User rule 2026-10-08 — every Sub0Llm library, DLL, test binary and CMake target carries the `sub0llm` prefix, never a bare `sub0_`; other Sub0 libraries should follow their own project prefix
metadata:
  type: feedback
---
"The library files should all be sub0llm prefix": `sub0_core.dll` is ambiguous and would collide if this
project were integrated beside another Sub0 library that made the same mistake. Renamed 2026-10-08:
`sub0llm_core`, `sub0llm_frontend`, `sub0llm_train`, `sub0llm_gen`, `sub0llm_backend_cuda`,
`sub0llm_openmp`, test binaries `sub0llm_tests`, `sub0llm_frontend_tests`, `sub0llm_*_tests`,
`sub0llm-cuda-selftest`.

**Why:** file names and CMake target names are global once two projects share a build or a bin directory.
**How to apply:** name any new target/artifact `sub0llm_*` (tools `sub0llm-*`). Older memory notes and the
work log still say `sub0_tests` / `sub0_core.dll`: read those as the new names. The `sub0::` namespace and
`SUB0_*` macros were not part of this change. Applies by analogy to Sub0MemPage, Sub0TieredCache, Sub0Pub
etc. (each uses its own project prefix). See [[sub0-camelcase-and-doxygen-interface-docs]].
