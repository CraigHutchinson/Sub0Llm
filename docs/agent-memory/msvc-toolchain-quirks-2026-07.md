---
name: msvc-toolchain-quirks-2026-07
description: "MSVC 19.51 (VS \"18\") toolchain regressions found while building the factspike96 config — omp for needs a signed loop var, OpenMP flag auto-detection is wrong, -march=native fails"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

Found while standing up a brand-new CMake config (`factspike96`) from scratch on 2026-07-19:

- **MSVC 19.51 hard-errors (`C3016`) on an UNSIGNED loop variable in `#pragma omp for`** — new/stricter
  than before. `#pragma omp simd` is more lenient: MSVC accepts `size_t` there, it just silently drops
  `reduction` clauses with a warning instead of erroring. Fixed a REAL, project-wide instance in
  `src/backend_cpu.cpp` (~line 1374, the gradient-reduction loop) by switching the loop var to
  `std::ptrdiff_t` with `static_cast<std::size_t>` at each array index. **Any new `#pragma omp for` loop
  in this codebase must use a signed loop variable** — this is a real compiler-strictness constraint, not
  a style choice.
- **MSVC 19.51's `find_package(OpenMP)` resolves to plain `-openmp` (2.0), not `-openmp:experimental`** —
  breaks `#pragma omp simd` (used in casing.hpp, muon.hpp, backend_cpu.cpp) with `error C7660: 'simd':
  requires '-openmp:experimental'`. Workaround used (deliberately scoped to this one config, NOT a
  project-wide `cmake/OpenMP.cmake` fix): explicit CMake cache override on reconfigure —
  `-DOpenMP_CXX_FLAGS=-openmp:experimental -DOpenMP_CXX_LIB_NAMES=`. This is a real, unresolved
  project-wide gap for any brand-new MSVC config — existing configs (`d196check`, `gsm8k`, `tok_cpu`, etc.
  — these actually build with **Clang**, not MSVC, per 2026-07-19 verification) were never exposed to it.
- **`-march=native` silently failed to take effect** for this from-scratch MSVC reconfigure even with a
  clean system+user PATH (confirmed reproducible, not environment pollution from an unrelated earlier
  mistake) — build falls back to generic/no-native-SIMD. Root cause NOT investigated, flagged only.

How to apply: any new from-scratch CMake config that ends up using **MSVC** (not the Clang toolchain most
existing configs on this machine use) should expect to need the OpenMP cache-flag override above, and
should verify native SIMD actually got enabled rather than assuming `-march=native` silently worked. See
also [[vs-install-path-18-not-2022]] for the correct MSVC install path on this machine.
