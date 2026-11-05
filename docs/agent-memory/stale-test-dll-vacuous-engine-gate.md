---
name: stale-test-dll-vacuous-engine-gate
description: READ FIRST when citing a neutral engine-suite result from 2026-09-23..2026-10-07 — sub0_tests loaded a stale tests/sub0_core.dll, so those "exact" gates could not see engine changes
metadata:
  type: project
---
Found 2026-10-08: `out/build/d196check/tests/sub0_core.dll` was dated 2026-09-23. Windows loads a DLL from
the exe's own directory first, tests build into `tests/`, the DLL into the build root, and nothing copied
it. Every `sub0_tests` "29,510,661 / 147, fingerprints unchanged" between those dates (O5 2b-3, O8, O9,
O10, O11, O12, O14, the storage work) ran the 23 September engine. Symptom that exposed it: a new test
target needing new exports exited 127 with no output.

Outcome: re-run against the fresh DLL at `1013f12` gives the same counts and fingerprints, so nothing was
hiding. Fixed in `tests/CMakeLists.txt` (target `sub0_core_beside_tests`, a dependency of every test that
links `sub0_core`); verified by planting the stale DLL and rebuilding.

**How to apply:** a silent exit 127 from a test = DLL load failure, check `tests/*.dll` dates. When a gate
claims "unchanged", ask what binary it actually loaded. Same family as "a failed build leaves old binaries
runnable" ([[quality-gate-not-implied-by-correctness-gate]], [[caps-bit-nothing-reads-is-not-a-guard]]).
