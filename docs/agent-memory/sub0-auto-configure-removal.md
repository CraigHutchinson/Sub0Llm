---
name: sub0-auto-configure-removal
description: SUB0_AUTO_CONFIGURE + sub0_generate_config CMake orchestration removed entirely (not just defaulted off); sub0llm-configure is now always run directly. Cleanup completed + verified 2026-07-03.
metadata:
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

The user had already started removing `SUB0_AUTO_CONFIGURE` (the CMake option that used to make
`cmake --build` transparently regenerate config headers via a `sub0_generate_config` custom target)
from `CMakeLists.txt` before this session picked it up; asked to "review the changes in the main
working tree... and cleanup the peripheral codes."

**What changed:** all CMake-side config orchestration is gone, not just defaulted off. Removed CMake
cache variables that only existed to seed the now-deleted `sub0_generate_config` command line:
`SUB0_CORPUS`, `SUB0_POS_ENCODING`, `SUB0_ROPE_THETA`, `SUB0_BF16`, `SUB0_PRECISION_GEMM`,
`SUB0_PRECISION_ACT`, `SUB0_CORPUS_PRETOK` (its AUTO-by-corpus-size-vs-RAM heuristic moved into
`sub0llm-configure` itself as `sub0::config::should_pretokenize`, see `config_util.hpp`).
`SUB0_TERNARY` is the ONE surviving CMake cache variable -- kept because `cmake/Backends.cmake` also
reads it directly for a real build-time backend-selection constraint (ternary is CPU-only; a GPU
build with `SUB0_TERNARY=ON` must fail at configure time, before nvcc ever runs).

**End state**: `sub0llm-configure --corpus <c> [--dmodel N ...]` is a plain executable with no
generated-header dependency, so it always builds first. The engine + stage-tool targets are
unconditionally defined (no `if(EXISTS ...)` gate) and simply fail to compile with a plain "file not
found" on the generated header until configure has actually been run once -- deliberately: a
CMake-tracked dependency on a file written by a process CMake doesn't orchestrate could only notice
staleness via mtime anyway, which the normal header-dependency scan already does for free. Why this
is the end state and not a defaults flip: a large-corpus reconfigure triggered silently by an
unrelated `cmake --build` was hard to tell apart from a hung build, and retrying (the natural
reaction) could race a second reconfigure against the first.

`scripts/workflow.ps1` (already updated by the user, functionally correct pre-session) wraps the
whole configure→build→tune→train→gen sequence as a plain script over the CLI invocations -- not a
CMake target, so it can't reintroduce the same orchestration problem.

**Docs updated to match** (were describing the old mechanism as still-live): `README.md` (Build
section, Configuration table, Positional-encoding section, "Using a larger corpus" section all
referenced `-DSUB0_CORPUS=`/`SUB0_POS_ENCODING`/`SUB0_CORPUS_TOK=AUTO` CMake vars that no longer
exist), `docs/CONFIGURE_ARCHITECTURE.md` (state-1 description + migration-stage-4 both described a
CMake auto-generate convenience that's gone), `docs/WORKFLOW_ARCHITECTURE.md` (already had the most
detail; fixed the "Build-stage gating" fork -- an `if(EXISTS ${GEN_HEADER})` design that was
considered but never actually built -- and item 5's stale "(stretch) workflow convenience target"
language, since `workflow.ps1` already fulfills that), `scripts/get_fineweb.py`'s docstring.

**Verification**: full rebuild (`cmake --build out/build/smoke_gpu`) clean from a fresh reconfigure,
full `ctest` suite 135/135 green (later re-run after further changes, still 135/135 except one
pre-existing unrelated flaky CPU gradient test, see below).

Also fixed in the same pass: `tools/sub0llm-tune.cpp` had a stray debug comment blaming a
CMake-dependency-tracking gap for "not rebuilding train.exe" -- the user clarified directly that this
was a MIS-DIAGNOSIS predating the real root cause (the resume-override-with-no-log-line bug, see
[[train-diagnostics-backlog]]), not a build-system issue at all. Just deleted the comment.

Related: [[corpus-tok-reuse-stamp]] (a genuinely new feature added to the configurator in this same
session, built on top of this cleaned-up CLI-first design) and [[architecture-solidification-plan]].
