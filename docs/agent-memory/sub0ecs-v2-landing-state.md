---
name: sub0ecs-v2-landing-state
description: "Sub0ECS v2 landing state as of 2026-10-05 — PR 4 (v2 to master) staged, follow-up PRs 5 and 6 onto v2, what is still open"
metadata:
  node_type: memory
  type: project
  originSessionId: 09648518-580b-4f7b-ba75-d6985e369c04
  modified: 2026-10-05T16:10:04.937Z
---

Sub0ECS (`D:\Craig\GitHub\Sub0ECS`, default branch `master`, no branch protection) as of 2026-10-05:

- `v2` holds the promoted library (`include/sub0ecs/`, header-only), doctest tests, nanobench comparison
  benchmarks (`bench/`, v1 frozen under `bench/baselines/v1`, tag `v1.0.0`), 12 CTest-run examples.
- **PR 4** `v2` -> `master`: the landing PR, staged, the user's to merge. Work continues as PRs onto `v2` so
  nothing waits on it.
- PRs 5 (component capacity), 6 (MSVC reference capture) and 7 (MSVC optimisation, `bench/tools/profile.py`, the
  profile-first workflow in `bench/BENCHMARKING.md`) are all in `v2` as of 2026-10-05 (6999a12).
- `master` has a ruleset requiring one approving review; Claude cannot approve or merge PR 4 and must not use the
  admin bypass. Fusion gains quoted from old GCC runs (3.5-5.3x) do not hold on this machine (0.95-1.7x):
  re-measure before quoting.
- 2026-10-06: PR 4 (v2 -> master) MERGED by the user; PR 8 merged into `v2`; PR 9 (`v2` -> `master`) open for the
  user's approval. `master` is the default branch and the product; `v2` is the staging branch Claude can merge into.
- **PR 8** `docs/library-first` (merged into `v2`, 2026-10-06): v1 comparator removed from the tree; benchmarks
  measured against `HandWritten` (paired baseline), `HandTuned` (SIMD ceiling), `OOP`, `NaiveObjects`, plus the
  store's own small-systems-fused cases; README/FINDINGS rewritten library-first. User direction: docs are about
  the library, not v1; the hand-written/heavily-optimised reference is the bar; flexibility is the USP and
  performance one of the benefits; benchmarks against EnTT/flecs themselves come later.
- Headline (100K, vs plain hand-written loop, range over MSVC/clang-cl/GCC): store 0.96-1.07x on iteration, fused
  small systems 0.95-1.03x (MSVC, GCC), hand-tuned ceiling 2.7-5.7x above, lookup by handle 0.23-0.28x.
- Harness trap: nanobench `compare()` sizes a group's pass count by its SLOWEST member; a very slow design made the
  fast ones run one cold pass. Fixed in `bench/harness/runner.cpp` (sized by the fastest).
- 2026-10-06 later: PR 9 merged by the user, `v2` branch deleted. **Work now branches from and targets `master`**,
  which needs the user's approving review (ruleset; no admin bypass), so Claude opens PRs and stops.
- **PR 11** `feature/core-aware-pool` (open): `Parallel` sized by performance cores (`cpu_topology.hpp`, after
  Sub0Llm's `cpu_affinity.hpp`), threads created on demand, per-worker wake-up, pinning as an opt-in
  (`Parallel::Options::pinToPerformanceCores`; user asked for opt-in after it measured slower than unpinned).
  Lessons: C++20 `atomic::wait` is very slow on MinGW libstdc++ (5-10x slower than 1 thread) - use mutex+cv;
  P-core pinning is the same or slower on Windows 11 for a right-sized pool. Open: width by work, not rows.
- 2026-10-06 end of session: PR 11 merged. **Open: PR 14** `fix/audit-correctness` (first readiness fixes from the
  header audit). **Issue #13** = `cpp-review` audit of the library (remaining findings listed in `docs/BACKLOG.md`
  "Readiness"); **issue #12** = conversion to the Sub0 style profile (profile declared in `STYLE_GUIDE.md`; next:
  include rooting, Doxygen tags, `[[nodiscard]]`, one rule per change). Merged remote branches are the user's to
  delete. `rg` is not on the bash PATH here: use the VS Code Insiders ripgrep binary for the profile's detect commands.
- 2026-10-06 later still: **PR 15** `style/sub0-profile` (stacked on PR 14) converts Sub0ECS to the Sub0 family style
  profile the same way Sub0Pipeline#29 did: STYLE_GUIDE.md declares `style-profile: sub0`, includes are
  `"sub0ecs/..."`, Doxygen tags + `[[nodiscard]]` on the user-facing API, `ISidePool`. Both PRs need the user's
  approving review. Cross-library differences the profile does not decide are listed on Sub0ECS issue #12.
- Open backlog (`docs/BACKLOG.md`): wider layout masks / option tags (§1b), async executor spike H10
  (`docs/research/executor-async.md`), TSan job, MSVC fusion inlining same-binary A/B.

**Why:** the user wants v2 work to keep flowing as PRs onto `v2` rather than block on the master merge.

**How to apply:** branch from `v2`, PR onto `v2`. Verify perf claims with paired ratios within one run
(cross-run times drift up to 40% on this laptop; v1's Update2 baseline is bimodal). Compare compilers only on one machine in rotated runs: MSVC is
1.0-1.5x behind GCC here, and neither vectorises the Update2 kernel (an earlier 3-7x claim was cross-machine).
VTune `hotspots` runs unelevated; `uarch` needs an elevated prompt ([[cpu-profiling-tooling-backlog]]).
See [[sub0-test-bench-tooling-doctest-nanobench]], [[single-responsibility-headers]].
