---
name: main-branch-consolidation-2026-08-30
description: "Sub0Llm now has a real `main` branch (GitHub default) off the active engine lineage; the old disjoint chapter-tutorial default branch and stale/spike branches are archived under archive/*, not deleted"
metadata: 
  node_type: memory
  type: project
  originSessionId: 84b5aaba-5d15-4c11-acba-06624e60ae97
  modified: 2026-08-30T22:28:44.443Z
---

Repo never had a real `main` before this. Two completely disjoint git histories shared the remote (no
common ancestor): `claude/llm-cpp23-repo-init-1f1Il` (347 commits, last touched 2026-06-24, was
GitHub's default branch, and is actually the dormant chapter-tutorial/diffusion track —
`chapters/`+`diffusion/` dirs, matches [[ch27-diffusion-track-plan]]'s "separate, dormant" framing) vs.
`claude/llm-transformers-1bit-qf3t3i` (471 commits, last touched 2026-07-30, the real active Sub0Llm
engine — matches essentially everything else in memory).

**What happened**: created `main` from the active engine lineage's tip, merged in two small
ahead-of-main worktree branches (`feature/modality-calibration`, one trivial `tests/CMakeLists.txt`
conflict resolved by keeping the superset), flipped GitHub's default branch to `main`
(`gh repo edit --default-branch main`), and deleted the now-superseded branch names
(`claude/llm-transformers-1bit-qf3t3i`, `claude/llm-cpp23-repo-init-1f1Il`, plus already-merged
`feature/tokenizer-v2`/`feature/tokenizer-throughput`/`feature/tokenizer-throughput-2`/
`lmhead-chunked-ce` — all fully-ancestor, zero data loss). Four stale worktree directories removed
(`.claude/worktrees/tutorspike`, `Sub0Llm-modality`, `Sub0Llm-tok-perf`, `Sub0Llm-v2`).

**Three branches preserved as `archive/*` (pushed to origin, not deleted) instead of merged:**
- `archive/chapter-education-diffusion` — the entire old default-branch lineage, kept for reference.
- `archive/spike-tutorspike` — TeacherSpike/Tutor adaptive-LR work ([[adaptive-lr-corpus-ledger-spike]]),
  explicitly a SPIKE per its own memory, deliberately not folded into `main`'s closing state.
- `archive/cuda-row-budget-pin-stale-2026-07-03` — **downgraded from "merge" to "archive-only" mid-task**:
  this branch forked from `a62af9b` (2026-07-03) and its 2 commits touch `memplan.hpp`/`backend_cuda.cu`/
  `cuda_tests.cpp`, all three rewritten by ~39,000 lines of subsequent GQA/LoopSplit/tokenizer/WSD work
  ([[nanbeige-features-progress]]). The user approved "merge the safe ones" believing this was low-risk;
  actually attempting the merge produced real 3-file conflicts in heavily-refactored CUDA code with no
  safe automatic resolution. Correctly the call to make unilaterally rather than force a risky merge:
  aborted it, archived instead, explained why. **Lesson: "N commits ahead" is not itself a safety signal
  — check how much the target has diverged underneath those files before promising a merge is safe.**

Verified: `main` builds and passes full test suite (114,495 assertions / 181 cases) after the merge
conflict resolution in `tests/CMakeLists.txt`, confirming the resolution was correct, not just clean.

**Also fixed in passing**: `.gitignore` was missing `data/cosmopedia.txt` (11GB) and `data/minipile.txt`
(5.9GB) — untracked and about to be swept into a commit. Added, same category as the existing
fineweb_edu/tinystories/gsm8k exclusions ([[measure-tokenizer-on-the-full-corpus-blend]]).
