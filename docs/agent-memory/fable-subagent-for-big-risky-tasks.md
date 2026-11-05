---
name: fable-subagent-for-big-risky-tasks
description: User reminder -- delegate to a heavier subagent (Agent tool, model="opus" -- NOT "fable", corrected 2026-09-02) when a task gets too big or risky to do directly in the main session
metadata:
  type: feedback
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
  modified: 2026-09-03T09:42:24.120Z
---

**CORRECTION 2026-09-02**: this account does not have Fable model access ("we don't have fable - Opus,
Sonnet, or Haiku"), even though the Agent tool's `model` schema still lists `"fable"` as an enum value.
Use **`model="opus"`** for this pattern going forward -- everything below describes the same delegation
pattern, written while Fable was the model actually used; read "Fable"/"fable" throughout as "the heavier
delegate model," now Opus, not a literal instruction to pass `model="fable"`. A subagent that itself
reports "dispatching to a Fable subagent" (as happened once, seen in [[pivot-to-existing-model-support-qwen4-preview]]'s
MoE Stage 0+1 dispatch) is describing an unavailable model and should be corrected the same way, pointed
at Opus instead.

User's original explicit reminder (2026-07-04): "remember we can delegate to fable subagent as/when a
task gets too big or risky." The delegation-when-too-big-or-risky POLICY stands; only the specific model
name changes.

**How to apply**: the Agent tool's `model` param -- use `"opus"` (was `"fable"`, no longer available).
Already used this session
precedent-wise before this reminder: [[cuda-wave6-orchestration-memory-findings]] describes a Fable
"end-to-end review" of the full CUDA training step, and two isolated-worktree Fable agents dispatched
on C1 (token-budget batching) and Wave 8 (chunked lm_head+CE) when those were judged too large/risky
for inline work -- both outcomes were reviewed and merged/deferred by the main session afterward, not
blindly trusted. Pattern to follow going forward: when a task is large in scope OR carries real
correctness/blast-radius risk (e.g. a CUDA kernel touching the hot training path, a big refactor across
many files), consider spawning a Fable subagent (optionally in an isolated worktree via
`isolation: "worktree"`) rather than pushing through it inline -- then review its actual diff/output
directly before merging, per the general "trust but verify" agent-delegation guidance.

**Candidates from the current backlog this could apply to**: the CUDA kernel implementations still
deferred for both gated-FFN (`TODO(gated-ffn-gpu)`) and Muon (`TODO(muon-gpu)`) -- both are exactly the
"touches the hot device training path, real correctness risk" shape this guidance is for.

**Worked example, phased (2026-07-04): tied-embeddings CUDA support.** Dispatched Fable for a 3-phase
pass -- Phase 1 audit+plan (no code), Phase 2 implementation, Phase 3 test/verify -- with an explicit
"stop and report after each phase" instruction so the plan could be reviewed before any code was
written. Phase 1 alone caught 2 real bugs a less careful audit would have missed (`memplan.hpp`'s
untied-hardcoded shape; a stale code comment claiming a configurator refusal didn't exist when it
already did). Phase 2/3 then found 2 MORE bugs during verification (a third+fourth independent
`Dims`-construction site the Phase 1 enumeration had missed) and caught a false-positive in its OWN new
test (per-element max-rel-diff blowing up near a zero-crossing at production scale -- switched to
whole-array relative-L2, the project's own established convention). See
[[weight-tying-tied-embeddings]] for the full outcome -- verified correct via independent hand-tracing
of the GEMM math, not just trusting the agent's report.

**`isolation: "worktree"` has proven unreliable TWICE now, in two different ways -- treat it as
not-to-be-trusted in this environment, default to the main tree with active coordination instead.**
1. (2026-07-04) Does NOT survive a `SendMessage` resume: the original spawn correctly used an isolated
   worktree for Phase 1 (audit-only, so it got auto-cleaned since no files changed), but resuming via
   `SendMessage` for Phase 2/3 put the agent directly in the MAIN working tree instead -- no separate
   worktree path was returned. Caused a real (harmless) collision: the agent noticed unrelated concurrent
   edits I was making to the same files (`layout.hpp`/`layout_tests.cpp`) and had to pause to investigate.
   No damage since Claude Code's Edit tool matches by content not line offsets.
2. (2026-07-06, a FRESH dispatch, not a resume this time) The assigned worktree was simply STALE from
   the start -- an old chapter-era snapshot of the repo (missing files that exist on the current branch,
   wrong git log). The agent correctly noticed, refused to work in it, and asked before proceeding
   rather than silently working against wrong ground truth.
Given both failure modes, the standing default now is: assume `isolation: "worktree"` may not give a
correct isolated copy, brief every phased dispatch for direct main-tree work with active coordination
from the start (see the git-stash incident below for why "isolated" mattered in the first place), rather
than treating the isolation flag as reliable groundwork to build a plan around.

**A real incident from relying on git tricks to fake isolation (2026-07-06):** mid-Phase-3 of the B1
chunked-lm_head/CE work, an agent ran `git stash push` scoped to its own 3 changed files to build an
isolated "old" (pre-change) binary for a throughput A/B. `git stash` can't isolate by author within an
uncommitted file, only by "differs from HEAD" -- it reverted a concurrent, unrelated change (my own
`DECAY_RANGES` work in the same file) too. Caught immediately, `git stash pop` restored everything,
independently reverified (rebuilt + reran the full suite myself, matched exactly). Lesson, now a
standing instruction given to every subsequent dispatch: NO git-state manipulation (stash, worktree add,
cherry-pick, reset, etc.) to fake before/after isolation in a shared tree -- copy the specific files to
a `.bak` and use a separate build directory instead. An agent that later found an unrelated, genuinely
pre-existing stale git artifact (a paused 3-day-old cherry-pick, unrelated to any of this session's
work) correctly stopped and flagged it rather than touching it -- that's the model to reinforce:
investigate and ask, never manipulate git state to route around an inconvenience.

**A new failure mode (2026-07-08, gated-FFN CUDA parity dispatch): a resumed agent can mistake ITSELF
for a delegate.** After Phase 1 (audit+plan) completed and was reviewed/greenlit via `SendMessage`, the
SAME agent's Phase 2 response described "dispatching to a Fable subagent" and reviewing its diff --
but `git status`/`git diff --stat` showed ZERO file changes. It had misread the greenlight message's
background context (which explained the PROJECT's general practice of delegating big CUDA work to
Fable) as an instruction to spawn ANOTHER delegate, rather than realizing it itself already IS the
Fable agent with direct Edit/Write tools and no one else to hand the work to. Caught immediately by the
standing "trust but verify" habit -- checking `git status` before believing a "done" report, not after.
Fix: sent an explicit correction stating in plain terms "there is no subagent for you to delegate to,
YOU are Fable, use your own tools, write the code yourself" and re-listed the concrete file-level work
items. Lesson for future phased dispatches: when resuming an agent with a greenlight message that
references "the project's practice of delegating to Fable" as CONTEXT for why it was dispatched, make
explicit that IT is the delegate and must act directly -- don't assume that context reads unambiguously
to the agent itself, especially across a resume boundary.

**Related reminder from the same day**: "remember to utilise the ncu tooling when it comes to the cuda
optimized version" -- when either of the two CUDA-kernel TODOs above actually gets built (whether
inline or via a Fable subagent), profile it with Nsight Compute the same way every other CUDA
kernel wave this project has shipped was profiled, don't just verify functional/numerical correctness
and stop. See [[ncu-vbs-memory-backup-blocker]] for the practical workaround already established
(profile against a smaller reference model, not the full production model -- VBS blocks ncu's
system-memory-backed replay fallback once VRAM gets tight) and [[cuda-kernel-optimization-waves]] /
[[training-step-nsys-occupancy-findings]] for the existing profiling-driven optimization precedent to
match.
