---
name: handover-docs-live-in-the-repo
description: START HERE for any new session or machine — the Sub0Llm handover, working practices and a restorable copy of this memory now live in the repo (written 2026-11-05)
metadata:
  type: reference
---
The state of the Qwen4-Next work, how to resume it on a new machine and the ranked open work are in the
repository, not only here: `docs/HANDOVER.md`. The project owner's standing directions, how to measure,
how to delegate and the environment traps are in `docs/WORKING_PRACTICES.md`. A dated, vetted snapshot of
this whole memory directory is `docs/agent-memory/`; on a new machine `python scripts/agent_memory.py restore`
seeds it (never overwrites), and `export` refreshes the snapshot from here.

**Why:** the user asked (2026-11-05) for everything needed to resume directly in a new session on a new
machine, including working practices that existed only as local memory.
**How to apply:** at every pause, update the docs if a direction changed, run `export`, review the diff
(the repo is public; the script refuses emails, credentials and private repository names), commit, push.
When this memory and the repo docs disagree, the repo docs are newer. See [[sub0llm-prefix-for-all-artifacts]],
[[defaults-are-the-best-options]], [[self-managed-memory-is-the-direction]], [[context-memory-planner-backlog]].
