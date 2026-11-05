# Agent memory — a restorable snapshot

Claude Code keeps an agent's working memory outside the repository, in
`~/.claude/projects/<project-path-slug>/memory/`. It does not move with the repository, so anything that
lived only there was lost with the machine. This directory is the repository's copy: one note per fact,
plus `MEMORY.md`, the index an agent loads at the start of every session.

**This is a dated snapshot, not documentation.** Notes record what was believed and measured when they were
written, including conclusions that later work superseded; the index marks the ones to read first. The current
state, the open work and the working practices are in [../HANDOVER.md](../HANDOVER.md) and
[../WORKING_PRACTICES.md](../WORKING_PRACTICES.md), and they win wherever they disagree with a note here.
Note names that mention `sub0_tests` or `sub0_core.dll` predate the 2026-10-08 rename to `sub0llm_*`.

Snapshot taken 2026-11-05, 214 notes, from a Windows dev machine; absolute paths in notes are that machine's.

## On a new machine

    python scripts/agent_memory.py restore          # seeds ~/.claude/projects/<slug>/memory; never overwrites
    python scripts/agent_memory.py restore --force  # overwrite notes that already exist
    python scripts/agent_memory.py where            # show the directory it will use

If Claude Code on the new machine keeps its memory somewhere other than the slug derived from this checkout's
path, pass `--dest DIR`.

## Refreshing it

    python scripts/agent_memory.py export

Review the diff, then commit. The repository is public, so `export` refuses to copy a note containing an e-mail
address, a credential-shaped string or the name of a private repository; extend `FORBIDDEN` in the script when
a new private project appears.

## Where to start

`MEMORY.md` is the index. Its first section is the active thread. "READ FIRST" marks a note carrying a caveat
that invalidates a number you might otherwise cite. The sections for the closed from-scratch training research
are background only.
