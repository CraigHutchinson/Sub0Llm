#!/usr/bin/env python3
"""Moves Claude Code's per-project memory between this machine's profile and the repository.

Claude Code keeps an agent's working memory outside the repo, in
`~/.claude/projects/<project-path-slug>/memory/` (an index, `MEMORY.md`, plus one note per fact). Anything
that exists only there is lost with the machine. `docs/agent-memory/` is the repository copy: a dated
snapshot that a new machine restores before its first session, so the agent starts with the same standing
preferences, host facts and hard-won warnings.

    python scripts/agent_memory.py export     # local memory  -> docs/agent-memory/   (then review and commit)
    python scripts/agent_memory.py restore    # docs/agent-memory/ -> local memory   (never overwrites)
    python scripts/agent_memory.py restore --force
    python scripts/agent_memory.py where      # print the local memory directory this would use

`--dest DIR` overrides the local directory for any command (use it if Claude Code on the new machine keeps
the memory somewhere other than the slug this script derives).

This repository is public, so `export` refuses to copy a note that contains something that must not be
published: an e-mail address, a credential-shaped string, or the name of a private repository. Extend
`FORBIDDEN` when a new private project appears.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "docs" / "agent-memory"
README = "README.md"          # the snapshot's own description; never exported over, never restored

FORBIDDEN = {
    "an e-mail address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}"),
    "a GitHub token": re.compile(r"\b(ghp_|gho_|github_pat_)[A-Za-z0-9_]{12,}"),
    "an API key": re.compile(r"\b(sk-ant-|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{12,})"),
    "a private repository": re.compile(r"\b(BikeDynamics|BikeLab|BikeTelemetry|PhantomFlow)\b"),
}


def local_memory_dir(override: str | None) -> pathlib.Path:
    """The directory Claude Code uses for this checkout's memory (an existing one wins, case-insensitively)."""
    if override:
        return pathlib.Path(override).expanduser()
    projects = pathlib.Path.home() / ".claude" / "projects"
    slug = re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
    if projects.is_dir():
        for existing in projects.iterdir():
            if existing.name.lower() == slug.lower():
                return existing / "memory"
    return projects / slug / "memory"


def notes(directory: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in directory.glob("*.md") if p.name != README)


def audit(note: pathlib.Path) -> list[str]:
    text = note.read_text(encoding="utf-8", errors="replace")
    return [f"{note.name}: contains {what}" for what, pattern in FORBIDDEN.items() if pattern.search(text)]


def export(local: pathlib.Path) -> int:
    if not local.is_dir():
        print(f"no local memory at {local}", file=sys.stderr)
        return 1
    source = notes(local)
    problems = [line for note in source for line in audit(note)]
    if problems:
        print("refusing to export -- this repository is public:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    keep = {note.name for note in source}
    removed = [old.name for old in notes(SNAPSHOT) if old.name not in keep]
    for old in removed:
        (SNAPSHOT / old).unlink()
    for note in source:
        shutil.copyfile(note, SNAPSHOT / note.name)
    print(f"exported {len(source)} notes from {local} to {SNAPSHOT} ({len(removed)} stale removed)")
    return 0


def restore(local: pathlib.Path, force: bool) -> int:
    source = notes(SNAPSHOT)
    if not source:
        print(f"no snapshot in {SNAPSHOT}", file=sys.stderr)
        return 1
    local.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for note in source:
        target = local / note.name
        if target.exists() and not force:
            skipped += 1
            continue
        shutil.copyfile(note, target)
        copied += 1
    print(f"restored {copied} notes into {local}; skipped {skipped} that already exist"
          + (" (use --force to overwrite them)" if skipped else ""))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["export", "restore", "where"])
    parser.add_argument("--dest", help="local memory directory, instead of the derived ~/.claude/projects/<slug>/memory")
    parser.add_argument("--force", action="store_true", help="restore: overwrite notes that already exist locally")
    args = parser.parse_args()
    local = local_memory_dir(args.dest)
    if args.command == "where":
        print(local)
        return 0
    return export(local) if args.command == "export" else restore(local, args.force)


if __name__ == "__main__":
    sys.exit(main())
