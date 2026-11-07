"""The git hooks: installing the pre-push gate and the post-checkout note, and chaining hooks that were already there."""

import json
import os
import shlex
import sys
from pathlib import Path

from . import levels
from .common import atomic_write_text, read_json, tooling_script
from .gitstate import git, git_head, log_event, perf_dir
from .pins import parse_pin


HOOK_MARK = "# sub0perf hook"
PREVIOUS_SUFFIX = ".user"  # a foreign hook that was in the way is renamed to <hook>.user and run first
HOOK_COMMENTS = {"pre-push": "performance regression gate (tools/perf/bench_check.py gate)",
                 "post-checkout": "notes the checkout; the first clean build records its measurement"}


def hook_body(name, python, script, comment):
    """The sh text of a hook. Paths are shell-quoted (spaces); it does nothing, with a message, when python or the tooling is gone."""
    lines = ["#!/bin/sh", f"{HOOK_MARK}: {comment}",
             f"python={shlex.quote(python)}", f"script={shlex.quote(script)}",
             'previous="$(dirname "$0")/' + name + PREVIOUS_SUFFIX + '"']
    guard = ['if [ ! -f "$python" ] || [ ! -f "$script" ]; then',
             '    echo "sub0perf: python ($python) or the tooling ($script) is missing; ' + name + ' hook skipped"',
             '    exit 0',
             'fi']
    if name == "pre-push":  # stdin carries the refs being pushed; the previous hook and ours both need it
        lines += ['refs="$(cat)"',
                  'if [ -f "$previous" ]; then',
                  '    printf \'%s\\n\' "$refs" | "$previous" "$@" || exit $?',
                  'fi', *guard,
                  'printf \'%s\\n\' "$refs" | "$python" "$script" gate --repo "$(git rev-parse --show-toplevel)" --refs-from-stdin']
    else:
        lines += ['if [ -f "$previous" ]; then',
                  '    "$previous" "$@" || exit $?',
                  'fi', *guard,
                  'if [ "$3" = "1" ]; then',
                  '    "$python" "$script" note-checkout --repo "$(git rev-parse --show-toplevel)"',
                  'fi',
                  'exit 0']
    return "\n".join(lines) + "\n"


def install_hooks_command(args):
    repo = Path(args.repo)
    d = perf_dir(repo)
    if d is None:
        print(f"sub0perf: {repo} is not a git repository; hooks not installed")
        return 0
    d.mkdir(parents=True, exist_ok=True)
    config_path = d / "config.json"
    config = read_json(config_path) if config_path.is_file() else {"targets": {}}
    config["targets"][args.target] = {"build_dir": str(Path(args.build_dir).resolve()), "exe": str(Path(args.exe).resolve()),
                                      "baseline": str(Path(args.baseline).resolve()), "config": args.config or ""}
    if getattr(args, "full_refs", None):  # tag-name patterns whose push runs the full level; "" restores the defaults
        patterns = [p for text in args.full_refs for p in levels.parse_patterns(text)]
        if patterns:
            config["full_refs"] = patterns
        else:
            config.pop("full_refs", None)
    if getattr(args, "quick_budget", None) not in (None, ""):
        config["quick_budget"] = float(args.quick_budget)
    if getattr(args, "pin", None):
        try:
            new_pins = [parse_pin(text) for text in args.pin]
        except ValueError as error:
            print(f"sub0perf: {error}")
            return 2
        keep = [p for p in config.get("pins", []) if (p["file"], p["var"]) not in {(n["file"], n["var"]) for n in new_pins}]
        config["pins"] = keep + new_pins
    atomic_write_text(config_path, json.dumps(config, indent=2) + "\n")
    hooks = Path(git(repo, "rev-parse", "--git-path", "hooks") or "")
    if not hooks.is_absolute():
        hooks = repo / hooks
    hooks.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable).as_posix()
    script = tooling_script().as_posix()
    for name, comment in HOOK_COMMENTS.items():
        path = hooks / name
        if path.is_file() and HOOK_MARK not in path.read_text(encoding="utf-8", errors="replace"):
            previous = hooks / (name + PREVIOUS_SUFFIX)
            if previous.exists():
                print(f"sub0perf: {path} exists and is not ours, and {previous.name} is taken; chain to the gate yourself:\n"
                      f'  "{python}" "{script}" gate --repo "$(git rev-parse --show-toplevel)" --refs-from-stdin')
                continue
            os.replace(path, previous)
            print(f"sub0perf: kept your {name} hook as {previous.name}; the installed hook runs it first")
        body = hook_body(name, python, script, comment)
        if path.is_file() and path.read_text(encoding="utf-8", errors="replace") == body:
            continue  # unchanged; Windows refuses to replace a hook that is running (a gate's own build)
        try:
            atomic_write_text(path, body)
        except PermissionError as error:
            print(f"sub0perf: could not update {path} ({error}); it is probably running, and the next configure retries")
            continue
        try:
            path.chmod(0o755)
        except OSError:
            pass
    return 0


def note_checkout_command(args):
    repo = Path(args.repo or ".")
    head = git_head(repo)
    log_event(repo, {"event": "checkout", "commit": head})
    d = perf_dir(repo)
    have = d and any((d / "records").glob(f"*/{head}.json")) if d else False
    if not have:
        print(f"sub0perf: checked out {head[:9]}; its benchmark record is made by the first clean build")
    return 0
