"""Dependency pins: a push must not name a dependency commit that is not on the dependency's remote.

A repository can pin a dependency by commit (`set(SOME_REVISION <sha>)` in a CMake file, say). Pushing a pin before the
dependency's own commit is pushed makes every CI run fail to fetch it. The pins are declared at hook install
(`install-hooks --pin file=...,var=...`) and kept in `.git/sub0perf/config.json` as

    "pins": [{"file": "cmake/Dependencies.cmake", "var": "X_REVISION", "repo": "owner/name", "checkout": "../X", "name": "X"}]

`repo` (a GitHub `owner/name`) is optional when `checkout` (a sibling clone, relative to the repository) has a GitHub
origin; `name` defaults to the checkout's folder or the repo's name. Only a pin file that the pushed range changes
is checked (a new branch has no known range, so its pins are), which keeps documentation-only pushes free.
"""

import re
import shutil
import subprocess
from pathlib import Path

from .gitstate import ZERO_SHA_CHARS, git, pushed_refs

PIN_KEYS = ("file", "var", "repo", "checkout", "name")


def parse_pin(text):
    """A `--pin` option, `key=value,key=value` (keys: file, var, repo, checkout, name), as a dict."""
    pin = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep or key.strip() not in PIN_KEYS or not value.strip():
            raise ValueError(f"bad --pin {text!r}: expected key=value pairs from {', '.join(PIN_KEYS)}")
        pin[key.strip()] = value.strip()
    if "file" not in pin or "var" not in pin:
        raise ValueError(f"bad --pin {text!r}: file= and var= are required")
    if "repo" not in pin and "checkout" not in pin:
        raise ValueError(f"bad --pin {text!r}: give repo= (owner/name) or checkout= (a sibling clone) to verify against")
    return pin


def pinned_sha(text, var):
    """The commit sha `set(<var> <sha> ...)` assigns in CMake text, or None (not a sha, or not set)."""
    m = re.search(r'\bset\(\s*' + re.escape(var) + r'\s+"?([0-9a-fA-F]{7,40})"?[\s)]', text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def github_slug(url):
    """`owner/name` of a GitHub remote url (https or ssh), or None."""
    m = re.search(r'github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?/?$', url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


def _run(command, cwd=None):
    """(return code, stdout, stderr); (127, '', message) when the program is missing."""
    try:
        r = subprocess.run(command, capture_output=True, text=True, cwd=cwd, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        return 127, "", str(error)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def on_remote(pin, sha, repo, run=None, have_gh=None):
    """Whether the dependency's remote has `sha`: True, False (known absent), or None (could not be verified).

    GitHub first (`gh api`) when the program is there and the repository is known; a 404 or 422 is "absent", any other
    failure (offline, not signed in) falls through to the sibling checkout's remote-tracking refs, which are as fresh
    as its last fetch. A commit the checkout does not have is not known to be absent: it may simply not be fetched.
    """
    run = run or _run
    have_gh = have_gh or (lambda: shutil.which("gh") is not None)
    checkout = (Path(repo) / pin["checkout"]).resolve() if pin.get("checkout") else None
    slug = pin.get("repo")
    if not slug and checkout and checkout.is_dir():
        slug = github_slug(run(["git", "-C", str(checkout), "remote", "get-url", "origin"])[1])
    if slug and have_gh():
        code, _, err = run(["gh", "api", f"repos/{slug}/commits/{sha}", "--silent"])
        if code == 0:
            return True
        if "404" in err or "422" in err or "No commit found" in err or "Not Found" in err:
            return False
    if checkout and checkout.is_dir():
        if run(["git", "-C", str(checkout), "cat-file", "-e", f"{sha}^{{commit}}"])[0] != 0:
            return None
        code, out, _ = run(["git", "-C", str(checkout), "branch", "-r", "--contains", sha])
        if code == 0:
            return bool(out.strip())
    return None


def pin_problems(repo, pins, refs, changed_paths, show_file, on_remote_fn=None):
    """(problems, warnings) for the pushed refs. Problems block the push; warnings are pins that could not be verified.

    `changed_paths(remote_sha, local_sha)` as in documentation_only; `show_file(local_sha, path)` is the file's text at
    that commit or None. Each distinct pinned sha is checked once.
    """
    on_remote_fn = on_remote_fn or (lambda pin, sha: on_remote(pin, sha, repo))
    problems, warnings, seen = [], [], set()
    for local_ref, local_sha, _, remote_sha in pushed_refs(refs):
        paths = None if set(remote_sha) == ZERO_SHA_CHARS else changed_paths(remote_sha, local_sha)
        for pin in pins:
            if paths is not None and pin["file"] not in paths:
                continue
            sha = pinned_sha(show_file(local_sha, pin["file"]) or "", pin["var"])
            label = pin.get("name") or (Path(pin["checkout"]).name if pin.get("checkout") else pin["repo"].rsplit("/", 1)[-1])
            if not sha or (label, sha) in seen:
                continue
            seen.add((label, sha))
            found = on_remote_fn(pin, sha)
            if found is False:
                problems.append(f"{local_ref} pins {pin['var']} = {sha[:12]} in {pin['file']}, which is not on {label}'s remote: "
                                f"push {label} first (if it was pushed from elsewhere, fetch it in the checkout)")
            elif found is None:
                warnings.append(f"could not verify that {label} has {sha[:12]} ({pin['var']} in {pin['file']}): no gh access and "
                                f"no fetched checkout that has it; make sure {label} is pushed")
    return problems, warnings


def show_file_at(repo):
    return lambda sha, path: git(repo, "show", f"{sha}:{path}")
