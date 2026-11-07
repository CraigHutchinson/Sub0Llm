"""Git state: refs, HEAD, clean tree, the local records under `.git/sub0perf`, and what a push changes."""

import json
import shutil
import subprocess
import time
from pathlib import Path

from .common import atomic_write_text, read_json

ZERO_SHA_CHARS = {"0"}


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def git_head(repo):
    return git(repo, "rev-parse", "HEAD")


def tree_clean(repo):
    """True when no tracked file differs from HEAD (untracked files do not change what is built)."""
    return git(repo, "status", "--porcelain", "--untracked-files=no") == ""


def perf_dir(repo):
    common = git(repo, "rev-parse", "--git-common-dir")
    if common is None:
        return None
    base = Path(common) if Path(common).is_absolute() else Path(repo) / common
    return base / "sub0perf"


def log_event(repo, event):
    d = perf_dir(repo)
    if d is None:
        return
    d.mkdir(parents=True, exist_ok=True)
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with open(d / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


KEEP_BINARIES = 3  # benchmark executables kept per target, newest first


def save_record(repo, target, record, only_if_missing=False, exe=None):
    """Keeps a clean tree's measurement as the record for its commit, and the benchmark executable that made it.

    The executable lets a later gate re-measure this commit's build beside the new one, in the same session
    (see reproduce.py); only the newest KEEP_BINARIES are kept.
    """
    d = perf_dir(repo)
    head = git_head(repo)
    if d is None or head is None or not tree_clean(repo):
        return None
    path = d / "records" / target / f"{head}.json"
    if only_if_missing and path.is_file():
        return path
    atomic_write_text(path, json.dumps({**record, "commit": head}, indent=1) + "\n")
    log_event(repo, {"event": "record", "target": target, "commit": head, "binary": keep_binary(d, target, head, exe)})
    return path


def keep_binary(perf, target, commit, exe):
    """Copies the benchmark executable to bins/<target>/<commit>/ and prunes older ones. True when kept."""
    if not exe or not Path(exe).is_file():
        return False
    where = perf / "bins" / target
    try:
        (where / commit).mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, where / commit / Path(exe).name)
        old = sorted((p for p in where.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in old[KEEP_BINARIES:]:
            shutil.rmtree(p, ignore_errors=True)
    except OSError:
        return False
    return True


def reference_binary(repo, target, sha):
    """The benchmark executable kept with the record of `sha`, or None (not kept, or pruned)."""
    d = perf_dir(repo)
    where = d / "bins" / target / sha if d and sha else None
    found = [p for p in where.iterdir() if p.is_file()] if where and where.is_dir() else []
    return found[0] if found else None


def reference_record(repo, target, upstream=None, limit=200):
    """The record of the nearest earlier commit (the upstream first, when it has one)."""
    d = perf_dir(repo)
    if d is None:
        return None, None
    records = d / "records" / target
    candidates = ([upstream] if upstream else []) + (git(repo, "rev-list", f"--max-count={limit}", "HEAD~1") or "").split()
    for sha in candidates:
        if sha and (records / f"{sha}.json").is_file():
            return sha, read_json(records / f"{sha}.json")
    return None, None


def accepted_since(repo, baseline_rel, reference_sha):
    """Benchmarks with an acceptance in HEAD's baseline that the reference commit's baseline lacks."""
    def entries(text):
        if not text:
            return []
        return json.loads(text).get("accepted", [])
    now = entries(git(repo, "show", f"HEAD:{baseline_rel}"))  # the committed baseline, not the working tree
    before = entries(git(repo, "show", f"{reference_sha}:{baseline_rel}") if reference_sha else "")
    names = set()
    for e in now[len(before):]:
        if isinstance(e.get("benchmarks"), dict):
            names |= set(e["benchmarks"])
        else:
            names.add("*")
    return names


def parse_push_refs(text):
    """The lines git passes a pre-push hook on stdin: (local ref, local sha, remote ref, remote sha)."""
    refs = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 4:
            refs.append(tuple(fields))
    return refs


def pushed_refs(refs):
    """The refs that send something (an all-zero local sha deletes the remote ref)."""
    return [r for r in refs if set(r[1]) != ZERO_SHA_CHARS]


def push_problems(refs, head, clean, resolve=lambda sha: sha):
    """Reasons the build in the working tree is not what a push sends; empty when it is (or only deletes are pushed).

    `resolve` turns a pushed object into its commit (an annotated tag points at a tag object).
    """
    pushed = pushed_refs(refs)
    problems = []
    if not pushed:
        return problems
    if not clean:
        problems.append("the working tree has uncommitted changes to tracked files, so the benchmarks would measure them and "
                        "not the commit being pushed; commit or stash them")
    for local_ref, local_sha, *_ in pushed:
        commit = resolve(local_sha)
        if commit != head:
            problems.append(f"{local_ref} ({commit[:9]}) is not what is checked out (HEAD is {(head or '?')[:9]}), so the build is not "
                            f"the pushed commit; check it out first")
    return problems


def is_documentation(path):
    """True for files that no build reads: Markdown anywhere, anything under docs/, licence files, and the
    provenance manifests under upstream/ (tools/provenance), and CI configuration (.github/, .ci/)."""
    name = path.rsplit("/", 1)[-1]
    return (path.endswith(".md") or path.startswith("docs/") or name.upper().startswith(("LICENSE", "LICENCE"))
            or (path.startswith("upstream/") and path.endswith(".json"))
            or path.startswith((".github/", ".ci/")))  # CI configuration: no build reads it


def documentation_only(refs, changed_paths):
    """True when every pushed update changes only documentation, so measuring could only find noise.

    `changed_paths(remote_sha, local_sha)` lists the files between two commits, or None when unknown (a new
    branch, or a remote commit not present locally); unknown means measure.
    """
    pushed = pushed_refs(refs)
    if not pushed:
        return False
    for _, local_sha, _, remote_sha in pushed:
        if set(remote_sha) == ZERO_SHA_CHARS:
            return False
        paths = changed_paths(remote_sha, local_sha)
        if paths is None or not all(is_documentation(p) for p in paths):
            return False
    return True


def changed_paths_between(repo, remote_sha, local_sha):
    """Files changed between two commits, or None when the remote commit is not present locally."""
    out = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", "-z", remote_sha, local_sha],
                         capture_output=True, text=True)
    return [p for p in out.stdout.split("\0") if p] if out.returncode == 0 else None
