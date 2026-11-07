"""Tests for pins: a push must not pin a dependency commit that is not on the dependency's remote."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from .. import common, gate, gitstate, hooks, pins
from .helpers import git_run, RepoTestCase

SHA = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"
ZERO = "0" * 40
PIN = {"file": "cmake/Dependencies.cmake", "var": "X_REVISION", "repo": "owner/dep", "name": "Dep"}


def cmake(sha):
    return f"set(X_REVISION {sha})\nCPMAddPackage(NAME Dep GIT_TAG ${{X_REVISION}})\n"


class ParsingTest(unittest.TestCase):
    def test_pin_option(self):
        pin = pins.parse_pin("file=cmake/D.cmake,var=X,repo=o/n,checkout=../dep,name=Dep")
        self.assertEqual(pin, {"file": "cmake/D.cmake", "var": "X", "repo": "o/n", "checkout": "../dep", "name": "Dep"})

    def test_bad_pin_options(self):
        for bad in ("", "file=a", "file=a,var=b", "file=a,var=b,colour=c", "var=b,repo=o/n", "file=a,var=b,repo="):
            with self.assertRaises(ValueError, msg=bad):
                pins.parse_pin(bad)

    def test_sha_is_read_from_the_set_command(self):
        self.assertEqual(pins.pinned_sha(cmake(SHA), "X_REVISION"), SHA)
        self.assertEqual(pins.pinned_sha(f'set( X_REVISION "{SHA.upper()}" CACHE STRING "")', "X_REVISION"), SHA)
        self.assertIsNone(pins.pinned_sha("set(X_REVISION main)", "X_REVISION"))
        self.assertIsNone(pins.pinned_sha(cmake(SHA), "OTHER_VAR"))
        self.assertIsNone(pins.pinned_sha(f"set(MY_X_REVISION {SHA})", "X_REVISION"))

    def test_github_slug(self):
        self.assertEqual(pins.github_slug("https://github.com/o/n.git"), "o/n")
        self.assertEqual(pins.github_slug("git@github.com:o/n.git"), "o/n")
        self.assertEqual(pins.github_slug("https://github.com/o/n"), "o/n")
        self.assertIsNone(pins.github_slug("https://example.com/o/n.git"))


class OnRemoteTest(unittest.TestCase):
    def run_gh(self, code, err=""):
        calls = []

        def run(command, cwd=None):
            calls.append(command)
            return code, "", err
        return run, calls

    def test_gh_finding_the_commit(self):
        run, calls = self.run_gh(0)
        self.assertTrue(pins.on_remote(PIN, SHA, ".", run, lambda: True))
        self.assertEqual(calls[0][:3], ["gh", "api", f"repos/owner/dep/commits/{SHA}"])

    def test_gh_not_finding_it_is_known_absent(self):
        for err in ("gh: Not Found (HTTP 404)", "gh: No commit found for SHA: abc (HTTP 422)"):
            run, _ = self.run_gh(1, err)
            self.assertFalse(pins.on_remote(PIN, SHA, ".", run, lambda: True), err)

    def test_gh_failing_for_other_reasons_is_unknown(self):
        run, _ = self.run_gh(1, "gh: could not resolve host (network)")
        self.assertIsNone(pins.on_remote(PIN, SHA, ".", run, lambda: True))

    def test_no_gh_and_no_checkout_is_unknown(self):
        run, calls = self.run_gh(0)
        self.assertIsNone(pins.on_remote(PIN, SHA, ".", run, lambda: False))
        self.assertEqual(calls, [])


class CheckoutTest(unittest.TestCase):
    """The fallback uses a real sibling checkout with a real (bare) remote."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.bare = root / "remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(self.bare)], check=True)
        self.dep = root / "dep"
        subprocess.run(["git", "clone", "-q", str(self.bare), str(self.dep)], check=True, capture_output=True)
        git_run(self.dep, "checkout", "-q", "-b", "main")
        self.pushed = self.commit("pushed")
        git_run(self.dep, "push", "-q", "origin", "main")
        self.local = self.commit("local only")
        self.main = root / "main"
        self.main.mkdir()
        self.pin = {"file": "cmake/Dependencies.cmake", "var": "X_REVISION", "checkout": "../dep", "name": "Dep"}

    def commit(self, message):
        git_run(self.dep, "commit", "-q", "--allow-empty", "-m", message)
        return gitstate.git_head(self.dep)

    def check(self, sha):
        return pins.on_remote(self.pin, sha, self.main, have_gh=lambda: False)

    def test_pushed_commit_is_on_the_remote(self):
        self.assertTrue(self.check(self.pushed))

    def test_local_only_commit_is_absent(self):
        self.assertFalse(self.check(self.local))

    def test_commit_the_checkout_does_not_have_is_unknown(self):
        self.assertIsNone(self.check(OTHER))


class PinProblemsTest(unittest.TestCase):
    def check(self, files, found, refs=None, changed=None):
        refs = refs or [("refs/heads/main", "a" * 40, "refs/heads/main", "b" * 40)]
        asked = []

        def on_remote(pin, sha):
            asked.append(sha)
            return found[sha]
        problems, warnings = pins.pin_problems(".", [PIN], refs, changed or (lambda r, l: ["cmake/Dependencies.cmake"]),
                                               lambda sha, path: files.get(path), on_remote)
        return problems, warnings, asked

    def test_pin_on_the_remote_passes(self):
        problems, warnings, _ = self.check({PIN["file"]: cmake(SHA)}, {SHA: True})
        self.assertEqual((problems, warnings), ([], []))

    def test_pin_not_on_the_remote_says_push_the_dependency_first(self):
        problems, warnings, _ = self.check({PIN["file"]: cmake(SHA)}, {SHA: False})
        self.assertEqual(len(problems), 1)
        self.assertIn("push Dep first", problems[0])
        self.assertIn(SHA[:12], problems[0])
        self.assertEqual(warnings, [])

    def test_unverifiable_pin_warns_and_does_not_block(self):
        problems, warnings, _ = self.check({PIN["file"]: cmake(SHA)}, {SHA: None})
        self.assertEqual(problems, [])
        self.assertIn("could not verify", warnings[0])

    def test_pin_file_not_in_the_pushed_range_is_not_checked(self):
        problems, warnings, asked = self.check({PIN["file"]: cmake(SHA)}, {SHA: False}, changed=lambda r, l: ["src/a.cpp", "README.md"])
        self.assertEqual((problems, warnings, asked), ([], [], []))

    def test_new_branch_and_unknown_range_are_checked(self):
        new_branch = [("refs/heads/n", "a" * 40, "refs/heads/n", ZERO)]
        self.assertEqual(len(self.check({PIN["file"]: cmake(SHA)}, {SHA: False}, refs=new_branch)[0]), 1)
        self.assertEqual(len(self.check({PIN["file"]: cmake(SHA)}, {SHA: False}, changed=lambda r, l: None)[0]), 1)

    def test_deleting_a_branch_checks_nothing(self):
        problems, _, asked = self.check({PIN["file"]: cmake(SHA)}, {SHA: False}, refs=[("(delete)", ZERO, "refs/heads/x", "b" * 40)])
        self.assertEqual((problems, asked), ([], []))

    def test_a_pin_that_is_not_a_sha_or_a_missing_file_is_ignored(self):
        self.assertEqual(self.check({PIN["file"]: "set(X_REVISION main)"}, {})[:2], ([], []))
        self.assertEqual(self.check({}, {})[:2], ([], []))

    def test_each_pinned_sha_is_looked_up_once(self):
        refs = [("refs/heads/a", "a" * 40, "refs/heads/a", "b" * 40), ("refs/heads/b", "c" * 40, "refs/heads/b", "d" * 40)]
        problems, _, asked = self.check({PIN["file"]: cmake(SHA)}, {SHA: False}, refs=refs)
        self.assertEqual((len(problems), asked), (1, [SHA]))


class GatePinsTest(RepoTestCase):
    """The hook entry point, with a real repository, sibling checkout and remote; no gh."""

    def setUp(self):
        super().setUp()
        root = self.repo.parent
        bare = root / "dep-remote.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        self.dep = root / "dep"
        subprocess.run(["git", "clone", "-q", str(bare), str(self.dep)], check=True, capture_output=True)
        git_run(self.dep, "checkout", "-q", "-b", "main")
        git_run(self.dep, "commit", "-q", "--allow-empty", "-m", "one")
        self.pushed = gitstate.git_head(self.dep)
        git_run(self.dep, "push", "-q", "origin", "main")
        git_run(self.dep, "commit", "-q", "--allow-empty", "-m", "two")
        self.unpushed = gitstate.git_head(self.dep)
        (self.repo / "cmake").mkdir()
        git_run(self.repo, "commit", "-q", "--allow-empty", "-m", "base")
        self.base = gitstate.git_head(self.repo)
        pin = {"file": "cmake/Dependencies.cmake", "var": "X_REVISION", "checkout": "../dep", "name": "Dep"}
        common.atomic_write_text(gitstate.perf_dir(self.repo) / "config.json", json.dumps({"targets": {}, "pins": [pin]}))
        patcher = mock.patch.object(pins.shutil, "which", lambda name: None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def commit_pin(self, sha, name="cmake/Dependencies.cmake"):
        (self.repo / name).write_text(cmake(sha), encoding="utf-8")
        git_run(self.repo, "add", ".")
        git_run(self.repo, "commit", "-q", "-m", "pin")
        return gitstate.git_head(self.repo)

    def gate(self, head, remote=None):
        args = type("A", (), {"repo": str(self.repo), "refs_from_stdin": True, "lock_timeout": 1.0})
        real = sys.stdin
        sys.stdin = io.StringIO(f"refs/heads/main {head} refs/heads/main {remote or self.base}\n")
        out = io.StringIO()
        try:
            with mock.patch("sys.stdout", out):
                code = gate.gate_command(args)
        finally:
            sys.stdin = real
        return code, out.getvalue()

    def test_pin_to_an_unpushed_commit_is_refused_and_logged(self):
        code, text = self.gate(self.commit_pin(self.unpushed))
        self.assertEqual(code, 1)
        self.assertIn("push Dep first", text)
        self.assertIn("pins-refused", (gitstate.perf_dir(self.repo) / "log.jsonl").read_text(encoding="utf-8"))

    def test_pin_to_a_pushed_commit_goes_on(self):
        self.assertEqual(self.gate(self.commit_pin(self.pushed))[0], 0)

    def test_a_push_that_leaves_the_pin_file_alone_is_not_checked(self):
        head = self.commit_pin(self.unpushed)
        self.assertEqual(self.gate(head, remote=head)[0], 0)  # nothing new to push: the range is empty

    def test_the_performance_skip_does_not_skip_the_pin_check(self):
        head = self.commit_pin(self.unpushed)
        with mock.patch.dict(os.environ, {"SUB0MATH_PERF_SKIP": "flaky"}):
            self.assertEqual(self.gate(head)[0], 1)

    def test_the_pin_skip_is_logged(self):
        head = self.commit_pin(self.unpushed)
        with mock.patch.dict(os.environ, {"SUB0MATH_PINS_SKIP": "pushed from another machine"}):
            self.assertEqual(self.gate(head)[0], 0)
        self.assertIn("pins-skip", (gitstate.perf_dir(self.repo) / "log.jsonl").read_text(encoding="utf-8"))

    def test_unverifiable_pin_warns_but_goes_on(self):
        code, text = self.gate(self.commit_pin(OTHER))
        self.assertEqual(code, 0)
        self.assertIn("could not verify", text)

    def test_documentation_only_push_is_not_checked(self):
        (self.repo / "notes.md").write_text("x", encoding="utf-8")
        git_run(self.repo, "add", ".")
        git_run(self.repo, "commit", "-q", "-m", "docs")
        self.assertEqual(self.gate(gitstate.git_head(self.repo))[0], 0)


class InstallPinsTest(RepoTestCase):
    def install(self, *pin_options):
        args = type("A", (), {"repo": str(self.repo), "build_dir": str(self.repo / "b"), "target": "tgt", "exe": str(self.repo / "b" / "x"),
                              "baseline": str(self.repo / "b.json"), "config": "", "pin": list(pin_options)})
        return hooks.install_hooks_command(args)

    def pins_in_config(self):
        return json.loads((gitstate.perf_dir(self.repo) / "config.json").read_text(encoding="utf-8")).get("pins")

    def test_pins_are_stored_and_reinstall_replaces_the_same_pin(self):
        self.assertEqual(self.install("file=a.cmake,var=X,repo=o/n"), 0)
        self.install("file=a.cmake,var=X,repo=o/m", "file=b.cmake,var=Y,checkout=../y")
        stored = self.pins_in_config()
        self.assertEqual([(p["file"], p.get("repo")) for p in stored], [("a.cmake", "o/m"), ("b.cmake", None)])

    def test_reinstall_without_pins_keeps_them(self):
        self.install("file=a.cmake,var=X,repo=o/n")
        self.install()
        self.assertEqual(len(self.pins_in_config()), 1)

    def test_a_bad_pin_installs_nothing(self):
        self.assertEqual(self.install("file=a.cmake"), 2)
        self.assertFalse((gitstate.perf_dir(self.repo) / "config.json").is_file())
