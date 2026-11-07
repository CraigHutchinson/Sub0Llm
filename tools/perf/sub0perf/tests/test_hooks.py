"""Tests for hooks."""

import shutil
import subprocess
import unittest
from pathlib import Path

from .. import gitstate
from .. import hooks as hk
from .helpers import RepoTestCase


@unittest.skipUnless(shutil.which("sh"), "needs a POSIX sh")
class HooksTest(RepoTestCase):
    def install(self):
        args = type("A", (), {"repo": str(self.repo), "build_dir": str(self.repo / "build dir"), "target": "tgt",
                              "exe": str(self.repo / "build dir" / "tgt"), "baseline": str(self.repo / "b.json"), "config": ""})
        hk.install_hooks_command(args)
        return Path(gitstate.git(self.repo, "rev-parse", "--git-path", "hooks"))

    def hooks_dir(self):
        hooks = Path(gitstate.git(self.repo, "rev-parse", "--git-path", "hooks"))
        return hooks if hooks.is_absolute() else self.repo / hooks

    def run_hook(self, name, *args, stdin=""):
        return subprocess.run(["sh", str(self.hooks_dir() / name), *args], input=stdin, capture_output=True, text=True, cwd=self.repo)

    def test_hook_text_quotes_paths_with_spaces(self):
        body = hk.hook_body("pre-push", "C:/Program Files/Python/python.exe", "C:/my tools/bench_check.py", "x")
        self.assertIn("python='C:/Program Files/Python/python.exe'", body)
        self.assertIn("script='C:/my tools/bench_check.py'", body)

    def test_missing_tooling_makes_the_hook_a_no_op_with_a_message(self):
        hooks = self.hooks_dir()
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "pre-push").write_text(hk.hook_body("pre-push", "/no/such/python", "/no/such/script.py", "x"), encoding="utf-8", newline="\n")
        result = self.run_hook("pre-push", "origin", "url", stdin="refs/heads/a " + "1" * 40 + " refs/heads/a " + "0" * 40 + "\n")
        self.assertEqual(result.returncode, 0)
        self.assertIn("missing", result.stdout)

    def test_install_creates_hooks_and_is_idempotent(self):
        self.install()
        first = (self.hooks_dir() / "pre-push").read_text(encoding="utf-8")
        self.install()
        self.assertEqual((self.hooks_dir() / "pre-push").read_text(encoding="utf-8"), first)
        self.assertIn(hk.HOOK_MARK, first)
        self.assertIn("--refs-from-stdin", first)
        self.assertTrue((self.hooks_dir() / "post-checkout").is_file())

    def test_reinstall_leaves_an_unchanged_running_hook_alone(self):
        self.install()
        original = hk.atomic_write_text

        def refuse_hooks(path, text):
            if Path(path).parent == self.hooks_dir():
                raise PermissionError("in use")  # what Windows does to a running hook
            original(path, text)

        hk.atomic_write_text = refuse_hooks
        try:
            self.install()  # unchanged hooks are skipped, so nothing is replaced
            (self.hooks_dir() / "pre-push").write_text("# stale ours " + hk.HOOK_MARK + "\n", encoding="utf-8")
            self.install()  # a changed one that cannot be replaced is reported, not raised
        finally:
            hk.atomic_write_text = original

    def test_foreign_hook_is_kept_and_chained_with_stdin_and_arguments(self):
        hooks = self.hooks_dir()
        hooks.mkdir(parents=True, exist_ok=True)
        out = self.repo / "seen.txt"
        (hooks / "pre-push").write_text(f'#!/bin/sh\necho "args=$@" > "{out.as_posix()}"\ncat >> "{out.as_posix()}"\n',
                                         encoding="utf-8", newline="\n")
        (hooks / "pre-push").chmod(0o755)  # a real hook is executable; git and the chain run it directly
        self.install()
        self.assertTrue((hooks / "pre-push.user").is_file())
        self.assertIn(hk.HOOK_MARK, (hooks / "pre-push").read_text(encoding="utf-8"))
        # The user's hook runs first with the arguments and stdin; ours then refuses (the pushed sha is not HEAD).
        result = self.run_hook("pre-push", "origin", "url", stdin="refs/heads/a " + "1" * 40 + " refs/heads/a " + "0" * 40 + "\n")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("push refused", result.stdout)
        seen = out.read_text(encoding="utf-8")
        self.assertIn("args=origin url", seen)
        self.assertIn("refs/heads/a", seen)

    def test_failing_user_hook_stops_the_push(self):
        hooks = self.hooks_dir()
        hooks.mkdir(parents=True, exist_ok=True)
        (hooks / "pre-push").write_text("#!/bin/sh\nexit 3\n", encoding="utf-8", newline="\n")
        (hooks / "pre-push").chmod(0o755)
        self.install()
        self.assertEqual(self.run_hook("pre-push", "origin", "url").returncode, 3)

