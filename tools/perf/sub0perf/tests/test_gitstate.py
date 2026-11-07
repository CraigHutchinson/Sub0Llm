"""Tests for gitstate."""

import json
import subprocess
import unittest

from .. import gitstate
from .helpers import git_run, RepoTestCase


class PushedRefsTest(unittest.TestCase):
    HEAD = "a" * 40
    OTHER = "b" * 40
    ZERO = "0" * 40

    def test_parse(self):
        text = f"refs/heads/main {self.HEAD} refs/heads/main {self.OTHER}\n\nrefs/heads/x {self.ZERO} refs/heads/x {self.OTHER}\n"
        self.assertEqual(gitstate.parse_push_refs(text), [("refs/heads/main", self.HEAD, "refs/heads/main", self.OTHER),
                                                    ("refs/heads/x", self.ZERO, "refs/heads/x", self.OTHER)])
        self.assertEqual(gitstate.parse_push_refs(""), [])

    def test_documentation_only_pushes_are_not_measured(self):
        update = [("refs/heads/main", self.HEAD, "refs/heads/main", self.OTHER)]
        docs = lambda *_: ["README.md", "docs/DESIGN.md", "tools/perf/README.md", "LICENSE"]
        code = lambda *_: ["README.md", "include/x.hpp"]
        self.assertTrue(gitstate.documentation_only(update, docs))
        self.assertFalse(gitstate.documentation_only(update, code))
        self.assertFalse(gitstate.documentation_only(update, lambda *_: None))  # remote commit unknown: measure
        new_branch = [("refs/heads/x", self.HEAD, "refs/heads/x", self.ZERO)]
        self.assertFalse(gitstate.documentation_only(new_branch, docs))  # nothing to diff against: measure
        self.assertFalse(gitstate.documentation_only([("refs/heads/x", self.ZERO, "refs/heads/x", self.OTHER)], docs))
        self.assertFalse(gitstate.is_documentation("docsy/a.cpp"))
        self.assertTrue(gitstate.is_documentation("upstream/bikelab.json"))  # provenance manifests are not built
        self.assertFalse(gitstate.is_documentation("upstream/patch.cpp"))
        self.assertTrue(gitstate.is_documentation(".github/workflows/ci.yml"))  # CI configuration is not built
        self.assertTrue(gitstate.is_documentation(".ci/local.json"))

    def refs(self, *shas):
        return [(f"refs/heads/b{i}", sha, f"refs/heads/b{i}", self.ZERO) for i, sha in enumerate(shas)]

    def test_head_pushed_from_a_clean_tree_is_fine(self):
        self.assertEqual(gitstate.push_problems(self.refs(self.HEAD), self.HEAD, True), [])

    def test_dirty_tree_is_refused(self):
        problems = gitstate.push_problems(self.refs(self.HEAD), self.HEAD, False)
        self.assertEqual(len(problems), 1)
        self.assertIn("uncommitted", problems[0])

    def test_pushing_something_that_is_not_head_is_refused(self):
        problems = gitstate.push_problems(self.refs(self.OTHER), self.HEAD, True)
        self.assertEqual(len(problems), 1)
        self.assertIn("refs/heads/b0", problems[0])

    def test_one_foreign_ref_among_several_is_refused(self):
        self.assertEqual(len(gitstate.push_problems(self.refs(self.HEAD, self.OTHER), self.HEAD, True)), 1)

    def test_deletes_are_ignored(self):
        self.assertEqual(gitstate.push_problems(self.refs(self.ZERO), self.HEAD, False), [])
        self.assertEqual(gitstate.push_problems(self.refs(self.ZERO, self.HEAD), self.HEAD, True), [])

    def test_tag_objects_resolve_to_their_commit(self):
        tag = "c" * 40
        self.assertEqual(gitstate.push_problems(self.refs(tag), self.HEAD, True, lambda sha: self.HEAD if sha == tag else sha), [])


def git_run(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True,
                   capture_output=True)


class AcceptedSinceTest(RepoTestCase):
    def baseline(self, *accepted):
        return json.dumps({"accepted": [{"benchmarks": {n: {}}} for n in accepted]})

    def test_reads_the_committed_baseline_not_the_working_tree(self):
        rel = "perf/baselines/h.json"
        (self.repo / "perf" / "baselines").mkdir(parents=True)
        (self.repo / rel).write_text(self.baseline(), encoding="utf-8")
        git_run(self.repo, "add", ".")
        git_run(self.repo, "commit", "-q", "-m", "one")
        ref = gitstate.git_head(self.repo)
        (self.repo / rel).write_text(self.baseline("A"), encoding="utf-8")
        git_run(self.repo, "commit", "-q", "-am", "two")  # A accepted in HEAD
        self.assertEqual(gitstate.accepted_since(self.repo, rel, ref), {"A"})
        (self.repo / rel).write_text(self.baseline("A", "B"), encoding="utf-8")  # uncommitted acceptance does not count
        self.assertEqual(gitstate.accepted_since(self.repo, rel, ref), {"A"})

    def test_baseline_missing_at_head_accepts_nothing(self):
        git_run(self.repo, "commit", "-q", "--allow-empty", "-m", "one")
        (self.repo / "b.json").write_text(self.baseline("A"), encoding="utf-8")
        self.assertEqual(gitstate.accepted_since(self.repo, "b.json", None), set())



class KeptBinaryTest(RepoTestCase):
    """The benchmark executable of a record is kept so a later gate can measure that build again."""

    def commit(self, text):
        (self.repo / "f.txt").write_text(text, encoding="utf-8")
        git_run(self.repo, "add", ".")
        git_run(self.repo, "commit", "-q", "-m", text)
        return gitstate.git_head(self.repo)

    def exe(self, content):
        path = self.repo.parent / "bench.exe"
        path.write_bytes(content)
        return path

    def test_record_keeps_the_executable_beside_it(self):
        sha = self.commit("one")
        gitstate.save_record(self.repo, "t", {"samples": {}}, exe=self.exe(b"build one"))
        kept = gitstate.reference_binary(self.repo, "t", sha)
        self.assertEqual(kept.read_bytes(), b"build one")
        self.assertEqual(kept.name, "bench.exe")

    def test_no_executable_means_no_binary(self):
        sha = self.commit("one")
        gitstate.save_record(self.repo, "t", {"samples": {}})
        self.assertIsNone(gitstate.reference_binary(self.repo, "t", sha))
        self.assertIsNone(gitstate.reference_binary(self.repo, "t", None))

    def test_only_the_newest_are_kept(self):
        import os
        shas = []
        for i in range(gitstate.KEEP_BINARIES + 2):
            shas.append(self.commit(f"c{i}"))
            gitstate.save_record(self.repo, "t", {"samples": {}}, exe=self.exe(f"build {i}".encode()))
            folder = gitstate.perf_dir(self.repo) / "bins" / "t" / shas[-1]
            os.utime(folder, (1000 + i, 1000 + i))  # coarse file times must not decide the order
        kept = [s for s in shas if gitstate.reference_binary(self.repo, "t", s)]
        self.assertEqual(kept, shas[-gitstate.KEEP_BINARIES:])

    def test_a_dirty_tree_keeps_nothing(self):
        sha = self.commit("one")
        (self.repo / "f.txt").write_text("changed", encoding="utf-8")
        gitstate.save_record(self.repo, "t", {"samples": {}}, exe=self.exe(b"x"))
        self.assertIsNone(gitstate.reference_binary(self.repo, "t", sha))
