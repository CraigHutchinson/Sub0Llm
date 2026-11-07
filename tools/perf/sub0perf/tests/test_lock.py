"""Tests for lock."""

import json
import os
import subprocess
import sys
import unittest

from .. import lock, platform_proc
from .helpers import git_run, RepoTestCase


class LockTest(RepoTestCase):
    def setUp(self):
        super().setUp()
        git_run(self.repo, "commit", "-q", "--allow-empty", "-m", "one")

    def lock(self, **kw):
        kw.setdefault("sleep", lambda s: None)
        return lock.PerfLock(self.repo, "test", **kw)

    def test_acquire_and_release(self):
        lock = self.lock()
        with lock:
            self.assertTrue(lock.path.is_file())
            self.assertEqual(json.loads(lock.path.read_text(encoding="utf-8"))["pid"], os.getpid())
        self.assertFalse(lock.path.exists())

    def test_released_after_an_exception(self):
        lock = self.lock()
        with self.assertRaises(RuntimeError):
            with lock:
                raise RuntimeError
        self.assertFalse(lock.path.exists())

    def test_live_holder_blocks_until_timeout(self):
        with self.lock():
            ticks = iter(range(0, 10000, 5))
            second = self.lock(timeout=20, clock=lambda: next(ticks))
            with self.assertRaises(lock.LockTimeout):
                second.__enter__()
            self.assertTrue(second.path.exists())  # the holder's lock survived

    def test_waiter_gets_the_lock_when_the_holder_releases(self):
        first = self.lock()
        first.__enter__()
        waits = []

        def sleep(s):
            waits.append(s)
            first.__exit__(None, None, None)
        second = self.lock(sleep=sleep)
        with second:
            self.assertEqual(len(waits), 1)

    def test_dead_holder_is_taken_over(self):
        lock = self.lock(alive=lambda pid: False)
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        lock.path.write_text(json.dumps({"pid": 99999999, "time": 0, "what": "old"}), encoding="utf-8")
        with lock:
            self.assertEqual(json.loads(lock.path.read_text(encoding="utf-8"))["pid"], os.getpid())

    def test_old_lock_of_a_live_pid_is_stale(self):
        lock = self.lock(alive=lambda pid: True, stale_s=100, clock=lambda: 1000.0)
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        lock.path.write_text(json.dumps({"pid": 1, "time": 1.0, "what": "old"}), encoding="utf-8")
        with lock:
            pass

    def test_pid_alive(self):
        self.assertTrue(platform_proc.pid_alive(os.getpid()))
        self.assertFalse(platform_proc.pid_alive(0))
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        self.assertFalse(platform_proc.pid_alive(child.pid))

