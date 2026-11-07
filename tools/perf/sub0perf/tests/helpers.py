"""Shared test fixtures: synthetic samples records, git helpers."""

import random
import subprocess
import tempfile
import os
import unittest
from pathlib import Path
from unittest import mock

from ..baseline import baseline_data
from ..judge import evaluate

ROUNDS = 5
NAMES = [f"Bench {i}" for i in range(8)]


def record(times, jitter=0.01, seed=1):
    """A samples record: `times` maps name to ns; constant calibration; small multiplicative jitter per round."""
    rng = random.Random(seed)
    meta = [{"round": r, "load": 0.0, "waited_s": 0.0, "quiet": True, "calibration_start_ns": 1000.0,
             "calibration_end_ns": 1000.0} for r in range(ROUNDS)]
    samples = {n: [t * (1.0 + rng.uniform(-jitter, jitter)) for _ in range(ROUNDS)] for n, t in times.items()}
    positions = {n: [0.5] * ROUNDS for n in times}
    return {"rounds": meta, "samples": samples, "positions": positions, "counters": {}, "epochs": 0}


def judge_against(base_times, now_times, seed=2):
    baseline = baseline_data(record(base_times, seed=1))
    rows, pairs, env = evaluate(record(now_times, seed=seed), baseline, 0.10, 0.05, 0.05, False)
    return {r[0]: r[5] for r in rows}, pairs, env


def git_run(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True,
                   capture_output=True)


class RepoTestCase(unittest.TestCase):
    def setUp(self):
        # Decisions must not depend on the host being on battery: tests fake mains unless they set the power state themselves.
        patch = mock.patch.dict(os.environ, {"SUB0PERF_FAKE_POWER": "ac"})
        patch.start()
        self.addCleanup(patch.stop)
        for name in ("SUB0MATH_PERF_ON_BATTERY", "SUB0MATH_PERF_POWER_STRICT"):
            os.environ.pop(name, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "my repo"  # with a space, as users have
        self.repo.mkdir()
        git_run(self.repo, "init", "-q")
