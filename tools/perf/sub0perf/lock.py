"""One measuring run per repository: a lock file under `.git/sub0perf`."""

import json
import os
import time

from . import platform_proc
from .gitstate import perf_dir


class LockTimeout(Exception):
    pass


class PerfLock:
    """Serialises measuring runs on one repository: `.git/sub0perf/lock`, created exclusively, holding pid and time.

    A lock whose process is gone, or that is older than `stale_s`, is taken over. Use as a context manager.
    """

    def __init__(self, repo, what, timeout=900.0, stale_s=6 * 3600.0, poll=2.0, alive=platform_proc.pid_alive, clock=time.time, sleep=time.sleep):
        d = perf_dir(repo)
        self.path = None if d is None else d / "lock"
        self.what, self.timeout, self.stale_s, self.poll = what, timeout, stale_s, poll
        self.alive, self.clock, self.sleep = alive, clock, sleep

    def _holder(self):
        try:
            info = json.loads(self.path.read_text(encoding="utf-8"))
            return int(info["pid"]), float(info["time"]), str(info.get("what", ""))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _stale(self):
        holder = self._holder()
        if holder is None:  # being written, or garbage: only old files count
            try:
                return self.clock() - self.path.stat().st_mtime > 60.0
            except OSError:
                return False
        pid, since, _ = holder
        return not self.alive(pid) or self.clock() - since > self.stale_s

    def __enter__(self):
        if self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = self.clock() + self.timeout
        announced = False
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._stale():
                    print("sub0perf: removing a stale lock")
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                if self.clock() >= deadline:
                    holder = self._holder()
                    raise LockTimeout(f"another sub0perf run holds the lock {self.path}"
                                      + (f" ({holder[2]}, pid {holder[0]})" if holder else ""))
                if not announced:
                    holder = self._holder()
                    print(f"sub0perf: waiting for another run ({holder[2]}, pid {holder[0]}) to finish" if holder
                          else "sub0perf: waiting for another run to finish", flush=True)
                    announced = True
                self.sleep(self.poll)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "time": self.clock(), "what": self.what}, f)
            return self

    def __exit__(self, *exc):
        if self.path is not None:
            holder = self._holder()
            if holder is not None and holder[0] == os.getpid():
                try:
                    self.path.unlink()
                except OSError:
                    pass
        return False
