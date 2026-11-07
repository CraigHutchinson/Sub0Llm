"""Constants and helpers every layer shares."""

import json
import os
import tempfile
from pathlib import Path

CALIBRATION = "__calibration"
REFERENCE_MARK = "hand-written"


def group_of(name):
    """The text before the first ", ": members of a group run back to back and pair with its reference."""
    return name.split(", ", 1)[0]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tooling_script():
    """The `bench_check.py` entry point, which consumers and installed hooks call by path."""
    return Path(__file__).resolve().parent.parent / "bench_check.py"


def atomic_write_text(path, text, *, overwrite=True):
    """Publish complete text atomically. With overwrite=False an existing path is never replaced.

    No-clobber publication uses a hard link in the same directory; unsupported filesystems refuse
    rather than fall back to a racy exists/replace sequence. Existing callers keep replace semantics.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        if overwrite:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
            os.unlink(tmp)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
