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


def atomic_write_text(path, text):
    """Writes `text` to `path` through a temp file in the same directory, so a reader never sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
