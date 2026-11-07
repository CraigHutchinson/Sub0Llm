#!/usr/bin/env python3
"""Verify the checked-in shared tooling matches its pinned Sub0Math provenance."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "tools/perf/UPSTREAM.json").read_text())
    failures = []
    expected = {entry["path"] for entry in manifest["files"]}
    sources = {str(path.relative_to(root)) for pattern in ("*.py", "*.hpp")
               for path in (root / "tools/perf").rglob(pattern)}
    failures.extend(sorted(sources - expected))
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            failures.append(entry["path"])
    if failures:
        raise SystemExit("Sub0Math vendor drift: " + ", ".join(failures))
    print(f"Verified {len(manifest['files'])} files from Sub0Math {manifest['commit']}")


if __name__ == "__main__":
    main()
