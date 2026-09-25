"""Build the perplexity quality-gate fixture from the local training corpora.

The gate (`scripts/run_perf_suite.py --stage ppl`, `docs/OPTIMIZATION_PROCESS.md`) scores this text
through the decode path and compares arms token by token. It is a blend -- educational prose,
textbook-style prose, mixed web text, math word problems -- so a precision change that only hurts one
kind of text still shows up.

The text itself is NOT committed: it is excerpted from local corpora whose redistribution terms this
repo has not verified. Instead the excerpt positions are fixed here and the result is pinned by SHA-256,
so any machine holding the same corpora reproduces the identical fixture, and a machine whose corpora
differ fails loudly instead of silently scoring different text.

Usage:  python scripts/make_ppl_fixture.py [--out PATH]      (default: out/quality/ppl_blend_v1.txt)
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (corpus, fraction of the file to seek to). Each excerpt starts at the first line break after that
# point, reads 2600 bytes, and is cut back to the last line break after byte 1200.
SOURCES = [
    ("data/fineweb_edu.txt", 0.37),
    ("data/cosmopedia.txt", 0.53),
    ("data/minipile.txt", 0.61),
    ("data/gsm8k.txt", 0.29),
]
VERSION = "ppl_blend_v1"
EXPECTED_SHA256 = "8e2bfd02c2a055b1f1656af11de8593808ef79cf64b7219b4d398fad052e63ff"


def excerpt(path: pathlib.Path, frac: float) -> str:
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(int(size * frac))
        f.readline()
        chunk = f.read(2600)
    cut = chunk.rfind(b"\n", 1200)
    chunk = chunk[: cut if cut > 0 else len(chunk)]
    return chunk.decode("utf-8", errors="ignore").strip()


def build() -> str:
    return "\n\n".join(excerpt(ROOT / p, frac) for p, frac in SOURCES) + "\n"


def ensure(out: pathlib.Path) -> pathlib.Path:
    """Write the fixture to `out` if absent or stale; raise if the corpora no longer reproduce it."""
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == EXPECTED_SHA256:
        return out
    missing = [p for p, _ in SOURCES if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"{VERSION}: corpora missing: {', '.join(missing)}")
    data = build().encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"{VERSION}: corpora produce sha256 {digest}, expected {EXPECTED_SHA256}. "
                         "The local corpora differ from the ones this fixture was pinned against, so "
                         "scores would not be comparable with recorded history.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "out" / "quality" / f"{VERSION}.txt")
    args = ap.parse_args()
    path = ensure(args.out)
    print(f"{VERSION}: {path} ({path.stat().st_size} bytes, sha256 {EXPECTED_SHA256})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
