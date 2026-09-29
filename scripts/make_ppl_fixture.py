"""Build the perplexity quality-gate fixture from the local training corpora.

The gate (`scripts/run_perf_suite.py --stage ppl`, `docs/OPTIMIZATION_PROCESS.md`) scores this text
through the decode path and compares arms token by token. It is a blend -- educational prose,
textbook-style prose, mixed web text, math word problems -- so a precision change that only hurts one
kind of text still shows up.

The text itself is NOT committed: it is excerpted from local corpora whose redistribution terms this
repo has not verified. Instead the excerpt positions are fixed here and the result is pinned by SHA-256,
so any machine holding the same corpora reproduces the identical fixture, and a machine whose corpora
differ fails loudly instead of silently scoring different text.

Versions (each pinned by SHA-256; history rows name the version they were scored on):
  ppl_blend_v1 -- one excerpt per corpus, ~2,400 tokens. Its 95% CI half-width is ~0.02 nats/token,
                  which a pure float-reassociation change already fills (O12 control run), so it cannot
                  resolve changes near the 0.03 gate.
  ppl_blend_v2 -- four excerpts per corpus, ~4x the tokens: half-width ~0.01. The default.

Usage:  python scripts/make_ppl_fixture.py [--version V] [--out PATH]   (default: out/quality/<V>.txt)
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (corpus, fraction of the file to seek to). Each excerpt starts at the first line break after that
# point, reads 2600 bytes, and is cut back to the last line break after byte 1200.
CORPORA = ["data/fineweb_edu.txt", "data/cosmopedia.txt", "data/minipile.txt", "data/gsm8k.txt"]
FIXTURES: dict[str, tuple[list[tuple[str, float]], str]] = {
    "ppl_blend_v1": (
        [("data/fineweb_edu.txt", 0.37), ("data/cosmopedia.txt", 0.53),
         ("data/minipile.txt", 0.61), ("data/gsm8k.txt", 0.29)],
        "8e2bfd02c2a055b1f1656af11de8593808ef79cf64b7219b4d398fad052e63ff"),
    # Corpus-interleaved so a --ppl-tokens cap still samples every kind of text.
    "ppl_blend_v2": (
        [(c, f) for f in (0.14, 0.39, 0.64, 0.89) for c in CORPORA],
        "54028b9fc85f78af2b85a8f7558b09d575bf0b5a254eb9bd0a570b1c356df422"),
}
VERSION = "ppl_blend_v2"


def excerpt(path: pathlib.Path, frac: float) -> str:
    size = path.stat().st_size
    with path.open("rb") as f:
        f.seek(int(size * frac))
        f.readline()
        chunk = f.read(2600)
    cut = chunk.rfind(b"\n", 1200)
    chunk = chunk[: cut if cut > 0 else len(chunk)]
    return chunk.decode("utf-8", errors="ignore").strip()


def build(version: str) -> str:
    return "\n\n".join(excerpt(ROOT / p, frac) for p, frac in FIXTURES[version][0]) + "\n"


def default_path(version: str = VERSION) -> pathlib.Path:
    return ROOT / "out" / "quality" / f"{version}.txt"


def ensure(out: pathlib.Path, version: str = VERSION) -> pathlib.Path:
    """Write the fixture to `out` if absent or stale; raise if the corpora no longer reproduce it."""
    sources, expected = FIXTURES[version]
    if out.exists() and hashlib.sha256(out.read_bytes()).hexdigest() == expected:
        return out
    missing = [p for p, _ in sources if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(f"{version}: corpora missing: {', '.join(missing)}")
    data = build(version).encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise ValueError(f"{version}: corpora produce sha256 {digest}, expected {expected}. "
                         "The local corpora differ from the ones this fixture was pinned against, so "
                         "scores would not be comparable with recorded history.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", choices=sorted(FIXTURES), default=VERSION)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()
    path = ensure(args.out or default_path(args.version), args.version)
    print(f"{args.version}: {path} ({path.stat().st_size} bytes, sha256 {FIXTURES[args.version][1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
