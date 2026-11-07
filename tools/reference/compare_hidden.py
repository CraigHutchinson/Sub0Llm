"""Compare corresponding raw little-endian f32 hidden dumps without NumPy."""
import argparse
import json
import math
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("actual")
parser.add_argument("reference")
parser.add_argument("--max-relative-l2", type=float, required=True)
args = parser.parse_args()
if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0:
    parser.error("tolerance must be finite and nonnegative")

def read(path):
    raw = Path(path).read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid f32 dump: {path}")
    values = struct.unpack(f"<{len(raw)//4}f", raw)
    if not all(map(math.isfinite, values)):
        raise ValueError(f"nonfinite dump: {path}")
    return values

actual, reference = read(args.actual), read(args.reference)
if len(actual) != len(reference):
    raise ValueError("dump dimensions differ")
delta = [a-b for a,b in zip(actual, reference)]
denominator = math.fsum(v*v for v in reference)
squared = math.fsum(v*v for v in delta)
relative = math.sqrt(squared/denominator) if denominator else (0.0 if not squared else math.inf)
result = {"values": len(actual), "max_abs": max(map(abs, delta)), "relative_l2": relative,
          "tolerance": args.max_relative_l2, "passed": relative <= args.max_relative_l2}
print(json.dumps(result))
raise SystemExit(0 if result["passed"] else 1)
