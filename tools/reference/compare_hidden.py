"""Compare corresponding raw little-endian f32 hidden dumps without NumPy."""
import argparse
import json
import math
import struct
from pathlib import Path

def read(path):
    raw = Path(path).read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"invalid f32 dump: {path}")
    values = struct.unpack(f"<{len(raw)//4}f", raw)
    if not all(map(math.isfinite, values)):
        raise ValueError(f"nonfinite dump: {path}")
    return values


def compare(actual_path, reference_path, tolerance):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    actual, reference = read(actual_path), read(reference_path)
    if len(actual) != len(reference):
        raise ValueError("dump dimensions differ")
    squared = math.fsum((a-b)**2 for a,b in zip(actual, reference))
    denominator = math.fsum(v*v for v in reference)
    relative = math.sqrt(squared/denominator) if denominator else (0.0 if not squared else math.inf)
    return {"values": len(actual), "max_abs": max(abs(a-b) for a,b in zip(actual, reference)),
            "relative_l2": relative, "tolerance": tolerance, "passed": relative <= tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual")
    parser.add_argument("reference")
    parser.add_argument("--max-relative-l2", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.max_relative_l2) or args.max_relative_l2 < 0:
        parser.error("tolerance must be finite and nonnegative")
    result = compare(args.actual, args.reference, args.max_relative_l2)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
