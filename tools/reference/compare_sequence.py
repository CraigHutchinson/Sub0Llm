"""Gate every requested position's f32 logits; prefixes use .posN before .logits."""
import argparse
import json
from pathlib import Path
from compare_hidden import compare


def compare_sequence(actual, reference, positions, tolerance, expected_values):
    if positions < 1 or expected_values < 1:
        raise ValueError("positive position and vocabulary counts required")
    results = []
    for position in range(positions):
        suffix = f".pos{position}" if position else ""
        result = compare(Path(f"{actual}{suffix}.logits"), Path(f"{reference}{suffix}.logits"), tolerance)
        if result["values"] != expected_values:
            raise ValueError(f"position {position}: expected {expected_values} logits, got {result['values']}")
        results.append({"position": position, **result})
    return {"passed": all(result["passed"] for result in results), "positions": results,
            "worst_relative_l2": max(result["relative_l2"] for result in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual_prefix")
    parser.add_argument("reference_prefix")
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--expected-values", type=int, required=True)
    parser.add_argument("--max-relative-l2", type=float, required=True)
    args = parser.parse_args()
    try:
        result = compare_sequence(args.actual_prefix, args.reference_prefix, args.positions,
                                  args.max_relative_l2, args.expected_values)
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        return 2
    print(json.dumps(result))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
