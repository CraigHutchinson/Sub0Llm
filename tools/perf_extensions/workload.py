"""Compatibility entry point; generic workload evidence is maintained upstream in sub0perf."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perf"))
from sub0perf.workload import main

if __name__ == "__main__":
    raise SystemExit(main())
