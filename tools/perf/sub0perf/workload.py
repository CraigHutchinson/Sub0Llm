"""Correctness-gated Linux whole-process evidence, with explicit RSS budgets.

Elapsed samples are diagnostics, not calibrated sub0perf regression verdicts or model tok/s.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time

from .common import atomic_write_text
from .gitstate import git_head, tree_clean
from .load import wait_for_quiet
from .lock import LockTimeout, PerfLock
from .power import power_state


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def execute(command, cwd, environment, timeout):
    """wait4 is per child: earlier correctness/warmup RSS cannot pollute measured samples."""
    if sys.platform != "linux":
        raise ValueError("workload RSS capture currently requires Linux wait4 (KiB)")
    with tempfile.TemporaryFile() as output:
        start = time.monotonic()
        proc = subprocess.Popen(command, cwd=cwd, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while True:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
                if pid:
                    proc.returncode = os.waitstatus_to_exitcode(status)
                    elapsed = time.monotonic() - start
                    break
                if time.monotonic() - start >= timeout:
                    raise TimeoutError(f"workload exceeded {timeout} seconds")
                time.sleep(0.01)
        finally:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                _, status, _ = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
        output.seek(0)
        output_hash = hashlib.sha256()
        for block in iter(lambda: output.read(1024 * 1024), b""):
            output_hash.update(block)
        if proc.returncode:
            raise ValueError(f"command failed with exit {proc.returncode}: {command[0]}")
        return {"elapsed_s": elapsed, "peak_rss_mib": usage.ru_maxrss / 1024,
                "output_sha256": output_hash.hexdigest()}


def command_at_root(command, root):
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("commands must be nonempty argument arrays")
    # Explicit executable paths avoid silently selecting a different binary through PATH.
    executable = Path(command[0])
    if not executable.is_absolute():
        executable = root / executable
    return [str(executable.resolve(strict=True)), *command[1:]]


def measure(spec_path, repo, rounds=3, timeout=120, max_load=0.05, settle=10):
    if rounds < 3 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("at least three rounds and a positive finite timeout required")
    if not 0 <= max_load <= 0.05 or not 0 <= settle <= 60:
        raise ValueError("max-load must be 0..0.05 and settle 0..60 seconds")
    spec_path, repo = Path(spec_path).resolve(), Path(repo).resolve()
    if git_head(repo) is None:
        raise ValueError("--repo must identify a Git checkout for shared locking/provenance")
    spec = json.loads(spec_path.read_text())
    root = spec_path.parent
    command = command_at_root(spec["command"], root)
    correctness = command_at_root(spec["correctness"], root)
    if not isinstance(spec["identity"], dict) or not spec["identity"]:
        raise ValueError("nonempty workload identity required")
    if not isinstance(spec["inputs"], dict) or not spec["inputs"]:
        raise ValueError("named dataset/configuration or fixture input files required")
    if spec["cache"] not in ("warm", "uncontrolled"):
        raise ValueError("only warm/uncontrolled cache supported; verified cold is not implemented")
    json.dumps(spec["identity"], allow_nan=False)
    inputs = {name: (root / path).resolve(strict=True) for name, path in spec["inputs"].items()}
    environment = dict(os.environ)
    overrides = spec.get("environment", {})
    if not isinstance(overrides, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()):
        raise ValueError("environment must map strings to strings")
    environment.update(overrides)
    captured_env = {key: environment[key] for key in sorted(set(overrides) | {
        "OMP_NUM_THREADS", "OMP_PLACES", "OMP_PROC_BIND", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}) if key in environment}
    with PerfLock(repo, "workload evidence", timeout=0):
        state = power_state()
        if state.untrusted or state.source == "fake":
            raise ValueError("real trusted power state required for workload evidence")
        hashes = {name: digest(path) for name, path in inputs.items()}
        binaries = {"measured": digest(command[0]), "correctness": digest(correctness[0])}
        spec_hash = digest(spec_path)
        verified = execute(correctness, root, environment, timeout)
        if spec["cache"] == "warm":
            execute(command, root, environment, timeout)
        samples = []
        for _ in range(rounds):
            load, _, quiet = wait_for_quiet(max_load, settle)
            if not quiet:
                raise ValueError("host remains busy; no workload evidence recorded")
            current = power_state()
            if current.untrusted or current.as_dict() != state.as_dict():
                raise ValueError("power state changed; no workload evidence recorded")
            samples.append({**execute(command, root, environment, timeout), "load": load})
        current = power_state()
        if current.untrusted or current.as_dict() != state.as_dict():
            raise ValueError("power state changed; no workload evidence recorded")
        # Refuse an artifact whose identity changed while the commands ran.
        if hashes != {name: digest(path) for name, path in inputs.items()} or binaries != {
                "measured": digest(command[0]), "correctness": digest(correctness[0])} or spec_hash != digest(spec_path):
            raise ValueError("workload inputs or executables changed during measurement")
    return {"schema": "sub0perf-workload-1", "scope": "whole-process invocation", "elapsed_verdict": "unjudged",
            "identity": spec["identity"], "inputs": hashes, "cache": spec["cache"], "warmups": int(spec["cache"] == "warm"),
            "arguments": command[1:], "correctness_arguments": correctness[1:],
            "correctness_binary_sha256": binaries["correctness"],
            "adapter_sha256": digest(__file__), "timer_poll_s": 0.01, "environment": captured_env, "affinity": sorted(os.sched_getaffinity(0)),
            "host": platform.node(), "system": platform.platform(), "power": state.as_dict(),
            "command": command, "binaries": binaries, "spec_sha256": spec_hash,
            "source": {"commit": git_head(repo), "tracked_clean": tree_clean(repo)},
            "correctness": {"command": correctness, "exit_code": 0, **verified}, "samples": samples}


COMPATIBILITY_FIELDS = (
    "scope", "identity", "inputs", "cache", "warmups", "arguments", "correctness_arguments",
    "correctness_binary_sha256", "adapter_sha256", "timer_poll_s", "environment", "affinity", "host", "system", "power",
)


def validate(record):
    """Reject incomplete/invalid evidence instead of interpreting it as a budget pass."""
    if not isinstance(record, dict) or record.get("schema") != "sub0perf-workload-1":
        raise ValueError("unsupported workload schema")
    if any(key not in record for key in COMPATIBILITY_FIELDS):
        raise ValueError("incomplete workload identity/method")
    if record["scope"] != "whole-process invocation" or record.get("elapsed_verdict") != "unjudged":
        raise ValueError("unsupported workload scope or elapsed verdict")
    correctness = record.get("correctness")
    if not isinstance(correctness, dict) or type(correctness.get("exit_code")) is not int or correctness["exit_code"] != 0:
        raise ValueError("failed/missing correctness gate")
    for key in ("identity", "inputs"):
        if not isinstance(record[key], dict) or not record[key]:
            raise ValueError(f"invalid {key}")
    for key in ("correctness_binary_sha256", "adapter_sha256"):
        value = record[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid {key}")
    for value in record["inputs"].values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid input hash")
    if record["cache"] not in ("warm", "uncontrolled") or type(record["warmups"]) is not int or record["warmups"] != int(record["cache"] == "warm"):
        raise ValueError("invalid cache method")
    if record["timer_poll_s"] != 0.01 or isinstance(record["timer_poll_s"], bool):
        raise ValueError("unsupported timer method")
    for key in ("arguments", "correctness_arguments", "affinity"):
        if not isinstance(record[key], list):
            raise ValueError(f"invalid {key}")
    if not isinstance(record["environment"], dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in record["environment"].items()):
        raise ValueError("invalid environment")
    for key in ("arguments", "correctness_arguments"):
        if not all(isinstance(value, str) for value in record[key]):
            raise ValueError(f"invalid {key}")
    if not record["affinity"] or not all(type(value) is int and value >= 0 for value in record["affinity"]):
        raise ValueError("invalid affinity")
    if not all(isinstance(record[key], str) and record[key] for key in ("host", "system")):
        raise ValueError("invalid host/system")
    json.dumps(record["identity"], allow_nan=False)
    if not isinstance(record["power"], dict) or record["power"].get("state") not in ("ac", "unknown") or record["power"].get("saver") or record["power"].get("throttled"):
        raise ValueError("invalid/untrusted power evidence")
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) < 3:
        raise ValueError("at least three measured samples required")
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("invalid workload sample")
        for key in ("elapsed_s", "peak_rss_mib"):
            value = sample.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid workload metric")


def check(record, baseline, max_rss_mib):
    validate(record)
    if not math.isfinite(max_rss_mib) or max_rss_mib <= 0:
        raise ValueError("positive finite RSS budget required")
    if baseline is not None:
        validate(baseline)
        for key in COMPATIBILITY_FIELDS:
            if record[key] != baseline[key]:
                raise ValueError(f"incompatible workload: {key}")
    peak = max(sample["peak_rss_mib"] for sample in record["samples"])
    return {"memory_verdict": "within-budget" if peak <= max_rss_mib else "over-budget",
            "peak_rss_mib": peak, "budget_mib": max_rss_mib,
            "compatibility": "matched" if baseline is not None else "not-compared",
            "elapsed_verdict": "unjudged"}


def outcome(status, **details):
    return {"schema": "sub0perf-workload-outcome-1", "status": status,
            "elapsed_verdict": "unjudged", **details}


class OutcomeParser(argparse.ArgumentParser):
    def error(self, message):
        print(json.dumps(outcome("refused", error=message, next_action="correct arguments or evidence; no timing verdict")))
        self.exit(2)


def main(argv=None):
    parser = OutcomeParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    run = commands.add_parser("measure")
    run.add_argument("--spec", required=True)
    run.add_argument("--repo", default=".")
    run.add_argument("--out", required=True)
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--max-load", type=float, default=0.05)
    run.add_argument("--settle", type=float, default=10)
    judge = commands.add_parser("check")
    judge.add_argument("--results", required=True)
    judge.add_argument("--baseline")
    judge.add_argument("--max-rss-mib", type=float, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "measure":
            if Path(args.out).exists():
                raise ValueError("--out must be a new evidence path; retain earlier records explicitly")
            # Existing shared helpers may print diagnostics; stdout remains one JSON outcome.
            with contextlib.redirect_stdout(sys.stderr):
                record = measure(args.spec, args.repo, args.rounds, args.timeout, args.max_load, args.settle)
            validate(record)
            atomic_write_text(args.out, json.dumps(record, indent=2, allow_nan=False) + "\n", overwrite=False)
            print(json.dumps(outcome("recorded", record=str(Path(args.out).resolve()), samples=len(record["samples"]),
                                     next_action="check with an explicit RSS budget; elapsed remains unjudged")))
            return 0
        result = check(json.loads(Path(args.results).read_text()),
                       json.loads(Path(args.baseline).read_text()) if args.baseline else None, args.max_rss_mib)
        print(json.dumps(outcome(result["memory_verdict"], **result,
                                 next_action=("inspect memory use or explicitly review the budget" if result["memory_verdict"] == "over-budget"
                                              else "memory only; use the calibrated gate for timing verdicts"))))
        return int(result["memory_verdict"] == "over-budget")
    except (OSError, ValueError, KeyError, TypeError, TimeoutError, LockTimeout) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
