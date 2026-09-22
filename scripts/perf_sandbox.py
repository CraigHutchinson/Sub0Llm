#!/usr/bin/env python3
"""Time-limited, best-effort exclusive-CPU sandbox for benchmarks (Windows).

WHY: docs/OPTIMIZATION_PROCESS.md S1 refuses to measure under >5% background load. After a reboot the
host sat at 6-19% for a long time (updaters, browser, desktop apps settling), so a baseline simply could
not run. Rather than wait for the machine to be idle, carve out most of it for the benchmark.

WHAT IT DOES, on enter:
  1. Confine every OTHER process this user can open to a small housekeeping set of E-cores
     ("reverse affinity") and drop it to BELOW_NORMAL priority. The benchmark keeps the rest.
  2. Switch to the High Performance power plan for the duration.
  3. Re-sweep periodically, because processes spawned mid-run start with full affinity.
  4. Arm a watchdog that restores everything after --max-seconds even if the benchmark hangs.
and on exit restores every process's original affinity/priority and the original power plan.

WHAT IT CANNOT DO -- stated plainly, because over-trusting it would reintroduce the exact error the
contention check exists to prevent:
  * Unelevated (the normal case here), it cannot touch SYSTEM services or protected processes
    (Defender's MsMpEng is PPL even to an admin), nor interrupts, DPCs or kernel threads.
  * So it REDUCES contention; it does not GUARANTEE exclusivity. Callers must VERIFY with
    bench_core_load() after entering -- the question becomes "are MY cores quiet", not "is the box".

Crash safety: the saved original state is written to out/perf_sandbox_state.json BEFORE anything is
changed, so `python scripts/perf_sandbox.py --restore` can undo a sandbox whose owner died mid-run.

Linux is deliberately NOT implemented here (AGENTS.md S8: no surface nothing on this host consumes).
The Linux design -- a cgroup v2 cpuset partition via `systemctl set-property --runtime` -- is recorded
in docs/OPTIMIZATION_PROCESS.md S1a so it can be built when it has a consumer.

Usage:
  python scripts/perf_sandbox.py --dry-run          # enter, report, verify, exit, verify restored
  python scripts/perf_sandbox.py --restore          # undo a sandbox left behind by a crash
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATE = ROOT / "out" / "perf_sandbox_state.json"

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_SET_INFORMATION = 0x0200
BELOW_NORMAL_PRIORITY_CLASS = 0x4000
HIGH_PRIORITY_CLASS = 0x0080
HIGH_PERFORMANCE_PLAN = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"   # Windows' built-in SCHEME_MIN

_k32 = ctypes.WinDLL("kernel32", use_last_error=True) if os.name == "nt" else None


def cpu_topology() -> dict[int, list[int]]:
    """EfficiencyClass -> logical processor indices, from the OS itself.

    Never assume the layout: on this host (Core Ultra 9 275HX) the 8 P-cores are logical
    [0,1,10,11,12,13,22,23] -- interleaved with the E-cores, not the first eight. A mask built on
    "cores 0-7 are P" would silently hand the benchmark mostly E-cores.
    """
    n = wt.ULONG(0)
    _k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
    buf = (ctypes.c_ubyte * n.value)()
    if not _k32.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation")
    out: dict[int, list[int]] = {}
    off = 0
    while off < n.value:
        size = int.from_bytes(bytes(buf[off:off + 4]), "little")
        # SYSTEM_CPU_SET_INFORMATION: LogicalProcessorIndex at byte 14, EfficiencyClass at byte 18.
        out.setdefault(buf[off + 18], []).append(buf[off + 14])
        off += size
    return {k: sorted(v) for k, v in out.items()}


def _mask(cpus: list[int]) -> int:
    return sum(1 << c for c in cpus)


def _pids() -> list[int]:
    arr = (wt.DWORD * 8192)()
    got = wt.DWORD(0)
    if not _k32.K32EnumProcesses(arr, ctypes.sizeof(arr), ctypes.byref(got)):
        return []
    return [arr[i] for i in range(got.value // ctypes.sizeof(wt.DWORD))]


def _get(pid: int) -> tuple[int, int] | None:
    h = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        pm, sm = ctypes.c_size_t(0), ctypes.c_size_t(0)
        if not _k32.GetProcessAffinityMask(h, ctypes.byref(pm), ctypes.byref(sm)):
            return None
        return pm.value, _k32.GetPriorityClass(h)
    finally:
        _k32.CloseHandle(h)


def _set(pid: int, mask: int | None, prio: int | None) -> bool:
    h = _k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_INFORMATION, False, pid)
    if not h:
        return False
    try:
        ok = True
        if mask is not None:
            ok &= bool(_k32.SetProcessAffinityMask(h, ctypes.c_size_t(mask)))
        if prio is not None:
            ok &= bool(_k32.SetPriorityClass(h, prio))
        return ok
    finally:
        _k32.CloseHandle(h)


def _active_plan() -> str | None:
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True).stdout
        m = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", out, re.I)
        return m.group(1) if m else None
    except Exception:
        return None


def bench_core_load(cpus: list[int], samples: int = 5, interval_s: int = 2) -> float | None:
    """Average load on the BENCHMARK's cores only -- the verification a sandbox needs.

    Total-system load is the wrong question once background is confined: the housekeeping cores are
    SUPPOSED to be busy. What matters is whether anything still lands on the cores we reserved.
    """
    paths = ",".join(f"'\\Processor({c})\\% Processor Time'" for c in cpus)
    ps = (f"$s = Get-Counter @({paths}) -SampleInterval {interval_s} -MaxSamples {samples}; "
          "($s.CounterSamples | Measure-Object CookedValue -Average).Average")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True,
                             text=True, timeout=samples * interval_s + 60).stdout
        return float(out.strip().splitlines()[-1])
    except Exception:
        return None


class Sandbox:
    """Context manager. Everything it changes is recorded before it is changed, and restored on exit,
    on watchdog expiry, or by --restore after a crash."""

    def __init__(self, housekeeping: int = 2, max_seconds: int = 3600, resweep_s: float = 10.0,
                 power_plan: bool = True):
        topo = cpu_topology()
        e_cores = topo.get(min(topo), [])
        allc = sorted(c for v in topo.values() for c in v)
        # Housekeeping = the highest-numbered E-cores: background gets the least valuable silicon.
        self.housekeeping = e_cores[-housekeeping:] if housekeeping else []
        self.bench_cpus = [c for c in allc if c not in self.housekeeping]
        self.hk_mask, self.bench_mask = _mask(self.housekeeping), _mask(self.bench_cpus)
        self.max_seconds, self.resweep_s, self.power_plan = max_seconds, resweep_s, power_plan
        self.saved: dict[int, tuple[int, int]] = {}
        self.plan_before: str | None = None
        self.skipped = 0
        self._stop = threading.Event()
        self._restored = False
        self._lock = threading.Lock()

    # -- sweep -------------------------------------------------------------------------------------
    def _sweep(self) -> None:
        me = os.getpid()
        for pid in _pids():
            if pid in (0, 4, me) or pid in self.saved:
                continue
            cur = _get(pid)
            if cur is None:
                continue
            with self._lock:
                self.saved[pid] = cur
                self._persist()                    # record BEFORE mutating -- crash-safe
            if not _set(pid, self.hk_mask, BELOW_NORMAL_PRIORITY_CLASS):
                with self._lock:
                    self.saved.pop(pid, None)       # could read but not write: leave it alone
                    self.skipped += 1

    def _persist(self) -> None:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"saved": {str(k): v for k, v in self.saved.items()},
                                     "plan_before": self.plan_before}), encoding="utf-8")

    def _resweeper(self) -> None:
        deadline = time.monotonic() + self.max_seconds
        while not self._stop.wait(self.resweep_s):
            if time.monotonic() > deadline:
                print(f"perf_sandbox: watchdog -- {self.max_seconds}s limit hit, restoring",
                      file=sys.stderr)
                self.restore()
                return
            self._sweep()

    # -- lifecycle ---------------------------------------------------------------------------------
    def __enter__(self) -> "Sandbox":
        if os.name != "nt":
            raise NotImplementedError("Windows only; Linux design in docs/OPTIMIZATION_PROCESS.md S1a")
        if STATE.exists():
            raise RuntimeError(f"{STATE} exists -- a previous sandbox did not restore. "
                               "Run `python scripts/perf_sandbox.py --restore` first.")
        if self.power_plan:
            self.plan_before = _active_plan()
            subprocess.run(["powercfg", "/setactive", HIGH_PERFORMANCE_PLAN], capture_output=True)
        # The orchestrator's own children inherit its affinity, so pin ourselves to the bench cores.
        _set(os.getpid(), self.bench_mask, None)
        self._sweep()
        threading.Thread(target=self._resweeper, daemon=True).start()
        return self

    def restore(self) -> None:
        with self._lock:
            if self._restored:
                return
            self._restored = True
            self._stop.set()
            for pid, (mask, prio) in self.saved.items():
                _set(pid, mask, prio)               # a process that has since exited just fails, fine
            if self.plan_before:
                subprocess.run(["powercfg", "/setactive", self.plan_before], capture_output=True)
            STATE.unlink(missing_ok=True)

    def __exit__(self, *exc) -> None:
        self.restore()

    def report(self) -> dict:
        return {"bench_cpus": self.bench_cpus, "housekeeping_cpus": self.housekeeping,
                "processes_confined": len(self.saved), "processes_unreachable": self.skipped,
                "plan_before": self.plan_before}


def restore_from_file() -> int:
    if not STATE.exists():
        print("nothing to restore")
        return 0
    st = json.loads(STATE.read_text(encoding="utf-8"))
    n = sum(_set(int(pid), m, p) for pid, (m, p) in st["saved"].items())
    if st.get("plan_before"):
        subprocess.run(["powercfg", "/setactive", st["plan_before"]], capture_output=True)
    STATE.unlink(missing_ok=True)
    print(f"restored {n}/{len(st['saved'])} processes (the rest have exited)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="enter, report, verify, exit, verify restore")
    ap.add_argument("--restore", action="store_true", help="undo a sandbox left behind by a crash")
    ap.add_argument("--housekeeping", type=int, default=2, help="E-cores left for everything else")
    ap.add_argument("--topology", action="store_true", help="print the P/E core map and exit")
    args = ap.parse_args()
    if args.topology:
        for k, v in sorted(cpu_topology().items()):
            print(f"EfficiencyClass {k} ({'P' if k else 'E'}-cores): {v}")
        return 0
    if args.restore:
        return restore_from_file()
    if args.dry_run:
        before = {pid: _get(pid) for pid in _pids()}
        with Sandbox(housekeeping=args.housekeeping, max_seconds=300) as sb:
            print(json.dumps(sb.report(), indent=2))
            print(f"load on bench cores: {bench_core_load(sb.bench_cpus)}%")
            print(f"load on housekeeping: {bench_core_load(sb.housekeeping)}%")
        # Flag only processes still carrying the sandbox's own mask+priority. A plain "differs from
        # before" test false-positives on short-lived processes whose PID exited or was reused mid-run.
        confined = (sb.hk_mask, BELOW_NORMAL_PRIORITY_CLASS)
        drift = [pid for pid, v in before.items() if v and v != confined and _get(pid) == confined]
        print(f"restore check: {len(drift)} process(es) still confined after exit"
              + ("" if not drift else f" -> {drift[:10]}"))
        return 1 if drift else 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
