"""Operating-system specifics: CPU pinning, priority, launching a measurement, pid checks, CPU time counters."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

_pin_cpu = None  # run every measurement on this one logical CPU (None = unpinned)


def set_pin_cpu(cpu):
    global _pin_cpu
    _pin_cpu = cpu


def pin_cpu():
    return _pin_cpu


def resolve_pin_cpu(option):
    """From --cpu: -2 = do not pin, -1 = the last logical CPU, otherwise that CPU."""
    if option == -2:
        return None
    return (os.cpu_count() or 1) - 1 if option == -1 else option


def launch(cmd, timeout=None):
    """Starts a measurement process at raised priority, pinned to the pin CPU when set, and waits for it.

    Pinning stops the scheduler moving the benchmark between cores mid-run (cold caches, different clocks);
    raised priority keeps ordinary background work from preempting it. With `timeout` (seconds) a process still running
    after that long is killed and subprocess.TimeoutExpired is raised.
    """
    if sys.platform == "win32":
        high_priority = 0x00000080  # HIGH_PRIORITY_CLASS
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=high_priority)
        if _pin_cpu is not None:
            ctypes.windll.kernel32.SetProcessAffinityMask(int(proc._handle), ctypes.c_size_t(1 << _pin_cpu))
        try:
            code = proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise
    else:
        cpu = _pin_cpu

        def pin():
            if cpu is not None and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, {cpu})
        code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=pin, timeout=timeout).returncode
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def cpu_times():
    """Returns (busy, total) CPU time counters for the whole machine, or None if unavailable."""
    if sys.platform == "win32":
        class FileTime(ctypes.Structure):
            _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]
        idle, kernel, user = FileTime(), FileTime(), FileTime()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        value = lambda t: (t.high << 32) | t.low
        total = value(kernel) + value(user)  # kernel time includes idle time
        return total - value(idle), total
    try:
        fields = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
        idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
        total = sum(fields)
        return total - idle, total
    except OSError:
        return None


def pid_alive(pid):
    """Whether a process with this id exists (never signals it: os.kill would terminate it on Windows)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, 0, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return kernel.GetLastError() == 5  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
