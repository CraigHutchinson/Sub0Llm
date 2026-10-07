"""Shared by the storage A/B drivers: the host's free memory, and a locked RAM ballast."""
from __future__ import annotations

import ctypes
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def available_gib() -> float:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 2**30


def start_ballast(ballast_exe: str, gib: int) -> subprocess.Popen:
    """Starts ballast.exe holding `gib` GiB locked, and returns once it reports the memory is held."""
    process = subprocess.Popen([ballast_exe, str(gib)], stdout=subprocess.PIPE, text=True)
    print(process.stdout.readline().strip(), flush=True)
    return process
