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


# The engine's measured working set (~19.5 GiB) rounded up, and headroom left for the OS.
ENGINE_GIB, SLACK_GIB = 20, 2
BALLAST_EXE = REPO / "out" / "tools" / "ballast.exe"


def ensure_ballast() -> pathlib.Path:
    """Builds ballast.exe once into out/tools/ and returns its path."""
    if not BALLAST_EXE.exists():
        BALLAST_EXE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["clang++", "-O2", str(pathlib.Path(__file__).with_name("ballast.cpp")), "-o", str(BALLAST_EXE)],
                       check=True)
    return BALLAST_EXE


def start_memory_pressure(room_gib: float, files: list[str], ballast_exe: str | None = None) -> subprocess.Popen:
    """Evicts `files`, then locks enough RAM that only `room_gib` is left for caching them beyond the engine.

    The eviction comes first so the files' own standby pages do not count as memory in use. Kill the
    returned process to release the memory.
    """
    import page_cache
    page_cache.evict_verified(files)
    available = available_gib()
    gib = max(1, int(available - ENGINE_GIB - room_gib - SLACK_GIB))
    print(f"available {available:.1f} GiB -> ballast {gib} GiB, leaving ~{room_gib:g} GiB for expert caching", flush=True)
    return start_ballast(ballast_exe or str(ensure_ballast()), gib)
