#!/usr/bin/env python3
"""Evict one file from the OS page cache, unelevated, and VERIFY it -- cold-cache runs without a reboot.

MECHANISM (Windows). Opening a file with FILE_FLAG_NO_BUFFERING makes the cache manager flush and purge
that file's cached pages, and closing the handle leaves them gone. No privilege is needed and nothing
else on the machine is disturbed -- unlike a system-wide standby-list purge, which needs admin and
throws away every other process's cache too. Measured on this host, 2026-09-22, on the 37 GiB `.moeq`
sidecar:

    sample re-read (cached)                      median   2.6 us   0% from disk
    after NO_BUFFERING open+close, same sample   median 110.3 us 100% from disk
    pages faulted in via a MEMORY MAP, unmapped  median   3.6 us  1% from disk
    ... then evicted the same way                median 106.2 us 100% from disk

and with a control, repeated twice: the same sample re-read without eviction stayed at 2.2-2.5 us (0% from
disk) immediately and after 1 s; with eviction it went to 100-113 us (100% from disk); re-read again, back to
~2 us -- i.e. the slow reads were real disk reads that re-populated the cache, not a handle artefact.

THIS IS UNDOCUMENTED BEHAVIOUR, and that is why verification is not optional. Microsoft's CreateFile
page documents FILE_FLAG_NO_BUFFERING only as the new handle's I/O semantics ("This flag does not affect
hard disk caching or memory mapped files") and says nothing about purging; the documented purge primitive,
CcPurgeCacheSection, is kernel-mode and "will not purge mapped files". What is observed here is a
file-system-driver side effect (a non-cached open of a file nothing maps triggers a flush+purge), which
a Windows update could change. If evict_verified() ever starts refusing, the documented fallback is a
system-wide standby purge -- NtSetSystemInformation(SystemMemoryListInformation, MemoryPurgeStandbyList),
which needs SeProfileSingleProcessPrivilege, i.e. an ELEVATED helper (the filtered UAC token lacks it; this
is the same fallback System Informer's memlists.c takes) -- or a reboot.

The mapped-pages pair matters: the engine memory-maps the sidecar, and mapped pages live in the file's
section object rather than the cache manager's own views. They are purged too -- PROVIDED nothing still
maps the file. A process cannot evict a file it has mapped itself, which is why eviction happens in the
harness BETWEEN runs, never inside the engine.

VERIFY, DON'T ASSUME. probe() times buffered 4 KiB reads at random page offsets: a page in memory reads
in a few us, one from NVMe in ~100-200 us. evict_verified() refuses to report success unless the probe
says the file is actually cold, so a purge that silently failed (another process holding a mapping -- an
indexer, a scanner, a leftover run) cannot produce a "cold" number that is really warm. The probe
re-caches the pages it samples (256 x 4 KiB = 1 MiB of a 37 GiB file), so it samples offsets the
benchmark is unlikely to need and its footprint is negligible.

VALIDATED AGAINST A REAL REBOOT (2026-09-22, fused decode, 3 tokens, --decode-only): post-reboot cold
2.955 s/token; evicted cold 2.816 and 2.695; warm 1.651. Eviction reproduces cold decode to within ~5-9%,
on the OPTIMISTIC side -- good for comparing arms against each other cold, but quote an absolute cold
number from a post-reboot run.

What eviction does NOT reproduce from a real post-reboot cold start: the NVMe drive's own cache, file-
system metadata (MFT) caching, and anything a prefetcher (SysMain) pulls back in on its own. For the
question this answers -- how fast is decode when expert pages must come from disk -- those are second-
order; a reboot remains the reference if they are ever suspected.

Linux equivalent (not implemented, no consumer here -- AGENTS.md S8): posix_fadvise(fd, 0, 0,
POSIX_FADV_DONTNEED), unprivileged, for clean unmapped pages; verify with mincore().
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import random
import statistics
import sys
import time

GENERIC_READ = 0x80000000
FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
PAGE = 4096
FROM_DISK_US = 20.0          # a read slower than this was not served from memory
COLD_THRESHOLD = 0.95        # evict_verified() requires at least this fraction of the probe from disk

if os.name == "nt":
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateFileW.restype = wt.HANDLE
    _k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD,
                                 wt.HANDLE]
    _k32.CloseHandle.argtypes = [wt.HANDLE]


def probe(path: str, samples: int = 256, seed: int | None = None) -> dict:
    """Fraction of `samples` random pages of `path` NOT resident in memory, plus read latencies."""
    pages = max(1, os.path.getsize(path) // PAGE)
    rng = random.Random(seed)
    lat = []
    with open(path, "rb", buffering=0) as f:
        for _ in range(samples):
            f.seek(rng.randrange(pages) * PAGE)
            t = time.perf_counter_ns()
            f.read(PAGE)
            lat.append((time.perf_counter_ns() - t) / 1000.0)
    lat.sort()
    return {"from_disk": round(sum(v > FROM_DISK_US for v in lat) / samples, 3),
            "median_us": round(statistics.median(lat), 1), "p90_us": round(lat[int(0.9 * samples)], 1)}


def evict(path: str) -> None:
    """Purge `path` from the page cache (Windows: a FILE_FLAG_NO_BUFFERING open + close)."""
    if os.name != "nt":
        raise NotImplementedError("Windows only; Linux design is in this module's docstring")
    h = _k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_ALL, None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING,
                         None)
    if not h or h == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), f"CreateFile(NO_BUFFERING) failed for {path}")
    _k32.CloseHandle(h)


def evict_verified(paths: list[str]) -> dict[str, dict]:
    """Evict every path, then prove each is cold. Raises if any is not -- never returns a warm "cold".

    Probes with a fresh seed each call, so it never measures pages an earlier probe itself cached.
    """
    for p in paths:
        evict(p)
    result = {p: probe(p, seed=time.perf_counter_ns()) for p in paths}
    warm = {p: r for p, r in result.items() if r["from_disk"] < COLD_THRESHOLD}
    if warm:
        raise RuntimeError(f"eviction did not take (something may still map the file): {warm}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("probe", "evict"):
        print("usage: page_cache.py probe|evict FILE...", file=sys.stderr)
        sys.exit(2)
    for path in sys.argv[2:]:
        print(path, evict_verified([path])[path] if sys.argv[1] == "evict" else probe(path))
