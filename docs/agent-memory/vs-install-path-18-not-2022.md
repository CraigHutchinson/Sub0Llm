---
name: vs-install-path-18-not-2022
description: "This machine's real Visual Studio install path is ...\\Microsoft Visual Studio\\18\\Community\\..., NOT ...\\2022\\..."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

The real Visual Studio (MSVC 19.51) install root on this machine is
`C:\Program Files\Microsoft Visual Studio\18\Community\...`. The `...\2022\...` path does **not** exist
and never has during any session. Verified via `Test-Path`: the "18" folder has existed since 2026-01-17,
about six months before this was discovered — this is not a mid-session VS update, it was simply always
the correct path.

**Why this matters**: earlier in a session, Claude hardcoded a `vcvars64.bat` call using the wrong
`...\2022\...` path, which silently failed every time ("the system cannot find the path specified")
without visibly breaking builds — the persistent shell already had a valid MSVC environment on `PATH`
from an earlier successful call, so nothing looked wrong until a build needing a FRESH `cl.exe`
resolution (a CUDA/nvcc build) broke. Claude initially misdiagnosed this as "Visual Studio silently
updated mid-session"; the user corrected this ("We didn't change any msvc setup - that change is odd,
2026 is still installed") — the real cause was Claude's own stale hardcoded path, not an environment
change.

How to apply: always use `...\Microsoft Visual Studio\18\Community\...` for `vcvars64.bat` / MSVC tool
paths on this machine going forward; don't assume "2022" from training-data priors about VS folder names.
See also [[msvc-toolchain-quirks-2026-07]].
