---
name: host-compilers-available
description: "C++ compilers installed on the dev box (CrogLegion) as of 2026-10-05 — MSVC 19.51, clang/clang-cl 22.1.8, native GCC 16.2 and 15.2 (WinLibs), GCC 15.2 in WSL — with paths and how to invoke each"
metadata:
  node_type: memory
  type: reference
  originSessionId: 09648518-580b-4f7b-ba75-d6985e369c04
  modified: 2026-10-05T18:11:17.975Z
---

Compilers on the dev machine (Core Ultra 9 275HX, Windows 11), verified 2026-10-05:

| Compiler | Version | Where / how |
|---|---|---|
| MSVC `cl` | 19.51 (VS 18 Community) | `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`; no `-march=native`, best is `/arch:AVX2` (CPU has no AVX-512) |
| Clang / `clang-cl` | 22.1.8 | `C:\Program Files\LLVM\bin` (on PATH). `clang-cl` uses the MSVC STL/ABI (run under vcvars); `/clang:-march=native` works |
| GCC (native, MinGW-w64 UCRT) | 16.2.0 | `%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin\g++.exe` (installed via winget 2026-10-05; NOT on a non-interactive shell's PATH — use the full path, add its `bin` to PATH, link `-static`) |
| GCC (native, MinGW-w64 MSVCRT) | 15.2.0 | same pattern, package `BrechtSanders.WinLibs.POSIX.MSVCRT_...` (was already installed) |
| GCC in WSL | 15.2.0 (`g++-15`) | `wsl -d Ubuntu-24.04`; plain `g++` is NOT installed there, `gcc-13` exists without a `g++-13`; cmake 4.2.3 + ninja present; sudo needs a password |

Notes:
- CMake 4.2.3 and Ninja are on the Windows PATH; winget is available for installs.
- For same-machine compiler comparisons prefer the native ones (same OS, pinnable to P-cores `0-1,10-13,22-23`); WSL2 is a VM that cannot be pinned.
- GCC defaults to `-ffp-contract=fast`: on this FMA-capable CPU with `-march=native` it changes results between differently shaped loops. Pin `-ffp-contract=off` when bit-identical results matter (found in Sub0ECS, see [[sub0ecs-v2-landing-state]]).
- v1 Sub0ECS code does not compile with Clang (member access on an incomplete type); GCC and MSVC accept it.
See also [[host-cpu-arrow-lake-hx]], [[vs-install-path-18-not-2022]].
