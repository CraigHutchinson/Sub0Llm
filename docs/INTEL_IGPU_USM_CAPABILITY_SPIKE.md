# Intel iGPU USM capability spike

Date prepared: 2026-09-08; compile gate executed 2026-09-09. This is a runtime-unexecuted, standalone
I19/S1 diagnostic. The ordinary and `SUB0_PROBE_PREPARED_COPY_API` variants both compiled and linked
with DPC++ 2025.3.3. No probe binary or GPU/CPU measurement ran.

## Questions and evidence boundaries

The probe reports SYCL USM allocation aspects plus the two atomic-access aspects exposed by SYCL 2020.
An atomic aspect is the device's runtime capability declaration; it is not evidence that simultaneous
CPU/GPU access is correct for an arbitrary algorithm. The later writable-coherence experiment still
needs explicit ordering, an allocation kind whose contract permits the access, and result validation.

When Level Zero headers and the loader import library are supplied, the same executable calls the
documented `zeInit`, `zeDriverGet`, `zeDeviceGet` and `zeDriverGetExtensionProperties` APIs. It obtains
the native Level Zero handle of the already-selected Intel `8086:7D67` SYCL device, finds the driver
that enumerates that exact handle, records its `zeDriverGetApiVersion` result, and records that driver's
extension names and major/minor versions. The SYCL driver-version string remains separately reported;
it is not treated as evidence of the Level Zero API version.
It does not substitute the first driver when the association cannot be established. The installed
oneAPI 2025.3 tree currently has no
`level_zero/ze_api.h`; therefore source support exists but header/API build availability is unknown
until an already-approved matching Level Zero development include tree and `ze_loader.lib` are
identified. The installed `C:\Windows\System32\ze_loader.dll` alone supplies runtime loading, not
declarations or a link-time import library. Header-only and library-only inputs are rejected rather
than producing a partial inventory build.

The installed SYCL headers separately define `SYCL_EXT_ONEAPI_COPY_OPTIMIZE=1` and declare
`prepare_for_device_copy` / `release_from_device_copy`. `-PreparedCopyApi` compiles, links and, only
with `-Run`, executes a balanced 4 KiB prepare/release pair. Keeping this mode separate prevents a
normal capability inventory from being mistaken for experimental API qualification. A successful
compile establishes API availability; a successful execution establishes only acceptance for that
range/context, not useful pinning, transfer speed, overlap, or physical zero-copy.

## Compile evidence and deferred runtime commands

Both compile-only commands below exited zero on 2026-09-09. The ordinary executable SHA-256 was
`AB291201B9E3D9D5B3F6645DD19C9B55828FD1EBDAD620AE439C6DDE1AD7F028`; the prepared-copy-API
variant was `B282DB36948AA96E58A227EC6A78D08D8C2697529D2000D7A7CAE5DF1203D743`. Source SHA-256 was
`7ADC9B2ABF247C57C84D8C94CAFA6BDB31D12CA71E96702015C29808E7AEAF4A`. The builds omitted direct
Level Zero inventory because matching development headers/import library remain unidentified.

From the isolated `intel-groundwork` worktree:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -PreparedCopyApi
```

The first command checks the ordinary SYCL probe's source/header/link availability and reports that
the Level Zero extension inventory was not built unless both development-file arguments are supplied. The second is the separate
experimental prepare/release compile/link gate. Neither executes the binary. In a reserved GPU window,
repeat each with `-Run`. If a matching official Level Zero development package is already available,
pass its parent include directory as `-LevelZeroInclude C:\path\to\include` and the exact import-library
file as `-LevelZeroLibrary C:\path\to\ze_loader.lib`. This enables the documented extension inventory.
Do not install or infer a header/library version merely to make the arm pass: missing development
files are a recorded unavailable build dependency.

No timings, concurrency tests, memory-pressure tests, imported mappings, or production engine changes
belong to this spike. Runtime results must be copied into a dated evidence record before any capability
is promoted in the Intel plan.
