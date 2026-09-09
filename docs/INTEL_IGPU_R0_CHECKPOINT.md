# Intel iGPU R0 checkpoint

Date: 2026-09-09. Branch execution commit: `6cefe89`. Target: Intel PCI `8086:7D67` on Windows,
interactive inference first. This checkpoint records bounded runtime groundwork; it does not select a
memory winner, qualify direct Level Zero, or authorize a production backend.

## Runtime gates executed

The system had no matching compiler/model process before the serialized window. These commands all
compiled their exact executable, recorded its hash, ran on the selected Level Zero Intel GPU, and
exited zero:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -Run
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -PreparedCopyApi -Run
pwsh -NoProfile -File scripts/intel/inventory/run-prepared-copy.ps1
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -WithOneDnn -Elements 257
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -WithOneDnn -CheckFailures
```

Raw manifests and logs are under [the dated runtime evidence](intel-groundwork/2026-09-09/runtime/).
Large generated fixtures and executables remain in the ignored output tree and are reproducible from
the recorded commands and hashes.

## Facts established on this tuple

- The selected device is Intel Graphics `0x7d67`, backend `level_zero`, runtime driver
  `1.15.39183+3`.
- Host, shared and device USM are reported and exercised. System USM is false.
- Atomic host-USM and shared-USM aspects are false. No concurrent CPU/GPU access scheme may assume
  atomic coherence from these aspects.
- The installed header exposes `SYCL_EXT_ONEAPI_COPY_OPTIMIZE=1`; a balanced 4 KiB
  `prepare_for_device_copy` / `release_from_device_copy` runtime call passed.
- The prepared-copy benchmark verified every element at 257 and 1,048,576 floats for ordinary mapped
  copy, prepared mapped copy and reusable host-USM staging. Changed source mappings were observed.
- The conservative mapped-file staging plus custom SYCL -> oneDNN -> custom SYCL controls still pass
  at 257 and 65,536 floats. All five failure-path checks pass.
- Direct Level Zero extension inventory was not built: matching development headers and
  `ze_loader.lib` were not supplied. The runtime loader alone does not close that dependency.

## Exploratory prepared-copy result

Warm medians from seven fixed-order, single-process samples:

| Elements / bytes | Mode | CPU staging, ms | H2D transfer, ms | Kernel, ms | Readback, ms | Validation-inclusive, ms |
|---:|---|---:|---:|---:|---:|---:|
| 257 / 1,028 | ordinary | 0 | 0.0504 | 0.0163 | 0.0461 | 0.1172 |
| 257 / 1,028 | prepared | 0 | 0.0503 | 0.0129 | 0.0460 | 0.1096 |
| 257 / 1,028 | host-USM staging | 0.0000 | 0.0480 | 0.0115 | 0.0450 | 0.1056 |
| 1,048,576 / 4 MiB | ordinary | 0 | 0.5312 | 0.1380 | 0.4265 | 1.7251 |
| 1,048,576 / 4 MiB | prepared | 0 | 0.4225 | 0.1493 | 0.4201 | 1.6472 |
| 1,048,576 / 4 MiB | host-USM staging | 0.1126 | 0.4347 | 0.1408 | 0.4132 | 1.7683 |

For the 4 MiB case, prepared transfer is 0.1087 ms lower than ordinary in these medians. Preparing
and releasing both ranges cost 0.6187 ms total, giving an exploratory break-even near six repeated
copies if the measured difference persists. The validation-inclusive difference is only about 4.5%.
This is a hypothesis for the independent-process I21 trial, not a selected allocation route: mode
order was fixed, samples share one process/runtime cache, and seven observations do not support p95.
At 1 KiB the transfer differences are negligible relative to submission noise.

## S0a dependency classification

The current DPC++ 2025.3.3 plus oneDNN 3.9.1 tuple qualifies only the measured SYCL handoff. oneDNN
v3.12 introduced experimental direct ZE; v3.13 tag `0e2a5bf` retains experimental status and documents
`dnnl_ze.hpp`, caller-owned native objects/USM, in-order immediate command lists only, and events whose
lifetime is bounded by the oneDNN stream. A future direct-ZE arm needs a separately pinned newer oneDNN
build and matching Level Zero SDK/import/runtime files. Until that tuple is acquired and audited,
direct ZE is **unavailable on the current development tuple**, not disproved on the hardware.

Primary references and the full evidence boundary remain in
[the tandem findings](INTEL_IGPU_TANDEM_SPIKE_FINDINGS.md) and
[release watchlist](INTEL_IGPU_RELEASE_WATCHLIST.md).

## R0 status

| Arm | Status | Consequence |
|---|---|---|
| I00 selected-device execution | qualified component | Continue engine-free Intel probes on this exact tuple |
| S0 SYCL custom/oneDNN composition | qualified component | Retain as the conservative provider/control route |
| S0a direct ZE composition | unavailable on current development tuple | Pin/build a newer isolated tuple before native interop code |
| S0b second-target portability | planned | No portable-executor claim |
| S0c provider A/B | prepared | Requires S0a outcome and I21 controlled trials |
| S1.1 capability inventory | measured, direct-ZE extension arm unavailable | USM atomics cannot be assumed; loader extensions remain unknown |
| S1.2 prepared-copy correctness | measured, performance inconclusive | Add independent-process/order controls before promotion |
| S1.3/S1.4 mapped import/coherence | conditional and planned | Begin only with documented import support and native dependencies |
| S1.5 scratch/prefetch/advice | planned | Can proceed on the existing SYCL tuple |
| S1.6 expert/IQ access trace | blocked on S2 real-IQ kernel | Cross memory modes only after the encoded kernel is correct |

R0 therefore authorizes I21 harness/schema preparation, I18/S2 real-IQ preparation and the S1.5
scratch/hint probe. It does not authorize I06 selection, production I07b/I11 wiring, a direct-ZE route,
or a full-model capacity conclusion.

## Next bounded actions

1. Complete I01's immutable dense/prefix/full-artifact admission record and I07b.0's consumer/toolchain
   map; these are parallel documentation gates.
2. Extend the prepared-copy runner/harness so provider/memory arms alternate across independent
   processes with a machine-readable I21 schema. Repeat 4 MiB and one representative selected-range
   size before selecting prepared copy.
3. Delegate S1.5 and S2/I18 as separate Sol spikes. S1.5 may run on this tuple; S2 first consumes real
   IQ fixture planes and proves emitted code/math before timing.
4. Keep direct ZE parked until a pinned v3.13-or-later oneDNN and matching Level Zero development tuple
   can be audited without contaminating the qualified SYCL control.

