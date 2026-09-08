# Windows USM applicability and next probes

Date: 2026-09-08. Codex review with a read-only Terra investigation. Documentation only;
no new runtime measurements. Applies to the exact Windows/compiler/driver tuple in
[groundwork results](INTEL_IGPU_GROUNDWORK_RESULTS.md), not all Intel GPUs.

## What the official guidance establishes

The [Intel USM guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/host-device-memory-buffer-and-usm.html)
is useful on Windows, but platform-specific sample plumbing and hardware assumptions need separating
from allocation contracts. Its [allocation discussion](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/unified-shared-memory-allocations.html)
uses discrete-memory/PCI examples; this is not evidence of this iGPU's physical transfer topology or
that device USM must win here.

The [SYCL USM contract](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:usm)
distinguishes runtime host/shared/device allocations from ordinary system allocations. Our measured
host/shared/device aspects are true; system is false. Therefore ordinary heap pointers and Windows
mapped views must not be submitted directly to standard SYCL kernels on the current configuration.
Shared allocation addressability does not remove synchronization or guarantee concurrent access.
Prefetch and memory advice are hints; a successful call alone establishes no performance benefit.

## Two separate opportunities

**Prepared copies.** Intel's [transfer guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/optimizing-data-transfers.html)
describes preparing externally allocated host ranges for repeated explicit transfers. Its Linux sample
uses POSIX allocation; our Windows experiment should use native allocation/mapping lifetime management.
The [experimental extension specification](https://github.com/intel/llvm/blob/65cc0cfe809f52169b92e6bea782ddc626ba0a87/sycl/doc/extensions/experimental/sycl_ext_oneapi_copy_optimize.asciidoc)
defines `prepare_for_device_copy` and `release_from_device_copy` in
`sycl::ext::oneapi::experimental`, returning `void`. Both declarations exist in installed compiler
2025.3 `include/sycl/usm.hpp:340`. The guide's introductory signatures differ; use the pinned spec and
installed headers. Check `SYCL_EXT_ONEAPI_COPY_OPTIMIZE`, compile/link, then test execution separately.
Preparation concerns explicit copies; it does not authorize kernel dereferencing of the host pointer.
Prepared ranges must not overlap, and release must match pointer/context. Keep this experimental API
inside the spike; it is not a production dependency decision.

**Imported host memory.** The [Level Zero external memory mapping extension](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_ExternalMemMap.html)
provides an external system-memory descriptor chained into host allocation. Availability must be
queried on our driver. File-backed mapping support remains platform/driver-dependent and unmeasured.
Do not substitute Windows external-handle sharing or device IPC for host-pointer import. Undocumented
`zexDriverImportExternalPointer` names are not a portable fallback contract. Preserve a working staged
path if the documented route is unavailable. Successful import and observed CPU/GPU coherence still
do not prove absence of internal copies, pinning cost or OS migration.

## Bounded I19 / S1 follow-ups

These refine existing packages, rather than adding a second memory workstream. All probes use private
fixtures, reusable allocations and reserved hardware time; no production model mapping is modified.

| Order | Question and scope | Completion evidence / decision |
|---|---|---|
| 1 | Record atomic/concurrent USM capabilities and Level Zero extension names/versions; compile/link prepared-copy API | Exact runtime manifest; unsupported is a valid result, never silently change backend |
| 2 | Compare ordinary explicit copy, prepared explicit copy and reused host-USM staging; begin at 4 MiB with generated read-only mapped input | Verify every output; separate prepare/release, first touch, transfer and kernel time; one-shot versus repeated-use break-even |
| 3 | If documented import is supported, compare heap control and bounded read-only mapped ranges using native Level Zero submission | Record allocation properties, return codes and exact mapping permissions; unsupported file mapping ends this arm |
| 4 | For a successful import, test CPU/GPU visibility on a separate writable scratch mapping | Wait before CPU observation and before freeing import, then unmap; report observed coherence, not physical zero-copy |
| 5 | Extend existing allocation comparison to shared/device write-only scratch and shared prefetch/advice controls | Correctness first; document accepted hints and independently measured effects; no concurrent CPU access unless its contract is verified |
| 6 | Replay bounded selected-expert ranges and dense M=1 versus prefill access, then cross viable memory modes with real IQ kernels from S2 | Include miss/hit mix, cache preparation, transferred bytes, synchronization and amortization; retain FP32 control separately |

I21 applies: alternate mode order across independent processes, separate startup from warm reuse,
record sample distributions and available budget/fault telemetry. Mark unobservable physical movement
unknown. Increase working sets only under the existing headroom/stop policy; reported global memory
and maximum allocation are not a sustainable capacity result.

At R0, record availability and correctness before selecting an allocation recipe. At R1, review request
latency and amortized setup with I10/I05 owners; retire losing arms or bound a specific follow-up.
Reopen the result after driver/compiler changes. I06 remains pending; this review qualifies neither
Windows mapped-file import nor a memory-mode performance winner.
