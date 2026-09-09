# Backend identity, portability and primitive providers

Date: 2026-09-09. Design clarification; proposed names, no new backend targets or configurator flags.
This complements the [Intel design](INTEL_IGPU_BACKEND_DESIGN.md) and
[release watchlist](INTEL_IGPU_RELEASE_WATCHLIST.md).

## What the upstream names mean

**oneAPI** is an open specification ecosystem; Intel's oneAPI Toolkit is a product bundle implementing
parts of it. It is broader than a particular runtime or neural-network library.
[oneAPI specifications](https://oneapi.io/spec/),
[Intel Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneapi-toolkit.html).
**SYCL** is the cross-vendor C++ programming standard. DPC++ is Intel's compiler implementation with
additional extensions; portable source still requires a working compiler/adapter/device combination.
[Khronos SYCL](https://www.khronos.org/sycl/).

**oneDNN** supplies neural-network primitives and graph optimizations for CPU/GPU engines. Its current
project describes multiple CPU architectures, Intel GPUs, and experimental NVIDIA/AMD GPU support.
Those GPU paths still depend on vendor software stacks. Open licensing and a cross-vendor API do not
establish equivalent operation coverage, Windows packaging or performance across targets.
[oneDNN project](https://github.com/uxlfoundation/oneDNN).

The current build guide distinguishes the GPU vendor from the runtime. Its direct OpenCL and Level
Zero runtime choices are Intel-GPU-only; SYCL is the recommended route for new GPU applications.
Thus oneDNN over Level Zero is a useful candidate *inside* our Intel backend, while a oneDNN/SYCL route
may be a component of a broader portable backend.
[Build options](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html).
This moving guide identifies itself as v3.14.0; it is not evidence that our installed 3.9.1 or the
previously reviewed v3.13 release implements every documented option. Pin documentation to the chosen
release when building. Direct ZE's experimental status in v3.12/v3.13 is recorded in the release watchlist.

The portable layer still needs Sub0Llm's IQ decoders, recurrent state, selection, scheduling and missing
operations. Our existing GPU oneDNN handoff is measured in the groundwork report; cross-vendor execution
and complete Qwen inference remain unqualified. A oneDNN CPU provider is also possible independently
of GPU backend identity, if a representative benchmark justifies it.

## Separate the engine choice from its ingredients

Sub0Llm owns model execution, encoded-weight interpretation, state, memory planning and correctness.
A primitive library can supply selected operations without becoming the engine backend's identity.
The existing `include/sub0/device_backend.hpp` is a vendor-neutral, link-time device seam; current
`cmake/Backends.cmake` still combines a CPU engine with optional CUDA training. These are observed
properties of this worktree, not evidence of a complete portable accelerator executor today.

| Dimension | Proposed vocabulary | Meaning |
|---|---|---|
| Engine backend | `cpu`, `cuda`, proposed `intel`, conditional `sycl` | Complete qualified implementation behind Sub0Llm's device seam |
| Device | Vendor, architecture, PCI identity, capabilities | The actual hardware that a build/session is qualified to execute on |
| Kernel/primitive provider | custom SYCL, Intel ESIMD, oneDNN, oneMKL, custom native binary | Supplies particular operations with declared type/layout constraints |
| Submission runtime | SYCL over a named adapter, direct Level Zero, CUDA | Owns submission, completion and compatible allocation/context contracts |
| Qualification | OS/compiler/runtime/library versions plus operation/format coverage | Establishes what works here; portability is never inferred from a name |

`intel` is a useful vendor specialization name even if some operations use oneDNN. `sycl` is a
reasonable name for a future implementation whose device execution and required custom operations
actually use a portable SYCL contract. Intel-only ESIMD or Level Zero dependencies must be isolated
from that required path. Neither name by itself promises an instruction set or performance level.
Avoid a production `oneapi` label: it does not identify which library, runtime or device is required.
Use `onednn` as provider metadata unless a genuinely separate complete executor earns that identity.
These are planning names to resolve at I07b/I11, not immediate source-directory renames.

## Tandem options to test

**One Intel backend with mixed providers** is the first experiment: custom IQ/recurrent kernels plus
oneDNN projections where they win, sharing compatible memory and synchronization. It preserves one
owner for state and scheduling. It does not require every op to use the same language or library.
A lower-level submission path must still demonstrate a benefit after handoff and preparation costs.

**A portable SYCL backend alongside Intel specialization** remains viable. It becomes a separate
qualified deliverable only when all required operations have working portable implementations and a
second concrete device/runtime consumer justifies maintenance. Shared orchestration is a candidate
for reuse; hardware layouts, kernels and numerical choices need not be forced into one template.
Portability preparation can begin earlier, but portability qualification needs real second-target
execution and matching operation/precision coverage.

**Two independent full executors** offer stronger isolation when ownership or runtime constraints
cannot compose. They also duplicate scheduling, cache behavior and correctness work. Keep this option
open if S0 proves incompatible contexts or unacceptable bridge costs; do not create it solely because
oneDNN and Intel-native APIs have different names.

Here tandem means available implementation choices and mixed primitive providers. Simultaneous
multi-backend execution, cross-device tensor migration and a runtime plugin registry are additional
features outside the initial one-device-session scope.

## Work-package decisions

S0 compares custom-kernel -> primitive -> custom-kernel handoffs with SYCL interop and experimental
oneDNN Level Zero interop. Record pointer provenance, context ownership, event dependencies, scratch
ownership, hidden allocations and packaging. Explicit-copy handoff is the control when sharing fails.
S2/I05 compares real IQ/custom math with compatible primitives: changing to generic INT4 is a separate
precision experiment, never a substitute for validating source encodings.

I06 records whether one context with mixed providers is viable and which alternatives remain open.
I07b/I11 chooses names and build facts only with the actual executor. Reuse the existing neutral seam;
provider choice is internal, selected at build/preparation time where practical, with reusable scratch
and no per-token provider discovery. Keep the benchmark's A/B choices available until R3 tests
integrated startup, TTFT, decode latency, packing cost and sustainable memory.

A conditional portability follow-up requires a named second hardware/OS tuple, supported primitive
matrix, custom-op strategy, budget and owner. It is not an implicit promise that every device
supported by oneDNN can run this model or its existing quantization unchanged.
