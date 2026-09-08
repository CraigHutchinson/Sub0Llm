# Native Intel execution: ISA, quantization and accessible memory

2026-09-08 follow-up. Research/design only. This is the prerequisite study for
[I18–I20](INTEL_IGPU_WORK_PACKAGES.md), following the user's preference for Sub0Llm's own native
execution rather than adopting another engine. No device capability or memory stress test ran here.

## Target identity and evidence standard

The adapter previously read from this machine's registry is Intel Graphics `8086:7D67`, driver
`32.0.101.8991`, on the 275HX. Intel's own device table maps `0x7D67` to `ArlHwConfig` and the name
Intel Graphics; it separately names `0x7D51` Intel Arc Graphics. Therefore use the device ID, not the
informal "Arc" label, to select documentation and kernels. The common `XE_HPG_CORE` source directory
is a driver implementation grouping, not proof that this SKU has discrete Arc's entire instruction set.
[Intel device table](https://github.com/intel/compute-runtime/blob/master/shared/source/dll/devices/devices_base.inl).

For every instruction family require three evidence columns: **documented for this device/revision**,
**reported by this installed runtime**, and **emitted and executed correctly by a small kernel**.
Compiler acceptance or an intrinsic's presence in a header proves neither hardware support nor speed.
DP4A and DPAS/XMX are different capabilities. Intel's Xe architecture comparison distinguishes them;
it must not be used to infer an ARL SKU's support by analogy.
[Intel architecture comparison](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html).

**Unresolved:** exact DPAS/XMX, DP4A, BF16 arithmetic and low-bit matrix support on this installed
device/driver. This pass verified identity and located capability/inspection interfaces, not those
instruction claims. A purported INT4 peak is not evidence that IQ4_NL bytes are executable INT4 operands.

## Native software layers to investigate

| Layer | Candidate | What the experiment decides |
|---|---|---|
| Kernel language | C++ SYCL with Intel ESIMD for selected kernels; ordinary SYCL for comparison | Control of vector width, gather/scatter, register pressure and memory access versus compiler optimization |
| Compilation | Intel DPC++/IGC, supported target-specific compilation and native-binary inspection | Actual generated instructions, spills, binary identity, startup compilation cost |
| Submission | Direct Level Zero command lists/queues/events versus SYCL over Level Zero | Host dispatch cost, reuse, synchronization, graph/list replay and explicit ownership |
| Dense library baseline | oneDNN / oneMKL where supported | Whether hand-written kernels beat a tuned library at the real M=1 and prefill shapes |
| Reference engines | llama.cpp CPU/SYCL, optional OpenVINO | Algorithm/quantization reference and external measurements; no adoption as Sub0Llm's executor |

ESIMD exposes explicit vectors and memory operations and can coexist with SYCL kernels. It is the
first vendor-specific kernel-language candidate. Direct Level Zero supplies module loading and
execution; it is not a kernel language. Native binaries still need compatible metadata/compiler output
and a supported driver. Windows scheduling and residency remain in the path: "native" here means
direct user-mode compute, not bypassing WDDM or writing a kernel driver.
[ESIMD design](https://intel.github.io/llvm/design/ESIMDDesignNotes.html),
[Level Zero module API](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/module.html).

Keep compiler intermediates and inspect generated machine code, using compatible IGC/IGA tooling.
Handwritten assembly, CM or XeTLA should be investigated only if supported on this target and an
identified kernel needs their extra control; none is a mandatory initial dependency. SPIR-V is an
intermediate representation, not evidence of the final machine instruction sequence.
[Intel compiler options](https://intel.github.io/llvm/UsersManual.html),
[Intel graphics assembler sources](https://github.com/intel/intel-graphics-compiler/tree/master/visa/iga).

## I18 instruction and quantization experiment specification

| Capability | Probe | Relevance |
|---|---|---|
| FP32/FP16 arithmetic and conversion | Checked FMA/reduction and conversion microkernels; inspect actual instructions | Dense GEMV, norms, gates and f16 unpack paths |
| Integer dot product | Signed/unsigned operand combinations, accumulation range, lane packing, tail handling | Weight/activation quantization and DP4A-style execution |
| Systolic/matrix operations | Query supported combinations, then tiny checked DPAS/joint-matrix kernel only when supported | Prefill tiles and potentially grouped experts; not assumed available |
| Integer shifts, bit operations and indexed gathers | Real IQ codebook/scales/sign decoding, coalesced versus scattered access | IQ1_S/IQ2_XXS/IQ4_NL cost can dominate the dot itself |
| Subgroups, shared local memory, barriers | Supported widths, reductions and bank/access behavior | Router/top-k, GDN reductions and QSA block selection |
| Register/scratch requirements | Tile/vector-width sweeps and disassembly/spill report | Avoid trading fewer loads for catastrophic spills at M=1 |

Where available, Level Zero exposes dot-product capability and input/output-type information alongside
ordinary compute/module properties. Negotiate API/extension availability; an old driver lacking a
query returns **unknown**, not a fabricated hardware answer. Its latest documentation lists both
non-systolic and systolic capability categories, which are not interchangeable.
[Level Zero device API](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/device.html).

Compare three mathematical paths against **the same decoded source weights**:

1. IQ decode fused with FP32/FP16 multiply and appropriate accumulation.
2. Bounded tile decode/pack into reusable scratch, followed by tuned arithmetic.
3. Integer-dot path with explicitly specified activation quantization, correction terms and scales.

Path 3 may change outputs even if stored weights are unchanged. WP4f now records a residual difference
attributed to llama.cpp's quantized activations after the converter fixes. Treat that as an explicit
precision choice, not a numerical target Sub0Llm must imitate. Preserve the decoded-weight reference,
and quantify activation-quantization error separately. Do not adopt a blanket 2.2% tolerance across
operators or layers. Packed caches preserve source provenance; lossy repacking is a separate quality gate.

Use real plane encodings and dimensions, M=1 first, then representative prefill blocks. Report useful
dot time **including unpack, codebook lookup, scales, loads and reductions**. Do not choose a format
solely on storage bits or a hardware instruction's theoretical TOPS.

## I19: how much host RAM can the iGPU actually use?

There is no verified single GiB answer yet. Separate these quantities:

| Quantity | Meaning and measurement |
|---|---|
| Physical host RAM | Shared resource for OS, CPU, iGPU, mappings and caches; historical host capacity is about 63.4 GiB |
| Addressable allocation classes | Device/shared/host/system-pointer support, atomics and concurrent-access flags from the installed compute runtime |
| Per-allocation limit | Query `maxMemAllocSize` / corresponding OpenCL limit; distinct from total usable memory |
| Driver-reported total and usable memory | Query supported device/extensions and record scope; not a reservation or guaranteed resident capacity |
| WDDM process budget and usage | DXGI budget/usage on the correctly matched adapter; varies with pressure and display work |
| Successfully allocated versus touched bytes | Commit/residency and first-touch faults; allocating a virtual range alone proves little |
| Sustainable working set | Largest representative access pattern meeting latency targets without paging/thrashing |

Level Zero distinguishes host, device, shared-single-device, shared-cross-device and shared-system
access. Its allocation limit is separate from these permissions. Use the installed API version rather
than assuming every latest extension exists.
[Memory access properties](https://oneapi-src.github.io/level-zero-spec/level-zero/1.5.0/core/api.html),
[allocation contracts](https://oneapi-src.github.io/level-zero-spec/level-zero/1.6.0/core/api.html).

DXGI supplies current process budget and usage; exceeding budget can produce performance penalties.
UMA has one physical pool in Microsoft's residency model. Use DXGI as OS-level context alongside
compute-runtime measurements, not as proof that an arbitrary CPU pointer can be dereferenced by a
Level Zero kernel. The familiar "half of RAM" figure is neither extra memory nor a measured universal
ceiling on every allocation path.
[DXGI budget query](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo),
[Windows residency](https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency).

Run bounded, scheduled sweeps with an explicit host headroom budget; stop on excessive faults/latency
or allocation failure. Measure sequential reads, random expert-plane reads, CPU-write/GPU-read
handoff and simultaneous CPU/GPU readers. Record CPU commit/working set, GPU budget/usage and touched
bytes separately. Do not infer performance from a successful giant allocation or exhaust RAM alongside
WP5b's full-model transplant.

### Placement and movement experiments

- **Resident source:** compare runtime-owned shared allocation against device allocation with staged
  upload; do not assume the latter means dedicated VRAM on an iGPU.
- **Mapped expert file:** WP5b owns the sidecar mapping. Test legal pointer import/access with that
  mapping's lifetime, alignment and runtime capability; otherwise copy selected encoded ranges into
  a bounded staging pool. CPU mmap alone is not GPU zero-copy.
- **Packed tiles:** double-buffer encoded/decoded tiles only if overlap shortens the measured critical
  path. Slots follow free → filling → ready → in-flight → reusable-after-completion. Never evict or
  overwrite a slot still referenced by queued work.
- **Cache policy:** key by source/plane/encoding/packing identity; account for both source and packed
  physical copies. Track warm reuse, routing changes and cold misses. Avoid per-token general heap
  allocation and unbounded packing caches.
- **Data motion:** compare GPU reads of shared bytes, copy-engine staging, CPU memcpy, and runtime
  prefetch/advice where supported. Measure bytes and waits; hints are not guarantees of migration.
- **Cross-vendor assist:** a later experiment needs host staging or proven interop between Intel and
  NVIDIA. No assumed peer access. Integrated-memory sharing does not imply CUDA pointer compatibility.

For each candidate report `prepare + bytes moved + compute + wait + fault/repack cost`; time actual
dependent work rather than summing theoretical bandwidth peaks. Preserve persistent GDN/QSA/KV state
on the executing device. A source mapping's owner must outlive all queued reads and cached views.

## I20: direct submission without changing the kernel

Compare the same checked device binary/kernel and stable buffers using ordinary SYCL submission,
SYCL graph reuse where supported, and direct Level Zero reusable/immediate command lists. Record
host submission time, device timestamps, completion latency, CPU occupancy and warm/cold module cost.
If precisely identical binaries cannot be loaded through both routes, document that confound and
separate code-generation effects from runtime overhead. Prove native module/argument ABI compatibility
before claiming ESIMD output can be dispatched directly.

Select native language, code generation, library use, allocation strategy and submission API
independently. A thinner submission API cannot fix a poor quantization kernel; conversely a good kernel
can lose behind avoidable per-token waits. This decomposition is the basis for the revised parallel plan.

## Evidence handoff

Use work-package spikes S0–S3 for initial native interop, mapped-memory, IQ math and replay proofs.
Record results through I21 using the [performance contract](INTEL_IGPU_PERFORMANCE.md). Checkpoints
R0/R1 update the recipe and dependencies before implementation; unsupported paths are closed with
evidence, not hidden behind a generic native-backend label. S4/S5 test architecture-chain and full
capacity assumptions before broader claims.

## New paper and the local probe distinction

The user supplied [Servat et al., 2607.26584v1](https://arxiv.org/html/2607.26584v1).
It evaluates OpenMP system-USM on a discrete Battlemage GPU under Linux Xe with oneAPI 2026.0.
Its discussion highlights migration, page granularity, faults and unnecessary movement of overwritten
scratch. These observations motivate tests; they do not establish the Windows iGPU's capabilities.

Our local SYCL/Level Zero probe reports `usm_shared=1`, `usm_system=0` and verifies a bounded staged
file-mapping round trip. Ordinary mapped pointers remain unqualified for direct device dereference.
The reported ~33.55 GiB global size and ~4 GiB maximum allocation are runtime properties, not a
measured usable-residency envelope. See the groundwork evidence report before planning capacity.

Project experiment updates (hypotheses, not paper results): compare runtime host/shared/device input
allocations with identical accesses and completion boundaries; keep overwritten scratch device-owned
as a baseline. Separate first touch, warmed reuse and CPU/GPU handoff. Add sparse selected-range and
aligned/unaligned patterns in the subsequent I19 wave; record unobservable faults/migration as unknown.
System-USM and a future Linux route require new capability checks. No OS/driver upgrade or OpenMP
backend promotion follows automatically from this paper.
