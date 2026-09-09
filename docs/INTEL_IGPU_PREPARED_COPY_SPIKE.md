# Prepared explicit-copy staging spike

Date: 2026-09-08; compile and bounded runtime gate executed 2026-09-09. All three modes passed
element-wise correctness at both sizes. The fixed-order single-process samples are exploratory and do
not select a production memory path. See [the R0 checkpoint](INTEL_IGPU_R0_CHECKPOINT.md).

## Question and sources

The spike asks whether Intel's experimental preparation call changes the cost of repeated explicit
copies from a generated read-only Windows mapping, relative to an ordinary explicit copy and a reused
host-USM staging allocation. The API contract and caveats are pinned in
[the Windows USM review](INTEL_IGPU_WINDOWS_USM.md): the extension specification is Intel LLVM commit
`65cc0cfe809f52169b92e6bea782ddc626ba0a87`, and the installed 2025.3 header declares the API in
`include/sycl/usm.hpp`. Preparation does not make the mapped pointer legal for kernel dereference.

Hypothesis: preparation may amortize setup over repeated copies. This is unverified on the target
Windows iGPU. A successful call does not prove pinning, residency, zero-copy, or reduced physical
movement. Reused host USM has an extra CPU copy by construction; whether it wins is empirical.

## Benchmark contract

The runner generates two distinct float32 files and maps each `PAGE_READONLY`/`FILE_MAP_READ`. Modes
alternate between both mappings during seven warm trials, so every element proves that a later copy
observes changed source data. The prepared mode prepares both disjoint mapped ranges before its first
sample and releases both only after the queue is drained. The ordinary and prepared modes copy a
mapping directly to device USM.
The staging mode first copies into one allocation made before sampling, then explicitly copies that
allocation to device USM. No measured call owns or resizes storage.

Both 257 floats and 1,048,576 floats (4 MiB) run. Every transfer feeds the same affine kernel, every
output element is copied back and checked exactly. Output separates per-range extension setup/release,
the staging CPU copy, explicit host-to-device transfer, kernel completion, readback, validation, and
validation-inclusive pipeline wall time. Cold and seven warm observations are labelled separately.
The inclusive value deliberately contains readback and every-element validation, while `transfer_ms`
isolates only the explicit host-to-device copy. Timing is captured before emitting each sample.

When `SYCL_EXT_ONEAPI_COPY_OPTIMIZE >= 1` is unavailable, the source records prepared mode as
unsupported but still runs and verifies the ordinary and reused host-USM staging baselines. Missing
Intel `8086:7D67` Level Zero execution and missing USM aspects fail rather than falling back. Repeated-use
break-even must be calculated from both ranges' observed setup and release costs and the per-copy
distributions after a reserved run; the source assumes no reuse count.

The three modes currently run in a fixed order within one process and the seven warm observations are
not independent process trials. These samples establish correctness and expose gross mechanism costs;
they do not support a winner claim. Any comparative result must use the reserved-run process
alternation and independent-trial controls referenced below.

## Compile and runtime evidence

The new compile-only runner path exited zero on 2026-09-09 without device enumeration, fixture
generation or GPU execution. Source SHA-256 was
`90A52D25218BB52FBCABBBC78DC8D08CF98261D493B1779DF0B99D2C09AD9F3D`; executable SHA-256 was
`DB2F22A8A48A20954DA0D191EB39CB7D8286388D8641CA814807CA4CC6EB414A`. The exact rebased
branch/base, toolchain, command and runner hashes are in the
[compile manifest](intel-groundwork/2026-09-09/compile-gates.json). Reproduce compilation with:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-prepared-copy.ps1 -CompileOnly
```

The reserved runtime command completed successfully and its raw manifests/logs are archived under
`docs/intel-groundwork/2026-09-09/runtime/`:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-prepared-copy.ps1
```

The runner captures the commit and dirty paths, compiler/device inventory, source/header/runner hashes,
fixture hashes, compile arguments, power plan, competing process snapshot, raw logs, and exit codes.
The R0 report publishes exploratory medians and explicitly avoids promotion. Apply process alternation
and independent-trial controls in [the performance contract](INTEL_IGPU_PERFORMANCE.md) before
comparing or promoting a mode.
