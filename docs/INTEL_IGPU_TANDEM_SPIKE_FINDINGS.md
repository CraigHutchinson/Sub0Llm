# Intel tandem S0a/S0b/S0c preliminary findings

Date: 2026-09-09. Terra produced an initial draft before hitting its usage limit; Codex completed
this source/evidence review under the documented fallback rule. **No new builds or hardware runs.**
S0a/S0b/S0c are partially prepared, not execution-qualified. No backend/provider is selected.

## S0a: native/library composition

**Measured previously:** the archived [257-element](intel-groundwork/2026-09-08/groundwork-257-probe.txt)
and [65,536-element](intel-groundwork/2026-09-08/groundwork-65536-probe.txt) runs establish custom
SYCL -> oneDNN ReLU -> custom SYCL on Intel 8086:7D67, DPC++ 2025.3.3 and oneDNN 3.9.1.
The probe owns its allocations, borrows them into library memory objects, and uses conservative waits.
This is SYCL over Level Zero, not a direct command-list comparison or proof of overlap.

**Documented in the moving v3.14 guide:** native interop can wrap caller driver/device/context,
command list and USM storage. Only in-order immediate lists are supported. Caller-created native
objects and supplied allocations remain caller-owned; returned execution events are tied to stream
lifetime. These constraints matter for teardown, replay and event reuse.
[Native interop guide](https://uxlfoundation.github.io/oneDNN/dev_guide_level_zero_interoperability.html).

**Version gap:** [v3.12](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.12) introduced
experimental direct ZE support, and [v3.13](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13)
retains experimental Level Zero support. This pass did not retrieve usable tagged v3.13 ZE headers;
the guessed raw header URLs failed. Consequently the exact release-specific signatures, ownership
and supported-list contract must be verified against the selected source checkout before authoring
a reproducer. Moving v3.14 API docs are leads, not a substitute for that check.

**Local prerequisite audit:** searching installed `oneAPI/dnnl/2025.3/include` finds no ZE interop
headers. The archived library version is 3.9.1; it does not qualify the new native route. No checked
custom direct-ZE kernel module is in the existing groundwork. These are two independent missing
inputs, not evidence that Windows ZE interop is impossible.

The next bounded experiment is therefore:

1. Pin a source tag/commit and compatible compiler/Level Zero development files; inspect that version's
   Windows build requirements and ZE headers. Build only the isolated primitive/probe targets once a
   build window is available; record configuration, dependency hashes and loaded runtime identity.
2. Produce a checked affine kernel module through the exact-target compiler. Verify direct submission
   independently at 257 and 65,536 elements before adding the library.
3. On one supported in-order immediate list, run affine -> f32 ReLU -> affine. Own allocations and
   input events explicitly; keep the library stream alive through every consumer of its returned events.
   Validate all elements against a CPU formula, then test changed inputs and repeated reuse.
4. Drain work before destroying primitives/memory wrappers/stream, then native events/lists, allocations
   and context in their required lifetime order. Exercise failed setup cleanup. Count scratch and
   preparation separately. Compare conservative waits with documented event handoff only after parity.

This is a concrete dependency/API gate before implementation, not a new backend abstraction. The
existing SYCL handoff and explicit-copy boundary are the controls.

## S0b: portable-provider coverage

The archived [dense manifest](intel-groundwork/2026-09-08/dense-manifest.json) selected only the Intel
Level Zero GPU. Its `sycl-ls` output was selector-filtered, so it does not establish the absence of other
platforms. The [adapter inventory](INTEL_IGPU_BACKEND_DESIGN.md#1-hardware-and-workload-boundary)
records an NVIDIA RTX 5070 Laptop GPU as a potential second target. No matching NVIDIA SYCL/oneDNN
runtime execution is qualified by these artifacts.

| Required surface | Existing Intel probe evidence | Remaining portable-executor evidence |
|---|---|---|
| f32 row-major dense projections | Selected M=1/32/128 shapes checked, strict f32 oneDNN vs simple SYCL | Other precision/layouts; complete head; exact second-target execution |
| Elementwise/reductions | Affine and ReLU only | RMSNorm, Q/K norm, activations, gates and reductions |
| Embeddings/RoPE/attention/KV | None in these probes | Indexing, positional conventions, masks, cache/reset and full state parity |
| GDN/QSA/gated residual | None | Recurrent/chunk state, sparse selection/tails and stream layouts |
| MoE/IQ | None | Routing/ties, selected-expert storage and real IQ1_S/IQ2_XXS/IQ4_NL fidelity |
| Provider memory contract | Small shared-context handoff only | Reusable scratch, packing/layout identity, completion and capacity on each target |

oneDNN documents experimental NVIDIA and AMD GPU support with vendor-stack dependencies.
[Project requirements](https://github.com/uxlfoundation/oneDNN). Next audit the selected release's
NVIDIA backend operation restrictions and the actual OS/compiler/adapter support for this laptop.
Do not assume a Linux plugin recipe works on Windows. If that tuple is unsupported, record it and
cost a separate supported OS/hardware experiment; do not silently switch the benchmark environment.

Portability admission requires the same tiny kernel and primitive handoff on a real second target,
then the required operation/format matrix. Source portability and library availability alone cannot
admit a full executor. oneDNN may also benefit CPU execution independently; that is a later bounded
provider experiment, not automatic GPU coverage.

## S0c: matched provider A/B protocol

Reuse the [dense raw results](intel-groundwork/2026-09-08/dense-probe.txt) and
[interpretation](INTEL_IGPU_GROUNDWORK_RESULTS.md). They are component baselines, with five in-process
pairs, no integrated request timing and no real IQ execution. K=2560,N=6144 is a projection-plane
control; it is not the 248320-wide vocabulary head.

| Arm | Purpose / controlled comparison |
|---|---|
| Custom-only SYCL projection chain | Provider baseline with the same math, inputs and output oracle |
| Custom SYCL -> oneDNN SYCL -> custom SYCL | Change projection provider while retaining device/context and surrounding work |
| Direct ZE custom -> oneDNN ZE -> direct ZE custom | Test native composition; retain identical custom binary where possible and record any code-generation differences |
| Explicit-copy handoff | Attribute actual bridge copies and staging cost; define which edge requires each copy |
| Library-only equivalent chain, conditional | Run only when every operation has an exact supported primitive; otherwise mark unavailable, not numerically substitute |

Use tiny K=96,N=384 and both non-square model planes 2560x640 and 640x2560 at M=1/32/128.
Preallocate caller storage and scratch; separate primitive creation/JIT, packing/reorder, first touch,
upload/readback, event/queue setup and warm work. Measure stage costs in separate diagnostic runs;
**do not insert stage-by-stage waits into the inclusive dependent-chain benchmark** unless they are
required by the tested implementation. Validate all outputs and changed-input/reset cases before
reporting timing. Alternate arm order across independent processes under I21, keeping runtime and
provider changes separate before testing their interaction.

A faster primitive that loses after packing, waits or bridge copies has not earned integration.
Correct shared ownership plus lower inclusive chain cost supports a mixed-provider Intel executor
experiment at I06/R3. Incompatible ownership or material bridge cost motivates a costed separate-executor
comparison; neither outcome is established now. Shape-dependent choices remain private benchmark
alternatives until the integrated decode/prefill path justifies them.

## Checkpoint

S0a: current SYCL handoff measured; native API/dependency gate remains open. S0b: local coverage audit
prepared; second-target support/operation qualification open. S0c: controlled experiment specified;
new harness/measurements pending. Next execution starts with native dependency and kernel validation,
not a new production interface, driver upgrade or claim that these spikes are complete.
