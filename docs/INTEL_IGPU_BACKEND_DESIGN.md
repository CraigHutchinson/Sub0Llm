# Intel iGPU inference: platform selection and backend design

Research date: 2026-09-08. **Design only; no backend selected or implemented.**
User priority: **interactive inference first; training later**.
Repository inspected at `90ec9c2b5dafb5034926d78e97ab7d8533295468`, with the uncommitted CPU Muon
changes present. WP4f remains independently owned; its active log reports unresolved layer-0 fidelity.

## Recommendation

Run a bounded comparison before committing to an Intel backend. The first contenders are
**llama.cpp SYCL, llama.cpp Vulkan, and OpenVINO**. OpenVINO has two distinct entry routes worth
checking: its llama.cpp backend and its native Runtime/GenAI stack. They have different model,
quantization, and state-management coverage. A result for one does not establish the other.

For a native Sub0Llm implementation, **SYCL with oneDNN and selective custom kernels is the leading
design hypothesis**, because we need control over Qwen4's unusual state, sparse attention, and encoded
expert weights. This is a judgment about implementation fit, **not a measured performance win**.
Vulkan remains a serious challenger. OpenVINO should win if its graph execution is faster at acceptable
accuracy and memory on the exact target, including any custom-operation or conversion cost.

Do not make a wholesale CPU/CUDA rewrite a prerequisite for evaluating these alternatives. External
benchmarks and small engine-free mechanism experiments can start independently. In the engine,
extend the existing device seam and split responsibilities incrementally after WP4f stabilizes.
Deliver the smallest useful inference path first; defer training, concurrent serving, and multi-device
scheduling until there is evidence that they are needed.

Execution tickets and dependency gates are in [Intel work packages](INTEL_IGPU_WORK_PACKAGES.md).
This document adds an Intel execution option to [BACKENDS.md](BACKENDS.md) and
[Qwen memory orchestration](QWEN4_MEMORY_ORCHESTRATION.md); it does not replace the latter's
cross-tier storage design or [Sub0Firn](SUB0FIRN_SPEC.md).

## 1. Hardware and workload boundary

### Verified here versus still unknown

Read-only registry inspection found:

| Item | Observation |
|---|---|
| CPU | Intel Core Ultra 9 275HX |
| Intel adapter | Intel Graphics, PCI `8086:7D67`, driver `32.0.101.8991` |
| NVIDIA adapter | RTX 5070 Laptop GPU, PCI `10DE:2D58`, driver `32.0.15.9636` |
| Developer commands on this shell's PATH | `sycl-ls`, `icx`, `icx-cl`, `clinfo`, `vulkaninfo` not found; this does not prove they are uninstalled |
| Prior project capacity measurements | About 63.4 GiB host RAM and 7.96 GiB NVIDIA VRAM; historical measurements in the orchestration doc, not fresh free-memory readings |

Intel's specification identifies this SKU's GPU as **Intel Graphics, four Xe cores, up to 1.9 GHz,
8 INT8 TOPS**, and separately its NPU as 13 INT8 TOPS. The published device ID matches the registry.
It is not a basis for extrapolating Arc 140V, B580, or datacenter results to this laptop.
[Intel 275HX specifications](https://www.intel.com/content/www/us/en/products/sku/242293/intel-core-ultra-9-processor-275hx-36m-cache-up-to-5-40-ghz/specifications.html).

Still measure: runtime visibility, actual EU/subgroup capabilities, matrix-instruction availability,
FP16/BF16 behavior, memory allocation limits and budgets, sustainable bandwidth, package power, and
display-load effects. Do not assume XMX from the Core Ultra brand. Intel's joint-matrix example
explicitly requires XMX hardware. [Intel joint-matrix guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-1/joint-matrix.html).

The iGPU uses the host memory system. **Its advertised shared-memory allowance is not additional RAM.**
USM may avoid explicit copies on an integrated device, but does not remove synchronization, bandwidth,
allocation, or residency costs. Measure device, shared, and host allocation strategies separately;
verify whether the chosen driver can import an existing mapping before claiming zero-copy model
storage. [Intel USM guidance](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/explicit-scaling-sycl.html).

### Three different deliverables

1. **Hardware viability:** a small supported model proves the driver/runtime and establishes local
   interactive latency. A small Qwen3.5 model is a useful GDN control, not a substitute Qwen4 gate.
2. **Exact architecture execution:** real Qwen4-preview component fixtures, then the same four-layer
   prefix and weights used by WP4f, including quantized experts. This proves selected decoder execution,
   not meaningful full-model generation or full-checkpoint fidelity.
3. **Useful model deployment:** a memory-feasible complete model, with its real tokenizer, chat
   template, PLE/n-gram path, and required state, producing measured useful interactive inference.
   The current 125B preview's feasibility remains a separate, much larger question.

The smallest downloaded preview GGUF is recorded as 67.56 GiB, including a 26.83 GiB n-gram table.
WP4e's four-layer artifact is 5.89 GiB f32 plus a 3.17 GiB encoded expert sidecar; its recorded process
peak is 14.32 GiB. These are historical results from [WP4](WP4_SCOPE.md), not an Intel memory plan.
The iGPU cannot turn a model larger than available host memory into a resident one. Excluding PLE or
using four layers must remain explicit in every comparison and must never be labeled full Qwen4 support.

## 2. Architecture-specific requirements

The public release is **Qwen3.8-Flash-Next**, whose model type is `qwen4_exp`; the repository's
"Qwen4-preview" name refers to that concrete model. The published config confirms hidden width 2560,
48 layers, 512 routed experts/top-10, four residual streams, GDN head counts 16/48, and attention head
counts 24/2 with head dimension 256. Re-fetched this pass; no inference about a future final Qwen4
release is needed. [Published config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json).

| Mechanism | What the backend must preserve | Performance work to investigate |
|---|---|---|
| GDN, three layers out of four | Causal depthwise convolution; grouped key/value heads; gate order and activations; recurrent state and reset semantics | Parallel prefill versus recurrent decode, fused gate/norm steps, persistent state, bounded chunk scratch |
| QSA, every fourth layer | Indexer, compressed blocks, top-k mask and incomplete tail, gated output, exact partial-RoPE convention | Fused selection/attention without a dense T-by-T materialization; retain pooled indexer state on device |
| Gated residual | Four streams and the low-rank read/write gates, including final collapse; not ordinary residual add | Fuse gates with elementwise work; preserve stream layout between sub-blocks |
| MoE | Router and normalization, top-10 ordering/ties, 512 routed plus shared expert, correct mixture weights | Group selected expert work, encoded-weight GEMV/dequant fusion, reuse packed storage across tokens |
| Dense projections/head | Q width 6144 differs from hidden width 2560; QSA query/gate projection is wider still; vocab 248320 | Separate M=1 GEMV from prefill GEMM; oneDNN/oneMKL and custom-kernel comparison; avoid transferring unnecessary prefill logits |
| PLE/n-gram, eventual full decoder | Hashing, table lookups, causal history and projection | Reuse existing tiered-storage work; measure CPU lookup plus a small transfer rather than putting the huge table on GPU |

Algorithm contracts and independent fixtures already live in `gdn_math.hpp`, `qsa_math.hpp`,
`gated_residual_math.hpp`, `moe_math.hpp`, and their mechanism docs. The new backend must validate
against those **and** the external oracle; comparing two implementations sharing the same mistaken
adapter is insufficient. In particular, DeepSeek hyper-connections/indexer support in a library is not
proof of identical Qwen gated-residual/QSA semantics.

### Quantization is part of the execution design

The actual expert sidecar stores mixed **IQ1_S / IQ2_XXS gate/up** and **IQ4_NL down** planes.
`moe_quant.hpp` currently resolves selected experts to f32 through a bounded cache. An Intel backend
cannot obtain those encoded weights from `upload_params(const float*)`: the routed experts are
deliberately absent from the f32 parameter arena in that build.

A native experiment should compare three explicitly labeled routes:

- Direct encoded-weight dot products, using each plane's real type and block layout.
- Bounded decode/pack of selected experts into reusable f16/f32 working storage.
- Offline conversion into a supported alternative representation, only with separate accuracy and
  capacity gates. Repacking losslessly and requantizing are different operations.

Do not materialize the entire expert pool in f32. Do not assume generic INT4 kernels can consume
GGML IQ encodings. Source storage is GGML `[out,in]` logically; the CPU arena uses `[in,out]`.
Any device packing descriptor must encode the mapping explicitly, and parity fixtures must include
non-square planes. Preserve original artifacts; any packed derivative needs a source checksum,
encoding/packing version, dimensions, and backend/compiler identity.

Two scale calculations illustrate what to measure; they are not latency predictions:

- One selected expert's three f32 planes occupy `3 * 2560 * 640 * 4 = 19,660,800` bytes
  (**18.75 MiB**). Top-10 is 187.5 MiB per layer if all ten are expanded without reuse.
- A single f32 recurrent-state matrix per GDN value head occupies
  `48 * 128 * 128 * 4 = 3 MiB` per GDN layer, **108 MiB across 36 layers**.
  Convolution history and prefill scratch are additional. Persistent state must not make a CPU round
  trip every token. Derive allocation sizes from the existing helpers during implementation.

## 3. Platform comparison

These are research conclusions as of the access date, not local runtime results. Upstream main branches
move: pin exact commits, runtime versions and drivers before reproducing or extending this work.

| Route | Strength for this project | Principal uncertainty or cost | Proposed disposition |
|---|---|---|---|
| **SYCL + oneDNN/custom kernels** | Fits C++ and explicit persistent storage; detailed control over QSA/GDN/quantized experts | Kernel engineering, compiler/runtime deployment, hardware-specific tuning | Leading native design candidate; prove it with a bounded experiment |
| **llama.cpp SYCL** | Existing GGUF execution and many relevant kernels; independent runnable comparison | Exact shapes/types, fallback and current driver's behavior must pass | First comparison lane |
| **OpenVINO Runtime / GenAI** | Graph compilation/fusion and existing inference tooling | Exact Qwen4 export, state and IQ-format path unproven; custom work may be substantial | Model-support feasibility gate, then benchmark |
| **llama.cpp OpenVINO** | Avoids writing an exporter for supported GGML graphs | Current IQ type coverage and stateful limitations directly affect our artifact | Separate support gate; useful control model baseline |
| **llama.cpp Vulkan** | Existing cross-vendor execution; Windows route independent of Intel C++ compiler | Exact quant/shape coverage, shader/driver behavior | First comparison lane, equal opportunity to win |
| **Native Vulkan** | Explicit resource/dispatch control, wider vendor reach | More resource/shader plumbing; yet another kernel implementation | Promote only if comparison justifies it |
| **Direct OpenCL / Level Zero** | Low-level control or runtime fallback | Neither supplies a complete LLM executor; more manual scheduling and kernels | Use inside a proven bottleneck, not a separate initial engine |
| **OpenMP target offload** | Potentially small experiments for isolated loops | Does not solve model residency, custom IQ kernels, or per-token launch overhead | Not the primary implementation hypothesis |
| **Windows ML / ONNX Runtime / DirectML** | Windows inference integration | Exact graph/export/provider support still required | Secondary contender if a working target export appears |
| **PyTorch XPU / Intel llm-scaler** | Research and later training/serving ecosystem | Different runtime/deployment model and unsupported-hardware risk on this HX iGPU | Later control; not the initial C++ dependency |
| **NPU** | A distinct Intel inference device worth tracking | Exact model/precision/state coverage and memory limits unproven | Deferred until iGPU and useful model choice are established |

### Evidence behind the shortlist

**SYCL is a programming model, oneAPI a toolchain/library ecosystem, and Level Zero a low-level
execution API.** They are not three interchangeable ready-made LLM backends. Intel documents the
SYCL-to-Level-Zero relationship; starting with Level Zero alone does not automatically provide better
execution. [Intel Level Zero overview](https://www.intel.com/content/www/us/en/developer/articles/technical/zero-in-on-level-zero-oneapi-open-backend-approach.html).

oneDNN exposes SYCL interoperability for custom kernels sharing its execution environment. Its build
options include SYCL, OpenCL, and Level Zero GPU runtimes. Use oneDNN primitives for suitable dense
operations and benchmark oneMKL BLAS as an alternative, rather than assuming all products need custom
kernels. Neither library name establishes support for our packed IQ formats.
[oneDNN interoperability](https://uxlfoundation.github.io/oneDNN/dev_guide_dpcpp_interoperability.html),
[build options](https://uxlfoundation.github.io/oneDNN/dev_guide_build_options.html).

llama.cpp documents Windows 11 SYCL builds and integrated-GPU support, and recommends testing FP16
versus FP32 because their speed and accuracy differ. Its source dispatches Gated DeltaNet and
lightning-indexer operations. That establishes useful implementation material, not complete Qwen4
support on this adapter. Record actual placement and tensor constraints in the experiment.
[SYCL documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md),
[SYCL implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-sycl/ggml-sycl.cpp).

Vulkan is an upstream build option; its current implementation also contains Gated DeltaNet and
lightning-indexer dispatch. An older missing-kernel issue is not evidence that today's source lacks
them. Correct execution of all Qwen4-specific shapes is still a runtime gate.
[Build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[Vulkan implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp).

The llama.cpp OpenVINO adapter translates GGML graphs into OpenVINO graphs. Its validation table
lists Qwen3.5 GPU success in stateless execution and failure/unsupported in stateful execution;
those modes are adapter-specific, not a statement that every OpenVINO API has the same restriction.
[OpenVINO backend documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md).

More decisively for our downloaded weights, the adapter's `supported_types` set includes common
K-quants but **omits IQ1_S, IQ2_XXS and IQ4_NL**. Its GDN support check also rejects particular
permuted inputs and multiple state snapshots. Inspect the full translated graph before predicting
fallback or failure; the source supports a narrower statement than "OpenVINO cannot run Qwen."
[Adapter support checks](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-openvino/ggml-openvino.cpp).

Native OpenVINO is a separate test. The verified-model page contains Qwen3.5 entries but no `qwen4`
match found this pass. GenAI's direct GGUF documentation describes limited-topology preview support.
Neither is evidence that our complete `qwen4_exp` artifact is already supported. Conversely, absence
from a verified list does not prove a graph cannot be exported and executed.
[Verified models](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html),
[GenAI inference](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html).

OpenVINO offers custom GPU operations and remote-tensor/context interoperability. These make a custom
path plausible, but require real integration work and synchronization validation. A C++ frontend
extension alone is not a GPU kernel implementation. Avoid fragmenting every layer across independent
OpenVINO and SYCL contexts; test sharing and boundary costs first.
[Custom GPU operations](https://docs.openvino.ai/2026/documentation/openvino-extensibility/custom-gpu-operations.html),
[remote tensor API](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__ocl__gpu__cpp__api.html).

Intel's IPEX-LLM repository was archived on 2026-01-28; it is not a sensible new maintenance foundation.
Intel llm-scaler instead describes a Linux/Ubuntu multi-GPU serving route, useful for a future serving
comparison but not evidence of native-Windows HX support. DirectML remains supported, with new
Windows inference development directed toward Windows ML. These facts influence investment order,
not the outcome of a performance test.
[IPEX-LLM](https://github.com/intel/ipex-llm),
[llm-scaler](https://github.com/intel/llm-scaler/blob/main/vllm/README.md),
[Microsoft DirectML guidance](https://learn.microsoft.com/en-us/windows/ai/directml/dml-get-started).

## 4. What the current backend division actually is

This is a source audit, not a reading of old "skeleton" comments:

| Surface | Current responsibility and constraint |
|---|---|
| `src/backends/cpu/backend.cpp` — 3,139 lines | Parameter/moment ownership, per-OpenMP-slot Worker arenas, autograd, model assembly, full forward, incremental caches, MoE store, optimizer and host memory report |
| `src/backends/cuda/backend.cu` — 6,823 lines | CUDA context/stream/BLAS ownership, kernels, launch wrappers, scratch plans, forward/backward/decode, optimizer, C exports and extensive parity/profiling hooks |
| `include/sub0/core.hpp` — 365 lines | Public CPU engine/Node/lifecycle API; generated-layout dependency, process/global ownership assumptions |
| `include/sub0/device_backend.hpp` — 255 lines | Existing neutral `sub0_dev_*` inline bridge to CUDA, capability struct, no-device stubs and mock evaluation route |
| `include/sub0/decode.hpp` | Selects device at setup, uploads the complete f32 arena; prefill calls `forward_one` for every prompt token and retrieves vocabulary logits |
| `include/sub0/eval.hpp`, `src/train_stage.cpp` | Other production consumers of device capabilities and lifecycle; training also handles parameter/moment synchronization |
| `cmake/Backends.cmake` | CPU engine always built; GPU means CUDA today; AUTO detects CUDA; HYBRID rejected |
| `cmake/SuperBuild.cmake` | Toolchain-isolation intent exists, but implementation currently creates only the host child; it is not a finished general multi-compiler orchestrator |

CUDA explicitly rejects GR, MoE, QSA, n-gram embeddings and `N_HEADS*D_HEAD != D_MODEL`.
GDN forward exists, but the capability function disables training and incremental decode in a GDN
build. **Sub0Llm CUDA is not currently an exact-Qwen4 performance baseline.** Upstream llama.cpp
CUDA is a separate candidate where its pinned build passes the target checks.

### Structural gaps relevant to Intel

1. **The device seam assumes an f32 parameter mirror.** It cannot describe encoded expert ranges or
   OpenVINO model artifacts. Extending this is more important than renaming files.
2. **Prefill and decode have one entry point.** Per-token full-vocabulary return during prefill prevents
   efficient chunk submission and avoids none of the CPU/device synchronization overhead.
3. **Ownership is implicit.** CPU parameters are process-wide; caches/TLS and Worker slots have different
   lifetimes. Device init/shutdown is called by generation, evaluation and training wrappers. Adding a
   third execution path without a single session owner risks stale state and premature teardown.
4. **Capabilities describe operations, not the complete artifact contract.** Backend availability,
   architecture support, exact encoded formats, context limits and precision must be checked together
   before accepting a model. A `supports_decode` bit alone is insufficient.
5. **Memory reports assume CUDA-style dedicated memory.** iGPU shared allocations and CPU source
   buffers can represent the same physical storage or duplicate it; summing advertised budgets is wrong.
6. **Compiler boundaries are only partly isolated.** The host uses C++26, CUDA C++20. Intel's chosen
   compiler need not parse all host headers; a new backend must not force the host off its compiler.

## 5. Proposed structure and migration

### Preserve these decisions

Keep generated `layout.hpp` as the shape/parameter source of truth, optional features off by default,
and build-time backend selection. Keep the cross-DLL seam C linkage with POD/fixed-width fields and
opaque pointers only; no STL, `Node`, SYCL queue, OpenVINO class or CUDA handle crosses it.
Keep the CPU reference and its algorithms separate from device implementations.

Select **one accelerator implementation per build initially**. CPU fallback remains available.
Runtime checks establish whether that compiled implementation can execute the loaded artifact; they
do not introduce a registry or a per-operation virtual dispatch. Do not add a `SUB0_DEVICE` choice
until its implementation and configurator/build consumers land together.

### Decompose by ownership and execution responsibility

Proposed destinations, created only when existing code moves into them:

```text
src/backends/cpu/
  state.hpp/.cpp       parameter storage, Worker ownership, private views
  ops.cpp              CPU forward operators
  backward.cpp         autograd traversal and backward operators
  model.hpp/.cpp       private Layer/Model construction and full-forward orchestration
  decode.cpp           incremental KV/GDN/QSA state and execution
  optimizer.cpp        AdamW/Muon and reduction
  api.cpp              existing sub0:: exported entry points
src/backends/cuda/
  context.hpp/.cu      stream, BLAS, allocation and graph ownership
   state.hpp/.cu        device allocations, cache/reset state and invalidation reasons
   weights.hpp/.cu      dense mirrors, fused projections and encoded-weight views
  kernels/            linear, attention, normalization, GDN, elementwise, optimizer
  forward.cu          inference launch sequences
  decode.cu           incremental execution
  train.cu            backward/training launch sequences
  api.cu              production C exports
  diagnostics.cu      existing backend-specific test/profiling exports
src/backends/intel/    created only for the selected native implementation
  context.*, weights.*, prefill.*, decode.*, kernels/, api.*
```

Tests stay under `tests/` and benchmarks under `benchmarks/`, consistent with this repository.
Private headers are not new public include surfaces. Do not split by arbitrary line count or mirror
every CPU op with a wrapper class. Small fused kernel helpers remain inline/template-local when
cross-TU calls would lose specialization. The directory names above are ownership targets, not an
instruction to expose every current global. Preserve the current ownership model during mechanical
moves, then introduce explicit private contracts at the boundaries below.

The first private contracts are deliberately small:

* CPU `WorkerState` owns parameter spans, activation/gradient arenas, the node pool, thread binding,
   and per-worker mechanism scratch. `Model` may hold pointers into that state, but no extracted file
   may create a second worker/layout owner. Decode caches remain separate because they reset per
   generation rather than per graph.
* CUDA `BackendContext` owns the stream, cuBLAS handle, allocation registry, synchronization, and
   forward/decode graph handles. `DeviceState` owns logical cache contents, binding tables, scratch
   capacities, and the reasons that invalidate them. Launchers receive narrow views of these objects;
   kernels do not reach process globals.
* CUDA graph invalidation is one private contract. Parameter upload, optimizer updates, TF32 changes,
   scratch reallocation, KV reset, and binding-buffer replacement must name which graph they invalidate.
   A buffer move is not complete until the dependent graph is discarded.
* `WeightView` distinguishes the dense parameter arena, fused/QKV mirrors, reduced-precision mirrors,
   and encoded expert ranges. An encoded range must never be represented as a generic f32 pointer.

Do not move CUDA diagnostics first: they currently depend on private kernels, launchers, global
buffers, stream state, and cuBLAS. Establish the context/state and a private diagnostic facade first;
then move diagnostics as a consumer of those contracts. Measure compile time, binary size and runtime
after each bounded extraction.

The `api` units consume each backend's implementation; forward/decode/train consume kernel launchers
and private storage. Kernels depend on math/layout descriptors, not on stages. `engine_core.cpp`
continues to own existing persistence/sampling logic. Extract shared **metadata and contracts**, not
a universal tensor graph or a common reduction implementation that changes reference numerics.

### Ownership and lifecycle for the first Intel integration

One generation/evaluation session owner acquires the backend, loads/prepares weights, reserves the
configured maximum cache/scratch, and releases resources after outstanding device work completes.
Implement that ownership as an internal non-copyable RAII object in the consuming stage/helper; its
creation must succeed before decode is callable. The initial supported contract is one model and one
active device session per process. No claim of concurrent gen/eval/train support.

Within the backend, one concrete context owns queues, compiled graphs/primitives, device allocations
and cache state. Borrowed host artifact descriptors need survive only until preparation finishes
unless explicitly retained; a retained mapping has an owner that outlives all queued reads. Scratch
and packed buffers remain stable until completion. Backend destruction frees backend allocations in
the backend DLL. Async exceptions are caught there and translated into a status plus caller-owned
diagnostic storage; they never cross the ABI.

Separate mechanical extraction from changing ownership. First preserve today's public lifecycle and
single-process behavior. Add the internal session guard when the Intel inference consumer is wired.
Only introduce public opaque session handles if a second simultaneous session becomes an actual
consumer; do not claim that reorganizing files makes current global state thread-safe.

### Minimal inference seam additions, with named consumers

| Addition | Contract | Consumer |
|---|---|---|
| Artifact preparation | Validate build layout, architecture fingerprints, tensor names/shapes, encoding and packing compatibility; accept bounded encoded ranges plus dense weights without expanding the whole pool | Generation setup / model loader; Intel context |
| Reserve/reset inference state | Prepare a maximum context/chunk plan outside the token loop; reset KV, QSA pooling/indexer, GDN convolution/recurrent state together | Generation session setup and new conversation |
| Chunk prefill | Ordered token IDs + absolute start position + valid length; update all state and return only requested final logits | Existing prefill loop in `decode.hpp`, replacing repeated single-token submissions when supported |
| Existing single-token decode | Submit one token, update state once, return logits under current sampling contract | Existing generation loop |
| Inference memory query | Separate unique host-resident bytes, dedicated-device bytes, shared allocations, scratch, packed caches and peak preparation requirements | `memplan` / setup feasibility report |

Define exact POD structures and ABI revision only when the first caller is implemented. The native
CUDA export flip should remove the bridge, preserve backend-private diagnostics, and test both the
CPU-only stubs and mock device. Version/size a changed boundary and reject mismatched DLLs; this is
**not** a checkpoint-format change. Model artifacts and architecture fingerprints keep their existing
semantics. Backend/packing identity belongs in build/derived-cache metadata, not the model architecture
fingerprint unless the actual mathematical model changes.

The ABI extension must be treated as an artifact/session contract, not just a longer list of
operations. Before an Intel inference caller is accepted, the backend must validate the complete
artifact description (layout fingerprint, tensor shapes, precision, encoded ranges and packing
identity), reserve state/scratch, and report memory by ownership category. `upload_params(const float*)`
remains a legacy dense path for CUDA parity and training; it is not the model-loading contract for an
encoded Intel artifact. Keep these descriptors private until a real generation/evaluation consumer
uses them, but make their version/size negotiation part of I07/I11 rather than a later retrofit.

Sampling initially stays on CPU for parity. Later, measure device top-k as a separate change:
callbacks and diagnostic consumers currently expect full logits, so a token-only return cannot silently
replace that API. An OpenVINO GenAI full pipeline similarly belongs behind a whole-generation adapter
if it owns sampling/tokenization; do not disguise it as the same low-level `forward_one` contract.

## 6. Performance hypotheses to test

- **Batch-one decode is not prefill GEMM.** Large GEMM peak throughput does not predict quantized
  matrix-vector speed, launch latency, sparse gathers, or router-selected expert throughput.
- **Keep a coherent chain on one device.** First compare CPU-only and iGPU-only sub-stacks, then a
  deliberate split at an MoE/block boundary. Tiny offloads around CPU-heavy stages can lose overall.
- **Fuse where traces justify it.** Quantized dequant-dot, norm/gating, and stable decode launch graphs
  are candidates. SYCL graph replay can reduce submission overhead, but dynamic top-k/expert selection
  must remain correct. [Intel SYCL graph explanation](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-offload-many-kernels-sycl-graph.html).
- **Use the right precision per mechanism.** Start with FP32 state/reductions for GDN, router and
  sensitive gates; trial FP16 dense weights/activations separately. Require nonfinite checks and
  multi-token drift measurements. Do not assume BF16/TF32/native low-bit acceleration on this device.
- **Count preparation too.** Lazy packing, JIT compilation, allocator caches and weight repacking can
  hide seconds or extra model copies behind attractive warm decode rates.
- **CPU+iGPU overlap can contend.** They share RAM bandwidth and package resources. Overlap independent
  selected experts only if the measured critical path improves after synchronization and contention.
- **Retain the NVIDIA comparison.** The optimal use of this laptop may be NVIDIA-heavy execution,
  CPU-only execution, or an Intel assist; using every device is not itself the objective.

A placement decision must use measured `compute + transfer + synchronization + contention` time.
For prefill-on-iGPU/decode-on-CPU, include copying/reformatting GDN state, convolution history, QSA
indexer/pooling state and KV, not just a KV transfer. Prefer a single placement per session initially.
For CPU+iGPU+CUDA later, use an explicitly selected composite implementation or scoped private
orchestrator; do not link two libraries exporting identical `sub0_dev_*` names or pretend OpenVINO
AUTO/HETERO schedules Sub0Llm's CUDA kernels.

## 7. Evidence required to choose a winner

**Gate A — platform:** real GPU enumerates, tiny operation and small model run correctly, supported
precisions/allocation limits recorded. A registry entry alone does not pass this gate.

**Gate B — correctness:** component fixtures at realistic small and exact head geometry; real encoded
expert planes of each used type; complete four-layer prefix compared to WP4f's corrected oracle.
Compare intermediate values, logits, top-k/router indices, masks, and every persistent state. Test
reset, second prompt, chunk boundaries, uneven tail, repeated decode, and context limit failure.
Missing required fixtures or unexpected CPU fallback is a reported failure, not a passing skipped test.

Use the existing reference tolerances first, record absolute/relative L2 and max error with denominator
floors, and freeze the chosen bounds before timing. Reduced precision has a separate quality gate;
do not demand bitwise GPU equality or relax tolerances after observing an inconvenient result. Tie
handling and actual selected-token disagreements need explicit investigation.

**Gate C — interactive measurements:** same token IDs, weights, architecture, context, precision and
sampling policy. Run CPU, SYCL, Vulkan and each eligible OpenVINO route in isolated builds. Include
upstream CUDA where valid. Distinguish startup/model preparation, first request, warm TTFT, prefill
tokens/s, and p50/p95 inter-token latency. Use prompts of 32/128/512/2048/4096 tokens where supported,
128–256 generated tokens, batch one first; batch four is a secondary diagnostic. Exact-scale prefix
timings remain prefix timings. Build larger-context variants explicitly; never exceed a baked limit.

Record at least five interleaved trials after warmup, temperature/power mode, driver/compiler/runtime
versions, per-op placement, allocation/copy counts, peak committed/working-set/shared-device memory,
page faults, and any supported power/energy counters. Separate cold/warm filesystem and compiled-model
caches. No simultaneous builds, benchmarks, training or model conversion on this laptop during timing.
Preparation and large-memory stress get their own scheduled runs.

**Gate D — integration value:** choose the fastest feasible route under the established correctness
and memory constraints. A provisional practical promotion threshold is a repeatable **20% reduction
in warm decode latency or TTFT**, with no material regression in the other (provisional 5%), relative
to the best comparable current route. These are proposed investment thresholds, not measured results.
If winners differ by prompt length, publish the tradeoff rather than an arbitrary composite score.
If iGPU loses, retain the useful refactor and evidence, and stop the native port. A quality-preserving
memory/energy advantage can justify a separate mode, but must be explicitly demonstrated.

## 8. Design review and outstanding decisions

Questions a new implementer would need answered are resolved in the plan: target is the preview text
decoder; initial owner is one session; encoded expert access is mandatory; chunk prefill is a consumed
addition; training is deferred. Exact runtime/toolchain versions, support results and placement are
experiments, not guessed API policy.

The plan-review pass identified these risks and incorporated the resolutions above:

1. **Duplicate device abstraction:** extend `device_backend.hpp`; do not create a second registry or
   virtual `TensorBackend` hierarchy. Basis: existing BACKENDS design and AGENTS §8/§10.
2. **Ownership gap across async work:** make preparation, stable borrowed storage and completion before
   destruction explicit; do not equate Worker slots with arbitrary host threads. Basis: cpp-review L1
   ownership rule and B18.
3. **Nominally neutral f32 upload excludes actual weights:** include encoded sidecar preparation and its
   generation caller in the same package. Basis: AGENTS §8/§10 and WP4e's current layout.
4. **Premature CPU/CUDA unification threatens specialization:** isolate mechanical moves and keep
   backend math/schedules separate; measure any shared-template extraction. Basis: AGENTS §2/§6.
5. **A generic "supports Qwen" claim hides missing behavior:** test the exact artifact and all recurrent
   states; qualify the reduced prefix. Basis: AGENTS §5/§7/§9.

No remaining design-policy blocker requires a user decision before the research work packages begin.
Hardware timing, concurrency behavior and final architecture fit still need review using the resulting
evidence; this document does not certify them. No drivers/SDKs were installed, large artifacts downloaded,
or device benchmarks run in this pass.
