# Intel release and experimental-feature watchlist

Reviewed 2026-09-09 by Terra; consequential compiler and graph claims cross-checked by the integration
owner. Research snapshot, not installed support or a promise of exhaustive coverage. No software was
installed and no new hardware capability was measured. This supplements the
[Windows USM review](INTEL_IGPU_WINDOWS_USM.md) and [execution policy](INTEL_IGPU_SPIKE_EXECUTION.md).

## Version boundary worth testing

Intel lists DPC++ **2026.1.1 (2026-07-29)** in its release history. The 2026.1 release adds native graph
recording; 2026.1.1 adds restricted host-task capture. Other relevant 2026 additions include handlerless
submission, directional USM prefetch, device clock access and SYCLBIN AOT linking. These are candidates
for reducing launch, movement or startup costs, not measured improvements here.
[Compiler release notes, updated 2026-08-11](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html).

Our existing measurements use 2025.3.3. Intel identifies 2026.0 as an ABI/API-breaking boundary requiring
recompilation for the newer SYCL runtime. Any comparison therefore uses separate build/output and
runtime environments, explicit loaded-library provenance and compatible primitive libraries. Preserve
the old baseline. Recheck numerical behavior and API migrations before comparing speed.
[2026 migration notice](https://www.intel.com/content/www/us/en/developer/articles/technical/sycl-breaking-changes-oneapi-dpcpp-compiler-2026.html).

The current Intel driver download page offers **32.0.101.8992** and lists the 275HX. Our measured
baseline remains 32.0.101.8991. This review did not establish a relevant compute fix, certification
status or regression result for 8992; keep driver uplift separate from compiler uplift so its effects
can be attributed. [Intel Windows graphics download](https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html).

## Prioritized candidates and remaining choices

Each row is a proposed experiment, not adoption. A released compiler can contain experimental APIs.
A current upstream branch can differ from that release; pin the chosen source revision before coding.

| Candidate / maturity | Relevant opportunity and uncertainty | Package, discriminating experiment and fallback |
|---|---|---|
| 2026 compiler/runtime, released; constituent extensions vary | Better code generation/startup/submission is plausible; exact local runtime compatibility untested | I00/S0: reproduce existing two-size correctness and oneDNN handoff in an isolated tuple; compare the same kernels before enabling new features. Keep 2025.3 baseline |
| Native SYCL graph capture, experimental | Can capture native and SYCL work together; requires supported backend and in-order immediate submission; executable graph update is unavailable | S3/I20: ordinary queue vs updatable SYCL graph vs immutable native capture; alternate routing/state inputs and reset. Test fixed pointers plus changing buffer contents separately from changed arguments/topology; ordinary submission remains valid |
| oneDNN 3.13, released; direct Level Zero runtime experimental | New native graph interop and quantized-layout optimizations may make a primitive/native hybrid useful; Windows/device/type coverage remains untested | S0/I05: compare SYCL primitive handoff with direct Level Zero primitive handoff, then M=1/prefill; keep existing oneDNN 3.9.1 and custom kernels |
| Prepared explicit copies, experimental | Already declared locally; Windows mapped-range benefit unmeasured | S1/I19: use the prepared-copy spike, including setup amortization; ordinary copy and reused host staging remain controls |
| Level Zero external system-memory mapping, optional extension | May avoid application staging; Windows read-only file-map acceptance and physical movement remain unknown | S1/I19: query extension, bounded native import then visibility checks; retain explicit staging. Never infer availability from a version string |
| ESIMD and matrix/dot instructions, Intel-specific; joint_matrix experimental | Instruction use, precision/layout suitability and speed on PCI 8086:7D67 are unqualified | S2/I18: exact-target compile/disassembly plus independently checked kernels at M=1/prefill; compare fused float decode, subgroup route and supported integer/matrix route with real IQ formats. Generic INT4 is not IQ fidelity |
| Directional prefetch / handlerless enqueue / device clocks / SYCLBIN, extension-specific | Potential movement, host-overhead and startup improvements; individual specifications and local support still need checking | I19/I20/I21: pin each extension, test one at a time on the same data/chain; host wall time, ordinary enqueue and current binary loading remain baselines |
| oneMKL 2026.1, released | Dense primitive control; its Linux-only system-USM feature cannot qualify Windows ordinary pointers | I05: only add a primitive comparison when its exact operation/precision helps; retain existing oneDNN and checked kernel |
| OpenVINO 2026 GPU LLM examples, external comparator | Published Windows INT4 examples motivate a comparison, but establish neither Qwen4 fidelity nor this PCI device's performance | Optional I04: equivalent supported model/workload, conversion precision and request boundaries; retain Sub0Llm execution ownership |

Native graph constraints above come from the [experimental graph specification](https://raw.githubusercontent.com/intel/llvm/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc),
reviewed on this date; this is a moving upstream URL. General graph update examples must not be applied
to native recording. Query `zeDriverGetApiVersion` and extensions: our recorded `1.15.39183+3` string
is not proof of Level Zero API version or graph support.
[Level Zero record/replay extension](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_RecordReplayGraph.html).

For copy/import sources and exact contracts, use the pinned links in the Windows USM review. Our empty
`matrix_combinations` result reports no combinations on the measured tuple; it proves neither absence
of hardware matrix units nor a working ESIMD workaround. Require independent instruction evidence.
[Intel target/AOT guidance](https://intel.github.io/llvm/GetStartedGuide.html).
The secondary library/comparator rows use [oneMKL release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/onemkl/2026.html)
and the [OpenVINO GPU LLM example](https://docs.openvino.ai/2026/model-server/ovms_docs_llm_quickstart.html).

## oneDNN changes that may affect the native design

The latest tagged release found was **v3.13 (2026-07-17)**. It adds native SYCL graph interoperability
and improves Intel-GPU matmul for specified activation/weight types and non-transposed layouts.
Its Level Zero persistent cache is experimental, as is grouped-memory NVFP4 support; none establishes
support for our IQ encodings. [oneDNN v3.13](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.13).

The v3.12 release introduced the experimental `ONEDNN_GPU_RUNTIME=ZE` route and small-M/N,
large-K matmul improvements. This directly motivates a second S0 interop arm: a checked primitive
sharing allocations/completion with our native submission, without assuming SYCL is the necessary
library boundary. Windows build support and allocation/context ownership must be established before
benchmarking this arm. [oneDNN v3.12](https://github.com/uxlfoundation/oneDNN/releases/tag/v3.12).

Keep that experiment bounded to one handoff and one representative projection before considering
an engine ABI change. Record primitive selection, input/weight layout, reorder/scratch bytes and
creation/JIT costs; a primitive win that requires costly repacking may lose at full integration.

## Review and promotion

S0 first separates toolchain effects from feature effects. S2/S3 retain competing code paths in private
benchmarks until R1 identifies useful options. R2 rechecks releases and interface churn before public
wiring. R3 reruns the meaningful alternatives on integrated inference, including changed expert
selections, state lifetimes, packing cost, memory headroom, startup and decode latency. R4 freezes the
qualified tuple; support on Linux, a discrete Arc GPU or a different model remains separately labelled.

Potential stabilization is a reason to watch a feature, not to schedule around its assumed arrival.
Unsupported or inconclusive outcomes carry a named next experiment or explicit deferral. Preview and
nightly builds need an identified relevant change and pinned revision; newer alone is insufficient.

Still open: exact bundled Unified Runtime/Level Zero revisions in the candidate compiler; current
preview-driver compute fixes and regression history; exact PCI ISA/XMX evidence; Windows external
mapping/handle interoperability; individual newer extension maturity and benchmark effect. No verified
beta-only advantage was established by this pass. Before any uplift, check these sources again and
record installer/library identity rather than treating this document as a permanent latest-version claim.
