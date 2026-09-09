# Standalone Intel groundwork

Run `scripts/intel/inventory/run-groundwork.ps1` from a fresh PowerShell process.
`-WithOneDnn` enables the library handoff; `-Elements 257` tests an uneven small extent;
`-CheckFailures` checks five negative cases. `-DenseBenchmark` and `-MemoryBenchmark` run
separate exploratory component benchmarks and require a reserved quiet measurement window.
The runner temporarily imports installed compiler paths in its process. It does not configure
or link the Sub0Llm engine. Outputs are uniquely named below `out/intel-review/`.

The probe accepts only Intel PCI 8086:7D67 on Level Zero. All other devices fail explicitly.
See [results and limitations](../../docs/INTEL_IGPU_GROUNDWORK_RESULTS.md).
API contracts were checked against the installed headers and the
[SYCL specification](https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html) and
[oneDNN interop guide](https://uxlfoundation.github.io/oneDNN/dev_guide_dpcpp_interoperability.html).
These are maintained measurement consumers, not a production backend or speculative engine API.

Additional preparation (not yet compiled or executed):

- [USM capability probe](../../docs/INTEL_IGPU_USM_CAPABILITY_SPIKE.md): exact-driver extension/API inventory and separate prepared-copy API qualification.
- [Prepared-copy comparison](../../docs/INTEL_IGPU_PREPARED_COPY_SPIKE.md): ordinary mapped copies, prepared ranges and reused host-USM staging.

Reserve compilation/measurement time before following those reports' commands. The
[release watchlist](../../docs/INTEL_IGPU_RELEASE_WATCHLIST.md) identifies newer and experimental
candidates; it does not change the installed baseline or qualify a replacement runtime.
