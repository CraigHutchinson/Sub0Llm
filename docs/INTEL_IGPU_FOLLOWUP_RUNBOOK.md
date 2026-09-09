# Intel iGPU follow-up execution runbook

Date: 2026-09-09. Target workload: interactive Qwen4 text inference on the local Intel iGPU;
training is deferred. This is the handoff entry point for the next session.

## What this branch establishes

`research/intel-groundwork` contains planning, standalone probes and archived measurements. It does
not add an Intel production target, alter default engine selection, or qualify a Qwen4 backend.
The branch was created from `90721bc`, rebased onto `e5af1ad`, then integrated `main` through
`6189121`. Archived 2026-09-08 measurements and the 2026-09-09 compile manifest retain their exact
original bases. Reconcile again only if `main` advances before merge, then repeat affected gates.

Measured on the named local tuple:

- Intel PCI `8086:7D67` executes SYCL through Level Zero using DPC++ 2025.3.3 and oneDNN 3.9.1.
- Host/shared/device USM allocations work; system USM is unavailable. Runtime-reported addressability
  does not prove arbitrary Windows mapping access, zero-copy residency, or sustainable capacity.
- Explicit mapped-file staging and custom SYCL -> oneDNN -> custom SYCL pass at two sizes.
- Ten f32 projection controls pass. They omit real IQ weights, recurrent/sparse mechanisms, full
  vocabulary head, request timing and full-model memory pressure.

Still open:

- direct Level Zero development/API availability and checked native kernel submission;
- runtime results for the compiled USM capability and prepared-copy probes;
- exact IQ1_S/IQ2_XXS/IQ4_NL code generation, correctness and performance;
- complete GDN/GR/QSA/MoE dependency chains and user-visible inference timing;
- sustainable memory envelope and a pinned full-artifact capacity calculation;
- the Windows host/DPC++ ABI boundary and real generation consumer;
- any second-target evidence needed to fund a portable SYCL executor.

The evidence taxonomy in [the spike policy](INTEL_IGPU_SPIKE_EXECUTION.md) is mandatory. Record
measured, documented, hypothesis and unknown claims separately. A missing dependency is an available
result state; do not replace it with a guessed API or an unrecorded fallback.

## Source-of-truth map

| Need | Source |
|---|---|
| Architecture, ownership and lifecycle | [backend design](INTEL_IGPU_BACKEND_DESIGN.md) |
| Package dependencies and completion gates | [work packages](INTEL_IGPU_WORK_PACKAGES.md) |
| Measurement protocol and promotion thresholds | [performance contract](INTEL_IGPU_PERFORMANCE.md) |
| Existing measurements and limitations | [groundwork results](INTEL_IGPU_GROUNDWORK_RESULTS.md) |
| Direct-ZE/provider/portability status | [tandem findings](INTEL_IGPU_TANDEM_SPIKE_FINDINGS.md) |
| Windows USM contracts | [Windows USM review](INTEL_IGPU_WINDOWS_USM.md) |
| Prepared but unexecuted probes | [USM capability](INTEL_IGPU_USM_CAPABILITY_SPIKE.md), [prepared copy](INTEL_IGPU_PREPARED_COPY_SPIKE.md) |
| Preview/experimental features to recheck | [release watchlist](INTEL_IGPU_RELEASE_WATCHLIST.md) |
| Prior plan findings | [whole-plan review](INTEL_IGPU_PLAN_REVIEW.md) |

If two documents disagree, dated raw manifests/results take precedence for observations, while the
work-package gates control what may be claimed complete.

## Resume preflight

1. Confirm the active worktree and branch. Never run this work from the main Qwen worktree.
2. Read `AGENTS.md` and `docs/ACTIVE_WORK_LOG.md`. Add or update an Intel row before holding a shared
   file or running a compiler/device/CPU-heavy workload for more than a few minutes.
3. Record current `main`, branch HEAD, merge base and dirty paths. Rebase only when concurrent owners
   no longer need the old base. Preserve main's active-log history when resolving that file.
4. B20 and its follow-up memory audit are merged. Recheck the active log for a successor before editing
   `moe_quant.hpp`, `transplant.hpp`, or `src/backends/cpu/decode.cpp`; the first research phases need
   no such edits.
5. Delegate each implementation spike to a Sol-class agent with bounded leaf ownership. Delegate
   external research/data gathering to Sonnet/Terra. The integration owner reviews and merges results.
6. Do not install or upgrade drivers, SDKs, runtimes or dependencies without separately authorized,
   reproducible acquisition work. Existing-tool and source audits can record an unavailable route.

## Ordered execution

Runtime status as of 2026-09-09: Phase 2 and the listed Phase 3 SYCL/USM controls passed and are
archived in [the R0 checkpoint](INTEL_IGPU_R0_CHECKPOINT.md). The conditional direct-ZE inventory did
not run because its development tuple is unavailable. Resume at the remaining Phase 1 records and
R0-authorized I21/S1.5/S2 preparation; do not repeat runtime controls without a changed tuple or a
specific reproducibility need.

### Phase 0 — make this groundwork PR reviewable

- Rebase onto the intended target branch and resolve the expected `AGENTS.md` and
  `docs/ACTIVE_WORK_LOG.md` conflicts from current main rather than restoring their older copies.
- Review the complete diff against the actual merge base. Confirm no production CMake or engine source
  consumes `tools/intel_probe/` or `benchmarks/intel/` spike headers/sources.
- Parse every changed PowerShell file and reproduce compilation of both standalone sources. Both
  sources compiled on 2026-09-09; repeat after rebase or any source/runner/toolchain change.
- Run the probes only in a reserved hardware window. If runtime evidence is excluded from this PR,
  keep their reports explicitly marked unexecuted and open a follow-up evidence commit.
- Update current branch/base/status statements and ensure the working tree is clean. Preserve archived
  measurement commits, hashes and tuple dates verbatim. Do not publish a PR until its actual base,
  title and diff are known.

Suggested PR scope: research package, reproducible standalone probes, archived early measurements and
the execution plan. Suggested title: `research(intel): prepare native iGPU backend groundwork`.
The description should state that production backend selection and Qwen qualification are absent.

### Phase 1 — freeze inputs and build boundaries

Run these in parallel only where their files do not overlap; none requires device timing.

1. **I01 artifact admission.** Pin WP5a `bb1fc70`, WP5b `e1bea23` and WP5c `2f57d14` through one
   immutable manifest: commit, generated config/fingerprint, source GGUF identity and conversion
   recipe, model and sidecar checksums, loader format, measured RSS, fixture origins, expected
   state/logits and tolerances. Hash every tokenizer asset, including vocabulary, merges, tokenizer
   JSON and `chat_template.jinja`; add PLE/n-gram paths, formats, origins and hashes when available.
   Record retrieval/rebuild commands plus WP5c's raw-prompt/no-chat and baked-context limitations.
   If current main changes an artifact, use its new exact revision and reopen affected gates. Admit
   the real dense control required by I11 separately from the corrected four-layer Qwen prefix: exact
   files, architecture/config, vocabulary head, token sequence, expected logits/state and tolerances.
2. **I07b.0 toolchain record.** Map every consumer of `HAS_CUDA`, `SUB0_BUILD_CUDA`,
   `SUB0_BUILD_DEVICE`, `ComputeBackend`, `sub0_dev_*` and CUDA-specific links. Pin host and DPC++
   compiler/STL/CRT, generator, oneDNN, Level Zero files, generated-header dialect, DLL exports and
   search path.
   Specify the first Intel DLL as inference-only and define its false capabilities.
3. **S0a dependency audit.** Identify the exact oneDNN tag/source and documented Windows direct-ZE
   support plus matching Level Zero headers/import library/runtime/loader. Record `native ZE unavailable
   on current tuple` if a compatible approved set cannot be identified. Continue with SYCL controls.
4. **Probe code review.** Apply cpp-review to ownership, completion, cleanup, integer bounds and result
   schemas in the prepared sources and runners before compiling them.

Phase 1 outputs are documents/manifests. Do not add a production backend selector or target here.

### Phase 2 — compile-only probe gates

From this worktree in an uncontended compilation window:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -PreparedCopyApi
```

If compatible Level Zero development files were already approved and pinned, add
`-LevelZeroInclude <include-parent>` and `-LevelZeroLibrary <exact-ze_loader.lib>`. Do not supply only
one. Capture compiler, headers, libraries, hashes, commands and exit status. Compile failure keeps the
corresponding runtime arm closed while preserving other controls.

Compile the prepared-copy runner without device execution:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-prepared-copy.ps1 -CompileOnly
```

Keep all build products under the isolated output tree. A successful build is not a measurement.

### Phase 3 — one serialized hardware window

Recheck the active log and competing processes, then execute in this order. Each command rebuilds and
runs the exact executable whose hash its manifest records:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -Run
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -PreparedCopyApi -Run
pwsh -NoProfile -File scripts/intel/inventory/run-prepared-copy.ps1
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -WithOneDnn -Elements 257
pwsh -NoProfile -File scripts/intel/inventory/run-groundwork.ps1 -WithOneDnn -CheckFailures
```

Run the direct-ZE extension inventory only if Phase 1 produced the compatible file tuple, using both
arguments on the same ordinary capability command:

```powershell
pwsh -NoProfile -File scripts/intel/inventory/run-usm-capabilities.ps1 -Run `
  -LevelZeroInclude <include-parent> -LevelZeroLibrary <exact-ze_loader.lib>
```

The dependency manifest hashes `ze_api.h` and the import library. Before this arm can qualify, the
runtime probe must also report the actually loaded `ze_loader.dll` path/file version and Level Zero
API version; never infer those from the import library. Review raw samples and correctness before
aggregating. Prepared-copy modes currently execute in one fixed-order
process, so they establish mechanism correctness and gross costs only. A winner requires I21's
alternating independent-process protocol. Archive a dated manifest and raw output for every attempt,
including unsupported/failure outcomes.

### R0 — authorize the next research wave

Publish a table for I00, every S0 arm and every S1 arm with `planned`, `prepared`, `compiled`,
`measured`, `unavailable`, `inconclusive`, or `qualified-component` status. R0 may pass with direct ZE
unavailable if the SYCL control is sound. It may not imply S2-S4, integration, performance promotion,
portability or full-model capacity.

Resolve these questions explicitly:

- What exact Windows host/DPC++/CRT candidate contract proceeds to the I07b.0 smoke proof, and what
  remains unproved until R2?
- Which allocation and explicit-copy baseline proceeds to real IQ work?
- Is direct ZE runnable now, deferred for dependencies, or rejected for this tuple?
- What exact fixture manifest feeds S2/S4 and the later full vocabulary head?
- Which active owners constrain shared files and hardware scheduling?

### Phase 4 — quantization and representative dependency chains

After R0, delegate one Sol agent per bounded spike and keep leaf ownership separate. Follow the
dependency order below; parallelism applies only to siblings whose named inputs already exist.

1. **Activate I21's harness/schema** with the first runnable control. Independent-process alternation,
   exact environment/artifact identity, asynchronous completion and raw result retention apply to all
   subsequent comparisons.
2. **Complete I19/S1 steps 3-5:** conditionally test documented mapped import and writable scratch,
   then shared/device write-only scratch and prefetch/advice controls. Unsupported import closes only
   those conditional arms.
3. **S2/I18:** use actual IQ1_S/IQ2_XXS/IQ4_NL planes at non-square gate/up/down shapes. Compare fused
   float decode-dot, bounded tile decode and integer-dot only with validated activation quantization.
   Retain emitted-ISA evidence or label instruction use unproven.
4. **S3/I20:** after I18 supplies a checked kernel/binary, compare ordinary SYCL, supported graph replay
   and direct ZE only if available, using the same dependent kernel and changing expert
   indices/arguments. Separate submission from device time.
5. **S0a:** after the direct-ZE dependency gate and checked native submission exist, execute the
   native custom -> oneDNN -> custom composition. If direct ZE is unavailable, retain the existing
   SYCL composition and explicit-copy controls as the result.
6. **Complete I19/S1 step 6:** cross viable memory modes with S2's real IQ kernel and representative
   selected-expert/dense access traces. This produces the safe memory recipe consumed by S5/I10.
7. **S4/I05:** after S2 and the selected submission/memory controls, add
   GDN/GR/QSA/router/selected-expert/state transitions and full vocabulary-head transfer
   to a representative chain. Label omitted operations; never call a partial chain TTFT.
8. **S0c:** after S0a correctness, compare custom, library and mixed providers. Hold runtime constant
   for provider comparisons; hold binary/math/packing constant for runtime comparisons where possible.
   Otherwise label the result `combined-route`; do not infer provider superiority.
9. **S5/I17a:** compute capacity from I01's real artifact/RSS and I19's safe envelope. Shared address
  space is counted once and is not extra RAM.

R1 records proceed/narrow/defer/stop for each lane. I06 produces separate Intel recipe and portability
decisions. A portable executor remains deferred until a second real target runs the complete required
operation/format matrix.

### Phase 5 — production integration, only after I06

1. Complete I07a's neutral CUDA export cleanup without changing CUDA behavior.
2. Close I07b.0 with a POD smoke DLL loaded by production `sub0_gen` in `src/gen_stage.cpp`, exercised
   through `sub0llm-gen`. CPU-only needs no Intel tools; CUDA remains CUDA-specific; Intel cannot enter
   training paths. `tools/sub0llm-qwen4-gen.cpp` is a direct CPU/WP5 oracle and is not proof of this seam.
3. Land I07b and I11 together through that same production stage using I01's admitted dense control.
   Use a session-owned full
   `[VOCAB]` logits buffer and include copy bytes/completion time under the existing CPU sampler and
   callback contract.
4. Implement I10's encoded preparation/memory accounting and I12-I14 behind exact fixture gates, then
   I15's corrected four-layer prefix through the production stage.
5. Tune only from the integrated profile in I22. After I17a capacity passes, I17b wires the pinned full
   WP5 artifact, real tokenizer/chat/EOS/PLE/history and streaming into the production stage or a named
   replacement reviewed as the new production consumer.
6. Qualify dense, prefix and full decoder separately in I23. Only I17b permits a useful full-Qwen claim.

Before I17b/I23, resolve WP5c's known Windows sampler stack hazard: `sample_token` currently requires
about 2.84 MiB of stack at vocabulary 248320, while only the Qwen4 tool has a 32 MiB linker workaround.
Either retain and verify an explicit stack setting on every production target that samples, or land the
preferred pre-sized reusable scratch fix with exact default/real-vocabulary tests. Do not infer that the
tool-specific workaround protects `sub0llm-gen`.

## Stop and reopen rules

- Stop production interface work if toolchain ABI, allocation ownership or artifact identity is
  unresolved at R2.
- Stop a benchmark arm on wrong-device fallback, missing required fixture, numerical mismatch,
  allocation failure or unsafe lifetime. Record the failure before changing the experiment.
- Reopen affected gates when compiler/runtime/driver, artifact checksum, math/packing, memory envelope
  or supported instruction changes. Preserve the old tuple and result.
- A correct component can justify another component experiment. It cannot authorize the full backend.
- Retire spike scaffolding from default builds once its finding has a production consumer and no
  `src/` or `tools/` code includes it, following `AGENTS.md` section 11.

## PR-readiness checklist

- [ ] Branch rebased on the intended published base; expected conflicts reconciled with current main.
- [ ] All changed and untracked paths reviewed; no unrelated main/Qwen/B20 work included.
- [ ] Documentation links resolve and status/date/base statements agree.
- [ ] PowerShell files parse; standalone C++ probes compile on the pinned tuple.
- [ ] Required runtime evidence is archived, or reports remain plainly marked unexecuted.
- [ ] No production build/config option exists without a real consumer.
- [ ] Default CPU/CUDA behavior is mechanically unaffected by unregistered standalone probes.
- [ ] Commit history is reviewable and working tree is clean.
- [ ] PR description lists measured evidence, absent production behavior, validation and known gaps.
