# Intel backend whole-plan review

Initial pass 2026-09-08; PR-readiness pass 2026-09-09 starting from branch `47c3cf2` and `main` at
`e5af1ad`. Reviewed committed plan `0e82ec3`, then revised documentation only. No Intel
implementation, benchmark or runtime qualification was performed. The C++ skill pass was read-only;
plan authoring followed its findings. Broader performance/delivery findings are project review,
not C++ style rules. Existing repository `.hpp`/`.cu` conventions take precedence over generic
skill examples. No speculative public C++ declarations were authored.

The branch was subsequently rebased onto `e5af1ad`; archived measurements preserve their original
`90721bc` provenance. The rebase and diff review are mechanical PR preparation, not evidence that the
backend is ready to implement.

## Proposed artifact and consumer audit

| Proposed artifacts | Named consumer / owner |
|---|---|
| CPU private state/ops/backward/model/decode/optimizer | Existing CPU API facade; I08 preserves shared parameter ownership |
| CUDA context/state/weights/kernel launchers/execution | Existing production API; diagnostics consume private contracts after I09 |
| Neutral ABI/build facts | Generation/evaluation/training setup and stubs; I07a cleanup, I07b with I11 |
| Intel context, dense kernels and API | I11 real dense generation consumer |
| Encoded views/packing/memory plan | Private I10/I11 experiments, public only with I15 encoded generation |
| GDN/GR/QSA/MoE kernels and state | I15 execution chain, after I12–I14 fixture checks |
| Chunk prefill/reset/session ownership | Existing generation setup/prefill/token loop; I15 |
| Hardware/ISA/memory/submission probes | Named experiments and I06 decision; benchmark artifacts, not production APIs |
| Benchmark schema/runner and optimization ledger | I21's first runnable probe, then I15/I17b; I22 consumes results |
| Full decoder and deployment qualification | I17b real tokenizer/PLE/generation consumer; I23 release evidence |

The existing `device_backend.hpp`, `decode.hpp`, layout/math helpers and provenance generator remain
the integration points. No second tensor engine, device registry or generic benchmark framework is
justified. Public POD/ABI layouts receive another review when actual declarations are proposed.

## Findings and revisions

1. **[MUST / L0-PLAN-STOP, resolved in plan] I11: “a real generation request runs”.**
   Its scope named dense projections/norm/RoPE but omitted attention, embeddings and FFN execution.
   The stated deliverable needed a complete operator chain (AGENTS §8/§9; skill L0 goal/artifact rule).
   Revision: I11 now names the complete dense path and requires one real control artifact/config.
   Lower-layer review of that missing path was deferred until this scope revision.

2. **[SHOULD / L0-PLAN, resolved in plan] I06: “warm TTFT and p50/p95 decode latency”.**
   The native chain exists only in I11/I15, which depend on I06. Requiring those native results first
   makes the decision unexecutable. Revision: component-based investment at I06; baseline at I15;
   integrated performance promotion at I22/I23. External results remain optional context.

3. **[SHOULD / L1-PLAN, resolved in plan] I05/I00/I01 broad directory ownership.**
   Independent lanes claimed overlapping `benchmarks/intel/` and `scripts/intel/` roots, contrary to
   the active-log coordination intent. Revision: mechanisms/inventory/fixtures leaf ownership;
   I21 owns common benchmark utilities and one integration owner controls shared build/API edits.

4. **[Project delivery gap, resolved in plan] I17 buried the useful decoder with deferred training.**
   Revision: I17a early capacity and I17b full text execution now have explicit consumers/dependencies
   and completion gates. Device sampling, vision/MTP/NPU and training remain separately deferred.
   An infeasible full artifact remains an honest open/failed feasibility result, not prefix success.

5. **[Project performance gap, resolved in plan] No durable measurement owner or timing contract.**
   Revision: I21 and the performance document define engine/user TTFT, async completion, raw paired
   trials, correlated token samples, cache modes, metadata, profiler overhead and regression evidence.
   Qualification cannot pass on unavailable hardware or missing required fixtures.

6. **[Project optimization gap, resolved in plan] Hypotheses lacked a bounded promotion process.**
   Revision: I22 ranks measured contributions, budgets tuning, validates math before timing, uses
   ablation and integrated reruns, and freezes winning recipes outside decode. Retained changes need
   precision/state, memory and multiscale evidence, not just a faster isolated GEMM.

7. **[Project lifecycle gap, resolved in plan] Deployment and failure behavior were scattered.**
   Revision: design lifecycle plus I23 cover compatible runtime/library contexts, preparation failure,
   cancellation boundaries, device loss, retained memory, redistributables, default builds and explicit
   qualification levels. Existing ownership prose was retained; no new session framework was invented.

8. **[Project planning gap, resolved in plan] No explicit experiment-to-plan feedback loop.**
   Revision: six bounded spikes S0–S5 have decision consumers, evidence cards and retirement rules;
   checkpoints R0–R4 revise assumptions, dependencies, estimates and qualification from actual results.
   Stable package IDs survive changes; only affected gates reopen when inputs change.

## 2026-09-09 PR-readiness findings

1. **[MUST / L0-PLAN-STOP, resolved as I07b.0] The proposed target had no executable route through
   CUDA-only build facts and consumers.** `GPU`, `AUTO`, `HAS_CUDA`, `SUB0_BUILD_CUDA`, generated
   `ComputeBackend`, `sub0_dev_*` and executable links must be mapped before adding Intel. The revised
   gate keeps CUDA facts CUDA-specific, defines a generic linked-device fact, makes the first Intel
   target inference-only and proves it through the real generation consumer.

2. **[MUST / L1-PLAN, resolved as I07b.0] The Windows compiler/DLL boundary was assumed.** The
   host-only superbuild and standalone DPC++ probes do not prove host C++26, generated headers,
   DPC++/oneDNN, STL/CRT, exception, allocator and unload compatibility. The revised plan requires a
   pinned toolchain record and POD smoke DLL loaded by the generation stage before production wiring.

3. **[MUST / dependency stop, resolved in S0a] Direct Level Zero lacked matching development files
   and a release-pinned API.** The current tuple has a runtime loader but no identified headers/import
   library, while local oneDNN 3.9.1 does not qualify newer experimental direct-ZE APIs. The plan now
   records the route unavailable when a compatible approved tuple cannot be pinned and continues with
   the already measured SYCL control.

4. **[MUST / evidence identity, resolved in I01/I17] Upstream WP5 inputs were described as active.**
   WP5a/b/c are merged, but execution must consume an immutable manifest with commit, config,
   tokenizer/model/sidecar checksums, loader format and RSS. Changed artifacts reopen only affected
   fixture/capacity/integration gates.

5. **[SHOULD / decision scope, resolved in I06] One Intel tuple cannot approve or disprove a portable
   executor.** The Intel native/mixed-provider choice and portable-SYCL funding decision are now
   separate. The latter requires a second real target and complete operation/format coverage.

6. **[SHOULD / inference boundary, resolved in I11/I15] Full logits ownership and transfer were
   implicit.** The initial path now requires a session-owned pre-sized vocabulary buffer, full-head
   oracle, completed copy bytes/time and preservation of CPU sampling/callback behavior.

7. **[SHOULD / comparison validity, resolved in S0c] Provider and runtime changes could be conflated.**
   Results must identify provider-only, runtime-only or combined-route comparisons and control
   compiler, binary, math and packing where possible.

8. **[Project handoff gap, resolved] The documents lacked one ordered resume path.**
   `INTEL_IGPU_FOLLOWUP_RUNBOOK.md` now records branch hygiene, facts/unknowns, compile/runtime order,
   R0 questions, subsequent spikes, stop/reopen rules and PR-readiness checks.

9. **[MUST / execution order, resolved] The initial handoff scheduled dependent S0/S1/S2-S5 work as
   peers.** The runbook now activates I21 first, completes conditional memory controls, feeds I18's
   checked binary to I20, runs native composition before provider A/B, crosses real IQ with memory,
   then executes the representative chain and capacity work.

10. **[MUST / consumer identity, resolved] “Real generation stage” could refer to two different paths.**
    I07b.0/I11/I15 now target production `sub0_gen`/`src/gen_stage.cpp` through `sub0llm-gen`.
    `tools/sub0llm-qwen4-gen.cpp` remains a direct CPU/WP5 oracle until I17b's explicit full integration.

11. **[MUST / full-model delivery, resolved] The plan omitted WP5c's measured sampler stack hazard.**
    I17b/I23 must cover every production sampling target with a verified stack requirement or,
    preferably, reusable pre-sized scratch. The Qwen4 tool's local 32 MiB workaround is insufficient.

12. **[SHOULD / artifact completeness, resolved] I17b's manifest omitted source conversion,
    per-tokenizer assets, chat template and PLE/n-gram identity.** I01 now requires their exact origins,
    hashes, formats and rebuild/retrieval recipes and records WP5c's raw-prompt/baked-context limits.

13. **[SHOULD / existing contract reuse, resolved] The consumer audit omitted
    `SUB0_BUILD_DEVICE`.** I07b.0 now audits it and prefers it as the generic linked-device predicate
    when its actual consumers confirm those semantics.

14. **[SHOULD / evidence chronology, resolved] Rebase wording could imply rewriting historical
    provenance.** The runbook updates only current branch/status statements and preserves archived
    commits, hashes and tuple dates. R0 selects a candidate ABI tuple; only I07b.0/R2 can prove it.

## Remaining evidence and review gates

The revised plan is actionable for preliminary research; it does not certify implementation readiness
for every conditional package. Exact ISA support, legal pointer/library interop, usable RAM, quality
budgets for lossy modes and speed remain experimental. I17c–e intentionally require future designs.

Before public wiring, R2 reviews the actual consumed ABI, resource ownership and dependency graph.
Before optimization, R3 requires a correct integrated baseline and frozen comparison conditions.
Before a useful full-model claim, R4 requires I17a/b plus I23 evidence. There is no agreed absolute
interactive-latency SLO yet; publish absolute measurements and avoid claiming one has been met.

No remaining finding above blocks the bounded spikes as now specified. New evidence can invalidate
assumptions; the review checkpoints exist to revise the plan rather than conceal that uncertainty.
