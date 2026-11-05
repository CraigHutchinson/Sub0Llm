# Memory Index

Hooks are terse — open the file for detail. "READ FIRST" = carries a caveat that invalidates a number
you might otherwise cite. Closed/absorbed items are linked from their successor note, not indexed here.

## Sub0Llm — active thread: pivot to existing-model support (Qwen4 preview)
- [**START HERE** — handover lives in the repo (2026-11-05)](handover-docs-live-in-the-repo.md) — `docs/HANDOVER.md` + `docs/WORKING_PRACTICES.md` + a restorable `docs/agent-memory/` snapshot (`scripts/agent_memory.py`); refresh at every pause
- [Older handover snapshot, 2026-09-07](wp4-handover-2026-09-07.md) — WP4a-f+WP5a/b/c ALL DONE; B19/I08/B20 merged, decode DISK-bound not CPU-bound (at the time).
- [Decode-throughput thread B24-B38](qwen4-decode-perf-b24-b35.md) — ACTIVE, `/goal`-bound ("token-gen near CPU theoretical performance," not yet met): BF16 backbone → disk-vs-compute-bound reframing → dequant fix → thread-count fix → transpose elimination (merged, ~5.5-6.0s→~3.38s/token) → real llama.cpp comparison (~4-6x still faster) → FP8/SIMD negative results → **B36/B37/B38 (2026-09-11): all three negative-result branches (B25/B33/B34) integrated into `main` as real, permanent, default-off, correctness-gated CLI options** (`--moe-io-mode`/`--prec-param fp8`/`--simd-reduce`), not left stranded — user's own "A/B shootout compliant, cleanly maintained" directive. B35 merged 2.4x; **O1 AVX2 kernel merged 2026-09-22: 0.705 s/token (2.14x)**; now **~7-8.5 tok/s long-run with every O1-O14 default on (measured 2026-10-07/08; defaults are AUTO in `resolve_decode_defaults`)**; native is quality-NEUTRAL by G-PPL (ppl 14.85 vs 15.03, 04af872). THREE agents share one tree — read `docs/ACTIVE_WORK_LOG.md` first.
- [Pivot: from-scratch training CLOSED, focus = Qwen3.8-Flash-Next as validation oracle](pivot-to-existing-model-support-qwen4-preview.md) — READ FIRST: docs/ROADMAP.md + docs/QWEN4_PREVIEW_REFERENCE.md. Full chronological detail (GDN/n-gram/Phase0/WP1-3 mechanism builds, WP4a-f's own stage-by-stage real-transplant story, every found-and-fixed defect and its commit) lives IN that file, not duplicated here — open it for depth rather than trusting a summary.
- [Resource-constrained inference: Reddit prior-art link, unread (WebFetch blocked)](resource-constrained-inference-prior-art-reddit.md) — check before WP4, may be redundant with existing feasibility docs
- [Sub0Log adoption backlog](sub0log-adoption-backlog.md) — separate project (github.com/CraigHutchinson/Sub0Log), typed/crash-survivable C++23 logging, design-only; flagged for later, not scheduled
- [**Storage core use case = n-gram table (regime 2)**](storage-core-use-case-ngram-regime2.md) — READ FIRST for MemPage/TieredCache work: owned cache targets data that never fits; n-gram rows first, MoE second
- [Sub0ECS v2 landing state](sub0ecs-v2-landing-state.md) — 2026-10-05: work targets `master` (needs the user's approving review, no admin bypass); PRs 14 (audit fixes) and 15 (Sub0 style conversion, stacked) open; issues #13 and #12 hold what remains; use `rotate.py` for any between-build comparison
- [Sub0MemPage project](sub0mempage-project-init.md) — S1b DONE 2026-09-29: opt-in `--moe-io-mode mempage` is the fastest COLD reader (0.241 vs IOCP 0.267 vs mmap 0.396 s/tok); warm reactive stays default; next S1c
- [Self-managed memory is the direction](self-managed-memory-is-the-direction.md) — RAM-constrained host: owned cache is the reference path, not a regime-2 special case; measure optimizations there too
- [Context-memory planner (backlog)](context-memory-planner-backlog.md) — size KV/context per chat (prompt + response + thinking), one planner for all runtime memory splits; design first
- [Owned expert cache state](owned-expert-cache-state.md) — READ FIRST for uncached I/O: 2026-10-06 uncached fills default in cache mode, ~10% over reactive in deep regime 2; a cached reader of the same file serializes non-cached reads on Windows
- [Real Qwen4 48-layer configure recipe](qwen4-real-axes-configure-recipe.md) — READ FIRST before building any tool against the real artifact: a PARTIAL recipe silently gives PARAM_FLOATS 2570717696 and a rejected model with no hint which axis is wrong
- [B21/B29 concurrency 'ceiling' = PIN ARTEFACT (B40)](b21-decode-concurrency-plateau.md) — READ FIRST: main thread pinned to 1 CPU confined every OpenMP team; fixed 2026-09-22, MoE threads now scale (185→39 ms)
- [Main branch consolidated 2026-08-30](main-branch-consolidation-2026-08-30.md) — repo now has a real `main` (GitHub default); old lineage + spikes archived under archive/*, not deleted

## Sub0Llm — LEGACY: from-scratch training research (closed 2026-08-30, see [[pivot-to-existing-model-support-qwen4-preview]])
Detail for every closed sub-thread below lives in its own file, deliberately not re-indexed line-by-line
here (LoopSplit arms A-E, the plateau/batch-confound history, the scratch/persistent-slot investigation,
model-sizing sweeps) — follow the pointers below into each file, which links onward internally.
- [Depth attention: Stages 0+1+2 DONE](depth-attention-stage0-done.md) — HANDOVER; CPU+CUDA green, but a gradient check ALONE cannot validate this op (presence test required) — lesson carried into the Qwen4-preview validation work
- [Nanbeige features: progress](nanbeige-features-progress.md) — GQA DONE+default; LoopSplit DONE but A/B-negative at this scale; GQA/LoopSplit/depth-attn/RoPE-scaling all "verified correct, never accuracy-validated" — gap left open, not being chased further
- [Nanbeige architecture reference](nanbeige-architecture-reference.md) — verified source for n-gram embeddings/mHC/LoopSplit/depth-attn; n-gram embeddings now feeds directly into the Qwen4-preview pivot
- [Nanbeige review findings](nanbeige-review-findings.md) — READ FIRST before trusting "suite green": 2 silent defects survived it
- [GQA A/B: three-pillar result](gqa-ab-three-pillar-result.md) — +8-13% tok/s, -50/-75% KV cache; quality unresolvable (seed noise)
- [LoopSplit 3-arm result + confound history](loopsplit-3arm-clean-result-looping-works.md) — READ [[loopsplit-3arm-batch-confound]] FIRST before citing any arm-C number
- [WSD cooldown, shipped + VALIDATED](wsd-cooldown-validated-tinystories-smoke.md) — still load-bearing for any future training, distillation students included
- [Truecasing collapse now UNCONDITIONAL (schemeV5)](truecasing-collapse-unconditional.md) — still load-bearing, freed 8.2% of vocab
- [Tutor/TeacherSpike adaptive-LR](adaptive-lr-corpus-ledger-spike.md) — SPIKE, archived not merged (`archive/spike-tutorspike`)
- [Scratch/persistent-slot substrate investigation](scratch-spikes-parked-2026-07-26.md) — READ FIRST if ever resumed; whole thread PARKED
- [Autosize research-grounded formula](autosize-research-grounded-formula.md) — per-corpus ledger, may be reusable for sizing small distillation students

## Sub0Llm — standing policy / preferences
- [`sub0llm` prefix for every artifact](sub0llm-prefix-for-all-artifacts.md) — libraries, DLLs, tests, targets; renamed 2026-10-08 (old notes' `sub0_tests`/`sub0_core.dll` = `sub0llm_*`)
- [Defaults are the best options](defaults-are-the-best-options.md) — recommended options become AUTO defaults (`resolve_decode_defaults`); default-off only while evaluating
- [Quality gate is not implied by the correctness gate](quality-gate-not-implied-by-correctness-gate.md) — run `cpp-review` before merging C++; load `cpp-write` before authoring. Green suites + a real perf number still missed a §1 hot-path violation and a dropped project convention (AGENTS.md §12)
- [Quality gate = paired perplexity (G-PPL)](quality-gate-is-perplexity-not-six-token-l2.md) — `--stage ppl`; the 6-token L2/argmax is a smoke check only (it wrongly flagged the native backbone)
- [No stale numbers in the tree](no-stale-numbers-in-tree.md) — superseded measurements are re-measured or removed (docs, comments, curated results); git is the archive
- [Three-pillar shootout policy](three-pillar-shootout-policy.md) — every A/B reports correctness + performance + memory
- [Measure tokenizer changes on the BLEND](measure-tokenizer-on-the-full-corpus-blend.md) — cosmopedia + minipile(code) + fineweb_edu, per-corpus AND combined. TinyStories is NOT verification
- [A caps bit nothing reads is not a guard](caps-bit-nothing-reads-is-not-a-guard.md) — TWO rounds: also put the refusal at the LOWEST callable seam, and grep for any function a comment claims exists
- [Boy-scout rule](boy-scout-rule-always-resolve-inconsistencies.md) + [resolve blocking pre-existing issues](resolve-preexisting-issues-blocking-our-work.md) — fix drift and adjacent breakage
- [Test/bench tooling = doctest + nanobench](sub0-test-bench-tooling-doctest-nanobench.md) — Sub0 family standard, latest via CPM; adopt incrementally, no forced migrations
- [Sub0 C++ style decisions](sub0-camelcase-and-doxygen-interface-docs.md) — camelCase, autobrief Doxygen (no @brief), "" own includes, #pragma once, nodiscard everywhere; constant prefix + sub0:: namespace still open; one shared style guide wanted
- [Single-responsibility headers](single-responsibility-headers.md) — one type/alternative per header, coherent dirs + umbrella headers; standing rule across Sub0 libs
- [Only add arguments we need](only-add-arguments-we-need.md) — no speculative knobs; [compile-time over runtime](compile-time-decisions-preferred.md)
- [Regression test on reproducible bug](regression-test-on-reproducible-bug.md) — pin the case, not just the fix
- [Verify correctness vs reference before perf](verify-correctness-against-reference-before-perf.md) — gate speedup claims on parity
- [Independent reimplementation catches identity-swap bugs](independent-reimplementation-catches-identity-swap-bugs.md) — fixture/gradient/parity checks share blind spots; a genuinely separate implementation doesn't
- [Memory architecture before mechanisms](memory-architecture-before-mechanisms.md) — design the cross-tier resource-placement architecture FIRST when multiple new capabilities must compose under one real budget, not as a deferred integration step
- [Optimization target is the hardware ceiling](optimization-target-is-hardware-ceiling.md) — aim past baselines
- [Hitting the target is not a reason to stop](hitting-the-target-is-not-a-reason-to-stop.md) — keep pushing while untested levers remain; refine iteratively, don't stop at the gate
- [Report decode speed in tokens/sec](report-decode-speed-in-tokens-per-sec.md) — tok/s from 2026-09-25 (native ~6.6, default ~4.6); phase costs stay in ms/token
- [Derive bounds from scale](derive-bounds-from-scale-not-fixed-thresholds.md) — caps scale with D/L/B, never fixed thresholds
- [RAII over manual cleanup](raii-cleanup-preferred-over-manual.md) + [keep SIMD primitives](keep-simd-kernel-primitives.md)
- [Python not committed](python-not-committed-cpp-library.md) — MISCATEGORIZED: this rule is from the diffusion track, not Sub0Llm; Sub0Llm's `scripts/` already commits small Python utilities, that's the real precedent
- [Fable subagent for big/risky tasks](fable-subagent-for-big-risky-tasks.md) — CORRECTED 2026-09-02: use model="opus", Fable unavailable on this account; always review its diff
- [Check captured logs first](check-captured-logs-before-process-archaeology.md) — before process archaeology
- [Hardware-efficiency iteration](hardware-efficiency-iteration-standing-task.md) + [carry forward niceties](carry-forward-chapter-niceties.md) — recurring
- [AGENTS.md established](agents-md-established.md) — repo-root pre-flight checklist
- [Deterministic-mechanisms thesis](deterministic-mechanisms-thesis.md) — STRATEGIC: mechanism-first over scale
- [Roadmap 2026-07-21](roadmap-2026-07-21.md) — supersedes the 2026-07-10 review
- [Capability roadmap: discrete experts](capability-roadmap-discrete-experts.md) — NOT classic MoE; blocked on marker design

## Sub0Llm — architecture / engine features (done)
- [Weight tying](weight-tying-tied-embeddings.md) / [QK-norm](qk-norm.md) / [attention sinks](attention-sinks.md) — all shipped, CPU+CUDA where applicable
- [Muon optimizer](muon-optimizer.md) + [its CUDA parity](muon-cuda-parity-in-progress.md) — hybrid Muon(matrices)+AdamW; 2 real hazards found
- [RoPE convention unified (half-split)](rope-convention-mismatch-forward-vs-forward-one.md) — checkpoints v2
- [EOS token + fixed marker scheme](eos-token-and-fixed-marker-scheme.md) — TOK_EOS 256, DRY doc-boundary scan
- [Central RunConfig / config.json](central-run-config-json.md) — X-macro recipe struct + simdjson read
- [Gen decode KV-cache optimization](gen-decode-kv-cache-optimization.md) — forward_one on both backends
- [Scoring uses batched forward()](scoring-use-batched-forward-not-forward-one.md) — not forward_one
- [GPU eval seam: DONE](gpu-eval-seam-needed-report-is-cpu-bound.md) — sub0_dev_forward_loss, parity-gated
- [Mock device backend = testable seam](mock-device-backend-test-seam.md) — PATTERN: reuse for the next device consumer
- [sub0_dev_* consumer migration](sub0-dev-seam-consumer-migration.md) — off the sub0_cuda_* externs
- [Model identity date-tag + table](model-identity-date-tag-and-comparison-table.md) — `models --metrics`
- [Checkpoint tools + best-retention](checkpoint-tools-and-best-retention.md) — `bundle` / `ckpt2model`
- [Architecture solidification (DRY)](architecture-solidification-plan.md) + [staged workflow](staged-workflow-config-architecture.md) — configure→tune→train→gen
- [Frontend layer testability](frontend-layer-testability-and-tooling.md) — engine-free pre-model layer
- [Arch features OFF by default: RESOLVED](arch-features-off-by-default-in-production.md) — defaults ON now
- [GGUF import feasibility](gguf-import-feasibility-review.md) — weight transplant + CUDA kernel open

## Sub0Llm — CUDA / performance
- [Attention kernel = throughput bottleneck](attention-kernel-throughput-bottleneck.md) — L1/TEX-bound; fix = FlashAttention tiling
- [CUDA kernel optimization waves](cuda-kernel-optimization-waves.md) — 5 waves, ~21% gain + int32-overflow bug
- [CUDA wave 6: orchestration + memory](cuda-wave6-orchestration-memory-findings.md) — chunked lm_head+CE: 42-43% VRAM cut
- [LM head tensor-core fix + graceful-stop](lm-head-tensor-core-fix-and-graceful-stop.md) — ~12-15% win
- [dq/dk register-spill fix](dq-dk-register-spill-fix.md) — 0 spill at production HD=64
- [CUDA binding-compose caps flip](cuda-binding-compose-caps-flip-merged.md) — device composes embed rows; ~40-57% throughput
- [Hybrid CPU/GPU execution](hybrid-cpu-gpu-execution-design.md) — source-routed split, ~2.3-2.6x over CPU
- [GPU failure-detection hardening](gpu-failure-detection-hardening.md) — [[nodiscard]] pass caught 9 more bugs
- [GPU batch undertuned / VRAM headroom](gpu-batch-undertuned-vram-headroom.md) — gpu_batch_estimate() DONE
- [Denormal stalls need FTZ](denormal-stalls-in-training-need-ftz.md) — Adam state; FTZ+DAZ fixed a 4.4x slowdown
- [Single-thread optimization + bench](single-thread-optimization-and-cycle-bench.md) — op_linear SIMD 2.05x
- [ncu depth vs nsys busy-time](ncu-depth-vs-nsys-busy-time.md) — "busy" ≠ efficient; [ncu VBS blocker](ncu-vbs-memory-backup-blocker.md) on big models
- [Thermal confounds A/B wall-clock](thermal-confounds-ab-wallclock-testing.md) — interleave trials per variant; run_perf_suite.py now builds once, rotates, cools (never rebuild before a sample)
- [Background-task child-process leak](background-task-child-process-leak.md) — assert process count before ANY throughput number
- [CPU profiling tooling](cpu-profiling-tooling-backlog.md) — VTune unelevated; 34.7% eff util, DRAM-bound; standalone-benchmark-+ `-target-pid` pattern found a real MoE defect code-reading missed
- [Host compilers available](host-compilers-available.md) — MSVC 19.51, clang/clang-cl 22.1.8, native GCC 16.2 + 15.2 (WinLibs, full path needed), GCC 15.2 in WSL; GCC needs `-ffp-contract=off` for bit-identical results
- [Host CPU: Arrow Lake-HX](host-cpu-arrow-lake-hx.md) — Core Ultra 9 275HX, 8P+16E, no SMT, 36MB L3; READ FIRST: DRAM 30 GB/s is 1-core, real 79 (P) / 91 GB/s (all)

## Sub0Llm — open bugs / unexplained
- [Stale test DLL made the engine gate vacuous](stale-test-dll-vacuous-engine-gate.md) — READ FIRST: sub0_tests loaded a 09-23 sub0_core.dll until 2026-10-08; fixed, re-run clean
- [GPU large-batch access violation](gpu-large-batch-access-violation.md) — crashes at batch 768/960 (d256); root cause OPEN
- [GPU tune vs production discrepancy](gpu-tune-benchmark-vs-production-discrepancy.md) — 63K vs 141K tok/s; OPEN
- [memplan overhead is 280 not 1650; HOST is the bigger half](memplan-overhead-280-not-1650-and-host-is-bigger.md) — READ FIRST: 1650 RETRACTED, headroom 512 was fine; host footprint was under-reported 27x
- [memplan VRAM-prediction gap at scale](memplan-vram-prediction-gap-at-scale.md) — over-estimates as batch grows at d768
- [win/s 'bug' RETRACTED](winrate-metric-suspect-2.5x.md) — STILL OPEN: mean window 215 tok vs 820 median doc
- [GPU illegal-access = hardware fault](gpu-illegal-access-hardware-fault-not-code-bug.md) — real nvlddmkm driver fault
- [CUDA padded-window gradient mismatch](cuda-padded-window-gradient-mismatch.md) — CLOSED; the accumulate-vs-overwrite precedent
- [Cross-test optimizer-state contamination](cross-test-optimizer-state-contamination.md) — reset_opt_state() per test
- [engine_tests gradient-check "dims-dependence"](engine-tests-gradient-check-dims-dependent.md) — OBSOLETE: only a TOY-VOCAB config misbehaves
- [gen "freeze" at --n >= SEQ_LEN](gen-freeze-at-seq-len-boundary.md) — FIXED: report 25min+ → 5s
- [Positional-arg trap: SECOND data loss](positional-arg-trap-second-data-loss.md) — corpus overwritten by a model.bin
- [Large-model Windows gotchas](large-model-windows-build-run-gotchas.md) — d768 heap-alloc; train hardlinks corpus.tok
- [MSVC toolchain quirks 2026-07](msvc-toolchain-quirks-2026-07.md) — omp-for needs signed loop var; VS path is [18, not 2022](vs-install-path-18-not-2022.md)

## Sub0Llm — tokenizer / corpus
- [Tokenizer schemeV4 merge: DONE](tokenizer-schemev4-merge-backlog.md) — LANDED 2026-07-30 (7792cbe+f0ae908), clean merge; marker set UNCHANGED so the EOS fix survived. Truecasing threshold still open
- [Tokenizer redesign COMPLETE](tokenizer-redesign-implementation-complete.md) — Stage 1+2+WS6+WS7 shipped
- [schemeV3: SPELL-unify](tokenizer-schemev3-spell-unify.md) — fixed 93.9% false positives
- [JOIN tokenizer strategy + result](join-tokenizer-strategy.md) — A/B: −27.6% tokens
- [Tokenizer dogfood+fuzz](tokenizer-dogfood-fuzz-and-source-pathologies.md) — tok_roundtrip_fuzz GREEN
- [Special-token extension review](tokenizer-special-token-extension-review.md) — design only, not implemented
- [Normalization + factored tokens](tokenizer-normalization-and-factored-tokens.md) — BACKLOG
- [Spelling via injected char↔word training](spelling-via-injected-char-word-training.md) — SPIKE
- [Corpus blending + spell integration](corpus-blending-and-spell-integration.md) — blend.hpp + --spell-mix
- [Configurator ingest optimization](configurator-ingest-optimization.md) — incremental BPE + scan cache (5x)
- [Out-of-core corpus pipeline](out-of-core-corpus-pipeline.md) — training mmaps corpus.tok
- [corpus.tok reuse stamp](corpus-tok-reuse-stamp.md) — configure skips re-tokenize when unchanged

## Sub0Llm — toolchain / JSON / C++26
- [C++26 reflection status](cpp26-reflection-migration-status.md) — builds at -std=c++26; reflection blocked on toolchain
- [JSON write-side simdjson](json-write-simdjson-and-cpp26-reflection.md) — BACKLOG: write side hand-rolled
- [JSON reads = forward on-demand](json-reads-forward-ondemand-not-dom-facade.md) — handlers + single forward pass
- [simdjson migration backlog](simdjson-migration-backlog.md) — drop nlohmann once unused
- [README architecture-links](readme-architecture-links-reference.md) — verify live before reuse

## Diffusion/chapter-education track (separate, dormant — Ch27-32, DiffusionGemma)
- [Diffusion track index](ch27-diffusion-track-plan.md) — Ch27 build/optimization, GPU forward arc, CUDA setup (sm_120)
- [Diffusion curriculum + quality index](ch28-curriculum-findings.md) — Ch28-31 plans + results, NELBO reframes
- [Diffusion architecture index](gist-is-compute-context-primitive.md) — gist/MERA, verbosity slider, denoiser toolbox
- [Optimizer alternatives backlog](optimizer-alternatives-backlog.md) — plain Adam only there; try AdamW/Muon
- [CPU threading model](cpu-threading-model.md) — cpu_topology, P-cores-first. Sub0Llm has no such module
- [Heterogeneous worker-split](heterogeneous-worker-split-followup.md) — P-cores get more windows

## Other / cross-cutting
- [Episodic multi-angle + specificity](episodic-multi-angle-and-specificity-plan.md) — next is low-rank frozen-base partition
- [Conditioning embedding amplitude](conditioning-embedding-amplitude.md) — must match token-embed scale (1/√D)
