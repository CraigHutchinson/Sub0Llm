# Handover — resuming the Qwen4-Next work in a new session or on a new machine

Performance-workflow update (2026-11-07, #48): start with [PERFORMANCE.md](PERFORMANCE.md).
It codifies the pinned Sub0Math `sub0perf` rig and the `perf`/`perf-profile` presets.

Written 2026-11-05. Nothing is in flight: the tree was clean, every branch worth keeping is merged or archived,
no benchmark or agent was running, and no file is held. Last engineering commit before this document:
`47a4cc4` (2026-10-08). Where a number below carries a date and a host, it is a measurement, not a promise:
re-measure before you cite it.

Read this first, then [WORKING_PRACTICES.md](WORKING_PRACTICES.md) (how this project is worked on, including
the standing directions that used to live only in an agent's local memory), then [AGENTS.md](../AGENTS.md)
(the pre-flight checklist for any code change).

## 1. What the project is now

Sub0Llm is a C++23 inference and training engine with compile-time model dimensions. Its from-scratch
training line was closed on 2026-08-30. The live work is **running the real Qwen3.8-Flash-Next model (a preview
of the Qwen4 architecture) on a CPU-first, RAM-constrained laptop, as fast as the hardware allows**, with the
engine as the validation oracle for its mechanisms (Gated DeltaNet, Gated Residual, 512-expert MoE, QSA).

The standing goal given by the project owner: **actual token generation should be near the CPU's theoretical
performance.** It is not declared met. Keep pushing while untested levers remain; see section 5.

## 2. Reading order

| Doc | What it holds |
|---|---|
| [ROADMAP.md](ROADMAP.md) | Direction and the history of the pivot |
| [QWEN4_PREVIEW_REFERENCE.md](QWEN4_PREVIEW_REFERENCE.md) | Verified facts about the model |
| [optimization/opportunities/README.md](optimization/opportunities/README.md) | Every decode optimization (O1-O14): status, evidence, and the current default flags |
| [optimization/opportunities/O13_storage_convergence.md](optimization/opportunities/O13_storage_convergence.md) | The ranked next steps, joining the storage and decode tracks |
| [OPTIMIZATION_PROCESS.md](OPTIMIZATION_PROCESS.md) | The measurement protocol and gates; [optimization/kpi_gates.json](optimization/kpi_gates.json) holds them in machine-readable form |
| [STORAGE_STACK_PLAN.md](STORAGE_STACK_PLAN.md) | The owned expert cache (Sub0MemPage and Sub0TieredCache) and its follow-ups |
| [BACKBONE_NATIVE_QUANT.md](BACKBONE_NATIVE_QUANT.md) | The native-quant backbone: sidecar format, kernels, quality verdict |
| [ACTIVE_WORK_LOG.md](ACTIVE_WORK_LOG.md) | Cross-agent coordination; the protocol is at the top |
| [agent-memory/](agent-memory/README.md) | The previous agent's working memory, restorable (section 4) |

## 3. State at the pause

**Everything below is merged to `main` and pushed.** Newest first.

| Work | What it is | Where to read |
|---|---|---|
| Rename | Every library, DLL, test binary and CMake target carries the `sub0llm` prefix (`sub0llm_core.dll`, `sub0llm_tests`, ...) | O-index row P2 |
| O14 | The 9 GiB bf16 parameter arena is a read-only view of the model file (`--param-arena`), auto-on for inference builds; bit-exact; 6-9 GiB less peak memory | [O14](optimization/opportunities/O14_mapped_param_arena.md) |
| O13 | Analysis joining the cache and decode tracks, with the expert-trace results | [O13](optimization/opportunities/O13_storage_convergence.md) |
| O12 | Q8_0 vector-accumulator kernel (`--backbone-q8-fast`), auto-on | [O12](optimization/opportunities/O12_q8_fast.md) |
| O11 | QSA indexer projection threaded; `QSA: indexer` 9.45 to about 1 ms/token | in [O10](optimization/opportunities/O10_subphase_roofline.md) |
| O10 | 18 profiler sub-phases; the roofline that ranked O11 and O12 | [O10](optimization/opportunities/O10_subphase_roofline.md) |
| O9 | Per-256 `ActSuper` kernels for Q4_K/Q5_K/Q6_K (`--backbone-act-super`), auto-on | [O9](optimization/opportunities/O9_act_super.md) |
| Defaults | The best measured options are the configurator's AUTO defaults (`resolve_decode_defaults`) | [WORKING_PRACTICES.md](WORKING_PRACTICES.md) section 1 |
| Quality gate | G-PPL on `ppl_blend_v2` (9,631 tokens); v1 (2,418) was too small to resolve this work | [OPTIMIZATION_PROCESS.md](OPTIMIZATION_PROCESS.md) section 2a |
| Perf suite | `--ppl-speed-rounds` (rotated long-run tok/s), `--ballast-room` (measure under memory pressure) | `scripts/run_perf_suite.py --help` |
| Test-gate fix | `tests/sub0llm_core.dll` is refreshed on every build; before 2026-10-08 the engine suite ran a 2026-09-23 engine | WORKING_PRACTICES section 6 |

The `main` build resolves, for a quantized-MoE inference build on CPU compute: fused expert dots, row-split
experts, 8 GEMV and 8 expert threads, OpenMP spin, native-quant backbone, per-256 activations, the fast Q8_0
kernel, and the mapped parameter arena. `--moe-io-mode` still defaults to `reactive`; the owned cache is
opt-in (`--moe-io-mode cache`), see section 5.

### Reference results

Measured 2026-10-07/08 on the dev machine (Core Ultra 9 275HX, 63 GiB RAM). Perplexity is deterministic
for a build, so it is the anchor for a new machine; speed is not.

| Check | Value |
|---|---|
| Default real-model build, `--stage ppl --arm "default:"` | ppl **11.3589** over 9,631 tokens (`ppl_blend_v2`) |
| Same build, first 2,000 tokens (`--ppl-tokens 2000`) | ppl **11.9443**; identical across heap/mapped arena, and across reactive/cache |
| Decode, long runs (2,000 to 9,631 tokens), memory-mapped experts | about 7-8.3 tok/s; run-to-run spread 5% or more |
| Decode, owned cache 17 GiB under `--ballast-room 10` | about 7.7-8.5 tok/s, about 11% over reactive in the same run |
| Decode, six tokens, warm | about 0.106-0.126 s/token |
| Phase costs, ms/token (warm, six tokens, early October) | GDN 33-36, routed experts 28-30, Gated Residual 18-20, QSA about 11, router+shared 8, lm_head 7.5 |

Neutral-build gates (config `--dmodel 196 --layers 11 --heads 7 --kv-heads 7 --seq 256`, below):
`sub0llm_tests` **29,510,661 assertions / 147 cases**, fingerprints forward `5a7382ea70d3913b`, grad
`7f44bdae18c313dd`, decode `d1625d19ed2258f1`; `sub0llm_frontend_tests` **231,180 / 304**.

A different CPU generation may change the last digits of the perplexity anchors (kernel paths are chosen at
compile time with `-march=native`). If yours differs, re-baseline once on the new machine and record it here.

## 4. Resuming on a new machine

Steps 1-6 are the checked path. The clean-directory configure and configurator build in step 5 were run from
nothing on 2026-11-05.

1. **Toolchain.** Windows 11 is the supported dev flow (the ballast, page-cache eviction and unbuffered-read
   tooling are Windows mechanisms). Clang 22 (`C:\Program Files\LLVM\bin`), CMake 4 or newer, Ninja, Python 3,
   Git. The CPU needs AVX2 and FMA; the dev CPU also has AVX-VNNI and VNNI-INT8, and `-march=native` uses them.
   Compiler and shell details are in [agent-memory/host-compilers-available.md](agent-memory/host-compilers-available.md).
   Git LFS is also needed: two small tracked data files (`data/english.txt`, `data/tokenizer_calibration.txt`)
   live in LFS, so run `git lfs install` before cloning.
2. **Clone** `https://github.com/CraigHutchinson/Sub0Llm`. Sub0MemPage and Sub0TieredCache are fetched by CMake at
   the commits pinned in `cmake/MemPage.cmake` and `cmake/TieredCache.cmake`; no sibling checkout is required.
3. **Restore the agent's memory** so a new session starts with the same standing preferences and warnings:
   `python scripts/agent_memory.py restore` (it never overwrites existing notes; `where` prints the target).
4. **Model files.** The real model is four files in one directory on the dev machine:
   `D:\ModelWeights\Sub0Llm-Qwen4-full48-bf16\qwen4_full48_q_bf16.bin` (9.2 GiB), `.bbq` (3.3 GiB), `.bbq.pair`
   and `.moeq` (37.1 GiB). They are produced from the unsloth `UD-IQ1_S` GGUF of Qwen3.8-Flash-Next
   (`D:\ModelWeights\Qwen3.8-Flash-Next-GGUF\UD-IQ1_S` on the dev machine) by the `sub0llm-transplant-q48`
   target (about 3 minutes, about 50 GB of disk):

       sub0llm-transplant-q48 --gguf <GGUF dir> --param-dtype 1 --out <dir>\qwen4_full48_q_bf16.bin ^
                              --backbone-quant <dir>\qwen4_full48_q_bf16.bin.bbq

   The `.bin` it writes should hash to SHA-256 `a2ceb97699d9072d3a6a1869644d0be5d0dc7243a09785e4400d7a6443af06a2`
   (byte-identical across independent transplants). Background and verification: [WP4_SCOPE.md](WP4_SCOPE.md)
   (WP5b) and [BACKBONE_NATIVE_QUANT.md](BACKBONE_NATIVE_QUANT.md) section 13c. Point the tools at the result
   with environment variables:
   `SUB0_QWEN4_MODEL` (the perf suite), `SUB0_QWEN4_MOEQ_PATH` and `SUB0_QWEN4_GGUF_DIR` (tests that use the
   real files; they skip when unset), `SUB0_QWEN_TOKENIZER_DIR` (the three Qwen tokenizer files).
5. **Build the real-model tools.** The `-DSUB0_COMPUTE=CPU` flag matters: the default AUTO finds `nvcc` and
   then fails because `cl.exe` is not on the PATH.

       cmake -S . -B out/build/wp5c_full48 -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release ^
             -DSUB0_COMPUTE=CPU -DSUB0_NATIVE=ON -DSUB0_BUILD_TESTS=OFF -DSUB0_BUILD_BENCHMARKS=OFF

   `scripts/run_perf_suite.py` then configures and builds each arm itself (the real-axes recipe is its
   `REAL_AXES` constant; a partial recipe yields a rejected model with no hint which axis is wrong).
6. **Training corpora for the quality fixture.** `data/*.txt` is not committed. Fetch with `scripts/get_*.py`
   (fineweb_edu, cosmopedia, minipile, gsm8k), then `python scripts/make_ppl_fixture.py`: it rebuilds
   `out/quality/ppl_blend_v2.txt` and fails loudly unless the SHA-256 is
   `54028b9fc85f78af2b85a8f7558b09d575bf0b5a254eb9bd0a570b1c356df422`.
7. **Baseline.** `python scripts/run_perf_suite.py --stage ppl --label baseline --arm "default:"` should
   print ppl 11.3589 over 9,631 tokens (section 3). It takes 20-30 minutes.
8. **Neutral gate build** (needs the corpus `data/gsm8k.txt`). In a second build directory with tests on,
   build `sub0llm-configure`, run it with `--corpus data/gsm8k.txt --dmodel 196 --layers 11 --heads 7 --kv-heads 7
   --seq 256`, build everything, and run `out/build/<dir>/tests/sub0llm_tests.exe` and `sub0llm_frontend_tests.exe`.
   Confirm the BUILD succeeded before reading counts. This step is the established practice but was not
   re-run from a clean directory in this session.

## 5. Open work, in the order I would take it

The project owner's directions behind this list are in [WORKING_PRACTICES.md](WORKING_PRACTICES.md) section 1.
The costed comparison is in [O13](optimization/opportunities/O13_storage_convergence.md) (sections 3 and 4).

1. **A memory planner, design first, with a measurement.** One module that splits RAM between context
   memory, the expert cache budget and headroom, at load and at chat start. Context memory (K/V caches, QSA
   keys, per-position scratch) is sized by the configured window; the owner wants it sized to prompt + maximum
   response + thinking limit, so a 32k-configured build does not hold 32k of memory. First step: measure what a
   real-axes build holds at a 32k window (every measurement so far used a 128-token window). The same planner
   supplies the cache budget that promoting cache mode needs.
2. **Make `--moe-io-mode cache` the default.** The owned cache is the reference path on this RAM-constrained
   host (about 11% over reactive under pressure, steadier run to run). Promotion needs a budget from free
   memory at load, a comparison with RAM to spare, the full suite run with the cache on, and a Linux fallback
   for the 64 MiB memlock limit. Checklist: [STORAGE_STACK_PLAN.md](STORAGE_STACK_PLAN.md), "Follow-ups".
3. **The n-gram table (S1c).** The owner named this as the next step after the optimization phase. It is the
   storage stack's core use case (102 GB of 320-byte rows, never fits in RAM) and the quality term both
   perplexity arms omit today. The current slot design spends 8 KiB per 320-byte row, so it needs a
   block-sharing design first. Design inputs: [NGRAM_EMBEDDING.md](NGRAM_EMBEDDING.md),
   [NGRAM_TABLE_TIERED_STORAGE.md](NGRAM_TABLE_TIERED_STORAGE.md), [SPECULATION_NGRAM_MOE_DESIGN.md](SPECULATION_NGRAM_MOE_DESIGN.md).
   My recommendation was planner first, because the n-gram row cache draws on the same budget; the owner has
   not yet confirmed that order.
4. **Two cheap memory tests.** Pin the 3.3 GiB `.bbq` mapping (it is read almost in full every token, yet its
   residency dipped to 90-94% in three of the runs), and try 2 MiB pages for it and for the expert pool.
5. **Small kernel work.** Thread the single-threaded small steps (hc_norm, the Gated Residual elementwise
   parts, the GDN b/a and conv steps), about 4-5 ms/token; explain why the K-quant GEMVs plateau near 53 GB/s
   against a 79 GB/s roof. Candidates are ranked in [O10](optimization/opportunities/O10_subphase_roofline.md).
6. **Parked on evidence, not abandoned.** A prefetch predictor (recency predicts 0.0% of cache misses, and a
   26 GiB cache ran no faster than 17 GiB, so miss waits are not what limits decode at the pressure measured);
   O7 expert kernels (slower in real decode, default off); the VNNI variant of the `ActSuper` kernels. Re-open
   any of them only with a new measurement.
7. **Earlier backlog, not started:** the design-doc measurements E1-E3 (route log, prompt-lookup acceptance,
   memory budget) and then speculative decoding / MTP (O6); P1, a lean default workflow for the optimization
   tooling (see the O-index).

A separate track exists on the `research/intel-groundwork` branch (Intel iGPU/USM groundwork by another agent,
docs `INTEL_IGPU_*`). It is paused and independent of the CPU decode work.

## 6. Traps that cost real time

Each is explained in [WORKING_PRACTICES.md](WORKING_PRACTICES.md); this is the short list.

- **Verify the build succeeded before reading test counts.** A failed build leaves old binaries runnable.
- **Check which binary a test loaded.** Windows takes `sub0llm_core.dll` from the test's own directory first.
- **An agent's reported speedup is a claim, not a measurement.** Re-run it (O9: reported +28-33%, measured +14-18%).
- **Check the whole machine for other benchmarks**, not just this repo's process names. Other projects' sessions
  run timing jobs here.
- **Never `grep -r` or `find` from the repo root.** `data/` holds hundreds of GB. Use `git grep`.
- **`-DSUB0_COMPUTE=CPU`** on every configure of this project's CPU builds.
- **Build directories made before 2026-10-08** hold `sub0_*` binaries under the old names; delete them.
- **Start worktrees and agents from the drive-letter case the repo was opened with** (`D:` against `d:`).

## 7. Leaving the tree cleanly

Before pausing: commit; merge to `main`; push; release the host (no ballast, benchmark or compile running);
mark any work-log row `done` or `paused`; refresh the agent-memory snapshot
(`python scripts/agent_memory.py export`, review the diff, commit). The next session should be able to start
from `git pull` and this document.
