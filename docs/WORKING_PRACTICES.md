# Working practices — how this project is worked on

**2026-11-07 workflow adoption:** [PERFORMANCE.md](PERFORMANCE.md) is the entry point for the pinned
Sub0Math `sub0perf` rig, quick/full iteration commands, baseline and evidence contracts. The detailed
model-specific recipes below remain applicable; historical host figures are not portable measurements.

Standing directions from the project owner, the measurement and delegation habits that came out of real
mistakes, and the environment traps of the dev machine. Until 2026-11-05 most of this lived only in an AI
agent's local memory, which does not travel with the repository. It is written here so any session on any
machine inherits it. The agent's full notes are in [agent-memory/](agent-memory/README.md); file names in
brackets below point at the note with the history and the reasoning.

[AGENTS.md](../AGENTS.md) is the pre-flight checklist for code changes. [HANDOVER.md](HANDOVER.md) is the
current state and the open work. This document is the "how".

## 1. Standing directions from the project owner

Dated, because the date says how current the direction is. The owner's words are quoted where they carried
the decision.

**Goal and ambition**
- The `/goal`: *actual token generation is near the CPU's theoretical performance.* Aim at the hardware
  ceiling, not at the last baseline [optimization-target-is-hardware-ceiling]. On the dev machine that
  ceiling is about 79 GB/s of DRAM from the P-cores and about 91 GB/s from all cores; the 30 GB/s figure that
  circulated for weeks was a single core [host-cpu-arrow-lake-hx].
- Hitting a target is not a reason to stop (2026-09-22). Report the gate, say what is still on the table,
  and keep working, taking the next lever from a fresh profile [hitting-the-target-is-not-a-reason-to-stop].
- Report decode speed in **tokens per second**, phase costs in ms/token (2026-09-25)
  [report-decode-speed-in-tokens-per-sec].

**Defaults and surface**
- **The defaults are the best options** (2026-09-29): "we should make the defaults the best options." A
  capability is off while it is being evaluated. Once it has passed its gates and been recommended, it
  becomes an AUTO default, resolved in one place (`resolve_decode_defaults` in `tools/configurator.cpp`): on
  for the build class it was measured on, the old value everywhere else, so other builds' generated headers
  stay byte-identical. An explicit flag always wins. Nobody should need to remember a list of recommended
  flags [defaults-are-the-best-options].
- Only add arguments and options that something consumes; decide at compile time what is known at compile
  time [only-add-arguments-we-need] [compile-time-decisions-preferred].
- **Every artifact carries the `sub0llm` prefix** (2026-10-08): libraries, DLLs, test binaries, CMake targets.
  A bare `sub0_core.dll` collides with any other Sub0 library that made the same choice
  [sub0llm-prefix-for-all-artifacts].

**Memory and the host**
- This host is RAM-constrained (63 GiB, with about 20 GiB already committed before the model loads, against
  49.5 GiB of model files). **Self-managed memory beats `mmap` over the long term** (2026-10-07): the owned,
  pinned expert cache is the reference path, not a special case for when RAM runs out. Measure every decode
  optimization under memory pressure as well as on an idle host [self-managed-memory-is-the-direction].
- Context memory should scale to the conversation (2026-10-08): size the K/V cache and per-position state to
  the prompt plus the maximum response plus the thinking limit, so a 32k-configured build does not hold 32k of
  memory and more model weights stay resident. It needs a planner module (or a config-fixed split) that also
  owns the expert-cache budget [context-memory-planner-backlog].
- The n-gram table is the storage stack's core use case, not an afterthought (2026-09-29); the owner named it
  as the next step once the current optimization phase resolves (2026-10-08), other open items to be weighed
  first [storage-core-use-case-ngram-regime2].

**Quality and honesty**
- The quality gate is **paired perplexity (G-PPL)** on the pinned `ppl_blend_v2` text; the six-token L2/argmax
  check is a smoke test only, and it wrongly flagged the native backbone once
  [quality-gate-is-perplexity-not-six-token-l2].
- Every A/B reports correctness, performance and memory, not whichever one improved [three-pillar-shootout-policy].
- **No stale numbers in the tree** (2026-10-06): a superseded measurement is re-measured or removed from docs,
  comments and curated results; git is the archive [no-stale-numbers-in-tree]. Measurements that stay must
  carry a date and a host. The exceptions are logs that are history by design (`ACTIVE_WORK_LOG.md`,
  `perf_history.jsonl`) and the dated snapshot in `agent-memory/`.
- A green suite and a good number do not substitute for a code-review pass: run the `cpp-review` standard over
  a diff before merging, and load `cpp-write` before authoring [quality-gate-not-implied-by-correctness-gate].
- Park, never revert: a negative result is kept behind its own `constexpr` toggle, after three honest passes
  (AGENTS.md section 13).
- Verify against a reference before optimizing, and keep an independent reimplementation as the check
  [verify-correctness-against-reference-before-perf] [independent-reimplementation-catches-identity-swap-bugs].
- Resolve inconsistencies you find, and blocking pre-existing issues, rather than routing around them
  [boy-scout-rule-always-resolve-inconsistencies] [resolve-preexisting-issues-blocking-our-work].

**Code style**
- The Sub0 family style (camelCase public API, Doxygen on interfaces, `#pragma once`, `""` for own includes,
  `[[nodiscard]]` audited, doctest and nanobench for new tests, single-responsibility headers) is decided for
  the sibling libraries [sub0-camelcase-and-doxygen-interface-docs] [sub0-test-bench-tooling-doctest-nanobench]
  [single-responsibility-headers]. **This engine's code is still snake_case on Catch2: match the surrounding
  code and do not start a migration nobody asked for.** The sibling style guides are in the Sub0Pipeline and
  Sub0ECS repositories.
- Small Python utilities are committed in `scripts/` (this repo's precedent), though a C++ library of the
  family should not depend on Python [python-not-committed-cpp-library].

## 2. Measuring

The full protocol is [OPTIMIZATION_PROCESS.md](OPTIMIZATION_PROCESS.md). The habits that matter, each learned
from a bad number:

- **Use the harness.** `scripts/run_perf_suite.py` builds every arm once, rotates the order, waits for a quiet
  machine and cools between samples. Hand-rolled measurement is where mistakes came from. Never rebuild
  between timed samples: a compile heats the CPU and the heat follows the arm schedule.
- **Long runs for speed, short runs for phase costs.** A three-token A/B had a 24-43% spread over an 11%
  effect and gave the wrong sign. Use `--stage ppl --ppl-speed-rounds N` (rotated 2,000-token runs) for tok/s
  and `--profile-phases 1` with `sub0llm-qwen4-forward --tokens 6` for the phase table.
- **Compare within one run.** Absolute tok/s moves with how much RAM the OS happens to spare that day (the
  same reactive arm read 4.9 to 7.1 tok/s across sessions). Perplexity is the parity check: it must be
  identical across arms that change where bytes come from, never what is computed.
- **Interleave and discard the cold run.** Heat and the OS file cache both favour whichever arm runs second.
  Use `--ballast-room GIB` to measure under memory pressure: it locks RAM so only that much is left for expert
  bytes, and evicts the model files before every run.
- **Check the whole machine for other benchmarks.** The suite's contention check matches process names
  (`sub0llm|clang|ninja|cmake`); a sibling project's compile-time benchmark on the same host is invisible
  to a check that only looks at this repo's names. Look at the process list for foreign `python` and `clang` jobs.
  Assert the process count before any number, and after stopping a script check that its children died
  [background-task-child-process-leak].
- **Thermal confounds are real** on this laptop-class CPU: at least two trials of each variant, interleaved
  [thermal-confounds-ab-wallclock-testing].
- **Read the captured log before process archaeology.** Long-running tools print their own progress; re-read
  the output file first [check-captured-logs-before-process-archaeology].
- **Rounding order alone moves perplexity** on a 2,418-token text by as much as the effects under test (a control
  that changed only summation order went from 14.70 to 14.90). The gate therefore uses the 9,631-token v2 text,
  and a verdict needs the paired confidence interval, not two bare perplexities.
- **An agent's measurement is a claim.** The O9 agent reported +28-33%; the primary re-measurement gave +14-18%.
  Quality, being deterministic, reproduces exactly; speed does not.

## 3. Delegating to agents

The owner's standing preference is to delegate research and implementation to Sonnet-class agents (or
`gpt-5.6-terra`) and to keep the review, integration and verification of consequential claims with the primary
agent [fable-subagent-for-big-risky-tasks, which also records that `opus` is the heavier model to use here].

- **Brief like a contract.** Name the files to read, the gates with exact numbers, the scope (what is theirs
  and what must not be touched), the deliverable, and that they report plainly whether the quality gate passed.
- **Isolated worktree, early WIP commits.** Agents hit session rate limits mid-run; committed work survives and
  uncommitted work does not. Tell them to commit in phase-sized steps, and resume a stopped agent by messaging
  it rather than starting over.
- **Tell them about the machine.** Give an agent the contention check to run before any build or timing, and
  tell it to add a work-log row before it holds the CPU or a ballast.
- **Review the diff yourself and re-run the gates.** Check the build succeeded, read the neutral-build counts
  and fingerprints, re-measure the headline number, and search for consumers in `tests/` and `tools/` as well as
  `include/` and `src/` (a missed test call site once broke the first build).
- **Do not let a delegate decide a default.** Agents leave a new option default-off and report; the primary agent
  flips the default after its own measurement.
- **Start agents from the drive-letter case the repository was opened with.** `D:` against `d:` broke
  worktree creation once.
- Research delegation does not authorize installs, edits to shared files, or contended hardware measurement.
- **Subagent brief checklist (token cost).** Every spawn pays about 75k tokens of fixed overhead (tools, MCP,
  system prompt, memory) before it does any work, so: batch independent questions into ONE spawn; keep a
  single lookup or command in the primary agent; give the agent a bounded question, a `file:line` citation
  requirement, "say unknown rather than guess", and a word cap on the report. A small model's claims about
  repo facts are leads, not findings: re-check each one you act on (a pointer-versus-real-file claim about
  `data/english.txt` was right but only because it was re-verified). Do not put a history rewrite or push
  in a delegate's brief; the permission layer blocks it, so do those yourself.

## 4. Changing code

[AGENTS.md](../AGENTS.md) is authoritative. The gates, in the order they are usually met:

1. Zero heap allocation in anything that runs per token (section 1 of AGENTS.md).
2. Decisions known up front are `constexpr` through the configurator (section 2).
3. New capability off by default and the neutral build byte-identical (section 4). The neutral config is
   `--dmodel 196 --layers 11 --heads 7 --kv-heads 7 --seq 256`; the generated header may differ only by the
   new constant. Gates: `sub0llm_tests` 29,510,661 / 147 with the three fingerprints,
   `sub0llm_frontend_tests` 231,180 / 304 (the current numbers are in [HANDOVER.md](HANDOVER.md)).
4. A numerical check independent of "it compiles and the loss drops" (section 6), at more than one scale (section 7).
5. For a shared surface, list every consumer repo-wide, tests and tools included (section 10).
6. G-PPL on the real model for anything that changes decode arithmetic, and a bit-exactness check (identical
   perplexity, paired difference exactly zero) for anything that should not.
7. `cpp-review` over the diff (section 12); an optimization measured negative gets three passes (section 13).

## 5. Git and files

- Commit often, in small steps; push only when asked, except at a handover where the next session needs it.
  Use the commit trailer the harness specifies for the current model.
- Stage specific paths. Several agents have shared one tree; `git add -A` can sweep in someone else's
  in-flight edits. Check `git status --short` before and after a commit.
- One owner per shared file at a time; add a row to [ACTIVE_WORK_LOG.md](ACTIVE_WORK_LOG.md) before holding a
  file or the CPU for more than a few minutes, and close it when done.
- Never commit model files, corpora, build output, IDE workspace files (`*.code-workspace` is ignored: it
  lists private repositories) or anything private. This repository is public.
- Files in `docs/agent-memory/` are vetted by `scripts/agent_memory.py export`, which refuses an e-mail address,
  a credential-shaped string or a private repository's name.
- Delete merged branches and finished worktrees promptly; one agent worktree held a 49 GiB copy of the model.
  Keep a parked experiment as an `archive/*` branch, never delete it.

## 6. Environment traps

- **A failed build leaves the old binaries runnable.** Confirm the build succeeded before reading test counts.
  Stale generated configs (missing a new constant) fail the build while old binaries still run and read fine.
- **The test DLL.** Windows loads `sub0llm_core.dll` from the test's own directory first. Until 2026-10-08
  nothing refreshed that copy, so from 2026-09-23 the engine suite ran a stale engine. It is now a dependency of
  every test that links the engine; a silent exit code 127 from a test binary is a DLL load failure
  [stale-test-dll-vacuous-engine-gate].
- **Build directories keep old names.** Anything built before 2026-10-08 has `sub0_*` binaries. Delete them.
- **`-DSUB0_COMPUTE=CPU`** on every CPU configure; AUTO finds `nvcc` and fails without `cl.exe` on the PATH.
- **Never recurse from the repo root.** `grep -r` and `find .` walk `data/` (hundreds of GB). Use `git grep` or
  scope the path. A stray recursive search disturbed a timing run once.
- **Non-cached reads queue behind a cached reader of the same file** on Windows for seconds afterwards. Do not
  read the expert sidecar through the OS cache around an uncached run [owned-expert-cache-state].
- **Shell.** Git Bash and PowerShell are both present. Heredocs mangle backslashes and `\t` in embedded
  Python: write scripts with a file-write tool. Set `PYTHONIOENCODING=utf-8` when Python prints non-ASCII.
  The LF/CRLF warnings on commit are harmless.
- **Windows large-image trap.** A config-sized static array in a DLL pushes `SizeOfImage` past what the loader
  maps (error 0xC000007B). Size arenas on the heap [large-model-windows-build-run-gotchas].
- **Generated headers are per build directory**, in `out/build/<dir>/generated/`. Compare them when two arms'
  banners look identical; a stale configurator silently produces the wrong arms.
- **The real-axes configure recipe must be complete** or the model is rejected without saying which axis is
  wrong; it lives in `scripts/run_perf_suite.py` (`REAL_AXES`) [qwen4-real-axes-configure-recipe].

### Linux verify recipe (keeps build and test output out of context)

```bash
cmake -S . -B out/build/<dir> -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
      -DSUB0_COMPUTE=CPU -DSUB0_NATIVE=ON -DSUB0_BUILD_TESTS=ON -DSUB0_BUILD_BENCHMARKS=OFF
cmake --build out/build/<dir> --target sub0llm-configure
out/build/<dir>/sub0llm-configure --corpus <corpus with <|endoftext|> markers> --dmodel 196 --layers 11 ...
scripts/verify.sh out/build/<dir>      # prints the build verdict and ctest summary only
```

`scripts/verify.sh` writes full logs to `<dir>/verify_build.log` and `verify_ctest.log`; read those only on a
failure. It caps ctest at 8 parallel jobs (`JOBS=n` to change): `-j$(nproc)` was OOM-killed on a 29 GiB host.
Architecture presets are files: `sub0llm-configure --list-profiles` shows what `profiles/*.profile` offers; add or
edit a file (format in `include/sub0/profile.hpp`) rather than the C++. `--profile <name|path|->`, `--profile-dir`
and `$SUB0_PROFILE_DIR` choose where it looks; explicit flags always override a profile.
Model weights: set `SUB0_MODELS_DIR` (or `--models-dir`). A profile's `model-*` keys say which weights it is for;
`sub0llm-configure --list-profiles` and `python3 scripts/models.py status` report found / NOT FOUND, `models.py check`
compares the profile's dims with the model's Hugging Face `config.json` (the source of truth; run it after editing a
profile), and `models.py fetch <profile>` lists the download and only downloads with `--yes`.
Read a large diff with `git diff --stat` first, then per-file, rather than inline. See also
[CORPUS_WORKFLOW.md](CORPUS_WORKFLOW.md) for the corpus-acquisition gap.

## 7. The dev machine

Everything measured in this project's records was measured here, so quote it when you compare.

- Intel Core Ultra 9 275HX (Arrow Lake-HX): 8 P-cores + 16 E-cores, no SMT, 36 MB L3, AVX2/FMA/AVX-VNNI/
  VNNI-INT8 and no AVX-512. DRAM about 79 GB/s from the P-cores [host-cpu-arrow-lake-hx].
- 63 GiB RAM, an RTX 5070 Laptop GPU (8 GB; the CUDA backend exists but decode work is CPU), Windows 11.
- Clang 22.1.8 builds this project. MSVC 19.51, GCC 15/16 and WSL GCC exist for comparisons
  [host-compilers-available].
- `D:` holds the model files (`D:\ModelWeights\...`), about 49.5 GiB for the real model.

## 8. Keeping this in step

- Work log: a row when you start something that holds a file or the CPU; close it on finish. The last entry
  states what is held, if anything.
- Agent memory: `python scripts/agent_memory.py export` refreshes `docs/agent-memory/` from the local memory
  directory (and vets it for a public repo); `restore` seeds a new machine. Do this at every pause, and review
  the diff before committing it.
- When a direction in section 1 changes, change it here in the same commit as the code that follows it.
