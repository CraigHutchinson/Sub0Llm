# Staged workflow architecture — user-driven configure → tune → train → gen

Builds on [CONFIGURE_ARCHITECTURE.md](CONFIGURE_ARCHITECTURE.md). Goal: each pipeline stage is a
**user-callable tool** that writes config the *next build* bakes in as `constexpr`, so CMake stops
mirroring what the configurator already does and a re-config is a tool run, **not** a CMake reconfigure.

## Target workflow

```
cmake --preset native                                            # 1. configure the build
cmake --build --preset native --target sub0llm-configure         #    -> just the configurator (no
                                                                   #       generated header needed)
./out/build/native/sub0llm-configure --corpus ./data/foo.txt     # 2. derive + emit the config headers
cmake --build --preset native                                    #    -> header now exists -> engine
                                                                   #       + stage tools build
./out/build/native/sub0llm-tune       # 3. measure + persist the tuned runtime knobs
cmake --build --preset native         #    -> bakes the tuned knobs (state 3)
./out/build/native/sub0llm-train      # 4. train (uses tuned knobs, else the configure estimates)
./out/build/native/sub0llm-gen --model <m> "Once there was a dog"   # 5. generate
```

`scripts/workflow.ps1` wraps steps 1-2 (and optionally 3-5) into one command. There is no CMake
orchestration of step 2 at all -- `sub0llm-configure` is a plain executable target with no generated
inputs, so it always builds; the engine/stage targets are unconditionally defined too, but simply fail
to compile with a clear "file not found" on the generated header until step 2 has actually run. A
scoped `--target sub0llm-configure` build is required first on a fresh checkout (an unscoped
`cmake --build` would otherwise also try, and fail, to build the not-yet-configurable engine targets).

Every stage is **auto-derived/defaulted** (memory-ladder GPU batch, P-cores−1 CPU width, vocab knee,
pretokenize-vs-on-demand by corpus size vs RAM, …); only `--corpus` is required. Every one of these
knobs is `sub0llm-configure`'s own CLI flag with its own sensible default -- CMake carries NONE of
them (see `SUB0_TERNARY`'s comment in `CMakeLists.txt` for why: a cache variable here would just be a
second, driftable source of truth for something the tool already owns outright, since it takes the
corpus path directly and needs no CMake orchestration to run). A rebuild between stages is the seam
that turns runtime-discovered facts into compile-time constants.

## Layering (one logic site, thin tools)

| Layer | Lives in | Used by |
|---|---|---|
| **frontend** (pre-model): config decisions, tokenizer, casing, unigram, memplan, registry | `sub0_frontend` (static) | every stage tool + the engine build |
| **build facts** (CMake-known: device caps, output paths, git sha) | `configure_file` → `sub0_build_facts.hpp` | baked into `sub0llm-configure` |
| **config emit** (smart derivation → `constexpr`) | `sub0::config` (pure, unit-tested) + the configurator's emitter | `sub0llm-configure` |
| **engine** (compute) | `sub0_core` = `engine_core` + `backend_cpu` (+ `backend_cuda`) | tune / train / gen |
| **stages** | `sub0_train`, `sub0_gen` | the stage executables |

## Decisions on the open forks

- **`configure_file` vs our C++ `config::parse` — use BOTH, at different layers.** `configure_file`
  bakes only the *facts CMake already knows* (device caps, output paths, sha) into the tool — dumb
  substitution, exactly its remit. The *derivation* (auto-size from corpus bytes, precision resolution,
  the vocab-curve knee, tune-cache fold) is real computation, already in `sub0::config` and unit-tested;
  `configure_file` cannot express it. **Do not** replace the C++ emitter with CMake string-substitution
  — we'd lose derivation, tests and extensibility. CMake feeds facts *in*; C++ emits the headers *out*.
- **Split configure into a separate "config-source-gen" binary? No.** The configurator already separates
  *decision* (`sub0::config`, pure) from *emission* (the writer); a second binary just adds a
  serialize/handoff with no gain. One `sub0llm-configure`, internally layered. Revisit only if a second
  producer ever needs the emitter.
- **Implicit stage-chaining (a stage kicks off the next build)? No (by default).** On Windows the running
  exe is locked, so a stage rebuilding/overwriting itself is fragile, and self-invoking `cmake --build`
  buries the build graph inside a binary. Keep stages pure (do the stage, exit). `scripts/workflow.ps1`
  chains the explicit `build → run → build` loop into one command for those who want it; it is a plain
  script over the same CLI invocations, not a CMake target, so it adds no orchestration for CMake to
  drift out of sync with.
- **Build-stage gating? Considered, not built.** An earlier draft of this plan had `if(EXISTS ${GEN_HEADER})`
  guard the engine + stage targets, with `CMAKE_CONFIGURE_DEPENDS` watching the header so its *appearance*
  would auto-reconfigure the next `cmake --build`. Superseded by item 2's final design below: the targets
  are unconditionally defined, and a missing generated header is just a plain compile error ("file not
  found") on `sub0llm-configure`'s output, same as any other missing header — no CMake-tracked dependency
  edge, no reconfigure-on-appearance magic, nothing to get out of sync.
- **`frontend_cuda` / `backend_cpu` discipline.** Keep `backend_*` strictly *compute*. Pre-model device
  facts come from CMake (`configure_file`), so a runtime `frontend_cuda` probe is only needed for the
  fully-decoupled state-1 self-probe — defer it. `sub0_frontend` holds the shared pre-model logic.

## Staged implementation plan

1. **Bake build facts into `sub0llm-configure`** (`configure_file` → `sub0_build_facts.hpp`); drop the
   mirrored CLI args from CMake entirely. ✅ **DONE** — device caps + output paths are baked as CLI
   defaults; every model/corpus/precision knob is the tool's own flag now (see below).
2. **Remove CMake orchestration of configure entirely — there is no lever, because there is nothing
   left to toggle.** ✅ **DONE (2026-07-03), superseding the SUB0_AUTO_CONFIGURE lever below.** The
   `sub0_generate_config` custom target, the `SUB0_AUTO_CONFIGURE` option, and the ON/OFF branching that
   used to build it are all GONE — along with the CMake cache variables that only existed to seed that
   target's command line (`SUB0_CORPUS`, `SUB0_POS_ENCODING`, `SUB0_ROPE_THETA`, `SUB0_BF16`,
   `SUB0_PRECISION_GEMM`, `SUB0_PRECISION_ACT`; `SUB0_CORPUS_PRETOK`'s AUTO-by-corpus-size-vs-RAM
   heuristic moved into `sub0llm-configure` itself, since the tool has `--corpus` directly and no longer
   needs CMake to hand it a path). Run `sub0llm-configure --corpus <c> [--dmodel N ...]` directly (or
   `scripts/workflow.ps1`) -- there is no `sub0_generate_config` target to build instead. The engine and
   stage-tool targets are unconditionally defined and simply fail to compile with a plain "file not
   found" if the generated header doesn't exist yet; there is no dependency edge to gate them on, by
   design -- a CMake-tracked dependency on a file written by a process CMake doesn't orchestrate would
   only be able to notice staleness via mtime anyway, which plain compilation already does for free via
   the normal header-dependency scan. **Why this is the end state, not just a defaults flip (superseding
   the OFF-by-default SUB0_AUTO_CONFIGURE compromise below):** a large corpus reconfigure triggered
   silently as a side effect of an unrelated `cmake --build` was hard to tell apart from a hung build,
   and retrying it (a natural reaction to an apparently-stuck build) could launch a second reconfigure
   concurrently with the first, racing on the same output files. Toggling the default to OFF reduced the
   frequency but not the possibility, since the machinery (and the temptation to flip it back ON) still
   existed; removing the machinery removes the failure mode outright. `sub0llm-configure` was already
   fully self-sufficient (build facts baked in, every knob a CLI default) before this -- the CMake side
   was pure duplication.
   <details><summary>Superseded: the SUB0_AUTO_CONFIGURE lever (removed 2026-07-03)</summary>
   Historical note, kept for context on prior build logs mentioning it. Each `sub0llm-<stage>` used to
   auto-depend on a `sub0_generate_config` custom target when `SUB0_AUTO_CONFIGURE=ON` (regenerate the
   header as a build artifact whenever the corpus/tool/tune-cache changed); `OFF` (the default since
   2026-07-02) made it an explicit on-demand target instead. Both modes still ran the configurator
   *through CMake*, which is the part now removed entirely.
   </details>
3. **Extract `sub0_frontend`** (config_util/memplan/casing/tokenizer/unigram/registry) so the tools and
   the engine share one site. ✅ **DONE** — the static lib `sub0_frontend` (was `sub0_tok`) compiles the
   tokenizer/unigram and exposes the header-only pre-model logic via its PUBLIC include; the configurator
   and the engine both link it, and the engine-free `sub0_frontend_tests` target covers it.
4. **Split the driver** into thin `sub0llm-{configure,train,gen,tune}` exes over shared runners. ✅
   **DONE** — `include/sub0/cli_stages.hpp` defines each `run_*` once; the umbrella `sub0llm` dispatches
   to them and the stage exes are thin `main()`s; diagnostics (vocab/bench/models/report/memplan) stay
   on the umbrella.
5. **(stretch)** a `workflow` convenience wrapper; the state-1 `frontend_cuda` self-probe.
   `scripts/workflow.ps1` **DONE** — a plain script chaining the CLI invocations (see above), not a CMake
   target; the strict `EXISTS`-gate this item originally paired it with is superseded/moot (see the
   "Build-stage gating" fork above — item 2 settled on unconditional targets + plain compile errors
   instead). `frontend_cuda` self-probe ⏳ still pending, deferred per the "`frontend_cuda` / `backend_cpu`
   discipline" fork above.

Stages 1-4 are done — the user-callable stage tools, the `sub0_frontend` extraction, and the shrunk,
non-mirroring CMake are all in place, plus the `workflow.ps1` convenience wrapper from item 5. Remaining:
only the state-1 `frontend_cuda` self-probe, deferred (not blocking).

## Frontend layer — testability + tooling (groundwork for the Stage-3 cleanup)

The **frontend** is the engine-free, pre-model logic the configurator + stage tools share but the engine
doesn't define: `config_util`, `memplan`, `registry`, `casing`, `tokenizer`, `unigram` (all header-only
+ std-only today). Its separation buys two things beyond tidiness:

### 1. Testability — the immediate, measured payoff
Header-only + std-only ⇒ it runs in the **fast engine-free** `sub0_frontend_tests` target (no engine build, no
GPU). First slice **done** (`tests/frontend_tests.cpp`): `memplan` (the VRAM clamp/cap math — monotone
footprint, `max_batch_for_vram` inversion, the clamp invariant) and `registry` (`corpus_tag` / `model_dir`
identity + `compatible()` — the `models --prune` rule) — the two areas this session changed, previously
tested only via the slow engine-linked target (`memplan`) or **not at all** (`registry`). 200 → 224
assertions. **Gaps still open**: `casing` edge cases (only covered indirectly via the tokenizer), registry
meta read/write + `scan` (needs a temp-dir I/O fixture), the configurator's header *emit* (integration only).

### 2. `sub0_frontend` lib (Stage 3) — one named home + one test target
Extract a `sub0_frontend` static lib (the headers above) that both the configurator and the stage tools
link, plus a single `sub0_frontend_tests`. Low-risk (a CMake + include reshuffle; the engine-free test
target already proves the value). `frontend_cuda` (a runtime device probe) is only needed for the fully
decoupled state-1 self-probe and stays deferred — CMake bakes the device facts today.

### Tokenizer / vocab as engine-free frontend tools — judgement: **YES**, for diagnostics + interchange
A `sub0llm-tokenizer` tool (links `sub0_frontend`, **not** the engine) is worth building:
- **Why engine-free is the point.** Inspecting/exporting a tokenizer needs only `sub0_frontend`. Today
  `sub0llm vocab` lives in the *train* stage lib, so you must build the whole engine to print a vocab
  table — wrong coupling. Moving it frees diagnostics from the engine and from a configured build.
- **Diagnostics**: `encode "text"` (show the token stream — debug the CamelCase-shatter / indented-code
  pathologies), `decode <ids>`, `roundtrip <file>` (ad-hoc lossless check on real content, vs the fuzz
  net), `vocab [--limit]` (moved off the engine).
- **Interchange**: `export --format json|tsv` (token↔id, piece, logp) for external/cross-tokenizer
  analysis and the planned diffusion-viz scrubber.
- **NOT a corpus preprocessor.** Bulk corpus→`corpus.tok` stays the configurator's job (it owns the scan,
  the out-of-core pipeline, the `.words` cache); don't duplicate it. `--dump-vocab` also stays in the
  configurator (it needs the corpus scan + the BPE-vs-Unigram A/B + the vocab curve). The tool operates on
  a **built** `tokenizer.tok` + arbitrary sample input — the lighter, post-build inspector/exporter.
- **Test payoff**: engine-free, its round-trip is the existing fuzz property, and the export format is
  schema-checkable — more coverage in the fast target.

**Workflow integration**: after a configure run, `sub0llm-tokenizer vocab/encode` sanity-checks the
tokenization *before* committing to a long train; `export` feeds external comparison + the viz scrubber.
Sequencing: do the `sub0_frontend` lib first (gives the tool + its tests a home), then the tool absorbing
`vocab` off the engine, then `encode/decode/roundtrip/export`.
