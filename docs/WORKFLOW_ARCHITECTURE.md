# Staged workflow architecture — user-driven configure → tune → train → gen

Builds on [CONFIGURE_ARCHITECTURE.md](CONFIGURE_ARCHITECTURE.md). Goal: each pipeline stage is a
**user-callable tool** that writes config the *next build* bakes in as `constexpr`, so CMake stops
mirroring what the configurator already does and a re-config is a tool run, **not** a CMake reconfigure.

## Target workflow

```
cmake --preset native                 # 1. configure the build; only the frontend tools can build yet
cmake --build --preset native         #    -> builds sub0llm-configure (no generated header needed)
./out/build/native/sub0llm-configure --corpus ./data/foo.txt    # 2. derive + emit the config headers
cmake --build --preset native         #    -> header now exists -> engine + stage tools build
./out/build/native/sub0llm-tune       # 3. measure + persist the tuned runtime knobs
cmake --build --preset native         #    -> bakes the tuned knobs (state 3)
./out/build/native/sub0llm-train      # 4. train (uses tuned knobs, else the configure estimates)
./out/build/native/sub0llm-gen "Once there was a dog"           # 5. generate
```

Every stage is **auto-derived/defaulted** (memory-ladder GPU batch, P-cores−1 CPU width, vocab knee,
…); only `--corpus` is required, and even that could default. A rebuild between stages is the seam
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
  buries the build graph inside a binary. Keep stages pure (do the stage, exit). The *gating* below makes
  the explicit `build → run → build` loop friction-free (no manual reconfigure); a thin top-level
  `workflow` convenience target can chain them for those who want one command.
- **Build-stage gating.** `if(EXISTS ${GEN_HEADER})` guards the engine + stage targets, and
  `CMAKE_CONFIGURE_DEPENDS` watches the header so its *appearance* auto-reconfigures on the next
  `cmake --build`. Fresh checkout → only `sub0llm-configure`; after it runs → tune/train/gen build. No
  explicit reconfigure between stages.
- **`frontend_cuda` / `backend_cpu` discipline.** Keep `backend_*` strictly *compute*. Pre-model device
  facts come from CMake (`configure_file`), so a runtime `frontend_cuda` probe is only needed for the
  fully-decoupled state-1 self-probe — defer it. `sub0_frontend` holds the shared pre-model logic.

## Staged implementation plan

1. **Bake build facts into `sub0llm-configure`** (`configure_file` → `sub0_build_facts.hpp`); drop the
   mirrored CLI args from the CMake `add_custom_command`. ✅ **DONE** — device caps + output paths are
   baked; the command keeps `--corpus` + the user-overridable knobs.
2. **Gate the stage targets on configure + a lever to fully decouple.** ✅ **DONE** — each
   `sub0llm-<stage>` auto-depends on `sub0_generate_config` (so it can't build before the config exists);
   `sub0llm-configure` itself has no such dependency (state-1 buildable). **`SUB0_AUTO_CONFIGURE`**
   (default ON) is the lever: ON keeps the one-shot convenience (the build regenerates config when the
   corpus/tool/tune-cache change); **OFF** is the pure staged workflow — the build never regenerates
   behind your back, `sub0_generate_config` becomes an explicit on-demand target, and the engine compiles
   against the existing header (gating only the dependency was insufficient — the header is a custom-command
   OUTPUT, so OFF swaps it for a COMMAND-only target). Even with ON the core decouple holds: dims live in
   tool-owned headers, so `sub0llm-configure …` + `cmake --build` re-sizes with **no CMake reconfigure**.
   The strict `EXISTS`-gate + `CMAKE_CONFIGURE_DEPENDS` (fresh checkout builds *only* the configurator)
   remains an optional stricter variant.
3. **Extract `sub0_frontend`** (config_util/memplan/casing/tokenizer/unigram/registry) so the tools and
   the engine share one site. ✅ **DONE** — the static lib `sub0_frontend` (was `sub0_tok`) compiles the
   tokenizer/unigram and exposes the header-only pre-model logic via its PUBLIC include; the configurator
   and the engine both link it, and the engine-free `sub0_frontend_tests` target covers it.
4. **Split the driver** into thin `sub0llm-{configure,train,gen,tune}` exes over shared runners. ✅
   **DONE** — `include/sub0/cli_stages.hpp` defines each `run_*` once; the umbrella `sub0llm` dispatches
   to them and the stage exes are thin `main()`s; diagnostics (vocab/bench/models/report/memplan) stay
   on the umbrella.
5. **(stretch)** a `workflow` convenience target; the strict `EXISTS`-gate; the state-1 `frontend_cuda`
   self-probe. ⏳ pending.

Stages 1, 2 (dependency form) and 4 are done — the user-callable stage tools + the shrunk, non-mirroring
CMake are in place. Remaining: the `sub0_frontend` extraction and the strict fresh-checkout gating.

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
  a **built** `tokenizer.bin` + arbitrary sample input — the lighter, post-build inspector/exporter.
- **Test payoff**: engine-free, its round-trip is the existing fuzz property, and the export format is
  schema-checkable — more coverage in the fast target.

**Workflow integration**: after a configure run, `sub0llm-tokenizer vocab/encode` sanity-checks the
tokenization *before* committing to a long train; `export` feeds external comparison + the viz scrubber.
Sequencing: do the `sub0_frontend` lib first (gives the tool + its tests a home), then the tool absorbing
`vocab` off the engine, then `encode/decode/roundtrip/export`.
