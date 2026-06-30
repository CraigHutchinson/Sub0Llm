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
   mirrored CLI args from the CMake `add_custom_command` (`--has-cuda/--cuda-arch/--gpu-*/-o`/paths…).
   Keeps `--corpus` + the user-overridable knobs. ← shrinks the CMake/configure duplication.
2. **Gate the engine + stage targets** on `EXISTS ${GEN_HEADER}` + `CMAKE_CONFIGURE_DEPENDS`; make the
   `add_custom_command` a *convenience* (still seeds a fresh checkout) but the documented path is the tool.
3. **Extract `sub0_frontend`** (config_util/memplan/casing/tokenizer/unigram/registry) so the tools and
   the engine share one site.
4. **Split the driver** into `sub0llm-{configure,tune,train,gen}` (thin `main()`s over the stage libs),
   gated per (2); fold the diagnostics (vocab/bench/models/report/memplan) into the nearest stage.
5. **(stretch)** a `workflow` convenience target; the fully-decoupled state-1 self-probe (`frontend_cuda`).

Stage 1–2 deliver the core want (user-called configure, simpler CMake); 3–4 are the thin-tools cleanup.
