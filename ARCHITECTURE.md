# sub0llm — Architecture Solidification Plan

> **Status: planning / future task.** This captures the target architecture and the migration path
> toward it. Nothing here blocks current chapter work; it's the map for paying down the duplication
> debt that has accumulated across chapter `main()`s as the engine matured.

## The problem we're solving

Training and inference logic is **duplicated across many `main()`s**, each re-implementing its own loop,
arg parsing, device setup, and checkpoint handling. The viz tools (`viz_gen`, `viz_train`) carry their
own training loops; `mera_train` carries the canonical one; `ch29` carries another. A fix or improvement
(FTZ init, cadence, denormal guard, a better sampler) has to be applied N times and drifts. **Each
subsystem's logic should be written ONCE in a module; the tools become thin front-ends that call it.**

Guiding principle (DRY): a tool should either **load a model we already trained** or **call our shared
training/inference module** — never re-derive the loop. Arg parsing, config resolution, device
placement, checkpoint I/O, the train step, and the generation loop are library concerns, not `main()`
concerns.

## Subsystems (the target decomposition)

| Subsystem | Responsibility | Today | Target |
|-----------|----------------|-------|--------|
| **engine-backend** | tensor kernels per device: CPU (AVX2/512), CUDA, OpenVINO | `src/backends/*` — exists, dispatched by Device | keep; add backends behind the same op interface |
| **engine-frontend** | the model graph for train + infer (autograd, nn modules, denoisers) | `include/sub0llm/nn`, `diffusion/.../nn` — exists | keep; one model type used by both train & infer |
| **training** | the train *framework*: loop, optimizer build, device-order, checkpoint/resume, eval cadence, early-stop | **scattered**: `mera_train` loop + `viz_gen::train` + `viz_train` + `ch29` loop | **one `Trainer` module**; `Checkpointer` + `schedule` + `Heartbeat` already extracted — fold the loop itself in |
| **inference** | the language *framework*: load a model dir, prompt → tokens, sampler/diffusion refine | `model_io::load_model_dir` + `sampler::refine_canvas` (good, templated) | keep; make every consumer go through `load_model_dir`, never re-train inline |
| **serve** | HTTP/REST, OpenAI-compatible endpoints for inference (and maybe job control) | `ch32_viz_server` (Boost.Beast, sync), `tools/server` (AR) | one server module; load a model dir; later: full Boost.ASIO async + simdjson throughout |
| **cli** | terminal: interactive + one-shot prompting | `tools/cli` (AR) | one CLI over the shared inference module; diffusion + AR |
| **cli-ui** | front-end: web now (`tools/viz`), native later | `tools/viz` static scrubber + server | keep web; talks to `serve` only |
| **daemon** | system service: resident/on-demand serve, status, tray, resume-after-reboot | — | new; supervises `serve` + persistent training jobs |
| **hub** | management: configure/start/resume training, job status & priority, corpus download/extend/configure | — | new; the control plane over `daemon` |

## Layering (who may depend on whom)

```
engine-backend  ─┐
engine-frontend ─┤→  training  ─┐
                 │   inference ─┤→  serve ─→ daemon ─→ hub
                 └─────────────┘     ↑          ↑
                       cli ──────────┘      cli-ui (web/native)
```

- **training** and **inference** are the two reusable cores. Everything above them is a thin adapter.
- A `main()` (cli, server, a chapter demo) should be ~arg-parse → call core → present. No loop bodies.

## Current duplication inventory (what to consolidate first)

1. **Training loop** — `mera_train.cpp::run<Model>()` is the reference (resume, device-order, Checkpointer,
   Heartbeat cadence, safety checkpoint). `viz_gen.cpp::train()` and `viz_train.cpp` each reimplement a
   simpler GPU loop. **Action:** extract `train::Trainer<Model>` from `mera_train::run`; viz tools call it
   (or just consume a checkpoint produced by `mera_train`).
2. **Arg parsing** — `config::resolve()` (reflected, layered, rejects unknown flags) is the canonical path
   and already used by `mera_train`/`ch29`. The viz tools use ad-hoc `arg_s/arg_i` + `cli::require_known`.
   **Action:** migrate viz tools onto `resolve()` (or a viz-scoped sub-config) so one schema drives all.
3. **Inline training in tools** — `viz_gen`/`viz_train` TRAIN a model to then visualize it. Per the DRY
   goal they should default to **loading** a trained dir (`load_model_dir`) and only optionally train.
   `viz_server` already loads a dir — make that the norm.
4. **Checkpoint/resume** — consolidated into `train::Checkpointer` (cadence + early-stop + honest resume
   + best-step + safety checkpoint). **Remaining:** migrate `ch29` onto it (TODOs in `ch29/main.cpp`).
5. **Device setup** — `init_cpu_compute()` (FTZ/DAZ) + model.to(device) + build-optimizer-after-move must
   be identical everywhere. **Action:** the `Trainer` owns the device-order contract once.

## Already-extracted reusable pieces (the seeds of the modules)

- `sub0diff::train::Checkpointer` — cadence + early-stop + resume I/O + best-step + rolling safety ckpt.
- `sub0diff::train::make_schedule` — corpus-scaled eval cadence (coverage rule) + sample sizes.
- `sub0diff::util::Heartbeat` — wall-clock cadence for logs / safety checkpoints (instantaneous rate).
- `sub0diff::config::RunConfig` / `resolve()` — reflected, layered, SHA-tagged config + unknown-flag reject.
- `sub0diff::nn::load_model_dir` — load any trained dir (flat or MERA), serve the best checkpoint.
- `sub0diff::nn::refine_canvas` — the templated diffusion sampler (one loop for every model type).

## Migration path (incremental, non-breaking)

1. **Extract `Trainer`** from `mera_train::run<Model>` (it already delegates checks to Checkpointer; lift
   the loop + device-order + train-step plumbing into `train::Trainer`). `mera_train` becomes arg-parse +
   `Trainer{cfg}.run()`.
2. **Thin the viz tools**: default to `load_model_dir`; keep an optional `--train` that calls `Trainer`.
   Deletes two duplicate loops.
3. **Unify arg parsing** under `resolve()` for the viz tools (one schema, one unknown-flag policy).
4. **Migrate `ch29`** onto `Checkpointer` (carry curriculum state in `Progress`; see its TODOs).
5. **serve**: factor a `serve` module shared by `viz_server` and `tools/server`; move to async Boost.ASIO
   + simdjson reads/writes throughout (currently sync Beast + nlohmann emit).
6. **daemon + hub**: new subsystems over `serve` — job registry (persistent training jobs with priority),
   resume-after-reboot, corpus management, status panels. Built last, on the stabilized cores.

## Cross-cutting conventions (apply as modules land)

- **One config schema** drives flags + JSON + config-SHA (reflection in `run_config.hpp`).
- **JSON**: simdjson on-demand forward parsing for reads (register handlers, single pass); emit stays
  nlohmann until a writer lands. Migrate readers off nlohmann DOM.
- **`init_cpu_compute()` on every thread/main** (FTZ/DAZ) — owned by `Trainer`/worker-pool, not copied.
- **Provenance**: `models/<name>_g<gitSHA>_c<configSHA>` so a dir reproduces from code+config.
- **No fixed magic thresholds**: cadences/caps scale with corpus/dims (coverage rule, pool caps).
