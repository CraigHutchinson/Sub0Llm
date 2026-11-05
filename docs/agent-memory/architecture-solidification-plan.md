---
name: architecture-solidification-plan
description: "DRY refactor roadmap — one training/inference module, thin tools; subsystem decomposition (engine/train/infer/serve/cli/cli-ui/daemon/hub); doc ARCHITECTURE.md"
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Future task (captured 2026-06-22, not yet started): solidify the sub0llm architecture so each
subsystem's logic is written ONCE and tools are thin front-ends — **kill the training-loop +
arg-parse duplication** across chapter `main()`s. Full map in **`ARCHITECTURE.md`** (repo root).

**DRY principle (user, verbatim intent):** a tool should either LOAD a model we already trained or CALL
our shared training/inference module — never re-implement the loop. Arg parsing, config resolution,
device placement, checkpoint I/O, train step, generation loop are LIBRARY concerns, not `main()` concerns.

**Subsystems:** engine-backend (cpu/cuda/openvino), engine-frontend (model graph for train+infer),
training (framework+process), inference (language framework+process), serve (http/openai), cli
(terminal), cli-ui (web now / native later), daemon (system service, tray, resume-after-reboot,
persistent jobs), hub (configure/status/priority of train+serve, corpus download/extend/configure).

**Biggest duplication to consolidate first:** the training loop — `mera_train::run<Model>()` is the
reference (resume + device-order + [[checkpointer]] Checkpointer + Heartbeat cadence + safety ckpt);
`viz_gen::train()` and `viz_train` reimplement simpler GPU loops. Extract `train::Trainer<Model>` from
`mera_train::run`, then thin the viz tools to default to `load_model_dir` (+ optional `--train`).

**Already-extracted seeds of the modules:** `train::Checkpointer`, `train::make_schedule`,
`util::Heartbeat`, `config::resolve()`/`RunConfig`, `nn::load_model_dir`, `nn::refine_canvas`.

Also on this track: full Boost.ASIO async + simdjson throughout for serve (currently sync Beast +
nlohmann emit, httplib already removed); migrate ch29 onto Checkpointer (carry curriculum in Progress).
Relates to [[ch29-checkpointer-migration]]-style follow-ups and the [[specialization-backend-roadmap]].
