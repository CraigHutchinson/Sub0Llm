# Design suite: the 27B performance goal

Written 2026-11-09 for the goal in #77. Each document is grounded in the repository as it is (verified facts are
marked, proposals are marked, unknowns name the measurement that settles them). Read in this order:

| Document | Question it answers | Tickets |
|---|---|---|
| [PERFORMANCE_MODEL.md](PERFORMANCE_MODEL.md) | What limits the model on this machine, and what is attainable? | #77, #74, #63 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How do config, weights, formats, projection, model math and measurement fit, and which seams make iteration modular? | #27-#30, #83 |
| [QUANT_LIBRARY.md](QUANT_LIBRARY.md) | How is a format or kernel added as a module, tested and selected? | #18, #31-#41, #82, #84 |
| [PREFILL_BATCHING.md](PREFILL_BATCHING.md) | How does batched prefill work and how is it proven? | #74 |
| [MEASUREMENT_PLATFORM.md](MEASUREMENT_PLATFORM.md) | How do we know what the machine can do and tie every number to it? | #78-#81, #75 |
| [WORK_PLAN.md](WORK_PLAN.md) | In what order, with what gates, does it all get done? | all |

Older documents stay as history and are superseded where these differ: `docs/QUANT_FORMATS.md`,
`docs/QUANTIZED_RESIDENT_DENSE.md`, `docs/CORPUS_WORKFLOW.md` (corpus work: #6). Measurement records live in
`docs/performance/`; the process rules are `docs/OPTIMIZATION_PROCESS.md` and `AGENTS.md`.

Maintenance: docs own design and evidence, tickets own status and discussion (#22). When a package lands, update its
section here (status line) and the plan's table; do not duplicate status in both places.
