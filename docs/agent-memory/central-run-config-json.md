---
name: central-run-config-json
description: "sub0::registry::RunConfig -- a central, X-macro-declared training-recipe struct persisted as config.json (simdjson::ondemand read, hand-rolled write) per model directory, fixing the class of bug that caused the d512 Muon->AdamW incident; live-verified; not committed as of 2026-07-10"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a6b481b-6211-474b-818a-995223d7d5ac
---

User's explicit ask, in direct response to the d512 incident ([[d512-fault-resume-optimizer-switch-bug]]):
"packing all the configuration into a central Struct... so we don't make a similar optimizer mistake
again... a config object in the code which saves to the model and vice versa for comparison...
config.json so we can use simdjson... new options must be added there and therefore become saved
implicitly to the model." Implemented same session, fully tested, not yet committed.

**What shipped:**
- `sub0::registry::RunConfig` (`include/sub0/registry.hpp`, next to the existing `ModelMeta`) —
  declared via one `SUB0_RUN_CONFIG_FIELDS(X)` macro listing `(type, name, default)` per field:
  `corpus`, architecture dims/flags (`d_model`..`qk_norm`), `optimizer`, `batch`, `lr`, `seed`. The
  struct definition, the JSON writer, AND the JSON reader are all generated from this ONE list —
  literally realizes "add it once, get it saved implicitly."
- `write_config_json` — inline in the header, hand-rolled JSON (a parser library buys nothing for
  emitting a flat object of scalars; simdjson is a parser, not a serializer).
- `read_config_json` — declared in the header, implemented in NEW `src/run_config.cpp` via
  `simdjson::ondemand` (single forward pass over the object's keys, one `if(key==...)` branch per
  macro row — matches [[json-reads-forward-ondemand-not-dom-facade]], this project's existing JSON
  philosophy, now actually exercised for the first time). Needed `SUB0_API`/`__declspec(dllexport)`
  on the declaration since it's the ONE non-`inline` function in an otherwise header-only file —
  everything else in registry.hpp compiles into each consumer's own TU, so this was the first thing
  in that header to actually cross the sub0_core.dll boundary and hit Windows' hidden-visibility
  default; caught immediately by a real link error, fixed by mirroring core.hpp's own SUB0_API macro
  (a self-contained copy, not an include of core.hpp — registry.hpp stays engine-free).
- **Dependency added for real**: `CPMAddPackage("gh:simdjson/simdjson@3.10.1")` (the codebase had a
  TODO comment anticipating exactly this for years — see [[simdjson-migration-backlog]], now partly
  superseded/started). Linked `simdjson::simdjson` PRIVATE to `sub0_core` only; consumers never see
  simdjson types, just the plain-std-typed `read_config_json` declaration.
- `train_stage.cpp` wiring: a `build_run_config` lambda is now the SINGLE place that assembles
  training-recipe values from compile-time constants + live CLI-resolved locals; `write_meta` derives
  meta.txt's overlapping fields FROM it (one source, two on-disk artifacts, can't drift apart from
  each other the way the original bug's two independently-set copies did). The actual fix: right
  before `AdamW opt(lr, optimizer==1)` is constructed, if `resumed` and a `config.json` already
  exists in the target model dir, its `optimizer` wins over whatever this invocation's CLI/default
  said (logged loudly), mirroring the ALREADY-correct batch/lr/seed-from-checkpoint precedent
  exactly. An old model dir with no config.json yet just falls through unchanged (no regression).

**Scope decisions, deliberate:**
- `meta.txt`/`ModelMeta`/`registry::scan()`/`compatible()` (cross-model-directory discovery/
  reporting) are UNTOUCHED — RunConfig is additive, not a meta.txt replacement. Kept the two
  concerns (discovery/reporting vs. resume-correctness) separate rather than risking the
  already-working, already-tested registry infrastructure in the same pass.
- `batch`/`lr`/`seed` ARE included in RunConfig (for completeness/comparison) but NOT re-enforced
  on resume via config.json — the `.ckpt` binary already correctly owns that job. Only `optimizer`
  (which had no checkpoint-binary home) gets the new resume-time enforcement.
- Deliberately excluded: `steps`/`keep`/`resume_mode` (per-invocation policy, no "correct" value to
  protect — matches train_stage.cpp's own pre-existing comment on why `steps` is never persisted).
- `gen`/`tune`/other stages NOT touched — no resume-correctness concern exists there.

**Testing**: `tests/run_config_tests.cpp` (new, wired into `sub0_tests` since `read_config_json`
lives in `sub0_core`, not the engine-free `sub0_frontend_tests` target) — round-trip, missing-file,
missing-directory, corrupt-file, forward-compatible-unknown-key, and JSON-escaping cases, all
passing (169/169 total suite). **Live end-to-end proof, not just unit tests**: trained a tiny model
with `--optimizer muon`, resumed it with NO `--optimizer` flag (the exact triggering condition of
the real incident) — train.log shows `config.json overrides this invocation's optimizer: adamw ->
muon` followed by the Muon banner, confirming the persisted choice won over the silently-defaulted
CLI value. All work done in the isolated CPU-only `out/build/tok_cpu` dir; `out/build/ts_cuda`
(mid-training for most of this) was never touched.

Not committed as of this writing. `out/build/ts_cuda`'s already-built binary predates this fix (it's
the same Jul 9 18:39 binary that ran d512's whole recovery) — a future rebuild of that dir would pick
this fix up automatically the next time it's reconfigured/rebuilt (not done yet, since a training run
was active in it through most of this work).

See [[d512-fault-resume-optimizer-switch-bug]] for the incident this fixes, [[json-reads-forward-ondemand-not-dom-facade]]
for the JSON-reading philosophy this follows, and [[boy-scout-rule-always-resolve-inconsistencies]]
for the general pattern of turning a specific bug into a structural fix.
