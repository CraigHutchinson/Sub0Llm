# Ch32 — running notes (things found while getting the engine working)

> Append findings/gotchas/TODOs as we go. Memory-management work is **deferred to a later milestone**
> (see the `denoiser-toolbox-and-memory-milestone` memory); note issues here, don't fix them now.

## Open items / things to make work

- ~~MERA model is N-specific~~ **DONE** — MERA takes `max_seq_len`; `forward(N)` rebuilds the pyramid per
  call, accepting ANY valid N ≤ max (blocks indexed by depth-from-finest; the top block handles whatever
  level falls ≤ w). Test covers N=64 + N=32 on one model.
- ~~fixed-N train/use mismatch~~ **RESOLVED by mixed-N training** (`ch32_mera_mixedn`, c8284d5). Train an
  identical MERA fixed@512 vs mixed on {128,256,512}; eval at each (3000 steps):
  `fixed 3.59/3.60/2.88 at N=128/256/512` (degrades ~25% off its training length) vs
  `mixed 2.545/2.535/2.532` (consistent at EVERY length, −29/−30/−12% vs fixed). Mixed even beats fixed
  at 512 — likely a multi-scale REGULARIZER: fixed@512 OVERFIT at 3000 steps (cf. the earlier MERA@512
  2000-step run = 2.53, matching mixed) while mixed stayed low. **Recommendation: train MERA mixed-N by
  default** — robust across lengths AND resists the documented early-overfit, with fewer total tokens.
- **Generation device** — sampler runs on CPU (reads host logits). GPU generation would need the
  sampler's softmax/commit on-device or batched D2H. Fine for now (generation is run-once).
- **Seam handling (2d)** — was diagnosed for single-level *hier* (function-word gap). MERA's multi-level
  decode already beats flat overall, so check whether MERA even has a residual seam issue before building
  halo/edge handling. (May be moot for MERA.)

## Deferred (memory milestone — noted, not fixed)

- Tensors >16 MB bypass `CudaPool` → per-step cudaMalloc/cudaFree at large N (the VRAM fluctuation).
- Pool disables caching under VRAM pressure (free < 2 GB) — fluctuation when scaling.
- Intra-step autograd graph build/teardown sets the PEAK — only gradient checkpointing reduces it
  (would lift MERA's ~16384 training ceiling).
- Per-step arena allocator = the principled end state (stable VRAM, predictable peak).

## Resources

- **Acronym / terminology index** — [`/ACRONYMS.json`](../../ACRONYMS.json): parsable glossary (project
  terms + common LLM terminology + external refs), keyed by UPPERCASE token. Doubles as the Viz
  tooltip source. Keep it updated as we coin/use terms.

## Planned

- **Interactive visualization & prompting** — scrub the diffusion "thought process" (iteration timeline)
  + drill into MERA tiered internals; param-adjustment comparison; verbosity slider. Design:
  [`../VISUALIZATION_DESIGN.md`](../VISUALIZATION_DESIGN.md). Phase A (cheap, first) = GenerationTrace
  JSON via the `refine_canvas` `on_iter` hook + a static `tools/viz/` web scrubber.

## Findings

- **Viz Phase B (true per-level activations) surfaced three real issues** (from scrubbing a traced run):
  1. *Early canvas looks "full of spaces."* These are TENTATIVE predictions at still-masked positions:
     with the whole canvas masked the model has no context and predicts the unigram mode everywhere —
     and in the **word-level** tokenizer the space token is the single most frequent type, so the
     prior is mostly spaces. Presentation: the viewer now renders whitespace as `␣` (was an invisible
     blank cell). Real signal underneath: word-level tokenization spends vocab/positions on standalone
     space tokens (visible as the `happy   little` double/triple gaps in decoded text) — wasteful, and
     it inflates the unigram floor the sampler bootstraps from.
  2. *First iteration commits spaces only.* Genuine model behavior, not a viz bug: spaces are the
     highest-confidence predictions before any context exists, so the `min_commit_frac` force-commit
     (sampler.hpp §SamplerConfig, the "commit garbage from near-unigram statistics" hazard) fills them
     first. The model lays down structural scaffolding (word boundaries) before content. Whether that
     *helps* or seeds a low-information skeleton is open — candidate experiment: exclude the space token
     from early force-commit, or down-weight whitespace in the commit-confidence ranking.
  3. **The coarse levels are a RIGID balanced tree — this is REAL, not a presentation artifact.** MERA's
     `pool()` mean-pools every `c` *consecutive* tokens (coarse slot j = fine [j·c, (j+1)·c)), so a
     coarse block can straddle a sentence/clause boundary. Language structure is an UNBALANCED tree
     (variable-length phrases/clauses/sentences); the power-of-c pooling imposes a fixed balanced c-ary
     tree that ignores syntax. This is exactly the "balanced-tree prior is weak for language" tension
     from the parked tree-predictor memory. The viewer faithfully shows the model's actual (rigid) tiling.
     **Next investigation:** content-adaptive / boundary-aware coarsening — let pool boundaries fall on
     learned or signal-derived breakpoints (e.g. soft pooling weighted by a predicted boundary score, or
     merge-by-similarity) so coarse slots align with linguistic units. Ties into the per-window
     content-adaptive (ACT) half of the verbosity-slider idea. Balanced MERA stays the efficient
     baseline; boundary-aware pooling is the accuracy lever to test against it.

- **MERA's NLL win does NOT robustly transfer to generation coherence (M2) at this scale.** 3-seed
  `ch32_hier_gen` (N=256, 64 gens): mera-gen content-recurrence beats flat-gen in 2/3 seeds (0.072 vs
  0.053; 0.059 vs 0.074; 0.076 vs 0.054) — mean 0.069 vs 0.060, but the per-seed variance swamps it
  (seed 8 flips). So MERA's clear, robust advantages stay in PREDICTION (NLL 3/3), compute, and context;
  generation coherence is comparable/noisy at 5M params on 540 paragraphs (the Chinchilla gap — samples
  are word-salad-ish for all variants). To get a real generation-quality signal: more samples (cut M2
  variance) and/or a bigger model + more data. Don't claim a MERA generation-quality win.

## Done

- **Reusable training-loop CHECKS module** (`sub0diff::train::Checkpointer`, `train/checkpointer.hpp` +
  `src/checkpointer.cpp`): bundles the discipline so every trainer inherits it instead of reimplementing
  (and drifting) — the coverage-rule eval cadence + averaged sample (`due()`/`eval_windows()`/
  `steps_bound()` from make_schedule), early-stop (best + patience + min-improve deadband), and the full
  honest-resume I/O (`load_weights`→`restore`→`record` over weights `step_*.ckpt` + Adam `step_*.opt` +
  progress `train_state.json`). The model device-order constraint stays the caller's (load on CPU →
  to(device) → build optimizer → restore). simdjson read confined to the .cpp. `ch32_mera_train` now
  delegates to it (deleted its inline TrainState/save/load/early-stop). Validated: train→resume rehydrated
  best/stalls and continued honestly; 52/52 tests green. **Coverage rule proven on a real run**: word-level
  TinyStories (vocab 2953) early-stopped at step 2700 (patience 10), best NELBO 2.31 @ step 1700, with the
  derived cadence (eval every 100 steps ≈ 1.4 epoch coverage, not the old fixed 500).
- **Robust resumable CUDA MERA trainer** (`ch32_mera_train`) — reuses the Ch29 consistent config layer
  (`sub0diff::config::RunConfig`) + binary checkpoint format. `--ckpt-dir X` ALONE reconstructs the exact
  arch from `run_config.json` (added BuildTime fields `model_type`/`mera_coarsen`/`mera_window`), reloads
  the latest `step_*.ckpt` + matching `step_*.opt` Adam moments, and continues. Validated on GPU: train
  0→400 (ckpt@200,400), then resume with only `--ckpt-dir` → rebuilt `MeraDenoiser V=1285 D=256 c=4 w=64`,
  loaded step 400, restored optimizer, continued 400→600 at ~37 steps/s. `GpuTrainer` templatized on Model
  (was Denoiser-only) for reuse. Two real bugs fixed en route:
  - **ch32 targets never called `sub0llm_apply_compile_options()`** (chapter-wide) → the executables
    missed AVX2/SIMD flags AND `SUB0LLM_CUDA` was undefined in their own TUs (so `#ifdef SUB0LLM_CUDA`
    device guards compiled the no-CUDA branch). Now applied to all ch32 targets.
  - **optimizer-before-`to(cuda)` ordering**: building Adam while params are on CPU then moving the model
    to GPU left Adam's (m,v) on the host → device-mismatch illegal access. Fixed: load-on-CPU →
    `to(cuda)` → build optimizer; and re-snapshot the param Variable-list at each save (Variable::to swaps
    storage, staling an up-front snapshot). NOTE: GpuTrainer's 13-arg masked-loss path illegal-accessed on
    MERA (the plain 7-arg `batched_diffusion_loss` is fine on GPU) — the lean trainer uses the 7-arg loop;
    whole-word/contiguous masking on MERA+GPU is an open follow-up.
- `init_cpu_compute()` (FTZ+DAZ) added to all new ch32 mains (`mera_train`/`viz_gen`/`viz_train`/
  `viz_server`) + the server's per-request httplib worker thread (per-thread MXCSR).
- **GPU-trainer ↔ lightweight-server separation + in-repo models + honest-resume state**:
  - `model_io::load_model_dir` now dispatches on `config.json` `model_type` → builds the flat Denoiser
    OR the MeraDenoiser (reads `mera_coarsen`/`mera_window`); `LoadedModel` holds whichever.
  - `ch32_viz_server --model-dir DIR` LOADS a trained model and serves it (no retraining) — serving
    factored into a `template<class Model> serve(...)` (MERA or flat); the inline-train path stays as a
    no-arg fallback. So the GPU trainer is one unit, the CPU httplib server another (sidesteps the
    clang/CUDA httplib link entirely). Validated: train on GPU → serve the dir on native CPU, /health +
    generate correct.
  - **`train_state.json`** sidecar for HONEST resume: `{step, best_nelbo, evals_since_best, code_sha,
    config_sha, updated_unix}`, written each checkpoint, rehydrated on resume (validated: resume picked
    up best_nelbo=5.12 / stalls=1 and continued the early-stop counter, not restarted). Read via
    simdjson on-demand (forward), written as a hand-built JSON string. Early-stop on `patience`.
  - **In-repo `models/`**: `--name foo` (a new `Data.name` consistent-layer field) → the dir becomes
    `models/foo_g<gitSHA>_c<configSHA>` (computed after vocab pin; deterministic tokenization ⇒ same
    name+config reproduces the same dir ⇒ resume-by-name). `models/README.md` documents the convention;
    `.gitattributes` routes `*.ckpt`/`*.opt` through Git LFS; `.gitignore` keeps generated model dirs out
    by default (force-add the keepers). Bug fixed: tokenizer was saved to a stale `tok_dir` captured
    before the `--name` dir override (config.json was correct, tokenizer/ went to the old default).
- `sum_squares` per-step cudaMalloc/cudaFree → persistent static scalar (commit 0f30f0e).
- Corrected: the `*_bench` multi-malloc functions are microbenchmarks, not the training hot path.
- MERA end-to-end generation works (templated sampler); generation-M2 vs flat measured (noisy, above).
- **Viz Phase A** — GenerationTrace + web scrubber (layers/timing/settled tok/s/tooltips), commit 64bc494.
- **Viz Phase B** — TRUE per-level activations: `MeraDenoiser::forward` optionally records per-slot RMS
  per level; `level_activations()` + an ADL `capture_levels()` hook feed the templated sampler (flat
  returns {} → viewer falls back to derived). Each Frame carries `level_rms` (one array per level, order
  = `meta.levels`); the hierarchy panel renders real activation magnitude (violet) when present. Validated:
  trace `level_rms` shapes `[128,32]` match `levels`; 52/52 tests green.
- **Viz Phase C** — interactive `ch32_viz_server`: trains a MERA once, serves `GET /health` +
  `POST /v1/generate_trace` (prompt + sampler/model knobs → fresh GenerationTrace) + the static viewer
  (mounts repo root, so one process, no Python). Viewer gained a live-generation panel + **A/B compare**
  (generate two settings, mark canvas divergence, iters/tok-s/divergent-count summary). Request parsing
  uses **simdjson on-demand directly** (forward, single-pass: `JsonFields` registers a typed handler per
  key, then walks the object once — no DOM/random-access), per the project JSON-read direction; trace
  serialization stays nlohmann (simdjson doesn't emit). `seq_len` snaps to a valid pyramid length so a
  knob never 500s. Validated on native CPU: valid/defaults/error(400) paths + prompt seeding all correct.
  Toolchain note: the clang/CUDA tree can't link httplib (clang++ GNU driver rejects bare `ws2_32.lib`
  positionals); the **native (Release CPU)** build links fine and trains the small viz model in seconds,
  so `--device cpu` is the supported path until that pre-existing toolchain issue is addressed.
- **Viz Phase D** — training-time trajectory (`ch32_viz_train`): snapshots a FIXED prompt+seed traced
  generation every K steps during training → `trajectory.json` ({meta, snapshots:[{step,nelbo,trace}]}),
  reusing `serialize_trace_json` per snapshot. Viewer detects `kind:"training_trajectory"` and shows a
  **training-step slider** above the iteration scrubber (only the model changes between steps, so it's a
  clean learning view). Validated on native CPU: 7 snapshots, NELBO 7.92→2.39 (learning ✓), step-1 text
  is unigram mush vs later structured prose; each snapshot trace carries `level_rms [128,32]`. Size note:
  per-frame `level_rms` dominates (3.4 MB for 7×65 frames) — if snapshot count grows, drop/subsample
  `level_rms` in trajectory mode. Open item: NELBO curve wobbles mid-run (t-averaged diffusion variance,
  cf. `diffusion-nelbo-ceiling-reframe`) — a per-t eval curve would read cleaner.
- **Viz Phase E** — the GIST readout: the coarsest (top) MERA level decoded into token space. `forward`
  optionally captures the top-level representation (the model’s compressed “plan”, `top_len` D-vectors);
  `gist_readout()` projects each slot through the tied embedding head (x·Eᵀ over real tokens) and returns
  the top-k tokens+scores per slot. An ADL `capture_gist()` hook feeds the templated sampler (flat → {});
  each Frame carries `gist`, serialized as `gist_tokens`/`gist_scores`. The viewer overlays the top-1
  token (amber) on the coarsest hierarchy row, full top-k on hover. **Finding (confirms the earlier scrub
  observation):** on a lightly-trained model EVERY gist slot reads ' ' (whitespace) by a wide margin —
  the coarse plan literally *is* whitespace, which is exactly why iteration 1 commits to spaces. The gist
  view makes that failure mode legible, and is the natural success metric for boundary-aware coarsening.
- **Boundary-aware coarsening (`--mera-gated-pool`, BuildTime A/B)** — the rigid mean-pool (every `c`
  rows averaged equally; a block can straddle a sentence/word boundary and blur across it) gets a
  content-weighted alternative: each `c`-block is combined by a softmax over a learned per-row score
  (`pool_proj_`, a (1,D) projection), so a block spanning a boundary can down-weight the wrong side — a
  soft, fixed-shape approximation of the “unbalanced tree”. `pool_proj_` is **zero-initialised**, so a
  gated model starts **bit-identical** to mean-pool (unit-tested) and only learns to deviate — a clean
  A/B. It’s a distinct architecture (own `config_sha` ⇒ own `models/` dir); `mera_gated_pool` rides the
  consistent config layer (run_config + config.json), so resume/serve reconstruct it. Open: run the
  matched A/B (gated vs mean) on TinyStories and read the gist — does the coarse plan stop collapsing to
  whitespace / align to word boundaries? (The gist view above is the readout for that experiment.)

- **Tokenizer quality pass + truecasing (full verification run, 2026-06-23).** Fixed a chain of word_level
  issues that fragmented the vocab: (1) LEADING quotes welded to words ('And, ''What's → 1000s of rare
  variants) — apostrophe now joins only inside a letter run; (2) multi-byte accents split words (piñata →
  pi+ñ+ata) — is_letter now accepts Latin-1/Extended letters; (3) the corpus typo concern measured at
  ~0.02% (negligible; a blind corrector would corrupt real words/names — dropped, hand-cleaned instead).
  Then **truecasing** (--truecase): lowercase each word's lemma + a <|cap|>/<|up|> marker, so Need/need/NEED
  and the high-frequency the/The collapse to one lemma id with de-fragmented statistics; round-trip is
  guaranteed by construction; PURE tokenizer transform (denoiser/loss/sampler untouched). Full corpus:
  vocab 11929 → 10314 (−13.5%), tokens +6.7% (the capitalised-word +1-marker cost). **Verification run**
  (MERA, seq128/win16/c4 → levels 128·32·8, B8, cuda-native): early-stopped at step 314090, **best NELBO
  1.514 @ step 224350** (patience-10 deadband saved it TWICE — it made its biggest gains at stalls 9, the
  classic diffusion sticky-basin/plateau-break dynamic). Served the best checkpoint and generated: output
  is **coherent TinyStories prose AND case restores correctly** ("named Lily", "She had", "The sun was
  shining", "One day", "They liked") — the model learned to place the case markers. Truecasing design
  VALIDATED end-to-end. Residual quality: small-model repetition ("the sun and the sun and…"), expected at
  this scale. Backlog (docs/factored_tokens_research.md): inflection as parallel factors, compositional
  stems as subwords, quote-glyph normalization, typo→nearest-id input retrieval.
