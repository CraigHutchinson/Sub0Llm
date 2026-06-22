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

- `sum_squares` per-step cudaMalloc/cudaFree → persistent static scalar (commit 0f30f0e).
- Corrected: the `*_bench` multi-malloc functions are microbenchmarks, not the training hot path.
- MERA end-to-end generation works (templated sampler); generation-M2 vs flat measured (noisy, above).
- **Viz Phase A** — GenerationTrace + web scrubber (layers/timing/settled tok/s/tooltips), commit 64bc494.
- **Viz Phase B** — TRUE per-level activations: `MeraDenoiser::forward` optionally records per-slot RMS
  per level; `level_activations()` + an ADL `capture_levels()` hook feed the templated sampler (flat
  returns {} → viewer falls back to derived). Each Frame carries `level_rms` (one array per level, order
  = `meta.levels`); the hierarchy panel renders real activation magnitude (violet) when present. Validated:
  trace `level_rms` shapes `[128,32]` match `levels`; 52/52 tests green.
