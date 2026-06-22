# Ch32 — running notes (things found while getting the engine working)

> Append findings/gotchas/TODOs as we go. Memory-management work is **deferred to a later milestone**
> (see the `denoiser-toolbox-and-memory-milestone` memory); note issues here, don't fix them now.

## Open items / things to make work

- ~~MERA model is N-specific~~ **DONE** — MERA now takes `max_seq_len` and `forward(N)` rebuilds the
  pyramid per call, accepting ANY valid N ≤ max (blocks indexed by depth-from-finest; the top block
  handles whatever level falls ≤ w). Test covers N=64 + N=32 on one model. **Caveat to watch:** if
  TRAINED at a single fixed N, the top block sees one top-length; using a different N at inference shifts
  the top-length (mild train/use mismatch). True variable-N robustness needs MIXED-N training (future).
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

## Findings

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
