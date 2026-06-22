# Ch32 — running notes (things found while getting the engine working)

> Append findings/gotchas/TODOs as we go. Memory-management work is **deferred to a later milestone**
> (see the `denoiser-toolbox-and-memory-milestone` memory); note issues here, don't fix them now.

## Open items / things to make work

- **MERA model is N-specific** — the level pyramid (lens, per-level blocks) is baked in at construction
  from `N`. A real engine needs variable-N handling: either construct per length, pad to the next valid
  pyramid size, or make the level structure dynamic. Fine for fixed-length training/benchmarks today.
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

## Done

- `sum_squares` per-step cudaMalloc/cudaFree → persistent static scalar (commit 0f30f0e).
- Corrected: the `*_bench` multi-malloc functions are microbenchmarks, not the training hot path.
