# Ch32 4b — Spread Commit-Order (gist-field sampler probe) Results

> BUILD_PLAN Phase-4b: a **training-free** test of the hierarchy's core claim — does committing a
> spatially **spread** coarse skeleton first (a "gist field" analog), instead of pure most-confident-first,
> reduce the repetition/topic-drift we see? Implemented as `SamplerConfig::CommitOrder::Spread` (greedy
> farthest-point in confidence order, then fill) + `ch30 --commit-order spread`. Measured locally
> (uncommitted script) on the word-TinyStories model (step 18k), ~170 canvases each, temp 0.5.

## Result — SPREAD HELPS (partial, free)

| metric | confidence (baseline) | spread | Δ |
|---|---|---|---|
| distinct-2 (↑ = less repetition) | 0.285 | **0.311** | +9% |
| distinct-3 | 0.487 | **0.544** | +12% |
| distinct-4 | 0.648 | **0.723** | +12% |
| within-canvas repeat-2 (↓ = less loop) | 0.204 | **0.155** | **−24%** |
| within-canvas repeat-3 | 0.105 | **0.074** | **−30%** |
| within-canvas repeat-4 | 0.050 | **0.039** | −22% |

Qualitatively: confidence loops ("The bird was happy … The bird was happy … The bird and"); spread is more
varied ("girl named Lily found a toy … wanted to play with her toys … was very curious and wanted to make a").

## Interpretation (the informative part)
Committing a spread coarse skeleton first **partially** fixes drift at DECODE time — a free ~25-30% cut in
looping — **but only partially** (spread still repeats some). That matches §13.8 exactly: a coarse anchoring
helps, but the model **fundamentally ignores distant context**, so a sampler trick can't fully fix what is a
training/representation limit. ⇒ two conclusions:
1. **Keep `--commit-order spread`** — a real, free quality lever (kept opt-in; not yet the default pending
   wider validation).
2. **Strong signal for P2 (gist conditioning):** if a *crude, untrained* spatial spread already cuts looping
   ~30%, a *trained* gist conditioner the model learns to actually condition on should help substantially
   more. The hierarchy premise gains a concrete, measured boost.

Next: P1 (char-composition codec, OOV) then P2 (gist conditioning) — now empirically motivated by both
M3 (corpus is power-law/critical) and 4b (coarse anchoring measurably reduces drift).
