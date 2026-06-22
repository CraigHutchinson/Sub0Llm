# Ch32 P2 2e — coarse-to-fine vs flat: the efficiency/context frontier

**Status: the compute/context win is VALIDATED and SCALES with N (10.7× fewer attention-ops, 64× less
attention memory, ~2× faster wall-clock at N=512, all growing with N). The accuracy gap is REAL and
currently LARGE (hier +29–41% masked NLL) but SHRINKS with N (41%→29%; content words 13%→7%). 2e
establishes the trade frontier the gist-as-coarsening reframe predicted (DESIGN_REVIEW_3); closing the
gap is the explicit job of 2c (IB-pooling) and 2d (seams).**

Model: [`hier_denoiser.hpp`](../../include/sub0diff/nn/hier_denoiser.hpp) — `HierDenoiser`: coarse
global pass over `N/c` mean-pooled slots (attention `O((N/c)²)`), broadcast back, then a fine pass of
`M = N/w` window-local blocks (block-diagonal `O(w²)` — the denoiser's existing batched forward).
Runner: [`hier_ab.cpp`](hier_ab.cpp) (`ch32_hier_ab`). Test: `[hier_denoiser]` (shape, finiteness,
trains). Pool/broadcast reuse the `compose_vocab` reshape + batched-matmul trick (`O(N·D)`, no B²
dense selector).

## Result (word-TinyStories, 540 train paras → 1721 vocab, D=256, depth 4 [flat] = 2 coarse + 2 fine [hier], 2000 steps, B=8)

| N | M=N/w | flat NLL | hier NLL | **gap** | content gap | **attn-ops ×** | wall-time | **attn-mem ×** |
|---|-------|---------:|---------:|--------:|------------:|---------------:|----------:|---------------:|
| 128 | 2 | 2.598 | 3.671 | +41.3% | +13.4% | 3.6× | 1.01× | 4× |
| 256 | 4 | 2.521 | 3.482 | +38.1% | +11.1% | 6.4× | 0.48× | 16× |
| 512 | 8 | 2.609 | 3.361 | +28.8% | +6.8%  | 10.7× | 0.59× | 64× |

(overall held-out masked NLL, nats; wall-time = hier ÷ flat; attn-ops/mem = flat ÷ hier)

## Reading it

- **Efficiency/context — VALIDATED, scales as predicted.** Attention work `O(N²) → O((N/c)²+N·w)`:
  3.6×→10.7× fewer ops as N grows; attention working set `O(N²)→O(w²)`: 4×→**64×** smaller; wall-clock
  ~**2× faster** at N≥256 (at N=128 the windows are too small for the GPU win to beat the pool/broadcast
  overhead — ties). The fine pass is the batch dim, so it parallelises across the M windows. This is the
  context-extension lever: a width-`w` window "sees" context N via the coarse plan at `O(w²)` cost.
- **Accuracy gap — real, large, but shrinking.** At full training the hier model is +29–41% NLL worse
  (the 40-step smoke test's "−1.6%" was an undertraining artifact — both models were barely trained).
  Crucially the gap **falls monotonically with N** (41→38→29%), and the **content-word gap falls faster**
  (13.4→11.1→6.8%) — i.e. the hierarchy's relative cost shrinks exactly where it matters (long sequences,
  topical words). Extrapolating, the curves head toward parity at larger N than we can train here.

## Why the gap is large (and what closes it → 2c/2d)

The naive `HierDenoiser` is deliberately minimal, and three things cost it accuracy — each a named
follow-up lever:
1. **Mean-pool coarse plan.** The global plan is an *untrained average* of token embeddings — a lossy
   summary that discards most long-range predictive structure. → **2c IB-pooling**: train the coarse
   representation to *retain the info predictive of the rest of the sequence*, not just average.
2. **No cross-window attention.** Fine windows communicate ONLY through the coarse plan; at N=128 that
   is 2 windows talking through 32 coarse slots — a hard bottleneck (hence the worst gap). As N grows
   the coarse plan is richer (128 slots at N=512) → gap shrinks. → richer/learned coarsening (2c) + a
   second coarsening level (P3 MERA) widen this channel.
3. **Coarse plan built from a 50%-masked input.** At eval noise 0.5 half the tokens are `[MASK]`, so the
   mean-pool coarse plan is half-garbage; flat attention can route around masks, mean-pool cannot. →
   a learned, mask-aware pooling (2c) recovers this.
4. **Window seams.** Fine windows are independent given the plan → edge tokens lack neighbour context
   (Ch28: edges recover ~40% vs ~62% interior, [[ch28-curriculum-findings]]). → **2d**: halo overlap /
   edge conditioning / a global refine sweep.

## 2c — mask-aware coarse pooling (RAN): closes the CONTENT gap, isolates seams as the rest

The first 2c lever — pool each coarse slot over its VISIBLE (non-`[MASK]`) tokens only, instead of a
uniform mean that averages in ~50% `[MASK]` garbage at noise 0.5 (`HierDenoiser(..., mask_aware=true)`,
free — pooling weights cost nothing). Same data/budget A/B, flat vs hier-uniform vs hier-mask-aware:

| N | flat | hier-unif | **hier-mask** | overall gap unif→mask | **content gap unif→mask** | function NLL (mask vs flat) |
|---|-----:|----------:|--------------:|----------------------:|--------------------------:|----------------------------:|
| 256 | 2.549 | 3.495 (+37.1%) | **3.295 (+29.3%)** | closes 21% | **+13.4% → +6.6%** | 2.008 vs 1.216 |
| 512 | 2.703 | 3.336 (+23.4%) | **3.262 (+20.7%)** | closes 12% | **+2.9% → +0.5%** | 1.944 vs 1.267 |

- **Content-word gap halves at N=256 and nearly vanishes at N=512 (+0.5%).** Once the coarse plan stops
  summarising `[MASK]` noise, it carries topic/content almost as well as full `O(N²)` attention — the
  coarsening hypothesis (long-range info compresses into a cheap plan) is confirmed for content.
- **The remaining gap is now almost ENTIRELY function words** (at N=512: content +0.5% but function
  +53%). Function words are *local grammar/glue*; what the `w=64` fine windows lack is cross-window flow
  at the **boundaries**. So 2c cleanly splits the gap: **content/long-range → solved by the coarse plan;
  function/local-grammar-across-seams → the remaining cost → 2d's job.**
- Compute win unchanged (6.4×/10.7× fewer ops, ~2× wall-clock) — mask-aware pooling is a free accuracy
  gain.

Next 2c+ levers (if needed beyond seams): a *learned* attention-pool / IB objective on the coarse plan;
this run shows even the trivial mask-aware mean nearly closes content, so the headroom there is small —
**2d (seams) is the higher-value next lever.**

## Generation M2 (RAN, 3 seeds): the hierarchy does NOT improve coherence — compute win only

The 2e/2c gates are NLL (prediction). The actual P2 goal is topic coherence in GENERATION (M2
content-recurrence). `ch32_hier_gen` trains flat + mask-aware hier at N=256, GENERATES 64 passages from
each via the templated sampler, and scores M2 vs corpus and the unigram-chance floor (recurrence gap
corpus−chance = 0.101).

| seed | flat-gen recurrence | hier-gen recurrence | distinct-2 flat / hier |
|------|--------------------:|--------------------:|-----------------------:|
| 7 | 0.045 (24% of gap) | 0.039 (18%) | 0.671 / 0.608 |
| 8 | 0.068 (47%) | 0.053 (33%) | 0.618 / 0.625 |
| 9 | 0.054 (33%) | 0.048 (27%) | 0.664 / 0.608 |

**Flat ≥ hier on content-recurrence in 3/3 seeds**, and hier is slightly *more* bigram-repetitive
(lower distinct-2) in 3/3. (A single earlier run showed hier 0.063 > flat 0.048 — a lucky generation
seed; the 3-seed check overturned it, the same way the 2b 3-seed check corrected its single run.)

**Verdict:** the coarse-to-fine hierarchy is a **compute/context primitive only**. It does NOT improve
generation coherence as a side effect — the shared coarse plan carries topic for *prediction* (2c
closed the content-NLL gap) but does not make *generation* reuse entities more than flat. Entity reuse
requires actively copying content words across distance, which neither model does well (~30% of the
gap) and the gist does not add. So the hierarchy's value is **efficiency/context at a small, 2c/2d-
shrinkable quality cost** — not a quality win. Do not claim coherence gains for it.

## Context-length ceiling (RAN): where flat dies, hier survives — and why hier walls too (→ P3)

`ch32_hier_ceiling` times one model at one seq-length per process (so an OOM can't corrupt the next
measurement) and a shell loop sweeps N. Small model (D=128, L=4, B=4), tok/s = B·N / s-per-step:

| N | flat | hier | hier ÷ flat |
|------|------------------------:|------------------------:|------------:|
| 512 | 29.6K tok/s | 59.6K tok/s | 2.0× |
| 1024 | 18.1K tok/s | 88.5K tok/s | 4.9× |
| 2048 | **66 tok/s (125 s/step)** | 86.6K tok/s | **~1300×** |
| 4096 | *crashed the box* (WDDM TDR / reboot from VRAM exhaustion) | 74.5K tok/s | — |
| 8192 | — | 1481 tok/s (22 s/step) | — |

- **flat practical ceiling ≈ N=1024.** At N=2048 it collapses to 66 tok/s — Windows WDDM oversubscribes
  VRAM (spills to system RAM) so the wall shows up as catastrophic *thrashing* (125 s/step), not a clean
  catchable OOM; at N=4096 it exhausted VRAM hard enough to take the machine down (TDR/reboot). On 8 GB
  this is the `O(N²)` attention-memory wall.
- **hier extends usable context ~4×** (to N≈4096 at 74.5K tok/s — the *same* throughput band as small N).
  At the N=2048 crossover hier is **~1300× higher throughput** than flat: the regime where the choice is
  "runs vs doesn't," not "faster vs slower." This is the context/compute headline, demonstrated.
- **But hier ALSO walls — and the reason is the P3 motivation.** The coarse pass is `O((N/c)²)`: it has
  its OWN `N²` term, merely scaled by `c²`. So single-level coarsening pushes the ceiling out by ~`c×`
  (c=8 → ~4× measured: flat dies ~2048, hier slows by ~8192) but does **not remove** it. **Removing it
  needs RECURSIVE log-depth coarsening — stack coarse levels so the top level is always small (P3
  MERA).** This sweep is the concrete, measured argument for building P3: a single gist level buys a
  ~`c×` context extension; only a `log(N)`-deep stack makes the coarse term cheap enough to scale freely.

## P3 — recursive MERA removes the COMPUTE wall (RAN); residual ceiling is memory

[`mera_denoiser.hpp`](../../include/sub0diff/nn/mera_denoiser.hpp) (`MeraDenoiser`) coarsens
RECURSIVELY (encode: disentangle + pool, up to a tiny top; decode: broadcast + skip + refine), so
total attention work is `Σ_k O((N/cᵏ)·w) = O(N·w)` — LINEAR, no level ever large. Same
`ch32_hier_ceiling` runner, `--model mera`. Full three-way ceiling (D=128, L≈4, B=4, c=8, w=64):

| N | flat | hier | **mera** |
|------|----------------:|---------------:|---------------:|
| 1024 | 18.1K tok/s | 88.5K | 51.5K |
| 2048 | 66 (dead) | 86.6K | 52.8K |
| 4096 | crashed box | 74.5K | 49.5K |
| 8192 | — | 1481 | **33.6K** |
| 16384 | — | — | 1822 (thrash) |
| 32768 | — | — | OOM |

- **The compute wall is GONE.** MERA holds ~50K tok/s *roughly constant* from N=1024→8192 — the
  linear-`O(N·w)` signature — where flat is long dead and hier's `O((N/c)²)` coarse pass has collapsed
  (1481 tok/s at 8192). **At N=8192 MERA is 22× faster than hier** and runs at lengths flat cannot
  approach. P3's core claim — recursive coarsening keeps every level small and removes the residual N²
  — is validated.
- **Crossover is ~N=4096.** Below it, hier is as fast or faster (its single coarse pass has less
  overhead than MERA's log-depth levels); above it, hier collapses and MERA stays flat. The recursive
  structure pays off exactly where it should — long sequences.
- **MERA's remaining wall is MEMORY, not compute.** N=16384 at 36 s/step is ~18× slower than the linear
  extrapolation (~2 s) = WDDM VRAM thrashing; OOM at 32768. The training autograd graph retains every
  level's activations (linear in N, but a large constant: ~7 blocks × intermediates), so on 8 GB it
  caps ~16384 — about 2× hier's ceiling. This is **separable engineering, not an architectural wall**:
  gradient checkpointing (recompute activations in backward instead of storing) buys unbounded *training*
  context, and *inference* (no autograd graph) scales much further already. The O(N·w) compute is the
  hard part and it is solved.

**P2→P3 progression (all measured):** flat `O(N²)` dies ~N=1024 → single-level hier `O((N/c)²)` extends
~4× then walls → recursive MERA `O(N·w)` holds constant throughput to ~8192 (22× over hier), memory-
bounded ~16384. Each phase's *measured* limitation motivated the next; P3 lands the linear-compute win.

### MERA also BEATS flat on accuracy (3 seeds) — not just compute

The 4-way NLL A/B (`ch32_hier_ab`, N=256, 2000 steps) — and a 3-seed repeat — show MERA is *more
accurate* than flat, not merely cheaper:

| seed | flat overall | MERA overall | MERA−flat | content | train: MERA vs flat |
|------|-------------:|-------------:|----------:|--------:|--------------------:|
| 7 | 2.494 | 2.452 | −1.7% | −4.4% | 80s vs 172s |
| 8 | 2.610 | 2.513 | −3.7% | −3.0% | 87s vs 177s |
| 9 | 2.587 | 2.496 | −3.5% | −3.7% | 81s vs 178s |

(single-level hier was +28–41% WORSE than flat; the recursive MERA is −2…−4% BETTER, and beats flat on
both content and function buckets.) Robust (3/3 seeds), and **not capacity** — at N=256 MERA has FEWER
blocks than flat (3 vs 4) yet wins, while training ~2.1× faster. This is the **M3 multi-scale hypothesis
vindicated in accuracy**: a U-Net/MERA processing the sequence at log(N) resolutions is a better
inductive bias for power-law language than flat single-resolution attention — better generalisation AND
cheaper. The single-level hier's accuracy loss came from its lossy single coarse pass + seams; the
recursive multi-level encode/skip/decode fixes that.

**Verdict — flat is strictly dominated.** MERA beats flat on accuracy (3/3), compute (~2×), and context
(linear vs N² wall); hier beats flat on speed/context but not accuracy. So **flat never wins on any
metric** — its role drops to the *reference/accuracy oracle* (educational baseline), not a deployment
option. Toolbox: **MERA = the default** (accuracy + scaling); **hier** only where small-N raw speed beats
accuracy (it has lower per-step overhead than MERA below ~N=4096); flat = reference.

## Frontier verdict & next

The gist-as-coarsening design is a **real efficiency/context primitive**: an order-of-magnitude compute
and memory win that grows with N, at an accuracy cost that *also* shrinks with N. It is not yet a free
win — at the N we can train, flat is ~30–40% better on NLL. The 2e benchmark is now the scoreboard for
2c/2d: **does a learned/IB coarse plan + seam handling shrink the gap toward parity while keeping the
compute win?** Two cheap, high-value variants to run next on `ch32_hier_ab`:
- **wall-clock-matched A/B** — hier trains ~2× faster, so at equal wall-clock it gets ~2× the steps;
  re-score the gap under a fixed *time* budget (not fixed steps), which is the deployment-relevant axis.
- **larger N / smaller w** — push N to where flat OOMs or stalls; report hier's max usable N (the
  context-extension headline) where flat cannot run at all.
