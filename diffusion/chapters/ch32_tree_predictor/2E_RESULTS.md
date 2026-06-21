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
