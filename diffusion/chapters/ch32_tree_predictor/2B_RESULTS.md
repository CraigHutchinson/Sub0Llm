# Ch32 P2 2b — gist conditioning: results (the feudal-signal kill-test)

> **⚠ Read with [`DESIGN_REVIEW_3.md`](DESIGN_REVIEW_3.md).** This experiment measured the gist's
> effect on per-token NLL *inside a single 64-token window* (N = w). That is the WRONG value
> proposition: the gist is a **coarsening / compute-decomposition primitive** (coarse plan → parallel
> fine sub-windows), and its accuracy job is *cross-window* coherence at N ≫ w. The within-window
> redundancy found here is *consistent with* that reframe (a coarse plan is redundant when the fine
> model already sees the whole window) — it is NOT a verdict that "gist conditioning is weak." The real
> P2 gate is the coarse-to-fine efficiency/context benchmark in DESIGN_REVIEW_3 §6.

**Status (within-window datum): the content-gist channel carries a REAL but TINY signal (real gist
beats its capacity-and-init-matched shuffled control in 3/3 seeds) and does NOT net-beat a flat
baseline — because in a bidirectional denoiser a gist pooled from VISIBLE tokens is largely REDUNDANT
with self-attention over those same tokens. This is the FLOOR of the gist's value (single window), not
its ceiling (cross-window, at scale). See DESIGN_REVIEW_3.**

Model: [`gist_denoiser.hpp`](../../include/sub0diff/nn/gist_denoiser.hpp) — `GistDenoiser` pools the
embeddings of the VISIBLE content words (input positions that are neither `[MASK]` nor function
types) into one vector `g` per window, projects it through a learned `W_g` (init **0** = identical to
the flat baseline at start), and adds it broadcast to every position. Runner:
[`gist_ab.cpp`](gist_ab.cpp) (`ch32_gist_ab`). Held-out masked NLL, bucketed content/function.

The control — **shuffled gist** (`GistDenoiser(..., shuffle_gist=true)`): same architecture, same
init (seed-matched), but each window's positions are routed to a DIFFERENT window's gist. Capacity is
held fixed; the gist↔window signal is destroyed. So **real-gist − shuffled-gist isolates the gist
SIGNAL from the extra `W_g` capacity** — the same discipline as M1's rarity split and M2's unigram
null.

## Result (word-TinyStories, 540 train paras → 1721 vocab, D=256 L=4 T=64, 2000 steps, 3 seeds)

| seed | baseline | gist | shuffled | gist−baseline | **gist−shuffled (signal)** |
|------|---------:|-----:|---------:|--------------:|---------------------------:|
| 7 | 2.570 | 2.574 | 2.685 | +0.2% | **−4.1%** |
| 8 | 2.537 | 2.546 | 2.558 | +0.4% | **−0.5%** |
| 9 | 2.481 | 2.500 | 2.533 | +0.7% | **−1.3%** |

(overall held-out masked NLL, nats; lower = better)

- **Real gist beats shuffled in 3/3 seeds** (−0.5…−4.1%, mean ≈ −2%) ⇒ the gist carries a genuine,
  capacity-controlled signal — it is NOT just the extra `W_g` parameters.
- **Real gist does NOT beat baseline** (+0.2…+0.7%, consistently a touch worse). The shuffled control
  is *worse* than baseline (wrong-topic conditioning actively hurts), and the real gist merely
  recovers most of the cost the gist pathway imposes — it does not exceed baseline.

## Why — the structural insight (bidirectional redundancy)

The first single-seed run showed −9.4% gist-vs-baseline; the 3-seed control proved that was mostly
**baseline variance** (the baseline alone ranges 2.48–2.57). The capacity-matched truth is a ~2%
signal that does not clear baseline. The reason:

**In a bidirectional denoiser every position already attends to every visible token.** A gist pooled
from the visible content words therefore hands the model information it can *already* extract via
self-attention — it is redundant, a minor computational shortcut at best (the ~2% over shuffled). The
ONLY content information attention lacks is the **masked tokens themselves**. So a gist that is (a)
leak-free and (b) available at inference (pooled from visible/committed tokens) is structurally
limited to that small redundant signal.

This is the key difference from the autoregressive *feudal* LM the design borrowed from: there a
manager's plan about the **future** is genuinely new information to a left-to-right worker. In
bidirectional diffusion there is no hidden "future" among the visible tokens — only the masked ones,
which a usable gist cannot encode without leaking.

## Next — the oracle upper bound (go/no-go for 2c/2d)

The decisive question: does a gist that DOES carry new info help? Test an **oracle gist** pooled from
the CLEAN content words (including the masked positions). If oracle-gist substantially beats baseline
(and its shuffled control), there is headroom — a *learned* gist (2c IB-pooling) that approximates the
oracle from visible context could capture some, and 2d feudal is worth building. If even the oracle
barely helps, gist conditioning in bidirectional diffusion is falsified and P2 should be re-scoped
(e.g. toward decode-time topic anchoring, cf. 4b spread, rather than train-time conditioning).
