# Ch32 DESIGN REVIEW 3 — the gist is a COMPUTE/CONTEXT primitive, not a within-window accuracy fix

> **Retrospective.** P2 framed the gist as a *topic-drift / accuracy* fix and the 2b kill-test measured
> its effect on per-token NLL *inside a single 64-token window* — where we found it is largely
> **redundant with bidirectional self-attention** (real gist beats a capacity-matched shuffled control
> by only ~2%, and does not net-beat baseline; [`2B_RESULTS.md`](2B_RESULTS.md)). That result is
> correct — and it tested the WRONG value proposition. The gist's real role is as a **coarsening
> operator** that enables **coarse-to-fine, parallel, sub-window generation**: it extends the effective
> context window and cuts compute, and its accuracy job is *cross-window* coherence, which a
> single-window experiment structurally cannot see. This review re-derives the gist on
> runtime/efficiency/context grounds and folds it back into the hierarchy plan.

## 1. The miss

The within-window redundancy is real: in a bidirectional denoiser **every position already attends to
every visible token**, so a gist pooled from the visible tokens of *that same window* hands the model
nothing attention can't already extract. We correctly measured ~0 net gain. But we then nearly
concluded "gist conditioning is weak." The error: a 64-token window is *below* the scale at which a
gist pays off. A coarse plan is worthless when the fine model already sees everything; it is decisive
when the fine model **cannot** see everything — i.e. when the full context N exceeds one window's
affordable attention span. 2b never created that regime (N = w = 64), so it could only see the
redundant component.

**Corrected claim:** the 2b within-window redundancy is *evidence for* this reframe, not against the
gist — it shows the gist must be judged at N ≫ w, by compute/context/x-window-coherence, not by
single-window NLL.

## 2. The reframe — coarse-to-fine parallel generation

Decompose generation of a length-N canvas into:

1. **Coarse pass** — build a low-resolution plan `G` over the FULL width N: downsample (pool every `c`
   tokens, or pool content words only) to `Nc = N/c` coarse slots and run a few denoiser/attention
   layers. `G` carries the long-range skeleton (topic, entities, structure) at `O((N/c)²)` cost.
2. **Fine pass** — partition the canvas into `M = N/w` sub-windows of width `w`. Denoise each
   sub-window with **full local attention (`O(w²)`) conditioned on the shared coarse `G`**. The
   sub-windows are **independent given `G`** → embarrassingly parallel.

The gist (P2) is exactly the coarse→fine conditioning channel of step 2; pooling content words is one
coarsening. This is the single-level case of P3's MERA (disentangle = local fine attention; coarsen =
the gist) — **P2 and P3 are the same operator at different depths.** (We had them as separate phases;
they are one mechanism.)

## 3. Runtime & efficiency (the math)

Let attention dominate (the `O(N²)` term). Per diffusion iteration, L layers, dim D:

| scheme | compute (work) | attention memory | parallelism |
|--------|----------------|------------------|-------------|
| **flat** | `O(L · N² · D)` | `O(N²)` | within one N×N attention |
| **coarse+fine** | `O(L · (N/c)² · D)` + `O(L · N · w · D)` | `O(w²)` per window | **M = N/w independent windows** |

Work speedup vs flat: `N² / ( N²/c² + N·w ) = 1 / ( 1/c² + w/N )`.

Worked example — N = 1024, window w = 64, coarsen c = 16:
- fine term `N·w = 65,536`; coarse term `(N/c)² = 4,096`; flat `N² = 1,048,576`.
- **work ≈ 15× less**; attention memory per window `w²/N² = 1/256` → **256× smaller** working set.
- the fine pass is `M = 16` independent windows → on `P` workers the fine wall-clock divides by `min(M,P)`.

On our hardware both axes are already in place:
- **GPU** — the denoiser's **batched forward already does block-diagonal attention over B windows in one
  call** (`forward(x, B, T)`, [`denoiser.cpp`](../../src/denoiser.cpp)). The M sub-windows ARE that
  batch dimension; one batched GPU pass denoises all of them, each conditioned on a broadcast `G`
  (`GistDenoiser` already broadcasts a per-window gist — make it the shared coarse `G`). Diffusion is
  GPU-favorable on our 8 GB box ([[diffusiongemma-reference-run]], [[cuda-first-class-iteration-time]]),
  and the batch-of-windows structure is *more* GPU-parallel than flat decode.
- **CPU** — per-window data-parallel execution already exists ([[cpu-threading-model]], the ch29
  per-window trainer); each P/E core denoises a sub-window. Heterogeneous split applies
  ([[heterogeneous-worker-split-followup]]): give P-cores more/larger windows.

So the coarse-to-fine generator is **largely assembling existing pieces** (batched forward + a shared
coarse gist), not new infrastructure.

## 4. Context-window extension

The coarse `G` injects whole-canvas information into each sub-window at `O(1)` conditioning cost, so a
sub-window of width `w` "sees" context N via `G` while paying only `O(w²)` attention. **Effective
context = N; per-window cost = w².** This is how the gist *extends* the usable context window under a
fixed memory/compute budget — complementary to the Ch25 long-context levers (KV cache, sliding window,
RoPE-NTK): those make a *single* long stream cheaper; the gist lets you *not run* the single long
stream at all, replacing `O(N²)` with `O(N·w)` + a coarse plan. For diffusion specifically (no
left-to-right KV reuse) this is the more natural scaling route.

## 5. Accuracy implications

- **Multi-scale match (M3).** Our corpora are power-law / critical ([`M3_RESULTS.md`](M3_RESULTS.md));
  most mutual information is short-range with a thin long-range tail. Local full attention captures the
  short range exactly; the coarse `G` carries the thin long-range tail. So the *expected* accuracy loss
  vs flat is small **iff `G` retains the long-range predictive bits** — which is precisely what the IB
  pooling objective (2c) is for. The gist's accuracy job is now well-posed: minimize the
  flat−hierarchical gap at large N, not beat flat at N = w (it can't and needn't).
- **Boundary seams (the real risk).** Independent parallel windows can disagree at edges — and we have
  *measured* this: Ch28 found window **edges recover ~40% vs ~62% interior** ([[ch28-curriculum-findings]]).
  Mitigations, cheap: (a) **halo/overlap** — widen windows by `h`, generate, discard the overlap
  (`w→w+2h`, modest cost); (b) **edge conditioning** — each window additionally sees its neighbors'
  committed boundary tokens or their gists; (c) a final **low-frequency global refinement** sweep over
  a coarsened full canvas to smooth seams. 4b's training-free *spread* commit already cut looping
  ~25-30% ([`4B_RESULTS.md`]) — evidence that coarse-first anchoring helps coherence at decode time.
- **The 2b redundancy is the floor, not the ceiling.** Within a window the gist ≈ attention (≤2% gain);
  *across* windows it is the ONLY long-range channel (attention doesn't cross window boundaries). So the
  gist's marginal value should *grow with N/w* — the opposite of what a single-window test shows.

## 6. The corrected metric & experiment plan

Judge the gist as a **compute/context primitive**, not by single-window NLL:

| axis | metric | baseline |
|------|--------|----------|
| **runtime** | tok/s and wall-clock at fixed N, vs flat; scaling with P workers | flat denoiser at same N |
| **memory** | peak attention working set; max N that fits 8 GB | flat `O(N²)` |
| **context** | max usable N at fixed budget | flat (OOM/too-slow ceiling) |
| **accuracy** | flat−hierarchical NLL gap **at large N** (where flat is the oracle, if it even fits); **M2 content-recurrence** of long generations; seam rate at window edges | flat at the largest N flat can still run |

**Decisive next experiment (replaces the "is 2b a pass" question):** a coarse-to-fine generator A/B at
N ≫ w — e.g. N = 512–1024, w = 64, c = 8–16:
1. **flat** denoiser at N (oracle accuracy, ceiling cost) — where it still fits/runs.
2. **coarse+fine** with a shared coarse gist, M parallel windows.
Report (a) tok/s + memory (expect the ~15× work / 256× memory wins), (b) the accuracy gap (NLL and M2
recurrence) — the gist passes if the gap is small while the compute/context win is large. The 2c IB
objective and §5 boundary fixes are the levers to *shrink the gap*; this benchmark is how we score them.

## 7. What this changes in the build plan

- **Re-scope P2's gate.** 2b's single-window NLL kill-test was the wrong gate; keep it only as the
  "within-window redundancy" datum. The real P2 gate is §6's coarse-to-fine **efficiency/context
  benchmark** with a bounded accuracy gap. Update BUILD_PLAN Phase 2 accordingly.
- **Merge P2↔P3 conceptually.** Gist (P2) = one-level coarsening; MERA (P3) = stacked log-depth
  coarsening. Build the single-level coarse-to-fine generator first (it reuses the batched forward),
  measure the compute/accuracy frontier, then stack only if the frontier warrants the depth.
- **2c (IB-pooling) gets a clear objective:** maximize long-range predictive info retained in `G` so the
  flat−hierarchical gap at large N is minimized — measured by §6, not single-window NLL.
- **Reuses, not new builds:** batched block-diagonal forward (GPU parallel windows), `GistDenoiser`
  broadcast conditioning (make `G` shared+coarse), cpu_topology per-window workers, the M2 metric (now
  measured on long generations), 4b spread/anchoring and Ch28 boundary findings for seams.

**Bottom line:** the gist is a divide-and-conquer primitive that buys ~order-of-magnitude compute and
memory headroom and extended context, at a *controllable* accuracy cost — and the within-window
redundancy we measured is exactly why it must be evaluated at scale, across windows, on runtime +
context + gap, not on single-window NLL.

---

## 8. Verbosity / time as a first-class slider (future extension)

The coarse-to-fine split also exposes a **user-facing quality/time/length control** that a flat model
cannot offer cleanly, because it separates *what to say* (the gist `G` — invariant meaning) from *how
much surface to spend saying it* (the fine realization). Two **orthogonal** axes fall out:

1. **Refinement depth — iterations `K`.** Time↑, local quality↑, *length fixed*. This already exists as
   sampler knobs (`max_iters`, `conf_threshold`, `entropy_bound`, `min_commit_frac` in
   [`sampler.hpp`](../../include/sub0diff/nn/sampler.hpp)). Diffusion refines in place, so more passes =
   more polish — but with a ceiling: on weak models iterative refinement can *lose* to one-step
   (error compounding, [[ch30-iterative-refinement-precondition]]), so depth must be gated on model
   strength.
2. **Realization length — verbosity `N` per gist.** Words↑, filler↑, clarity↑ (to a point), time↑ (but
   *parallel* across windows). **This is what the gist newly enables:** fix `G`, then infill it into a
   *terse* short canvas or a *verbose* long one — same meaning, different elaboration. Without a plan,
   "generate longer" just drifts; with `G` held fixed, length controls only the verbosity of realizing
   a fixed meaning (the classic plan→surface-realization NLG split, now native to the diffusion
   hierarchy, and natural because diffusion picks its canvas length up front).

**Why orthogonal:** you can ask for terse-but-polished (low `N`, high `K`) or verbose-but-rough (high
`N`, low `K`). A flat AR model conflates the two (more tokens = more compute = the only knob).

### Two forms
- **Global slider (v1).** One control `0..1` → `(K, N/Nc)`: low = terse/fast/rough, high =
  verbose/slow/clear. A user dial, or an API param like `max_tokens` but *meaning-preserving* (it
  changes elaboration, not content).
- **Per-window dynamic (the extension you flagged).** `G` already scores each coarse slot's content; so
  **allocate length and iterations per fine window by content density** — content-rich slots (many
  content words, high masked entropy) get more tokens + more passes; filler/function regions get fewer.
  This is per-window *adaptive computation time* (Graves ACT), a natural fit because the windows are
  already independent and the sampler already has per-window entropy/confidence stopping. Connects to
  the LoopedGPT `forward_k` runtime budget (Ch17) and the thinking-budget idea (Ch16). Implementation
  note: variable per-window length means ragged windows → pad to the batch max with masked padding (the
  block-diagonal forward assumes equal `T`), or bucket windows by length.

### Risks / where it breaks
- **Verbosity ≠ quality monotonically.** Too much length → filler hallucination, dilution, and the M2
  *looping/repetition* failure (we already have the metric). There is an information-density sweet spot;
  past it, clarity falls. **M2 content-recurrence + distinct-n are exactly the guardrails** to find the
  knee.
- **Depth ceiling** (above): the `K` axis has negative returns on weak checkpoints; re-measure as models
  strengthen.

### How to test it (once HierDenoiser has a generator)
Sweep `(K, N)` for a *fixed* gist and plot the **verbosity–clarity–time surface**: tokens, wall-time,
M2 content-recurrence + distinct-n (clarity/looping), and info density (content tokens / total). The
deliverable is the knee of that surface and a default slider mapping. This needs HierDenoiser
generation first (the sampler is currently `Denoiser`-only) — a near-term build once the 2c/2d accuracy
gap is acceptable. Until then this is a recorded design axis, not a claim.

**Why it matters:** it turns the hierarchy's compute decomposition into a *product* knob — a
meaning-preserving terseness/verbosity/latency dial — that is only coherent *because* the gist holds the
content fixed while length and depth vary. It is the user-facing payoff of separating plan from
realization.
