# Ch32 — Build-Out Plan: Testable Increments

> Turns the design ([`README`](README.md) vision, [`DESIGN_REVIEW`](DESIGN_REVIEW.md) synthesis,
> [`DESIGN_REVIEW_2`](DESIGN_REVIEW_2.md) out-of-box) into **building blocks** that are each
> independently buildable and **falsifiable** — every increment has a test that can *kill* the design
> cheaply, and is measured against baselines we already have (char/word, Shakespeare/TinyStories).

## ⏯ STATUS / RESUME HERE (updated 2026-06-20) — hierarchy track is PAUSED mid-build
The premise and mechanism are validated; the actual P-phase build hasn't started. Pick up at **P1**.

**Done:**
- **Premise (M3): VALIDATED** — language is power-law/critical on our corpora (TinyStories α=1.88,
  Shakespeare α=1.79; Markov-1 control decays exponentially). See [`M3_RESULTS.md`](M3_RESULTS.md). The
  flat denoiser (exponential decay) structurally can't match it ⇒ the hierarchy is justified.
- **4b (gist-field sampler probe): DONE** — `--commit-order spread` (training-free) cuts within-canvas
  looping ~22–30% and raises distinct-n ~10% on word-TinyStories. See [`4B_RESULTS.md`](4B_RESULTS.md). A
  *crude untrained* coarse-anchor already helps ⇒ strong signal that a *trained* gist (P2) helps more.
- All design docs committed (README, DESIGN_REVIEW, DESIGN_REVIEW_2, BUILD_PLAN, M3/4B results).

**Progress:**
- **Phase 0 M1 (OOV-cliff) DONE** (commit 6928738) — `eval/oov_cliff.hpp`, `--oov_cliff`. Measured
  on word-TinyStories: **NLL_common 1.91 vs NLL_rare 15.44 → 8.08× cliff** (rare types predicted
  ~8× worse, *above* uniform — actively mispredicted). The word-level OOV weakness is real and large
  → P1 is justified. See [`M1_RESULTS.md`](M1_RESULTS.md). This is the **gate for 1c** (ratio→~1).
- **P1 steps 1a+1b DONE** (commit d3e4302) — `CharComposer`+`CharDecoder` codec
  ([`char_codec.hpp`](../../include/sub0diff/nn/char_codec.hpp)); round-trips spellings >90% + order-
  sensitive (anagrams diverge), `[char_codec]` test. See [`P1_RESULTS.md`](P1_RESULTS.md). Key finding:
  the non-AR decoder needs LEARNED POSITION QUERIES (RoPE alone → 35%, +queries → >90%).

- **P1 1c model stack BUILT** (commits 2e6ea15 + compose_vocab + loss-generalise) — `CodecDenoiser`
  (`E = lookup + alpha·compose_vocab(spellings)` for input AND the weight-tied LM head; alpha init 0 =
  plain word-level baseline), `compose_vocab` whole-vocab primitive, and `diffusion_loss` generalised
  on the model type. All tested in isolation (compose parity, trains + alpha activates). See
  [`P1_RESULTS.md`](P1_RESULTS.md).

- **P1 1c OOV A/B RAN — the naive codec FAILS the kill-test** (commit pending, runner `ch32_oov_ab`).
  Step-budget-matched baseline vs CodecDenoiser on word-TinyStories: cliff **6.50× → 8.08× (WORSE)**;
  in-vocab NLL −1.9% (slightly better). The shared composer is optimised by a common-word-dominated
  loss (~14:1), so it helps common words and the additive gate leaves rare words *more* confusable.
  This is a falsification, not a pass. See [`1C_RESULTS.md`](1C_RESULTS.md).
- **P1 1c follow-up (1) rare-weighted 2×2 RAN — rare-weighting helps but does NOT rescue the codec**
  (runner `ch32_oov_ab`, now full 2×2; `tok_weight` wired into `batched_diffusion_loss`). Rare-aware
  loss lowers the cliff on *both* models (baseline 6.36→**4.85×**, codec 8.10→5.66×), but under the
  identical objective the plain baseline still beats the codec. Culprit isolated: the **additive
  gate** retains the noisy rare lookup. See [`1C_RESULTS.md`](1C_RESULTS.md) §Follow-up (1).
- **P1 1c follow-up (2)+(3) convex+pretrain RAN — char-composition FALSIFIED at the PREMISE level.**
  With a near-perfect spelling autoencoder (composer pretrained to recon-CE **0.026**), *replacing*
  the rare lookup with the composed vector makes rare words **much worse** (NLL_rare 23.1 vs baseline
  12.1; cliff **7.68× vs 4.72×**). Mechanism isolated: the composed vector is a faithful *spelling*
  encoding, the embedding is *weight-tied to the LM head*, so spelling-similar words ("cat"/"cap")
  collide at the output. **Spelling geometry ≠ prediction geometry.** Three integrations all lose to
  a plain rare-weighted baseline. See [`1C_RESULTS.md`](1C_RESULTS.md) §Follow-up (2)+(3).

**P1 verdict (closed):** the OOV cliff is real (M1) and **reducible by the rare-weighted objective
alone** (best arm of all, ~4.7×) — adopt that. **Char-composition as an OOV *embedding* mechanism is
the wrong tool and does NOT enter P2/P3** (spelling≠prediction geometry under a tied head). Any future
sub-word/OOV info must enter as an *input-side feature the model re-embeds* (concatenated, not tied,
not substituted into the output projection) — a different design with its own kill-test, outside the
P1 codec. **Do not build P2 on char-composition.**

- **Phase-0 M2 (topic-drift) DONE + BASELINED** — `eval/topic_drift.hpp`, runner `ch32_m2_drift`,
  `[topic_drift]` test. Validated coherence signal = **content recurrence** (entity reuse ≥2×). On
  word-TinyStories: corpus 0.121 is **5.7× above** the unigram-chance floor 0.021 (real structure),
  and the **flat model sits AT the floor (0.021)** — zero entity-persistence. distinct-n ≈ corpus
  (not looping). Methodological catch: window-rearrangement controls are useless on lexically-
  homogeneous TinyStories; the valid control is a *distribution-matched* unigram null. See
  [`M2_RESULTS.md`](M2_RESULTS.md). **P2 gate: lift model recurrence 0.021 → 0.121 without looping.**

- **P2 2b (gist conditioning) RAN + REFRAMED** — `GistDenoiser` (visible-content-word gist, learned
  proj init 0), runner `ch32_gist_ab` with a capacity+init-matched **shuffled-gist control**. 3 seeds:
  real gist beats shuffled in 3/3 (~2% NLL) ⇒ real signal, NOT capacity — but does NOT net-beat a flat
  baseline. Cause: **in a bidirectional denoiser a gist of VISIBLE tokens is redundant with
  self-attention over them** ([`2B_RESULTS.md`](2B_RESULTS.md)). **RETROSPECTIVE
  ([`DESIGN_REVIEW_3.md`](DESIGN_REVIEW_3.md)):** the single-window NLL kill-test tested the WRONG thing
  — the gist is a **coarsening / compute-decomposition primitive** (coarse plan → parallel fine
  sub-windows): ~15× less work, ~256× less attention memory, extended context, at a controllable
  cross-window accuracy cost. The within-window redundancy is the FLOOR (consistent with the reframe),
  not a verdict. **P2/P3 are the same coarsen operator at different depths.**

**Resume order after 1c:** **P2** (gist conditioning: content-word `is_content` table + IB-pooling +
feudal training) → **P3** (MERA log-depth, gated on the M3 gap). Plus ungated side probe 4a (holographic
capacity numerics).

**Key change since pause — GPU training is now WIRED** (Stage 4 Phase 7, [[cuda-first-class-iteration-time]],
commit e1064ba): `ch29 --device cuda` trains the Denoiser end-to-end on ONE GPU stream (GpuTrainer;
batched_diffusion_loss + Adam + clip + checkpoint all on/around CUDA). Verified on TinyStories
(NELBO 4.85→4.06, clean exit). **The reason to pause P1 on CPU is GONE — the P-phases can now iterate
fast on GPU** (`--device cuda`). One CPU-only tail remains: the post-training recall sweep falls back
to CPU (logits/argmax host-deref); fine for now (run-once), GPU recall probe is a LOW follow-up.

**Baselines banked for the A/Bs:** char-TinyStories (`/d/tmp/ch29_tinystories`, NELBO 1.12), word-TinyStories
(`/d/tmp/ch29_word_tiny`, NELBO 1.77), char/word/BPE-512 Shakespeare. Decode default fixed (temp 0.9,
min_commit 0.03) so samples reflect real model quality. Analysis corpus = TinyStories
([[tinystories-default-analysis-corpus]]).

## Principles
1. **Each increment ships a measurable result**, not just code.
2. **Falsification-first:** every block states *"kills the design if…"*. Cheapest killers first.
3. **Reuse-first:** the word-level `Denoiser`, tokenizers, ch30 sampler, recall-probe, corpora already
   exist. New code is leaf modules + losses + metrics, not a rewrite.
4. **Baselines are banked:** char-TinyStories@30k, word-TinyStories@~20k, char/word-Shakespeare,
   BPE-512 — so increments are A/B'd, not judged in a vacuum.
5. **Gate:** the *full build* (P1–P3) stays PARKED behind the flat-diffusion training research. But
   **Phase 0 + the §4 free probes are UNGATED** — they de-risk the biggest claims now.

---

## Phase 0 — Instrumentation (the testbench). **UNGATED — do these first, no architecture needed**
You cannot test P1–P3 without these three metrics. All are eval-only tools over existing models.

| # | metric (building block) | what it measures | pass bar later | run now? |
|---|---|---|---|---|
| **M1** | **OOV-cliff** — hold out rarest X% of word types as "OOV-at-test"; report `NLL_oov / NLL_invocab` | does word-level fall off a cliff on unseen words? | P1 drives ratio → ~1 | yes (quantifies the word-level weakness) |
| **M2** | **topic-drift** — entity/content-word persistence + repetition rate over a generated passage | does coherence decay across a passage? | P2 lowers it vs P1 | yes (baseline the drift we keep seeing) |
| **M3** | **correlation decay** — MI (or token-prediction-improvement proxy) vs distance `d`=1..seq; fit log-log (power-law) vs log-linear (exp) | the criticality claim (Review II §1) | P3 moves model curve toward corpus power-law | **yes — the key de-risk** |

**M3 is the single most valuable ungated experiment.** Measure the correlation-decay curve of (a) the
**corpus** (Shakespeare + TinyStories) and (b) our **trained char/word models**. The whole MERA/criticality
reframe predicts: corpus = power-law, flat model = exponential (falls below corpus at large `d`). If that
gap is real → the hierarchy is justified by data. **If the corpus is NOT power-law at our scale, or the
flat model already matches it → the criticality motivation collapses and we should NOT build the MERA
hierarchy.** One eval tool, decides the most ambitious claim. Build: `tools/corr_decay/` (reuse the
recall-probe's corpus loading + the model forward).

*Phase-0 exit:* three metrics implemented + a baseline table over existing models. Then P1+ become A/Bs.

---

## Phase 1 — Char-composition codec (the OOV fix). **First real build**
Folds level C into embed/decode so word-level diffusion is OOV-proof (DESIGN_REVIEW §6).

| step | building block | test | kills design if… |
|---|---|---|---|
| 1a | `CharComposer` (Conv1d over a word's char-embeds → word vector) | autoencoder: compose→decode a word round-trips | can't reconstruct frequent words |
| 1b | `CharDecoder` (word state → char logits) | round-trip + spell held-out OOV words | can't spell unseen words from state |
| 1c | wire codec around the word `Denoiser` (compose in, decode out), lookup/composed **gated blend** | **M1**: OOV ratio→~1 AND in-vocab NLL ≥ plain word-level (no regression); **grammar speed** ≈ word-level (~2–4k steps on TinyStories) | OOV cliff persists, OR in-vocab regresses, OR grammar slows to char-level pace |

**P1 is the headline increment:** word-level grammar speed *plus* char-level OOV robustness, in one model.
*Exit:* M1 cliff removed, grammar speed retained. ~1 new module pair + a gate; testable in isolation (1a/1b)
before integration (1c).

---

## Phase 2 — Gist as a COMPUTE/CONTEXT primitive (re-scoped). **Builds on P1**
The gist is a **coarsening operator** for coarse-to-fine, parallel sub-window generation — NOT a
within-window accuracy fix (DESIGN_REVIEW_3, after the 2b retrospective). Judge it by runtime +
context + a bounded cross-window accuracy gap, not single-window NLL.

| step | building block | test | kills design if… |
|---|---|---|---|
| 2a | `is_content` table (frequency split) | **DONE** — `content_type_mask`; M2 uses it | — |
| 2b | `GistDenoiser` (content gist, learned proj) + shuffled-gist control | **DONE/REFRAMED** — real signal vs control but within-window-redundant; see DESIGN_REVIEW_3 | (re-scoped: single-window NLL was the wrong gate) |
| **2e** | **coarse-to-fine generator**: coarse plan `G` over full N → `M=N/w` parallel fine windows conditioned on `G` (reuse batched block-diagonal forward + shared broadcast gist) | **the real gate** — tok/s & memory vs flat at N≫w (expect ~15× work / ~256× attn-mem), and small flat−hierarchical NLL / M2-recurrence gap | the accuracy gap at large N is large AND not closable by 2c / boundary fixes |
| 2c | **IB-pooling**: train `G` to retain long-range predictive info | shrinks the flat−hierarchical gap (2e) vs mean-pool `G` | IB `G` no better than mean-pool on the 2e gap |
| 2d | boundary/seam handling (halo overlap, edge conditioning, global refine sweep) | window-edge recovery ≈ interior (cf. Ch28 40% vs 62%); seam rate ↓ | seams can't be closed without erasing the compute win |

*Exit:* the coarse-to-fine generator hits an order-of-magnitude compute/memory/context win at N≫w with
a small, 2c/2d-shrinkable accuracy gap vs flat — i.e. the gist earns its place as a scaling primitive.

---

## Phase 3 — MERA log-depth hierarchy (the criticality fix). **Builds on P2; the ambitious one**
Only attempt if **M3 (Phase 0)** confirmed the corpus is power-law and the flat model isn't.

| step | building block | test | kills design if… |
|---|---|---|---|
| 3a | (gate) confirm M3 gap on the P2 model | flat/shallow model curve is exponential vs corpus power-law | no gap → skip P3 entirely |
| 3b | `MeraLayer` = `disentangle()` (local attn) + `coarsen()` (pool), stack `log2(seq)` deep | **M3**: model correlation curve moves toward corpus power-law | curve doesn't budge → hierarchy doesn't help criticality |
| 3c | coherence-at-distance eval (long passages) | long-range coherence ↑ vs P2 | no coherence gain despite curve change |

*Exit:* M3 curve shifted + measurable long-range coherence gain — the design's deepest claim, validated or
falsified on one curve.

---

## Phase 4 — High-upside side probes (parallel, cheap, partly UNGATED)

| # | probe | test | cost / gate |
|---|---|---|---|
| 4a | **Holographic/VSA** (Review II §2): `HyperVector` bind/bundle/unbind | **capacity test first** (no training): at our `dim`, how many bound levels recover cleanly? then a tiny diffuse-in-HRR-space run | UNGATED numerics; build only if capacity passes |
| 4b | **CA / gist-gradient sampler tweak** (Review II §4): condition each position on a slowly-varying gist gradient + persistent local commit | **M2** drift on the *current* model — no training | **UNGATED, free, run now** on word-TinyStories |

---

## Sequencing & dependency graph
```
P0 (M1,M2,M3) ── ungated, FIRST ──┐
   │  M3 gates P3                  │
   ▼                              ▼
  P1 (OOV codec) ─→ P2 (gist) ─→ P3 (MERA)        4b (CA tweak) ── ungated, anytime
   gate M1          gate M2       gate M3,3b       4a (HRR cap)  ── ungated numerics
```
- **Do now (ungated, ~days):** M1, M2, **M3 (the criticality decision)**, 4b (free sampler test), 4a
  capacity numerics. These either *validate the design's premises on real data* or *kill the riskiest
  parts before any build*.
- **Then (gated by the diffusion-training research):** P1 → P2 → P3, each an A/B against banked baselines
  with a falsifiable exit.

## What each increment de-risks (one line each)
- **M3:** is the hierarchy justified at all? (criticality real on our data?)
- **P1:** does char-composition buy OOV robustness *without* losing word-level's speed? (the core trade)
- **P2:** can a cheap content-word gist actually cut topic-drift? (the remaining quality gap)
- **P3:** does log-depth structure change the *correlation physics*, not just the loss?
- **4a/4b:** are the exotic representations/samplers worth their risk, tested cheaply first?

## Effort/risk snapshot
| block | new code | risk | gate |
|---|---|---|---|
| P0 metrics | 3 eval tools | low | ungated |
| P1 codec | 2 leaf modules + gate | low-med | research |
| P2 gist | 1 module + 2 losses | med (objective design) | research |
| P3 MERA | new layer + stack plumbing | high (the ambitious bet) | research + M3 |
| 4a HRR | HyperVector type + probe | high upside / high risk | ungated numerics |
| 4b CA | sampler tweak only | low | ungated, now |
