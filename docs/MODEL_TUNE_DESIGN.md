# Model-Dimension Tune — Design

Status: **proposal** (not yet implemented). Author note: grounded in the existing
`sub0::tune` search core, `sub0::memplan` footprint model, the model registry
(`meta.txt`), and the `report` stage. Validated assumptions come from the
tinystories dimension sweep recorded at the end of this doc.

---

## 1. Motivation

We already have an **inner tune**: `sub0llm tune` searches *compute* knobs (GPU
batch, TF32, attention-backward strategy, CPU threads) to maximise throughput for a
**fixed** model. It uses `sub0::tune::maximize` — a robust coordinate-descent search
with Explore / Refine / Confirm phases and a median-of-samples confirmation.

What we lack is an **outer tune**: a search over the *model architecture*
(`d_model`, `n_layers`, `n_heads`, `d_ff`, and optionally `seq_len` / `vocab`) to
**minimise validation loss** subject to memory and data-budget constraints. Today
this is done by hand: edit dims → reconfigure → rebuild → train → eyeball `report`.

The goal: make architecture search **the same kind of tune** — propose a point,
evaluate it, record the result, propose the next — but where one "evaluation" is a
*configure + build + probe-train* rather than a kernel timing.

Two empirical facts from this session make it tractable:

1. **Short-budget ranking predicts final ranking.** At ~1.5 epochs the eventual
   loser was already clearly behind (d128 @ 2.70 vs d96 @ 2.39). We do **not** need
   to train to the floor to compare dimensions.
2. **The naive heuristic was wrong; data settled it.** Chasing `head_dim 64` by
   dropping heads (d128 L5 H2 → val 2.44) lost to keeping heads and widening
   (d160 L5 H4 → val 1.46). An automated, data-driven search would have found this.

---

## 2. Constraint: dimensions are compile-time `constexpr`

`D_MODEL`, `N_LAYERS`, `N_HEADS`, `D_FF`, `SEQ_LEN`, `VOCAB` are baked into
`sub0_config.hpp` by the configurator. This is deliberate (cache locality, folded
hot loops, no per-op dim plumbing). Consequence: **each candidate architecture needs
a reconfigure + rebuild.** The outer search therefore spans process boundaries and is
naturally **orchestrated by CMake itself** (the project's super-build already builds the
tree per-dims; see §7), not a single in-process loop.

(The `TODO(dynamic-training-mode)` in `train_stage.cpp` proposes a `SUB0_TUNING`
build where dims become runtime parameters; that would later collapse the rebuilds
and allow a true in-process `sub0llm model-tune`. This design targets the
**baked-dims** world first and treats dynamic mode as Phase 3.)

---

## 3. Two-layer architecture

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  cmake -P cmake/ModelTune.cmake   (orchestrator — CMake-native loop)   │
 │  (driven by the `model-tune` custom target; no bespoke shell tooling) │
 │     loop:                                                              │
 │       next = sub0llm report --propose      # the "brain": history→point │
 │       break if CONVERGED                                              │
 │       cmake --preset native -D<next dims>  # reconfigure (isolated dir) │
 │       cmake --build --preset native        # rebuild                 │
 │       sub0llm tune --backend gpu           # INNER compute-tune       │
 │       cmake --build --preset native        # bake tuned batch        │
 │       sub0llm train --steps <probe budget> # PROBE-train → meta.txt   │
 │       sub0llm report <model>               # record metrics+samples   │
 └──────────────────────────────────────────────────────────────────────┘
                 │ reads/writes
                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  model registry (models/*/meta.txt, report.txt)                       │
 │  + models/model_tune_history.tsv  (append-only, one row per probe)    │
 └──────────────────────────────────────────────────────────────────────┘
```

- **Layer 1 — `cmake/ModelTune.cmake`** (run via `cmake -P`, exposed as a `model-tune`
  custom target): the orchestrator. It is the only piece that knows how to reconfigure +
  rebuild. **It is written in CMake, not a shell script** — see §7 for why the super-build
  the repo already has is the natural home and how the adaptive loop folds in. It reuses
  the existing `tune` / `train` / `report` subcommands unchanged, just sequenced.
- **Layer 2 — `report --propose` (the brain)**: a **pure, stateless** function of the
  recorded history → the next dimension point (or "converged"). The *history is the
  state*; there is no hidden optimiser state on disk. This mirrors the project's
  pure-`tune` philosophy and makes the proposer deterministic and unit-testable.

---

## 4. The pure core: `model_search.hpp`

A new engine-free header (sibling of `tune.hpp` / `coherence.hpp`), unit-tested in
isolation. It does **not** know about GPUs, builds, or files — only ladders, a
feasibility predicate, and an observed history.

```cpp
namespace sub0::model_search {

struct Dims { int d_model, n_layers, n_heads, d_ff, seq_len, vocab; };

struct Observation {            // one probe result (parsed from the registry)
    Dims   dims;
    double val_nelbo;           // primary metric at the probe budget
    double bits_per_byte;       // tokenization-normalised (cross-vocab comparable)
    long long tokens_seen;      // for equal-budget fairness checks
    bool   feasible = true;     // fit VRAM + param band when trained
};

// Per-axis ascending candidate ladders (caller fills d_model/layers; heads DERIVED).
struct Space {
    std::vector<int> d_model;   // e.g. {64,96,128,160,192,256,320,384,448,512}
    std::vector<int> n_layers;  // e.g. {2,3,4,5,6,8,10,12,16,20,24}
    // n_heads candidates are derived per d_model: divisors giving head_dim in [hd_lo,hd_hi]
    int hd_lo = 32, hd_hi = 128;
    // d_ff defaults to 4*d_model; seq_len/vocab fixed unless explicitly laddered.
};

// Hard constraints a candidate must satisfy to be PROPOSED at all.
struct Budget {
    int    vram_mb_usable;      // from sub0_cuda_free_vram_mb() - headroom (see tune)
    long long train_tokens;     // corpus train split
    double tok_per_param_lo = 5, tok_per_param_hi = 40;   // Chinchilla band
    int    min_batch_fit = 64;  // memplan::max_batch_for_vram must reach this
    int    param_floats_cap = 0;// optional absolute ceiling
};

enum class Phase { Explore, Refine, Confirm, Converged };

struct Proposal {
    bool   converged = false;
    Dims   next{};              // the next point to build+probe (if !converged)
    Dims   best{};              // best observed so far
    Phase  phase = Phase::Explore;
    std::string rationale;      // human line: which axis moved and why
};

// THE entry point. Pure: same history -> same proposal. No I/O.
Proposal propose_next(const Space&, const Budget&,
                      const std::vector<Observation>& history,
                      const Dims& current);

// Helpers (also unit-tested):
std::vector<int> head_candidates(int d_model, int hd_lo, int hd_hi);  // divisors in band
bool feasible(const Dims&, const Budget&);                            // VRAM + param band
}
```

### Search strategy (mirrors `tune::maximize`)

1. **Explore** — coarse coordinate descent. Starting from `current` (or the corpus
   default), evaluate one step along each axis (next d_model up, next n_layers,
   each feasible head count) and move to the best improver. Infeasible neighbours are
   skipped (objective = +inf), exactly like the compute tuner skips VRAM-over batches.
2. **Refine** — once an axis stops improving, halve the step (zoom the ladder around
   the leader) and re-probe the immediate neighbourhood.
3. **Confirm** — re-probe the top-K finalists at a **longer** probe budget (the
   architecture analogue of the median-of-samples confirmation), to defend against a
   short-probe ranking flip.
4. **Converged** — no feasible neighbour improves `val_nelbo` by > ε (e.g. 0.5%)
   across all axes, or the candidate budget is exhausted.

The proposer reconstructs which points are tried from `history` each call, so it is
fully resumable: kill the script, restart, it picks up from the recorded models.

### Head-count rule (the lesson, encoded)

`head_candidates(d_model)` returns divisors of `d_model` giving `head_dim ∈ [32,128]`.
The **default policy prefers keeping/raising head count when growing width** rather
than dropping heads to chase a larger `head_dim` — the d128-H2 regression proved that
trade harmful. Explore may still *probe* lower head counts when explicitly widening,
but the proposer never *recommends* dropping below the current best's head count
unless a probe shows it strictly better.

---

## 5. The objective: a fair probe-train

One "evaluation" = train the freshly-built model for a **fixed short budget** and read
validation loss. Fairness rules (so dims, not confounds, are compared):

- **Equal tokens seen, not equal steps.** Batch differs per dims (the inner tune sets
  it). Compare `val_nelbo` at a fixed `tokens_seen` target (e.g. 2 epochs' worth), not
  a fixed step count, because batch×steps = tokens.
- **Fixed seed** and fixed LR policy (`lr = base·sqrt(batch/8)` is already
  batch-aware, which is what we want).
- **Primary metric = `val_nelbo`**; **secondary = `bits_per_byte`** (normalised by
  `bytes/token`, so it is comparable even if `vocab` is on the ladder).
- **Plateau-aware probe:** reuse `coherence::trend_plateaued` — stop the probe early if
  it plateaus *before* the budget (cheap win), else stop at the budget. The probe never
  needs the true floor.
- **Confirm phase** re-trains the top-K to ~2× budget to break ties.

Probe budget is itself a knob (`--probe-epochs`, default ~2). The session data shows
~1.5–2 epochs already separated d96 / d128 / d160 cleanly.

---

## 6. `report --propose` (the brain) — CLI

Extend the existing `report` stage with a tuning mode:

```
sub0llm report [model] --propose
                       [--probe-epochs N]      # objective budget (default 2)
                       [--max-candidates K]    # stop after K probes (default 20)
                       [--d-model  lo:hi]      # restrict the d_model ladder
                       [--layers   lo:hi]
                       [--heads-band lo:hi]    # head_dim band (default 32:128)
                       [--json]                # machine-readable proposal
```

Behaviour:

1. Print the **human report** for `model` as today (architecture, quality, **samples**,
   per-knob verdicts) and append a row to `models/model_tune_history.tsv`.
2. Load the **comparable history**: every registered model with the *same*
   `corpus / seq_len / vocab`, varying `d_model / n_layers / n_heads / d_ff`, with its
   `best_val_nelbo`, `bits_per_byte`, `tokens_seen`, `status` (from `meta.txt` /
   `report.txt`).
3. Build `Space` + `Budget` (VRAM from `sub0_cuda_free_vram_mb()` − headroom; param band
   from `train_tokens`), call `model_search::propose_next(...)`.
4. Emit one of:
   - `PROPOSE d_model=160 n_layers=5 n_heads=4 d_ff=640   # widen (keep 4 heads); head_dim 24→40`
   - `CONVERGED d_model=160 n_layers=5 n_heads=4   best val_nelbo=1.461`
   plus the ready-to-run `cmake --preset native -D...` line (and `--json` form for the
   script).

This is the **same ask/tell shape** as the inner tuner, just at the model layer.

---

## 7. Orchestration in CMake (fold into the super-build — no bespoke tooling)

The repo already has a **super-build** (`cmake/SuperBuild.cmake`, `SUB0_SUPERBUILD`): a
recursive self-call that `ExternalProject_Add`s the *same source tree* once per
component with forwarded cache vars — and it already forwards **`SUB0_D_MODEL`,
`SUB0_N_LAYERS`, `SUB0_N_HEADS`, `SUB0_SEQ_LEN`** into each child, collecting artifacts
into a shared bin dir. In other words, *"build this tree N times with different model
dims, isolated, into one place"* is **already the mechanism**. So the orchestrator does
not need a separate language — it folds into CMake, giving a single cross-platform entry
point (`cmake --build --preset native --target model-tune`) instead of a `.ps1`.

There are two CMake-native shapes; the design uses **(B)** as the driver and offers
**(A)** for exhaustive sweeps.

### (A) Declarative super-build grid — for a STATIC sweep

A `SUB0_MODEL_TUNE_GRID` mode where the top-level enumerates a fixed ladder/grid and
emits one `ExternalProject_Add` per candidate (each a child build of this tree with its
own `-DSUB0_D_MODEL=… -DSUB0_N_HEADS=…`), whose **build/test steps run the probe-train +
report**. Children are serialised with `ExternalProject_Add_StepDependencies` because the
GPU is a single shared resource; a final aggregation target reads the history and prints
the ranking. This reuses the existing super-build verbatim and is the cleanest fit for a
*coordinate-sweep-as-grid* or an exhaustive ladder.

Limitation: **CMake's dependency graph is fixed at configure time**, so a declarative
super-build cannot *react* to a probe result to choose the next point — it can only run a
**pre-enumerated** set. That is fine for a grid, not for adaptive ask/tell.

### (B) CMake `-P` orchestration script — for the ADAPTIVE loop (default)

The adaptive coordinate descent needs "propose → build → probe → propose" where each
point depends on the previous *runtime* result. CMake's **script mode (`cmake -P`)** is a
full imperative interpreter with `while`/`foreach` and `execute_process`, so the loop
lives in CMake, not a shell:

```cmake
# cmake/ModelTune.cmake  —  run via:  cmake -P cmake/ModelTune.cmake
#                          or target:  cmake --build --preset native --target model-tune
set(EXE "${BIN}/sub0llm")
foreach(i RANGE ${MAX_CANDIDATES})
  # 1. ASK: the pure proposer reads the registry history -> next point (or CONVERGED).
  execute_process(COMMAND ${EXE} report --propose --probe-epochs ${PROBE_EPOCHS} --json
                  OUTPUT_VARIABLE J)
  string(JSON CONVERGED GET "${J}" converged)
  if(CONVERGED)
    break()
  endif()
  string(JSON D GET "${J}" next d_model)   # + n_layers / n_heads ...

  # 2. CONFIGURE + BUILD the proposed dims (baked constexpr -> rebuild). Isolated build
  #    dir per candidate mirrors the super-build's per-child isolation.
  execute_process(COMMAND ${CMAKE_COMMAND} --preset native
                          -DSUB0_D_MODEL=${D} -DSUB0_N_LAYERS=${L} -DSUB0_N_HEADS=${H})
  execute_process(COMMAND ${CMAKE_COMMAND} --build --preset native)

  # 3. INNER compute-tune for THESE dims, then bake it.
  execute_process(COMMAND ${EXE} tune --backend gpu)
  execute_process(COMMAND ${CMAKE_COMMAND} --build --preset native)

  # 4. TELL: probe-train to the budget (meta.txt records best val + tokens_seen).
  execute_process(COMMAND ${EXE} train --steps 0 --probe-epochs ${PROBE_EPOCHS})

  # 5. RECORD: report writes metrics + samples into the model dir and history.tsv.
  execute_process(COMMAND ${EXE} report --propose)   # also prints the human report
endforeach()
```

A thin `add_custom_target(model-tune COMMAND ${CMAKE_COMMAND} -P
${CMAKE_SOURCE_DIR}/cmake/ModelTune.cmake -D... )`, added only when `SUB0_MODEL_TUNE=ON`,
makes it a first-class build target. `string(JSON ...)` (CMake ≥ 3.19) parses the
proposer's `--json`, so no text munging.

### Why CMake, not a shell script

- **No extra tooling / language**: it is the same `cmake` already required to build;
  cross-platform (Windows/Linux) for free, unlike `.ps1`.
- **Reuses the super-build's proven dim-forwarding** and per-child isolation pattern.
- **Single entry point**: `--target model-tune` sits beside the normal build targets.
- The **adaptive intelligence stays out of CMake** (in the pure `report --propose`), so
  CMake is only the sequencer — which is exactly what `cmake -P` is good at, and avoids
  pushing search logic into a build system.

### What CMake should NOT do here

Do not encode the search math (ladders, coordinate descent, feasibility) in CMake — that
belongs in the pure, unit-tested `model_search.hpp` behind `report --propose`. CMake only
*drives* (configure/build/run/serialise). This keeps the brain testable and the orchestrator
a dumb, reliable loop.

> On Windows the rebuild relinks DLLs — the loop is strictly sequential, so nothing holds
> them during a rebuild. The inner `tune`/`train` run to completion before the next
> `cmake --build`, matching the manual workflow used to validate this design.

---

## 8. Data model / retrospective history

- **Source of truth**: the registry. `meta.txt` already carries
  `d_model/n_layers/n_heads/seq_len/vocab/steps/epochs/tokens_seen/best_val_nelbo/status`.
  Add `bits_per_byte` and `probe_epochs` (small additions to `ModelMeta`).
- **`report.txt`** (already saved per model this session) keeps the human metrics +
  samples for eyeballing.
- **`models/model_tune_history.tsv`** — append-only, one row per probe:
  `timestamp, git_sha, d_model, n_layers, n_heads, d_ff, params, tokens_seen,
   val_nelbo, bits_per_byte, status`. This is what `--propose` reads and what a human
  (or a notebook) reads to plot the search retrospectively **without rebuilding**.

---

## 9. Convergence & guardrails

- **Stop** when no feasible neighbour beats the best by > ε (default 0.5% rel
  `val_nelbo`), OR `--max-candidates` reached, OR all neighbours infeasible.
- **Feasibility** gates every proposal: `memplan::max_batch_for_vram(...) ≥ min_batch_fit`
  (won't propose a model that can't train without WDDM spill) and `tokens/param` inside
  the Chinchilla band (won't propose a model the corpus can't feed or that wastes data).
- **Head floor**: never *recommend* fewer heads than the current best (the d128 lesson),
  only *probe* them when explicitly widening.
- **Determinism**: fixed seed end-to-end; the proposer is pure.

---

## 10. Why this is "the same `tune`, one level up"

| aspect | inner `tune` (compute) | outer model-tune (architecture) |
|---|---|---|
| search core | `tune::maximize` coordinate descent | `model_search::propose_next` (same shape) |
| knobs | batch, TF32, attn-bwd, threads | d_model, n_layers, n_heads, d_ff |
| objective | measured tok/s (ms) | probe-train val_nelbo (minutes) |
| feasibility skip | VRAM-over batch → skip | VRAM/param-band → skip |
| robustness | median-of-samples confirm | re-probe top-K longer (confirm) |
| driver | in-process lambda | CMake `-P` loop (`model-tune` target) + stateless proposer |
| result | baked into `sub0_config.hpp` | baked dims + recorded in registry/history |

---

## 11. Phasing

- **P0 — pure proposer.** `include/sub0/model_search.hpp` + `tests/model_search_tests.cpp`
  (validate `head_candidates`, `feasible`, and `propose_next` over a synthetic history
  that reproduces the d96/d128/d160 ranking). No build/train wiring yet.
- **P1 — `report --propose`.** Wire the proposer to the registry + emit PROPOSE/CONVERGED
  + `--json`. Add `bits_per_byte`/`probe_epochs` to `ModelMeta` and the history TSV.
- **P2 — `cmake/ModelTune.cmake`** (adaptive `cmake -P` loop) + a `model-tune` custom
  target, plus `train --probe-epochs`. Optionally the declarative `SUB0_MODEL_TUNE_GRID`
  super-build mode for exhaustive sweeps. End-to-end on tinystories; produce the ranking
  table. No new language/tooling — it folds into the existing super-build.
- **P3 — confirm phase** (re-probe top-K longer) + head-utilisation early indicator
  (mean attention entropy + head similarity) folded into `report` to prune head counts
  without a full probe.
- **P4 (later) — dynamic-training-mode** (`SUB0_TUNING` runtime dims) collapses the
  rebuild, enabling an in-process `sub0llm model-tune` that calls `tune::maximize`
  directly with the probe-train as its objective.

---

## 12. Risks & open questions

- **Probe fidelity vs cost.** Too-short probes rank noisily; mitigated by equal-tokens
  comparison + the confirm phase. (Empirically 1.5–2 ep sufficed here, but harder
  corpora may need more.)
- **Rebuild dominates wall-clock.** ~30s rebuild + minutes/probe × ~axes×steps
  candidates. Coordinate descent keeps the candidate count low; dynamic mode (P4)
  removes the rebuild entirely.
- **Comparability scope.** `--propose` must restrict to one `corpus/seq/vocab` family;
  cross-family comparison is meaningless (different loss scales).
- **Metric choice.** `val_nelbo` for same-vocab search; `bits_per_byte` when `vocab` is
  on the ladder. Document which is primary in each run.

---

## Appendix — empirical anchor (tinystories, seq 256, this session)

| d_model | n_layers | n_heads | head_dim | params | converged val NELBO | ppl | bits/byte |
|--------:|---------:|--------:|---------:|-------:|--------------------:|----:|----------:|
| 96  | 5 | 4 | 24 | 0.98M | 1.985 | 7.28 | 1.37 |
| 128 | 5 | **2** | 64 | 1.55M | **2.44** ❌ | 11.5 | — |
| 160 | 5 | 4 | 40 | 2.24M | **1.461** ✅ | 4.39 | 1.018 |

Lesson encoded in the search: **head count beats head_dim here**; widen while keeping
heads. The model-tune would have found d160 H4 automatically and rejected d128 H2 after
a ~1.5-epoch probe.
