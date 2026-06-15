# Chapter 31 — The Optimizer Sandbox

**Question.** Ch29's recall ceiling tracked the gradient's *within-level consistency* almost 1:1
(exact-count masking raised consistency 0.31→0.51, and recall ~15%→19%; @10%-noise 0.30→0.42).
That pointed the finger at the optimizer: a diffusion step is a noisy average over near-orthogonal
noise levels, and how an optimizer *digests* that noise might set the ceiling. We had also blamed
Muon's repeated failure (ch29: full-coverage NELBO 5.68, word-START **0.2%**) on "orthogonalization
amplifying the high-variance diffusion gradient." This chapter tests both claims in isolation.

**Method.** Strip the LLM away; keep only the gradient pathology in a problem with a known optimum,
and run the **production** `Adam` / `AdamW` / `Muon` (the exact code Ch29 ships) on it.

- Parameter `W ∈ R^{m×n}` — a matrix, so Muon orthogonalizes it (its real regime). Optimum `W=W*`.
- `T` noise levels, each owning a **disjoint column block** → per-level gradients are exactly
  orthogonal across levels (the probe's ≈0 cross-level cosine) and sum to the true gradient.
- Within-level noise `σ·R` sets the **consistency knob**, swept to the empirical {1.0,0.9,0.7,0.5,0.3}.
- `shared-t` (one level/step) vs `independent-t` (B workers, own levels) = Ch29's `--shared-t` vs baseline.
- Controls: **depth-2** (`Y=W₂·W₁`, non-convex + layer-coupled, Muon on both layers) and an
  **LR-sensitivity** sweep (the headline tables report best-over-LR; real Ch29 ran a single Muon LR).

## Findings

1. **The noise pathology does NOT break Muon.** Across every consistency level Muon tracks Adam to
   within ~15% (muon/adam 0.73–1.16); at zero noise Adam edges it (orthogonalization is unneeded on
   a well-conditioned problem). No blow-up, no divergence.
2. **Depth-2 coupling does NOT break Muon either** (0.72–1.16× Adam). Non-convexity + two coupled
   layers orthogonalized simultaneously is still fine.
3. **Both optimizers are similarly LR-sensitive** (best→worst over the grid: Adam 0.12→0.94, Muon
   0.11→0.84). Muon's Ch29 default (lr 0.02) is its *best* on the toy — the LR choice wasn't the bug.
4. **The consistency-collapse is real, optimizer-independent, and faithfully reproduced**: as `W→W*`
   the signal shrinks and within-level consistency falls (e.g. 0.50→0.07, matching the real probe's
   "0.31 init → 0.026 converged"). The final residual rises monotonically as `c0` falls
   (~0.05 at c0=1.0 → ~0.19 at c0=0.3) **for both Adam and Muon equally**.

## Conclusions

- **The stored explanation is falsified.** "Orthogonalization amplifies the high-variance diffusion
  gradient" is *not* supported: in isolation Muon is robust to the noise, the cross-level
  orthogonality, and shallow coupling. Ch29's catastrophic Muon result must come from full-transformer
  specifics the sandbox omits — 6-layer coupling, RMSNorm/softmax/residual signal-propagation, the
  embedding/output AdamW-group tuning, or the lack of warmup. **Next real test:** a *fair* Muon re-run
  on Ch29 — warmup + a small matrix-LR sweep — before closing the book on it.
- **The optimizer is not the crux; gradient consistency is.** Adam ≈ Muon at every point — swapping
  optimizers does not move the wall. What moves it is the gradient's within-level consistency, which
  bounds *both* optimizers identically. We already bought +4pt recall by raising consistency
  0.31→0.51 (exact-count masking). The sandbox says the lever for the *next* gain is the same:
  **raise consistency further** (more exact/even corruption, per-t variance reduction, larger
  effective batch per level), not a new optimizer. The consistency-collapse near convergence
  (0.5→0.07) is the ceiling to attack.

## Reproduce

```bash
cmake --build --preset native --target ch31_optimizer_sandbox
./build-native/bin/ch31_optimizer_sandbox        # ~seconds; prints all three experiments
```
