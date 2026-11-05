---
name: optimizer-alternatives-backlog
description: Backlog — support AdamW + try Muon/modern optimizers; Muon aligns with the Ch29 optimization-ceiling lever
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

Backlog (user-requested 2026-06-14): the codebase optimizer is **plain Adam** (`include/sub0llm/nn/optimizer.hpp`: lr 1e-3, β 0.9/0.999, eps 1e-8, NO weight decay, constant LR). AdamW is the de-facto standard; **Muon** and other modern optimizers are worth supporting + trying.

**Priority read (mine):**
- **AdamW** (decoupled weight decay) — ~2-line add to the existing Adam. But it targets *generalization* (overfitting), and Ch29 measured the model *underfits* (train≈held-out), so low expected value for raising recall *here* — still worth having as the standard default.
- **Muon** (Jordan 2024: Newton-Schulz-orthogonalized momentum on 2D weight matrices, hybrid with AdamW for embeddings/scalars/norms) — HIGHER implementation cost (Newton-Schulz iteration + param routing by shape) but genuinely targets the *fit/optimization* ceiling. This aligns with [[ch29-model-quality-founded-dims]] **axis F** (the open "optimization fragility / real scaling blocker" lever — wider models get stuck in the unigram basin from a bad init; the recall peak at D=128 is an *optimization* peak, not proven a capacity peak). Muon famously improves small-transformer convergence (nanoGPT speedruns). So it's not just a nicety — it's a candidate to actually break the ~15%/~11% content-prediction ceiling. Worth a real A/B, not just backlog.
- Others to consider: Lion (cheap, sign-based), Sophia (2nd-order-ish).

When picked up: add as a new `Optimizer` alongside Adam (shared `step()`/`zero_grad()` interface), wire a `--optimizer adam|adamw|muon` flag into ch29 main, A/B on the fast proxy (--paragraphs 10000) comparing word-level + word-START recall and the wider-model trainability (does Muon rescue D=192 from the unigram basin?). [[verify-correctness-against-reference-before-perf]]

**DONE + RESULTS (2026-06-15):** Built it — `Optimizer` base interface, `Adam` extended with decoupled weight decay (= AdamW), `Muon` (Newton-Schulz orthogonalised momentum; tested exposed `newton_schulz_orthogonalize` kernel; embedding/head→AdamW routing via a `force_adamw` mask), `make_optimizer` factory, `--optimizer` flag + 4 unit tests. Real perf fix found en route: NS does ~855 TINY matmuls/step, each paying intra-op GEMM thread-pool fork/join overhead → disabling intra-op threading during NS = **13× Muon speedup** (general lesson: tiny matmuls must run serial). **Proxy A/B (V=512, 20k steps budget-matched, each at its sensible LR): AdamW WINS** — word-level 11.1% / START 11.1% / NELBO 4.39, beating Adam (9.7% / 10.0% / 4.50). **Muon LOSES** — naive all-2D collapsed (embedding orthogonalisation); even with embedding→AdamW it escapes the unigram basin ~3-4× SLOWER than Adam (NELBO 5.88 @ step 8500 vs Adam 5.67). WHY (through-line from the [[ch31-diffusion-optimization-sandbox-plan]] grad-probe): the diffusion gradient is HIGH-VARIANCE (within-level consistency ≈0.31) + noise-direction-rich; Muon drives ALL singular directions to magnitude 1 INCLUDING noise → amplifies it, whereas Adam/AdamW's 1/√v DAMPS noisy coords. So the lever is **variance reduction**, not a fancier optimizer. SHIPPABLE: make AdamW the default (+1.4pt). Muon parked (needs warmup / variance-reduced gradients).
