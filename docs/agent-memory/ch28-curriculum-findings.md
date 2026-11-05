---
name: ch28-curriculum-findings
description: Ch28 diffusion curriculum A/B results — frontier-point + token-gated auto mode wins; window-edge recall deficit measured
metadata: 
  node_type: memory
  type: project
  originSessionId: 9f29e15e-c786-4872-a953-a7472888b707
---

Ch28 denoiser curriculum sweep (2026-06-12, char-corpus, T=24, 337K params) settled three things:

1. **Concentration beats dilution**: training mass AT the noise ceiling ("point") beats uniform
   band sampling at every eval noise level. A step's learning signal is its masked-token count
   (5% noise ≈ 1 token, 50% ≈ 12), so band sampling wastes half the steps on weak signal.
2. **Winner = `auto` mode** (default in ch28 main.cpp): frontier-point sampling + per-noise-bucket
   loss EMAs judged in MASKED TOKENS (min_tokens=1500/decision → dwell ∝ 1/noise, token-linearized)
   + self-termination when the target ceiling's bucket plateaus. Self-stopped at 203,860 steps;
   recall 91.7/88.0/79.6/60.2/36.4% @ 10/25/40/60/80% noise = 59.7% overall (best of 9 variants).
   Buckets compared only against their own history — "loss rose because noise rose" cannot read
   as regression (fixes the oscillation the single-EMA controller had).
3. **Window-edge recall deficit**: first/last positions recover at ~40/37% vs 61.6% interior
   (one-sided context, ~4-position ramp at each edge). For Ch30/31 this motivates overlap or
   committed-block KV context (DiffusionGemma's block-AR provides exactly the missing left context).

**How to apply:** Ch29's formal objective should keep frontier/token-weighted noise sampling in
mind vs plain U[0,1] t-sampling; cite these numbers when choosing. Modes still selectable:
`ch28_diffusion_intro [auto|point|linear|mix|adaptive] [ceiling]`. See [[ch28-diffusion-track-plan]].
