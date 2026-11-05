---
name: ch29-pause-state-and-data-matched-decision
description: Where the Ch29/Ch30 thread was paused (2026-06-17) + the smaller-model-vs-more-data decision frame to resume on
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

**CORRECTION 2026-06-19:** the "data is the coherence ceiling" claim here is SUPERSEDED — char-level diffusion is coherent on the same corpus; BPE-512 was the issue. See [[char-level-tokenization-lever]].

Paused 2026-06-17 to resume the Stage-4 CUDA work first (see [[specialization-backend-roadmap]]), then come back to THIS (Ch30 + model quality).

**Run state:** the founded+curriculum W=16 run was STOPPED at **k=36/63, step 318860** — checkpoint dir `D:/tmp/ch29_curric_founded` (has `config.json`, `run_config.json`, `tokenizer/`, `step_000318860.ckpt`+`.opt`+`train_state.txt`). Honest-resume intact (could continue with bare `--ckpt-dir`), but the high-k climb past ~t=0.4 is net-negative per-level ([[diffusion-loss-weighting-and-level-cadence]]), so stopping there cost little.

**Ch30 check:** `ch30_iterative_sampling --model-dir D:/tmp/ch29_curric_founded` LOADS + runs the full iterative pipeline fine (V=514, T=64) — no format breakage — but generation is **word-salad** (data-starved/underfit; great 1-token infill, poor free generation). Enough to exercise Ch30's iterative-vs-one-step lesson, not for coherent prose.

**Decision frame (user Q: "to solve data-starved, just train a smaller model?"):** half-right.
- A data-matched SMALLER model fixes the MEMORIZATION symptom → honest clean-generalizing metrics (eval tracks train, real recall, clean curriculum signal), cheaper. For the diffusion-mechanics chapters + a well-behaved Ch30, this is the right call.
- It does NOT buy coherent generation — the quality ceiling is DATA quantity/diversity, not params (the minimal-case proved capacity already fits the data; the bottleneck is generalizing from too little). Lever for coherence = MORE DATA (full 7222-para corpus via GPT-2 tokenizer / external text), not fewer params. Aligns with [[ch29-model-quality-founded-dims]] ("data + gradient quality binding, not capacity").
- Chinchilla sizing: ~2M tokens → ~75–100K params (≈ D64/L2–3), ~60× smaller than founded. (Diffusion's multi-noise corruption is a mild augmenter — tolerates a bit more, not orders of magnitude.)
- **Pending, not yet run:** a data-matched A/B (`--embed-dim 64 --n-layers 3 --d-ff 256`, founded proportions, same corpus) to show the eval/train gap closing + the honest coherence ceiling. Build on the debug tree at a quiet point (avoid colliding with the active Stage-4 working tree).

**Also pending:** Ch30 config-module rework (`load_model_dir` → read `run_config.json` via the RunConfig module + `config.json` fallback; drop the nlohmann dup) — captured in [[simdjson-migration-backlog]].
