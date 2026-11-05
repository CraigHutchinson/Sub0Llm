---
name: diffusion-engine-visualization-plan
description: planned tool — interactively prompt the MERA diffusion engine and SCRUB the thought process (iteration timeline × MERA level hierarchy); param-adjustment comparison; verbosity slider
metadata: 
  node_type: memory
  type: project
  originSessionId: f6f73404-ddcc-4bc1-b640-2b89a8b5c475
---

User idea (2026-06): build an interactive **visualize + prompt** tool for the MERA diffusion engine —
scrub the model's "thought process" as a timeline during/after generation, see the **tiered (MERA-level)
internals** as it iterates, and watch how **parameter adjustment** reshapes the trajectory.

Why this engine fits: (1) diffusion generation is ALREADY a timeline — `refine_canvas` re-predicts the
whole canvas each iteration and commits confident positions, so text "snaps into focus" (each iteration =
a frame to scrub); (2) MERA is a spatial HIERARCHY (coarse plan → fine windows), a 2nd axis. So:
**time (iterations) × scale (levels).**

Plan (full doc: `diffusion/VISUALIZATION_DESIGN.md`):
- Core artifact = a **GenerationTrace** JSON emitted by the C++ engine: per-iteration frames {tokens,
  masked, committed-this-frame, confidence, entropy, commit_order, remasked} + optional per-level
  digests {len, gist proj2d, attn top-k}. Heavy tensors stay out; ship digests / re-run one forward for
  the inspected frame.
- Hooks: Phase A is nearly FREE — extend the existing `refine_canvas` `on_iter(canvas, iter)` callback to
  emit a Frame (the loop already computes confidence/entropy/commit/remask). Phase B = opt-in
  `ForwardTrace*` on `MeraDenoiser::forward` for per-level internals (null in hot path = zero overhead).
- Viewer = lightweight **web scrubber** in `tools/viz/` (HTML+JS — the no-Python rule is core-only; JS
  ok); served via a new `POST /v1/generate_trace` on `tools/server`. Scrub bar over iterations, canvas
  confidence heatmap, expandable MERA pyramid panel, prompt box + sampler/model knobs, side-by-side
  trace DIFF for parameter comparison (e.g. spread vs confidence commit-order — makes the 4b looping
  result visible; N-sweep = the [[verbosity-time-slider-from-gist]] dial with M2 guardrails live).
- Phasing: A timeline+scrubber → B level internals → C prompt+knobs+diff → D training-time trajectory
  replay across checkpoints. Each independently useful; A is the first build.

Status: DESIGNED, not built. Build Phase A first (GenerationTrace + on_iter Frame + tools/viz scrubber).
Relates to [[gist-is-compute-context-primitive]], [[verbosity-time-slider-from-gist]].
