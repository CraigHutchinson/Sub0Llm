# Diffusion engine — interactive visualization & prompting (design)

> Goal: prompt the MERA diffusion engine interactively and **scrub the "thought process"** — replay
> generation iteration-by-iteration and drill into the **tiered (MERA-level) internals**, as a
> comprehension aid and a way to see how parameter changes reshape the trajectory.

## 1. Why this engine is uniquely visualizable

- **Generation is already a TIMELINE.** Unlike autoregressive decode (one token at a time, left→right),
  the diffusion reverse process (`refine_canvas`) re-predicts the WHOLE canvas each iteration and commits
  only the confident positions, so the text emerges in confidence/spread order across the canvas. Each
  iteration is a frame; the sequence of frames is the "thought process" to scrub.
- **The model is a HIERARCHY (MERA).** Each forward pass coarsens N → N/c → … → a tiny top, then refines
  back down. So at any frame there is a *pyramid* of representations (coarse plan → fine windows) — a
  second axis to expand into. **Two axes: time (iterations) × scale (levels).**

## 2. The core artifact — a GenerationTrace (engine-side, C++)

Everything hangs off a structured, replayable trace emitted by the engine as JSON (heavy tensors stay
out; digests go in). One trace per generation:

```
GenerationTrace {
  meta:   { prompt, vocab, D, c, w, N, levels[], sampler{temp, conf_threshold, min_commit_frac,
            remask_threshold, commit_order, max_iters}, seed }
  frames: [ Frame, … ]                         // one per refine iteration (the timeline)
  final:  { text, n_iters, seconds }
}
Frame {
  iter
  tokens[N]            // decoded token (or id) at each position this iteration
  masked[N]            // still [MASK]? (0/1)
  committed[N]         // committed THIS frame? (0/1) — drives the "snap into focus" animation
  confidence[N]        // max prob per position (heatmap)
  entropy[N]           // per-position entropy (where the model is unsure)
  commit_order[]       // positions committed this frame, in order (confidence vs spread)
  remasked[]           // positions re-opened this frame (low-confidence remask)
  // OPTIONAL tiered internals (heavy → digest only, or on-demand for a selected frame):
  levels: [ { len, gist_digest{norm, mean_entropy, proj2d[len][2]}, attn_topk[] }, … ]
}
```

`proj2d` = a cheap 2-D projection (e.g. first 2 principal directions, or 2 fixed random projections) of
each level's per-slot vector, so the coarse plan is plottable without shipping D-dim tensors.

## 3. Instrumentation hooks (minimal engine changes)

- **Iteration timeline (Phase A) — almost free.** `refine_canvas` already has an `on_iter(canvas, iter)`
  callback and already computes per-position confidence + entropy + the commit/remask sets internally.
  Extend the callback to receive a `Frame` (or a richer struct) so the trace is captured with NO change
  to the sampling logic — just surface what the loop already has. The whole Phase-A timeline (tokens,
  masked, committed, confidence, entropy, commit order) needs only this.
- **Tiered internals (Phase B) — opt-in forward trace.** Add an optional `ForwardTrace*` (default null)
  to `MeraDenoiser::forward`; when present, record per-level `{len, gist_digest, attn_topk}` as the
  encode/decode passes run. Null in the hot path → zero overhead for normal generation. Capture full
  internals only for the FRAME the user is inspecting (re-run that one forward with the trace on),
  keeping the per-iteration trace light.

## 4. The viewer (UI)

Engine = C++ (emits JSON). Viewer = a lightweight **web scrubber** (single HTML+JS file in `tools/viz/`;
JS is fine — the no-Python rule is for the core library). It loads a trace and offers:

- **Scrub bar over iterations** — drag to replay the canvas snapping into focus; play/pause/step.
- **Canvas panel** — the N positions as a grid/strip; color = committed-this-frame / confidence heatmap;
  hover a position to see its predicted token + confidence + entropy over time.
- **Hierarchy panel** — expand a frame into the MERA pyramid: each level as a row (coarse at top, fine at
  bottom), `proj2d` scatter of its slots, attention top-k, info flow up (coarsen) / down (refine).
- **Prompt box + knobs** — set the prompt and the sampler/model params, (re)generate, get a new trace.
- **Acronym tooltips** — any term/acronym shown in the UI (MERA, NELBO, gist, conf_threshold, …) gets a
  hover tooltip with its expansion + definition, sourced from [`/ACRONYMS.json`](../ACRONYMS.json) (keyed
  by UPPERCASE token; `glossary.terms[token.toUpperCase()]`). That index is the project glossary
  (project terms + common LLM terminology + external refs) and is maintained alongside the code.

Serving: reuse `tools/server` (the OpenAI-compatible HTTP server) — add a `POST /v1/generate_trace`
endpoint returning the `GenerationTrace` JSON for a prompt+params; the web viewer is a static page that
calls it. No new server framework.

## 5. Parameter adjustment — the comparison view

The payoff the user asked for: **see how parameter changes reshape the thought process.**

- **Sampler knobs** (`temperature`, `conf_threshold`, `min_commit_frac`, `remask_threshold`,
  `commit_order` confidence↔spread, `max_iters`): re-generate and **diff two traces side-by-side** —
  same prompt+seed, two settings; scrub both in lockstep; highlight where the trajectories diverge
  (e.g. spread vs confidence commit order — the 4b looping result becomes *visible*).
- **Model/length knobs** (`N`, `c`, `w`): now that MERA is variable-N, vary N at a fixed gist and watch
  the **verbosity slider** (DESIGN_REVIEW_3 §8) — terse vs elaborated realization of the same plan, with
  M2 recurrence / distinct-n shown live as guardrails.
- **Training-time** (later, Phase D): snapshot traces at successive checkpoints to watch the engine
  *learn to denoise* — the same prompt's trajectory sharpening across training steps.

## 6. Phasing (build order, each independently useful)

| phase | deliverable | engine change | value | status |
|-------|-------------|---------------|-------|--------|
| **A** | iteration timeline JSON + web scrubber (canvas + confidence heatmap + commit order) | extend `refine_canvas` `on_iter` to emit a Frame | watch diffusion "snap into focus"; smallest, highest ratio | **done** (64bc494) |
| **B** | MERA per-level internals (true per-slot activation RMS per level) | opt-in slot-RMS capture on `MeraDenoiser::forward` + ADL `capture_levels` | see the tiered hierarchy think | **done** (4b7d24e) |
| **C** | prompt box + param knobs + A/B trace diff | `ch32_viz_server` `POST /v1/generate_trace` (simdjson on-demand reads) | parameter-adjustment comparison; verbosity slider | **done** |
| **D** | training-time trajectory replay across checkpoints | checkpoint-tagged traces | watch it learn | pending |

## 7. Data-volume notes (so it stays cheap)

- Per-iteration trace (tokens + masked + committed + confidence + entropy) ≈ a few × N floats × ~30
  iters — tiny; ship always.
- Full per-level activations (N×D × levels × iters) are LARGE — never store wholesale. Use `proj2d` +
  scalar digests for the always-on trace; re-run one forward with full `ForwardTrace` only for the frame
  being inspected.
- Keep the trace schema versioned (it is a contract between the C++ emitter and the JS viewer).

## 8. First step

Phase A: a `GenerationTrace` writer + the `on_iter`→Frame extension + a static `tools/viz/` HTML
scrubber, validated on a real MERA generation (we already have `ch32_hier_gen` producing samples). It is
a small, self-contained build that immediately makes the diffusion process legible — and everything
later (hierarchy, params, training) extends the same trace.
