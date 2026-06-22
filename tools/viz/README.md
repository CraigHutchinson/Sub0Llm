# sub0llm Viz — diffusion "thought process" scrubber (Phase C)

Replay a MERA/flat diffusion generation **iteration-by-iteration**: watch the canvas snap into focus,
see per-position confidence/entropy, the MERA **level hierarchy** (true per-level activations), **timing**,
and a **settled tok/s** metric. With the live server (Phase C) you can also **set a prompt + sampler/model
knobs and generate on the fly**, and **A/B compare** two settings (same prompt+seed) to see how a parameter
reshapes the trajectory. Tooltips for terms/acronyms come from [`/ACRONYMS.json`](../../ACRONYMS.json).

See the design: [`diffusion/VISUALIZATION_DESIGN.md`](../../diffusion/VISUALIZATION_DESIGN.md).

## Two ways to run

### A) Live server — prompt box + param A/B (Phase C, recommended)

One process trains a small MERA once, then serves both the API and the viewer. **No Python needed.**

```bash
# Release CPU build links cleanly everywhere and trains the small model in seconds:
cmake --build build-native --target ch32_viz_server
./build-native/diffusion/chapters/ch32_tree_predictor/ch32_viz_server.exe --device cpu --port 8080
#   --device cuda  trains on GPU (needs a CUDA build; the clang/CUDA tree currently can't link
#                  httplib — use --device cpu on the native build until that toolchain issue is fixed)
#   --steps S  --seq_len N (also the trained MAX length)  --coarsen c  --window w  --embed_dim D
```

Then open **http://localhost:8080/tools/viz/** — the page auto-detects the server (same origin). Use the
**live generation** panel: type a prompt, set the knobs, click **generate → A**, change a knob, click
**generate → B**, then tick **mark A↔B divergence** to see exactly where the two trajectories differ on
the canvas. The compare line reports iters / settled tok/s / divergent-token count for A vs B.

### B) Static trace file — offline scrubber (Phase A/B)

**1. Build the trace generator** (CUDA build; needs the MSVC env, see project CLAUDE.md):

```bash
cmake --build build-cuda --target ch32_viz_gen
```

**2. Generate a traced passage** → writes `tools/viz/trace.json` and prints a self-check:

```bash
# from the repo root
./build-cuda/diffusion/chapters/ch32_tree_predictor/ch32_viz_gen.exe \
  --model mera --seq_len 128 --steps 2000 --paragraphs 600
# options: --model {mera|flat}  --seq_len N  --coarsen c  --window w  --steps S
#          --temp_x100 90  --out tools/viz/trace.json
```

The self-check at the end reports: frames written, final masked count (should be 0), the level pyramid,
settle time, **settled tok/s**, and the final decoded text. `STATUS: OK` means the trace is valid.

**3. View it.** A browser can't `fetch` a local file over `file://`, so either:

```bash
# (a) serve the repo root, then open the page (auto-loads ./trace.json + ./ACRONYMS.json tooltips)
python -m http.server 8000        # any static server works; this is just the simplest
#   -> open  http://localhost:8000/tools/viz/
```

or **(b)** open `tools/viz/index.html` directly and click **“trace”** to load `tools/viz/trace.json`
with the file picker (tooltips need the served route).

### C) Training trajectory — watch it learn (Phase D)

Snapshot a fixed-prompt+seed traced generation every K steps during training, so you can scrub a
**training-step** axis and watch the same prompt's denoising sharpen as the model learns:

```bash
cmake --build build-native --target ch32_viz_train
./build-native/diffusion/chapters/ch32_tree_predictor/ch32_viz_train.exe \
  --device cpu --steps 3000 --snap-every 300 --seq_len 128 --prompt "once there was a"
#   -> writes tools/viz/trajectory.json (self-check prints per-snapshot nelbo + text)
```

Serve the repo root and open the page; it auto-loads `trajectory.json` and shows a **training step**
slider above the iteration scrubber. Because the prompt and seed are fixed, only the model changes
between steps — early snapshots are noisy unigram mush, later ones structured prose, and the per-step
**NELBO** is shown as the model improves.

## What you see

- **Scrub bar / play** — drag across refine iterations; the canvas fills in commit order.
- **Canvas** — each position: settled tokens solid; *tentative* (still-masked) guesses italic with a
  red→green **confidence** heatmap; green border = committed this iter; amber = remasked. Hover a cell
  for pos / confidence / entropy / token / predicted.
- **Hierarchy** — the MERA levels (fine→coarse). For a MERA trace each slot is colored by the level’s
  **true per-level activation RMS** (violet, Phase B) — the model’s actual internal representation at
  that resolution; for flat traces (no hierarchy) it falls back to the derived settled-fraction (green).
  Note the coarse tiling is **rigid** (every `c` consecutive tokens pool to one slot), so a coarse block
  can straddle a sentence boundary — see ch32 NOTES (boundary-aware coarsening is the open lever).
- **Metrics** — % settled, committed-this-iter, iter time (ms), **settled tok/s** (running + overall),
  total settle time.

## Quick validity check (no browser)

```bash
python -c "import json;d=json.load(open('tools/viz/trace.json'));print(d['meta']['n_frames'],'frames; levels',d['meta']['levels'])"
```
