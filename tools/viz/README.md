# sub0llm Viz — diffusion "thought process" scrubber (Phase B)

Replay a MERA/flat diffusion generation **iteration-by-iteration**: watch the canvas snap into focus,
see per-position confidence/entropy, the MERA **level hierarchy**, **timing**, and a **settled tok/s**
metric. Tooltips for terms/acronyms come from [`/ACRONYMS.json`](../../ACRONYMS.json).

See the design: [`diffusion/VISUALIZATION_DESIGN.md`](../../diffusion/VISUALIZATION_DESIGN.md).

## How to run

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
