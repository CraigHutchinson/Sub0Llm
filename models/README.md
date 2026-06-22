# models/ — trained checkpoints (in-repo, provenance-tagged)

Each trained model lives in its own directory named for its **identity**:

```
models/<name>_g<gitSHA>_c<configSHA>/
```

- `<name>` — human label (`--name`)
- `g<gitSHA>` — short git SHA of the **code** that produced it (`SUB0DIFF_CODE_SHA`)
- `c<configSHA>` — FNV-1a over the **BuildTime** config fields (the specialized-build tag)

So the directory name alone tells you which code + which architecture produced the weights, and the same
`--name` + same config deterministically reproduces the same directory (→ resume-by-name just works).

## Produce one (`ch32_mera_train`, GPU)

```bash
cmake --build build-cuda --target ch32_mera_train
./build-cuda/diffusion/chapters/ch32_tree_predictor/ch32_mera_train.exe \
  --model-type mera --device cuda --name tinystories_mera --word-level \
  --corpus data/tinystories_clean.txt --seq_len 128 --steps 20000 --eval_every 500
```

Each model dir contains:

| file | purpose |
|------|---------|
| `config.json` | model arch (model_type, vocab, D, layers, heads, d_ff, seq_len, mera_coarsen/window, mera_gated_pool) |
| `run_config.json` | the FULL resolved run config (every knob) + `_code_sha` + `_config_sha` — the resume source |
| `train_state.json` | dynamic progress (step, best_nelbo, **best_step**, evals_since_best) — **honest** resume of the early-stop history; `best_step` is the checkpoint the server serves |
| `tokenizer/` | `vocab.json` + `merges.txt` |
| `step_*.ckpt` | weights (latest step = newest) |
| `step_*.opt` | Adam moments for the matching step |

## Resume (after a crash, or to train longer)

```bash
# same --name + same config → same dir → continues from the latest step with Adam + best-tracking intact
ch32_mera_train --name tinystories_mera --word-level --corpus data/tinystories_clean.txt --steps 40000
# or point straight at the dir:
ch32_mera_train --ckpt-dir models/tinystories_mera_g<sha>_c<sha> --steps 40000
```

## Serve a trained model (no retraining — the lightweight server loads the dir)

```bash
ch32_viz_server --model-dir models/tinystories_mera_g<sha>_c<sha> --port 8080
#   open http://localhost:8080/tools/viz/
```

## Storage

Checkpoints are binary and can be large, so `*.ckpt` / `*.opt` / `*.bin` are routed through **Git LFS**
(`.gitattributes` here). Run `git lfs install` once before committing a model. By default this folder's
`.gitignore` keeps generated model dirs **out** of git (so throwaway experiments don't bloat the repo) —
force-add the one you want to keep:

```bash
git lfs install
git add -f models/tinystories_mera_g<sha>_c<sha>
```
