---
name: qwen4-real-axes-configure-recipe
description: "The COMPLETE sub0llm-configure recipe for the real Qwen4 48-layer axes -- partial recipes silently yield a wrong PARAM_FLOATS and a rejected artifact"
metadata:
  node_type: reference
  type: reference
---

To build any tool against the real 48-layer artifact (`D:\ModelWeights\Sub0Llm-Qwen4-full48-bf16\
qwen4_full48_q_bf16.bin`), the configure line must be COMPLETE. Omitting any axis below yields
`PARAM_FLOATS 2570717696 / NUM_PARAMS 626` instead of the correct **`4915107200 / 1074`**, and
`load_model` then rejects the artifact with only *"model was built with a different (constexpr) config"*
— no hint which axis is wrong. This cost real time twice in one session.

```
sub0llm-configure --dmodel 2560 --layers 48 --heads 24 --seq 128 --kv-heads 2 --head-dim 256 \
  --rotary-dim 64 --rope-theta 10000000 --gdn-full-attn-stride 4 --gdn-key-heads 16 \
  --gdn-value-heads 48 --gdn-key-head-dim 128 --gdn-value-head-dim 128 --hc-count 4 --hc-lowrank 320 \
  --qsa-indexer-n-heads 4 --qsa-indexer-kv-heads 1 --qsa-indexer-head-dim 128 \
  --qsa-indexer-budget 2048 --qsa-indexer-compress-ratio 4 --tie-embeddings 0 --d-ff 640 \
  --moe-quant-experts 1 --num-experts 512 --experts-per-tok 10 --prec-param 1 --vocab-exact 248320
```

**Traps specifically:**
- `--rotary-dim 64` is NOT `--head-dim 256`; defaulting it to the head width is wrong and silent.
- `--loop-middle-layers 4` appears in the WP4 handover but is **wrong here** — do not pass it.
- `--gdn-key-head-dim/--gdn-value-head-dim` are 128, not the 256 that aliasing head-dim would give.
- `--tie-embeddings 0` is required (the default is on).
- The five `--qsa-indexer-*` axes must all be passed together.

**Fastest recovery if it drifts again**: difference the generated `sub0_corpus.hpp` of a known-good
build dir (`out/build/wp5c_full48/generated/`) against yours — that is how this was recovered.
Canonical copy lives in `docs/MOE_QUANT_DOT.md` §6h.

**Decode-performance flags (2026-09-22, O1+O2):** append `--moe-quant-dot 1 --decode-gemv-threads 8 --decode-omp-spin 1
--moe-decode-threads 10` for the fast decode path (0.313 s/token steady state). `--profile-phases 1` adds the
phase table to sub0llm-qwen4-forward's output.
Update (O3, 63d22ad): add `--decode-omp-spin 1` -> 0.213 s/token (spin alone -20%, spread halved).

## 2026-09-26 — current recommended decode flags (supersede earlier ones in this note)
`--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 8 --decode-omp-spin 1 --backbone-quant-dot 1`
- `--moe-decode-threads 8` now beats 10 (~3 ms on routed experts) with the native backbone.
- `--backbone-quant-dot 1` is opt-in (quality L2 0.2926 / argmax 3/6 vs bf16 batched path); it needs the
  paired `<model>.bbq` + `.bbq.pair` beside the .bin (present for the ModelWeights artifact).
- Add `--profile-phases 1` for per-phase ms. Result: ~6.6 tok/s median, 6.9 tok/s clean rounds.

## 2026-09-26 (later) — add --moe-row-split 1 (O8, merged 0179e9e)
Recommended: `--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 8 --decode-omp-spin 1 --backbone-quant-dot 1 --moe-row-split 1`
- O8 is bit-exact (ppl identical) and +11% long-run decode (5.16 -> 5.71 tok/s over 2,418 tokens).
- A worktree's neutral build dir may lack newer generated constants: regenerate with
  `sub0llm-configure --corpus D:/Craig/GitHub/Sub0Llm/data/gsm8k.txt --dmodel 196 --layers 11 --heads 7 --kv-heads 7 --seq 256`
  (verified: only the new constant differs). A FAILED build leaves the old test binaries runnable --
  always check the build succeeded before reading suite counts.

## 2026-09-29 — add --backbone-act-super 1 (O9, merged 555fa12/6ca9725)
Recommended: `--moe-quant-dot 1 --decode-gemv-threads 8 --moe-decode-threads 8 --decode-omp-spin 1 --backbone-quant-dot 1 --moe-row-split 1 --backbone-act-super 1`
- Per-256 ActSuper activations for the Q4_K/Q5_K/Q6_K native roles. G-PPL PASS (ppl 14.70 vs 14.85, dNLL -0.0098, CI -0.032..+0.012).
- Long-run decode +14% in my own passes (5.72 -> 6.53 tok/s); the subagent's +28-33% did not reproduce — cite +14%.

## 2026-09-29 (later) — decode flags are now AUTO DEFAULTS
The recipe alone (no decode flags) builds the best decode: MOE_QUANT_DOT, BACKBONE_QUANT_DOT, BACKBONE_ACT_SUPER,
MOE_ROW_SPLIT and DECODE_OMP_SPIN are on, with both thread counts at 8. Pass `--X 0` to override. See [[defaults-are-the-best-options]].

## 2026-11-05 — configuring a build directory from nothing (verified)
`cmake -S . -B out/build/<dir> -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DSUB0_COMPUTE=CPU -DSUB0_NATIVE=ON -DSUB0_BUILD_TESTS=OFF -DSUB0_BUILD_BENCHMARKS=OFF`.
**`-DSUB0_COMPUTE=CPU` is required**: AUTO finds nvcc and fails ("Cannot find compiler 'cl.exe'"). Then
`run_perf_suite.py` configures and builds each arm. Model path override: `SUB0_QWEN4_MODEL`. Full steps: repo `docs/HANDOVER.md`.
