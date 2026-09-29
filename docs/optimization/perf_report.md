# Sub0Llm performance report

Generated 2026-09-29T10:17:17Z -- label `defaults-best`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| default | 14.7035 | 2.6881 | 0.4876 | 6.98 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
