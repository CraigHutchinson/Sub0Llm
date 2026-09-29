# Sub0Llm performance report

Generated 2026-09-26T21:29:26Z -- label `O8-threads-2`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | t12 vs t16, 2418 tokens | +0.0000 nats/token (95% CI +0.0000..+0.0000), ppl x1.0000, top-1 agree 100.0% | PASS |
| `G-PPL` | t8 vs t16, 2418 tokens | +0.0000 nats/token (95% CI +0.0000..+0.0000), ppl x1.0000, top-1 agree 100.0% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| t16 | 14.8488 | 2.6979 | 0.4864 | 4.85 |
| t12 | 14.8488 | 2.6979 | 0.4864 | 5.49 |
| t8 | 14.8488 | 2.6979 | 0.4864 | 5.15 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
