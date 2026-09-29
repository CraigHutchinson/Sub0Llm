# Sub0Llm performance report

Generated 2026-09-29T09:54:08Z -- label `O9-verify2`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | super vs base, 2418 tokens | -0.0098 nats/token (95% CI -0.0318..+0.0122), ppl x0.9902, top-1 agree 86.6% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| base | 14.8488 | 2.6979 | 0.4864 | 5.94 |
| super | 14.7035 | 2.6881 | 0.4876 | 6.91 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
