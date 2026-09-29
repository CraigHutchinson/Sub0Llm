# Sub0Llm performance report

Generated 2026-09-29T14:19:34Z -- label `O12`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PPL` | base vs q8fast, 2418 tokens | -0.0103 nats/token (95% CI -0.0302..+0.0096), ppl x0.9897, top-1 agree 86.8% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Perplexity (ppl_blend_v1, decode path)

| Arm | Perplexity | Mean NLL | Top-1 | Decode tok/s |
|---|---:|---:|---:|---:|
| q8fast | 14.8559 | 2.6984 | 0.4888 | 6.66 |
| base | 14.7035 | 2.6881 | 0.4876 | 6.26 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
