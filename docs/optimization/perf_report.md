# Sub0Llm performance report

Generated 2026-09-22T10:50:49Z -- label `O3-spin`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PERF` | spin vs nospin | -20.2% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Throughput (interleaved)

| Arm | Median s/token | Spread | Runs |
|---|---:|---:|---|
| nospin | 0.267 | 28.5% | 0.267, 0.340, 0.264 |
| spin | 0.213 | 14.1% | 0.205, 0.213, 0.235 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
