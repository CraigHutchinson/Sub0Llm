# Sub0Llm performance report

Generated 2026-09-21T19:48:06Z -- label `smoketest`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Throughput (interleaved)

| Arm | Median s/token | Spread | Runs |
|---|---:|---:|---|
| fused | 1.406 | 0.0% | 1.406 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
