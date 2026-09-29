# Sub0Llm performance report

Generated 2026-09-29T20:49:45Z -- label `S1b-cold`

## Gate panel

| Gate | Detail | Value | Status |
|---|---|---|---|
| `G-PERF` | pipelined vs reactive | -32.7% | PASS |
| `G-PERF` | mempage vs reactive | -39.1% | PASS |
| `G-HASH` | neutral-build decode fingerprint unchanged | n/a | n/a |
| `G-SUITE-ENGINE` | sub0_tests assertion count | n/a | n/a |
| `G-SUITE-FRONTEND` | sub0_frontend_tests assertion count | n/a | n/a |
| `G-PARITY` | forward vs forward_one bit-exact | n/a | n/a |
| `G-QUALITY` | logit L2-relative vs the unfused path | n/a | n/a |
| `G-PPL` | decode-path perplexity vs the baseline arm, paired per token | n/a | n/a |
| `G-COMPETITOR` | llama.cpp gap on the same host and model | n/a | n/a |

## Throughput (interleaved)

| Arm | Median s/token | Spread | Runs |
|---|---:|---:|---|
| reactive | 0.396 | 14.4% | 0.438, 0.381, 0.408, 0.385 |
| pipelined | 0.267 | 16.1% | 0.251, 0.251, 0.294, 0.283 |
| mempage | 0.241 | 9.1% | 0.239, 0.244, 0.261, 0.239 |

History: `perf_history.jsonl`. Policy: `docs/OPTIMIZATION_PROCESS.md`.
